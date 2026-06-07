// ZINC CUDA kernels — ports of the Vulkan compute shaders. NVRTC-compiled at
// runtime for the running device's arch. Authored to the ZINC dispatch ABI:
// bound buffers come first (device pointers), then one by-value push struct.
//
// This is the start of the CUDA kernel library that forward_cuda.zig will
// orchestrate. Kernels here are correctness-first ports (clean parallelization);
// the fused/tiled perf variants come later (M3). Each kernel is validated
// numerically against an independent CPU reference (see src/cuda/kernels_test.c).

// ---- shared device helpers --------------------------------------------------

// IEEE half -> float, no <cuda_fp16.h> dependency (keeps NVRTC self-contained).
__device__ __forceinline__ float zinc_half_to_float(unsigned short h) {
    unsigned sign = (unsigned)(h >> 15) & 1u;
    unsigned exp = (unsigned)(h >> 10) & 0x1Fu;
    unsigned mant = (unsigned)h & 0x3FFu;
    unsigned f;
    if (exp == 0u) {
        if (mant == 0u) {
            f = sign << 31;
        } else {
            // subnormal: normalize
            int e = 1;
            while ((mant & 0x400u) == 0u) { mant <<= 1; e--; }
            mant &= 0x3FFu;
            f = (sign << 31) | ((unsigned)(127 - 15 + e) << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        f = (sign << 31) | (0xFFu << 23) | (mant << 13);
    } else {
        f = (sign << 31) | ((exp - 15u + 127u) << 23) | (mant << 13);
    }
    return __int_as_float((int)f);
}

// GGML Q4_K 6-bit scale/min unpack (j in 0..7), canonical llama.cpp form.
__device__ __forceinline__ void zinc_q4k_scale_min(int j, const unsigned char* q,
                                                    unsigned char* d, unsigned char* m) {
    if (j < 4) {
        *d = q[j] & 63u;
        *m = q[j + 4] & 63u;
    } else {
        *d = (q[j + 4] & 0xFu) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

__device__ __forceinline__ float zinc_warp_reduce_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

// Block reduce — valid result in thread 0. blockDim must be a multiple of 32.
__device__ __forceinline__ float zinc_block_reduce_sum(float v) {
    __shared__ float sh[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    v = zinc_warp_reduce_sum(v);
    if (lane == 0) sh[wid] = v;
    __syncthreads();
    int nwarps = (blockDim.x + 31) >> 5;
    v = (threadIdx.x < nwarps) ? sh[lane] : 0.0f;
    if (wid == 0) v = zinc_warp_reduce_sum(v);
    return v;
}

__device__ __forceinline__ float zinc_warp_reduce_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float zinc_block_reduce_max(float v) {
    __shared__ float shm[32];
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    v = zinc_warp_reduce_max(v);
    if (lane == 0) shm[wid] = v;
    __syncthreads();
    int nwarps = (blockDim.x + 31) >> 5;
    v = (threadIdx.x < nwarps) ? shm[lane] : -3.4e38f;
    if (wid == 0) v = zinc_warp_reduce_max(v);
    return v;
}

// Block sum reduction with result broadcast to ALL threads (blockDim mult of 32).
__device__ __forceinline__ float zinc_block_reduce_sum_all(float v) {
    __shared__ float shs[32];
    __shared__ float bc;
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    v = zinc_warp_reduce_sum(v);
    if (lane == 0) shs[wid] = v;
    __syncthreads();
    int nwarps = (blockDim.x + 31) >> 5;
    v = (threadIdx.x < nwarps) ? shs[lane] : 0.0f;
    if (wid == 0) v = zinc_warp_reduce_sum(v);
    if (threadIdx.x == 0) bc = v;
    __syncthreads();
    float r = bc;
    __syncthreads();
    return r;
}

// ---- rms_norm (port of rms_norm_mul.comp) -----------------------------------
// y = weight * (x / sqrt(mean(x^2) + eps)). One block per token.
struct RmsPush { unsigned N; float eps; };

extern "C" __global__ void rms_norm(const float* x, const float* w, float* y, RmsPush pc) {
    unsigned token = blockIdx.x;
    const float* xt = x + (size_t)token * pc.N;
    float* yt = y + (size_t)token * pc.N;

    float ss = 0.0f;
    for (unsigned i = threadIdx.x; i < pc.N; i += blockDim.x) {
        float v = xt[i];
        ss += v * v;
    }
    ss = zinc_block_reduce_sum(ss);

    __shared__ float rms_inv_sh;
    if (threadIdx.x == 0) rms_inv_sh = rsqrtf(ss / (float)pc.N + pc.eps);
    __syncthreads();
    float rinv = rms_inv_sh;

    for (unsigned i = threadIdx.x; i < pc.N; i += blockDim.x) {
        yt[i] = w[i] * (xt[i] * rinv);
    }
}

// ---- dmmv_q4k (port of dmmv_q4k.comp) ---------------------------------------
// y[row] = sum_k dequant(W[row][k]) * x[k], W in Q4_K. One block per output row.
// Offsets are in BYTES (matching the Vulkan push constants); acc_mode 1 => y +=.
struct DmmvPush { unsigned M, K, a_offset, x_offset, y_offset, acc_mode; };

extern "C" __global__ void dmmv_q4k(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) {
    unsigned row = blockIdx.x;
    if (row >= pc.M) return;
    unsigned bpr = pc.K >> 8;                       // blocks per row (K / 256)
    const unsigned* arow = a_u32 + (pc.a_offset >> 2) + (size_t)row * bpr * 36u;
    const float* xrow = x + (pc.x_offset >> 2);

    float sum = 0.0f;
    for (unsigned e = threadIdx.x; e < pc.K; e += blockDim.x) {
        unsigned b = e >> 8;                          // which 256-elem block
        unsigned within = e & 255u;                  // 0..255 in block
        const unsigned* blk = arow + (size_t)b * 36u;
        unsigned d_dmin = blk[0];
        float d = zinc_half_to_float((unsigned short)(d_dmin & 0xFFFFu));
        float dmin = zinc_half_to_float((unsigned short)(d_dmin >> 16));
        const unsigned char* scales = (const unsigned char*)(blk + 1);   // 12 bytes
        const unsigned char* qs = (const unsigned char*)(blk + 4);       // 128 bytes

        unsigned chunk = within >> 6;                // 0..3 (64-elem chunk)
        unsigned wc = within & 63u;
        unsigned half = wc >> 5;                      // 0 low nibble, 1 high nibble
        unsigned l = wc & 31u;                        // 0..31
        unsigned char sc, mn;
        zinc_q4k_scale_min((int)(chunk * 2u + half), scales, &sc, &mn);
        unsigned char qb = qs[chunk * 32u + l];
        unsigned nib = (half == 0u) ? (qb & 0xFu) : (unsigned)(qb >> 4);
        float wv = d * (float)sc * (float)nib - dmin * (float)mn;
        sum += wv * xrow[e];
    }

    sum = zinc_block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        unsigned yi = (pc.y_offset >> 2) + row;
        if (pc.acc_mode != 0u) y[yi] += sum; else y[yi] = sum;
    }
}

// ---- swiglu (port of swiglu.comp) -------------------------------------------
// y = silu(gate) * up, silu(x) = x / (1 + exp(-x)). One element per thread.
struct SwigluPush { unsigned N; };

extern "C" __global__ void swiglu(const float* gate, const float* up, float* y, SwigluPush pc) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pc.N) return;
    float g = gate[idx];
    float silu = g / (1.0f + expf(-g));
    y[idx] = silu * up[idx];
}

// ---- scale_accumulate (port of scale_accumulate.comp) -----------------------
// a[i] += scale * b[i]  (residual add). a is read-write.
struct ScaleAccPush { unsigned N; float scale; };

extern "C" __global__ void scale_accumulate(float* a, const float* b, ScaleAccPush pc) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pc.N) return;
    a[idx] += pc.scale * b[idx];
}

// ---- sigmoid_scale_acc (port of sigmoid_scale_acc.comp) ---------------------
// a[i] += sigmoid(c[0]) * b[i]  (MoE shared-expert sigmoid gating). a read-write.
struct SigmoidAccPush { unsigned N; };

extern "C" __global__ void sigmoid_scale_acc(float* a, const float* b, const float* c, SigmoidAccPush pc) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pc.N) return;
    float gate = 1.0f / (1.0f + expf(-c[0]));
    a[idx] += gate * b[idx];
}

// ---- dmmv_f32 (port of dmmv_f32.comp) ---------------------------------------
// y[row] = sum_k W[row][k] * x[k], W f32. One block per output row. (Router,
// SSM alpha/beta projections.) Reuses DmmvPush; offsets in BYTES.
extern "C" __global__ void dmmv_f32(const float* w, const float* x, float* y, DmmvPush pc) {
    unsigned row = blockIdx.x;
    if (row >= pc.M) return;
    const float* wrow = w + (pc.a_offset >> 2) + (size_t)row * pc.K;
    const float* xrow = x + (pc.x_offset >> 2);
    float sum = 0.0f;
    for (unsigned k = threadIdx.x; k < pc.K; k += blockDim.x) sum += wrow[k] * xrow[k];
    sum = zinc_block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        unsigned yi = (pc.y_offset >> 2) + row;
        if (pc.acc_mode != 0u) y[yi] += sum; else y[yi] = sum;
    }
}

// ---- dmmv_q8_0 (port of dmmv_q8_0.comp) -------------------------------------
// y[row] = sum_k dequant(W[row][k]) * x[k], W in Q8_0 (34-byte blocks: f16 d +
// 32 int8). dequant: d * qs[i]. (SSM in/out proj, shared-expert weights.)
extern "C" __global__ void dmmv_q8_0(const unsigned char* a, const float* x, float* y, DmmvPush pc) {
    unsigned row = blockIdx.x;
    if (row >= pc.M) return;
    unsigned bpr = pc.K >> 5;                          // blocks per row (K / 32)
    const unsigned char* arow = a + pc.a_offset + (size_t)row * bpr * 34u;
    const float* xrow = x + (pc.x_offset >> 2);
    float sum = 0.0f;
    for (unsigned e = threadIdx.x; e < pc.K; e += blockDim.x) {
        unsigned blk = e >> 5;                          // which 32-elem block
        unsigned i = e & 31u;                           // 0..31 in block
        const unsigned char* blkp = arow + (size_t)blk * 34u;
        unsigned d_bits = (unsigned)blkp[0] | ((unsigned)blkp[1] << 8);
        float d = zinc_half_to_float((unsigned short)d_bits);
        signed char q = (signed char)blkp[2u + i];
        sum += (d * (float)q) * xrow[e];
    }
    sum = zinc_block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        unsigned yi = (pc.y_offset >> 2) + row;
        if (pc.acc_mode != 0u) y[yi] += sum; else y[yi] = sum;
    }
}

// ---- dmmv_q5k (port of dmmv_q5k.comp) ---------------------------------------
// Q5_K block (256 elems, 176 bytes): [0..3] d+dmin(f16); [4..15] 6-bit scales;
// [16..47] qh (1 high bit/elem); [48..175] qs (4-bit low). 5-bit q = nibble +
// (qh_bit<<4); value = d*scale*q - dmin*min (same get_scale_min_k4 as Q4_K).
extern "C" __global__ void dmmv_q5k(const unsigned char* a, const float* x, float* y, DmmvPush pc) {
    unsigned row = blockIdx.x;
    if (row >= pc.M) return;
    unsigned bpr = pc.K >> 8;
    const unsigned char* arow = a + pc.a_offset + (size_t)row * bpr * 176u;
    const float* xrow = x + (pc.x_offset >> 2);
    float sum = 0.0f;
    for (unsigned e = threadIdx.x; e < pc.K; e += blockDim.x) {
        unsigned b = e >> 8, within = e & 255u;
        const unsigned char* blk = arow + (size_t)b * 176u;
        float d = zinc_half_to_float((unsigned short)((unsigned)blk[0] | ((unsigned)blk[1] << 8)));
        float dmin = zinc_half_to_float((unsigned short)((unsigned)blk[2] | ((unsigned)blk[3] << 8)));
        const unsigned char* scales = blk + 4;     // 12 bytes
        const unsigned char* qh = blk + 16;        // 32 bytes
        const unsigned char* qs = blk + 48;        // 128 bytes
        unsigned chunk = within >> 6;              // 0..3
        unsigned half = (within & 63u) >> 5;       // 0..1
        unsigned l = within & 31u;                 // 0..31
        unsigned char ql = qs[chunk * 32u + l];
        unsigned nib = (half == 0u) ? (ql & 0xFu) : (unsigned)(ql >> 4);
        unsigned bit = (qh[l] >> (2u * chunk + half)) & 1u;
        unsigned q5 = nib + (bit ? 16u : 0u);
        unsigned char sc, mn;
        zinc_q4k_scale_min((int)(chunk * 2u + half), scales, &sc, &mn);
        sum += (d * (float)sc * (float)q5 - dmin * (float)mn) * xrow[e];
    }
    sum = zinc_block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        unsigned yi = (pc.y_offset >> 2) + row;
        if (pc.acc_mode != 0u) y[yi] += sum; else y[yi] = sum;
    }
}

// ---- dmmv_q6k (port of dmmv_q6k.comp) ---------------------------------------
// Q6_K block (256 elems, 210 bytes): [0..127] ql (low 4 bits); [128..191] qh
// (high 2 bits); [192..207] scales (16 int8); [208..209] d(f16). 6-bit q =
// (ql_nibble | qh_bits<<4); value = d * int8_scale * (q - 32).
extern "C" __global__ void dmmv_q6k(const unsigned char* a, const float* x, float* y, DmmvPush pc) {
    unsigned row = blockIdx.x;
    if (row >= pc.M) return;
    unsigned bpr = pc.K >> 8;
    const unsigned char* arow = a + pc.a_offset + (size_t)row * bpr * 210u;
    const float* xrow = x + (pc.x_offset >> 2);
    float sum = 0.0f;
    for (unsigned e = threadIdx.x; e < pc.K; e += blockDim.x) {
        unsigned b = e >> 8, within = e & 255u;
        const unsigned char* blk = arow + (size_t)b * 210u;
        float d = zinc_half_to_float((unsigned short)((unsigned)blk[208] | ((unsigned)blk[209] << 8)));
        unsigned half = within >> 7;               // 0..1 (128-elem half)
        unsigned wh = within & 127u;
        unsigned l = wh & 31u;                     // 0..31
        unsigned group = wh >> 5;                  // 0..3 (q1..q4)
        const unsigned char* ql = blk + (size_t)half * 64u;
        const unsigned char* qh = blk + 128u + (size_t)half * 32u;
        const signed char* sc = (const signed char*)(blk + 192u + (size_t)half * 8u);
        unsigned is = l >> 4;                       // 0 or 1
        unsigned qhb = qh[l];
        unsigned q; unsigned sci;
        if (group == 0u) { q = (ql[l] & 0xFu) | (((qhb >> 0) & 3u) << 4); sci = is + 0u; }
        else if (group == 1u) { q = (ql[l + 32u] & 0xFu) | (((qhb >> 2) & 3u) << 4); sci = is + 2u; }
        else if (group == 2u) { q = (ql[l] >> 4) | (((qhb >> 4) & 3u) << 4); sci = is + 4u; }
        else { q = (ql[l + 32u] >> 4) | (((qhb >> 6) & 3u) << 4); sci = is + 6u; }
        sum += (d * (float)sc[sci] * ((float)q - 32.0f)) * xrow[e];
    }
    sum = zinc_block_reduce_sum(sum);
    if (threadIdx.x == 0) {
        unsigned yi = (pc.y_offset >> 2) + row;
        if (pc.acc_mode != 0u) y[yi] += sum; else y[yi] = sum;
    }
}

// ---- softmax_topk (port of softmax_topk.comp) -------------------------------
// MoE router: pick top-k experts from logits, output [ids(k) | renorm-softmax
// weights(k)] (weights as float-bits). Single block of 64 threads, shared-mem
// winner select (no ballot). out[0..k-1]=ids, out[k..2k-1]=weight bits.
struct TopkPush { unsigned n_experts; unsigned k; };

extern "C" __global__ void softmax_topk(const float* logits, unsigned* out, TopkPush pc) {
    __shared__ float s_logits[256];
    __shared__ float s_val[64];
    __shared__ unsigned s_idx[64];
    const float NEG_INF = __int_as_float(0xff800000);
    unsigned tid = threadIdx.x;
    for (unsigned i = tid; i < pc.n_experts; i += 64) s_logits[i] = logits[i];
    __syncthreads();
    for (unsigned ki = 0; ki < pc.k; ki++) {
        float best = NEG_INF; unsigned bidx = 0;
        for (unsigned i = tid; i < pc.n_experts; i += 64)
            if (s_logits[i] > best) { best = s_logits[i]; bidx = i; }
        s_val[tid] = best; s_idx[tid] = bidx;
        __syncthreads();
        if (tid == 0) {
            float gb = NEG_INF; unsigned gi = 0;
            for (unsigned t = 0; t < 64; t++)
                if (s_val[t] > gb) { gb = s_val[t]; gi = s_idx[t]; }
            out[ki] = gi;
            out[pc.k + ki] = __float_as_uint(gb);
            s_logits[gi] = NEG_INF;
        }
        __syncthreads();
    }
    if (tid == 0) {
        float maxl = NEG_INF;
        for (unsigned i = 0; i < pc.k; i++) maxl = fmaxf(maxl, __uint_as_float(out[pc.k + i]));
        float wsum = 0.0f;
        for (unsigned i = 0; i < pc.k; i++) {
            float w = expf(__uint_as_float(out[pc.k + i]) - maxl);
            out[pc.k + i] = __float_as_uint(w); wsum += w;
        }
        float inv = (wsum > 0.0f) ? 1.0f / wsum : 0.0f;
        for (unsigned i = 0; i < pc.k; i++)
            out[pc.k + i] = __float_as_uint(__uint_as_float(out[pc.k + i]) * inv);
    }
}

// ---- rope (port of rope_fused.comp) -----------------------------------------
// RoPE with partial rotation (IMRoPE): rotate first rope_dim dims/head in pairs,
// copy the rest. freq from inv_freq buffer when freq_base_bits==0, else computed.
// One block per head.
struct RopePush { unsigned stride, rope_dim, n_heads, position, freq_base_bits, attn_scale_bits; };

extern "C" __global__ void rope(const float* x, float* y, const float* inv_freq, RopePush pc) {
    unsigned tid = threadIdx.x;
    unsigned head = blockIdx.x;
    unsigned base = head * pc.stride;
    unsigned half_rot = pc.rope_dim >> 1;
    float freq_base = __uint_as_float(pc.freq_base_bits);
    float attn_scale = pc.attn_scale_bits != 0u ? __uint_as_float(pc.attn_scale_bits) : 1.0f;
    bool use_buf = (pc.freq_base_bits == 0u);
    for (unsigned i = tid; i < half_rot; i += blockDim.x) {
        float xi = x[base + i];
        float xih = x[base + i + half_rot];
        float freq_i = use_buf ? inv_freq[i]
                               : (1.0f / powf(freq_base, (float)(2u * i) / (float)pc.rope_dim));
        float theta = (float)pc.position * freq_i;
        float ct = cosf(theta) * attn_scale;
        float st = sinf(theta) * attn_scale;
        y[base + i] = xi * ct - xih * st;
        y[base + i + half_rot] = xi * st + xih * ct;
    }
    for (unsigned i = pc.rope_dim + tid; i < pc.stride; i += blockDim.x)
        y[base + i] = x[base + i];
}

// ---- argmax (port of argmax.comp) -------------------------------------------
// Greedy sample: token_id = argmax_i logits[i] (lowest index wins ties).
// Single-block reduction (the .comp's two-phase split is a perf detail for huge N).
struct ArgmaxPush { unsigned N; };

extern "C" __global__ void argmax(const float* logits, unsigned* token_id, ArgmaxPush pc) {
    __shared__ float s_val[32];
    __shared__ unsigned s_idx[32];
    unsigned tid = threadIdx.x;
    float best = -3.4e38f; unsigned bidx = 0u;
    for (unsigned i = tid; i < pc.N; i += blockDim.x) {
        float v = logits[i];
        if (v > best) { best = v; bidx = i; }
    }
    unsigned lane = tid & 31u, wid = tid >> 5;
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_down_sync(0xffffffffu, best, o);
        unsigned oi = __shfl_down_sync(0xffffffffu, bidx, o);
        if (ov > best || (ov == best && oi < bidx)) { best = ov; bidx = oi; }
    }
    if (lane == 0u) { s_val[wid] = best; s_idx[wid] = bidx; }
    __syncthreads();
    if (tid == 0u) {
        float gb = s_val[0]; unsigned gi = s_idx[0];
        unsigned nwarps = (blockDim.x + 31u) >> 5;
        for (unsigned w = 1u; w < nwarps; w++)
            if (s_val[w] > gb || (s_val[w] == gb && s_idx[w] < gi)) { gb = s_val[w]; gi = s_idx[w]; }
        *token_id = gi;
    }
}

// ---- moe_weighted_acc (port of moe_weighted_acc.comp) -----------------------
// a[i] += sum_j weight_j * b[j*src_stride + i]. Weights from the softmax_topk
// routing buffer (routing[n_used .. 2*n_used-1], as float bits).
struct MoeAccPush { unsigned N, n_used, src_stride; };

extern "C" __global__ void moe_weighted_acc(float* a, const float* b, const unsigned* routing, MoeAccPush pc) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pc.N) return;
    float sum = 0.0f;
    for (unsigned j = 0; j < pc.n_used; j++) {
        float w = __uint_as_float(routing[pc.n_used + j]);
        sum += w * b[(size_t)j * pc.src_stride + i];
    }
    a[i] += sum;
}

// ---- ssm_conv1d (port of ssm_conv1d.comp) -----------------------------------
// Depthwise causal 1D conv (d_conv taps) + SiLU, via a circular state buffer.
// One thread per channel. Updates `state` in place (writes current_input into
// the state_offset slot). conv_kernel is f16 or f32 per kernel_is_f16.
struct ConvPush { unsigned conv_channels, d_conv, kernel_is_f16, state_offset; };

extern "C" __global__ void ssm_conv1d(const float* current_input, const unsigned char* conv_kernel,
                                      float* state, float* out_data, ConvPush pc) {
    unsigned ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= pc.conv_channels) return;
    unsigned d_conv_1 = pc.d_conv - 1u;
    float ci = current_input[ch];
    float sum = 0.0f;
    for (unsigned ki = 0; ki < pc.d_conv; ki++) {
        unsigned k_idx = ch * pc.d_conv + ki;
        float kw = (pc.kernel_is_f16 != 0u)
                       ? zinc_half_to_float(((const unsigned short*)conv_kernel)[k_idx])
                       : ((const float*)conv_kernel)[k_idx];
        float sv;
        if (ki < d_conv_1) {
            unsigned slot = pc.state_offset + ki;
            if (slot >= d_conv_1) slot -= d_conv_1;
            sv = state[(size_t)slot * pc.conv_channels + ch];
        } else {
            sv = ci;
        }
        sum += kw * sv;
    }
    out_data[ch] = sum / (1.0f + expf(-sum));            // SiLU
    state[(size_t)pc.state_offset * pc.conv_channels + ch] = ci;
}

// ---- ssm_gated_norm (port of ssm_gated_norm.comp) ---------------------------
// Per head: out = (o / rms(o)) * norm_weight * silu(z). One block per head.
struct GatedNormPush { unsigned d_inner, dt_rank, head_v_dim, d_state, norm_per_head; };

extern "C" __global__ void ssm_gated_norm(const float* o, const float* z, const float* norm_weight,
                                          float* out, GatedNormPush pc) {
    unsigned h = blockIdx.x;
    unsigned base = h * pc.head_v_dim;
    float ss = 0.0f;
    for (unsigned i = threadIdx.x; i < pc.head_v_dim; i += blockDim.x) { float v = o[base + i]; ss += v * v; }
    ss = zinc_block_reduce_sum(ss);
    __shared__ float rms_inv_sh;
    if (threadIdx.x == 0) rms_inv_sh = rsqrtf(ss / (float)pc.head_v_dim + 1e-6f);
    __syncthreads();
    float rinv = rms_inv_sh;
    for (unsigned i = threadIdx.x; i < pc.head_v_dim; i += blockDim.x) {
        float nv = o[base + i] * rinv;
        unsigned norm_idx = (pc.norm_per_head != 0u) ? (base + i) : (i % pc.d_state);
        nv *= norm_weight[norm_idx];
        float zv = z[base + i];
        out[base + i] = nv * (zv / (1.0f + expf(-zv)));
    }
}

// ---- kv_cache_write (port of kv_cache_write.comp) ---------------------------
// Append K and V vectors into their caches at dst_offset (= physical_token*kv_dim).
struct KvWritePush { unsigned kv_dim, dst_offset; };

extern "C" __global__ void kv_cache_write(const float* k_src, float* k_dst, const float* v_src, float* v_dst, KvWritePush pc) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < pc.kv_dim) {
        k_dst[pc.dst_offset + i] = k_src[i];
        v_dst[pc.dst_offset + i] = v_src[i];
    }
}

// ---- naive_attention (decode: softmax(QK^T)V, GQA + attention sinks) --------
// Correctness-first decode attention (the flash/paged version is M3). One block
// per query head, contiguous KV cache [seq_len, n_kv_heads, head_dim]. Dynamic
// shared holds seq_len scores. Sink: sinks[sink_offset+head] (NaN = no sink).
struct AttnPush { unsigned head_dim, n_heads, n_kv_heads, seq_len, attn_scale_bits, sink_offset; };

extern "C" __global__ void naive_attention(const float* q, const float* k, const float* v,
                                           const float* sinks, float* out, AttnPush pc) {
    extern __shared__ float s_scores[];           // size = seq_len floats (dynamic)
    __shared__ float s_m, s_rescale, s_inv;
    unsigned head = blockIdx.x;
    unsigned tid = threadIdx.x;
    unsigned hd = pc.head_dim;
    unsigned kv_head = head / (pc.n_heads / pc.n_kv_heads);
    const float* qh = q + (size_t)head * hd;
    float scale = pc.attn_scale_bits != 0u ? __uint_as_float(pc.attn_scale_bits) : rsqrtf((float)hd);

    // Pass 1: scores = scale * (q . k_i), track max.
    float lmax = -3.4e38f;
    for (unsigned i = tid; i < pc.seq_len; i += blockDim.x) {
        const float* ki = k + ((size_t)i * pc.n_kv_heads + kv_head) * hd;
        float dot = 0.0f;
        for (unsigned d = 0; d < hd; d++) dot += qh[d] * ki[d];
        float score = dot * scale;
        s_scores[i] = score;
        lmax = fmaxf(lmax, score);
    }
    lmax = zinc_block_reduce_max(lmax);
    if (tid == 0) s_m = lmax;
    __syncthreads();
    float m = s_m;

    // Pass 2: e_i = exp(score_i - m), sum.
    float lsum = 0.0f;
    for (unsigned i = tid; i < pc.seq_len; i += blockDim.x) {
        float e = expf(s_scores[i] - m);
        s_scores[i] = e;
        lsum += e;
    }
    lsum = zinc_block_reduce_sum(lsum);
    if (tid == 0) {
        float sum = lsum, rescale = 1.0f, final_sum = lsum;
        float sink_val = sinks[pc.sink_offset + head];
        if (sink_val == sink_val) {   // sink present (NaN == NaN is false)
            float sink_max = fmaxf(m, sink_val);
            rescale = (sum > 0.0f) ? expf(m - sink_max) : 0.0f;
            final_sum = sum * rescale + expf(sink_val - sink_max);
        }
        s_rescale = rescale;
        s_inv = (final_sum > 0.0f) ? 1.0f / final_sum : 0.0f;
    }
    __syncthreads();
    float rescale = s_rescale, inv = s_inv;

    // Pass 3: out[d] = (sum_i e_i * V[i,d]) * rescale * inv.
    for (unsigned d = tid; d < hd; d += blockDim.x) {
        float acc = 0.0f;
        for (unsigned i = 0; i < pc.seq_len; i++)
            acc += s_scores[i] * v[((size_t)i * pc.n_kv_heads + kv_head) * hd + d];
        out[(size_t)head * hd + d] = acc * rescale * inv;
    }
}

// ---- ssm_delta_net (port of ssm_delta_net.comp) -----------------------------
// Gated delta-net autoregressive selective scan. One block per (head,row), one
// state column per thread (blockDim == head_v_dim, must be a multiple of 32).
// State [dt_rank][head_v_dim][head_v_dim] carries across the n_tok token loop:
// per token — L2-norm Q/K per group, gate=exp(softplus(alpha+dt_bias)*ssm_a),
// beta=sigmoid; decay state*=gate; sk=<state,k>; d=beta*(v-sk); state+=k*d;
// readout o=<state,q>. Unfused scalar form (perf/coopmat fusion is M3).
struct DeltaNetPush {
    unsigned d_inner, dt_rank, head_v_dim, d_state, n_group;
    unsigned ssm_a_is_f16, dt_bias_is_f16, has_dt_bias, has_ssm_a;
    unsigned n_tok, conv_stride_tok, ab_stride_tok, y_stride_tok;
};

extern "C" __global__ void ssm_delta_net(
    const float* conv_out, const unsigned char* dt_bias, const float* alpha,
    const float* beta, const unsigned char* ssm_a, float* state, float* out_data,
    DeltaNetPush pc) {
    unsigned h = blockIdx.x;
    unsigned row = blockIdx.y;
    unsigned col = threadIdx.x;                  // 0..head_v_dim-1
    if (h >= pc.dt_rank || row >= pc.head_v_dim) return;
    unsigned hv = pc.head_v_dim;
    unsigned qk_dim = pc.d_state * pc.n_group;
    unsigned k_len = (hv < pc.d_state) ? hv : pc.d_state;
    size_t row_base = ((size_t)h * hv + row) * hv;
    float rs = state[row_base + col];            // state[h][row][col]

    float dt_bias_val = 0.0f;
    if (pc.has_dt_bias != 0u)
        dt_bias_val = pc.dt_bias_is_f16 ? zinc_half_to_float(((const unsigned short*)dt_bias)[h])
                                        : ((const float*)dt_bias)[h];
    float ssm_a_val = 0.0f;
    if (pc.has_ssm_a != 0u)
        ssm_a_val = pc.ssm_a_is_f16 ? zinc_half_to_float(((const unsigned short*)ssm_a)[h])
                                    : ((const float*)ssm_a)[h];
    unsigned k_hi = (pc.n_group == pc.dt_rank) ? h : (h % pc.n_group);

    __shared__ float s_g, s_b;

    for (unsigned t = 0; t < pc.n_tok; t++) {
        unsigned conv_base = t * pc.conv_stride_tok;
        unsigned q_off = conv_base + k_hi * pc.d_state;
        unsigned k_off = conv_base + qk_dim + k_hi * pc.d_state;
        unsigned v_off = conv_base + 2u * qk_dim + h * hv;

        // L2-normalize Q/K per group (sum-sq reduced across cols), scale Q.
        float qi = (col < k_len) ? conv_out[q_off + col] : 0.0f;
        float ki = (col < k_len) ? conv_out[k_off + col] : 0.0f;
        float sumq = zinc_block_reduce_sum_all(qi * qi);
        float sumk = zinc_block_reduce_sum_all(ki * ki);
        float sq = qi * (rsqrtf(fmaxf(sumq, 1e-12f)) / sqrtf((float)pc.d_state));
        float skv = ki * rsqrtf(fmaxf(sumk, 1e-12f));

        if (col == 0) {
            float a = alpha[t * pc.ab_stride_tok + h] + dt_bias_val;
            float sp = logf(1.0f + expf(a));               // softplus
            float gate_val = (pc.has_ssm_a != 0u) ? (sp * ssm_a_val) : (-sp);
            s_g = expf(gate_val);
            s_b = 1.0f / (1.0f + expf(-beta[t * pc.ab_stride_tok + h]));
        }
        __syncthreads();
        float g = s_g, b = s_b;

        float v_val = conv_out[v_off + row];
        rs *= g;                                            // decay
        float sk = zinc_block_reduce_sum_all((col < k_len) ? rs * skv : 0.0f);
        float d = b * (v_val - sk);
        if (col < k_len) rs += skv * d;                     // rank-1 update
        float o = zinc_block_reduce_sum_all((col < k_len) ? rs * sq : 0.0f);  // readout
        if (col == 0) out_data[t * pc.y_stride_tok + h * hv + row] = o;
        __syncthreads();
    }
    state[row_base + col] = rs;                             // write final state
}
