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

// ---- rms_norm_residual ------------------------------------------------------
// Fused gemma post-norm + residual add: hidden[i] += w[i] * (x[i] / rms(x)).
// Collapses the (rms_norm -> scale_accumulate) pair used after attention and
// the FFN in the gemma decode path (the residual scale is always 1.0 there)
// into one launch, also dropping a full n_embd write+read round-trip of the
// normalized vector. `hidden` is the read-write residual accumulator. One
// block per token (decode: token=1).
extern "C" __global__ void rms_norm_residual(const float* x, const float* w, float* hidden, RmsPush pc) {
    unsigned token = blockIdx.x;
    const float* xt = x + (size_t)token * pc.N;
    float* ht = hidden + (size_t)token * pc.N;

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
        ht[i] += w[i] * (xt[i] * rinv);
    }
}

// ---- rms_norm_residual_scale ------------------------------------------------
// rms_norm_residual that also applies the gemma per-layer output scale s[0] to
// the whole residual stream: hidden[i] = s[0] * (hidden[i] + w[i]*x[i]/rms(x)).
// Folds the standalone scalar_mul (blk.N.layer_output_scale.weight) into the
// post-ffn norm+residual on the dense gemma path, removing one tiny launch + one
// command submission per layer. Bit-identical to the two-kernel path: the
// residual add is the same FMA producing the same f32 value scalar_mul would
// have re-loaded, then a plain multiply by s[0].
extern "C" __global__ void rms_norm_residual_scale(const float* x, const float* w, float* hidden, const float* s, RmsPush pc) {
    unsigned token = blockIdx.x;
    const float* xt = x + (size_t)token * pc.N;
    float* ht = hidden + (size_t)token * pc.N;

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
    float scale = s[0];

    for (unsigned i = threadIdx.x; i < pc.N; i += blockDim.x) {
        ht[i] = (ht[i] + w[i] * (xt[i] * rinv)) * scale;
    }
}

// ---- rms_norm_residual_norm / rms_norm_residual_scale_norm ------------------
// Fuse a block's INPUT rms_norm into the PRECEDING block's output norm+residual.
// On the dense gemma path each layer runs four single-block n_embd reductions:
//   pre-attn rms_norm -> ... -> post-attn rms_norm_residual (writes hidden) ->
//   pre-ffn  rms_norm -> ... -> post-ffn  rms_norm_residual_scale (writes hidden)
// A post-norm-residual's `hidden` is exactly the input the very next pre-norm
// reads, and both are ONE-block (grid {1,1,1}) reductions over the same n_embd
// vector — so the next pre-norm can run in the SAME launch, right after the
// residual add (a __syncthreads makes the just-written `hidden` visible to the
// phase-2 reduction). This removes one tiny launch per norm boundary:
//   post-attn-residual + pre-ffn-norm  (within the layer)
//   post-ffn-residual  + pre-attn-norm (across the layer boundary, into the
//                                        NEXT layer's `attn_norm.weight`)
// = ~2 launches/layer on the dense gemma-31b decode path (~119/token over 60
// layers; only layer 0's pre-attn norm and the last layer's post-ffn stay
// standalone). BIT-IDENTICAL to the two-kernel path: phase 1 is the exact
// rms_norm_residual[_scale] arithmetic (same FMA, same reduction), phase 2 is
// the exact rms_norm arithmetic re-reading `hidden` from global — the same
// values the standalone pre-norm would have read. The intervening __syncthreads
// barriers also make reusing zinc_block_reduce_sum's shared scratch race-free.
extern "C" __global__ void rms_norm_residual_norm(
    const float* x, const float* w_post, float* hidden,
    const float* w_pre, float* pre_out, RmsPush pc) {
    unsigned token = blockIdx.x;
    const float* xt = x + (size_t)token * pc.N;
    float* ht = hidden + (size_t)token * pc.N;
    float* pt = pre_out + (size_t)token * pc.N;

    // phase 1: post-norm + residual (identical to rms_norm_residual)
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
        ht[i] += w_post[i] * (xt[i] * rinv);
    }
    __syncthreads(); // hidden fully written + visible before phase-2 reduction

    // phase 2: pre-norm of the updated residual (identical to rms_norm)
    float ss2 = 0.0f;
    for (unsigned i = threadIdx.x; i < pc.N; i += blockDim.x) {
        float v = ht[i];
        ss2 += v * v;
    }
    ss2 = zinc_block_reduce_sum(ss2);
    __shared__ float rms_inv_sh2;
    if (threadIdx.x == 0) rms_inv_sh2 = rsqrtf(ss2 / (float)pc.N + pc.eps);
    __syncthreads();
    float rinv2 = rms_inv_sh2;
    for (unsigned i = threadIdx.x; i < pc.N; i += blockDim.x) {
        pt[i] = w_pre[i] * (ht[i] * rinv2);
    }
}

extern "C" __global__ void rms_norm_residual_scale_norm(
    const float* x, const float* w_post, float* hidden, const float* s,
    const float* w_pre, float* pre_out, RmsPush pc) {
    unsigned token = blockIdx.x;
    const float* xt = x + (size_t)token * pc.N;
    float* ht = hidden + (size_t)token * pc.N;
    float* pt = pre_out + (size_t)token * pc.N;

    // phase 1: post-norm + residual + per-layer output scale (== rms_norm_residual_scale)
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
    float scale = s[0];
    for (unsigned i = threadIdx.x; i < pc.N; i += blockDim.x) {
        ht[i] = (ht[i] + w_post[i] * (xt[i] * rinv)) * scale;
    }
    __syncthreads(); // hidden fully written + visible before phase-2 reduction

    // phase 2: pre-norm of the updated residual (identical to rms_norm)
    float ss2 = 0.0f;
    for (unsigned i = threadIdx.x; i < pc.N; i += blockDim.x) {
        float v = ht[i];
        ss2 += v * v;
    }
    ss2 = zinc_block_reduce_sum(ss2);
    __shared__ float rms_inv_sh2;
    if (threadIdx.x == 0) rms_inv_sh2 = rsqrtf(ss2 / (float)pc.N + pc.eps);
    __syncthreads();
    float rinv2 = rms_inv_sh2;
    for (unsigned i = threadIdx.x; i < pc.N; i += blockDim.x) {
        pt[i] = w_pre[i] * (ht[i] * rinv2);
    }
}

// ---- rms_norm_rope ----------------------------------------------------------
// Fused gemma per-head Q/K norm + RoPE: collapses the (per-head rms_norm ->
// rope) pair into ONE launch, dropping a full head_dim write+read round-trip of
// the normalized head. Bit-identical to the two-kernel path: same rms (weight
// shared across heads, applied per head_dim), then NEOX partial rope from the
// layer's inv_freq table with attn_scale=1.0 (the gemma q/k rope path always
// uses freq_base_bits==0 / attn_scale_bits==0). One block per head; the
// normalized head is staged in dynamic shared memory (head_dim floats) so the
// rope pair-reads see the post-norm values regardless of x/y aliasing.
// `dst_offset` lets the K head write its result straight into the KV cache at
// position*kv_dim (Q passes 0 and writes head-contiguously into its own buffer),
// folding away the K half of the separate kv_cache_write launch.
struct RmsRopePush { unsigned head_dim; float eps; unsigned rope_dim; unsigned position; unsigned dst_offset; };

extern "C" __global__ void rms_norm_rope(const float* x, const float* w, const float* inv_freq, float* y, RmsRopePush pc) {
    unsigned head = blockIdx.x;
    const float* xt = x + (size_t)head * pc.head_dim;
    float* yt = y + (size_t)pc.dst_offset + (size_t)head * pc.head_dim;
    extern __shared__ float sh[]; // pc.head_dim normalized values

    float ss = 0.0f;
    for (unsigned i = threadIdx.x; i < pc.head_dim; i += blockDim.x) {
        float v = xt[i];
        ss += v * v;
    }
    ss = zinc_block_reduce_sum(ss);

    __shared__ float rms_inv_sh;
    if (threadIdx.x == 0) rms_inv_sh = rsqrtf(ss / (float)pc.head_dim + pc.eps);
    __syncthreads();
    float rinv = rms_inv_sh;

    for (unsigned i = threadIdx.x; i < pc.head_dim; i += blockDim.x)
        sh[i] = w[i] * (xt[i] * rinv);
    __syncthreads();

    unsigned half_rot = pc.rope_dim >> 1;
    for (unsigned i = threadIdx.x; i < half_rot; i += blockDim.x) {
        float xi = sh[i];
        float xih = sh[i + half_rot];
        float theta = (float)pc.position * inv_freq[i];
        float ct = cosf(theta);
        float st = sinf(theta);
        yt[i] = xi * ct - xih * st;
        yt[i + half_rot] = xi * st + xih * ct;
    }
    for (unsigned i = pc.rope_dim + threadIdx.x; i < pc.head_dim; i += blockDim.x)
        yt[i] = sh[i];
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

// ---- dmmv_q5_1 (gemma4 MoE ffn_down_exps) -----------------------------------
// y[row] = sum_k dequant(W[row][k]) * x[k], W in Q5_1 (24-byte blocks of 32:
// f16 d, f16 m, 4-byte qh high-bits, 16-byte qs nibbles). 5-bit q = nibble |
// (qh_bit<<4); value = d*q + m. (UD-Q4_K_M ships ffn_down_exps as Q5_1.)
extern "C" __global__ void dmmv_q5_1(const unsigned char* a, const float* x, float* y, DmmvPush pc) {
    unsigned row = blockIdx.x;
    if (row >= pc.M) return;
    unsigned bpr = pc.K >> 5;                          // blocks per row (K / 32)
    const unsigned char* arow = a + pc.a_offset + (size_t)row * bpr * 24u;
    const float* xrow = x + (pc.x_offset >> 2);
    float sum = 0.0f;
    for (unsigned e = threadIdx.x; e < pc.K; e += blockDim.x) {
        unsigned blk = e >> 5;                          // which 32-elem block
        unsigned i = e & 31u;                           // 0..31 in block
        const unsigned char* blkp = arow + (size_t)blk * 24u;
        float d = zinc_half_to_float((unsigned short)((unsigned)blkp[0] | ((unsigned)blkp[1] << 8)));
        float m = zinc_half_to_float((unsigned short)((unsigned)blkp[2] | ((unsigned)blkp[3] << 8)));
        unsigned qh = (unsigned)blkp[4] | ((unsigned)blkp[5] << 8) | ((unsigned)blkp[6] << 16) | ((unsigned)blkp[7] << 24);
        const unsigned char* qs = blkp + 8;
        unsigned nib = (i < 16u) ? (unsigned)(qs[i] & 0xFu) : (unsigned)(qs[i - 16u] >> 4);
        unsigned bit = (qh >> i) & 1u;
        unsigned q5 = nib | (bit << 4);
        sum += (d * (float)q5 + m) * xrow[e];
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

// ---- embed_lookup_q4k (Effort 25 cycle 5: GPU-side embedding dequant) -------
// Dequantize one Q4_K row of token_embd.weight (the row for token `tok[0]`) into
// out[0..K], replacing the per-token CPU dequant + full-row H2D with a GPU
// dispatch reading the token id from a tiny device buffer (so it captures as the
// decode graph's first node). Bit-identical math to the CPU `dequantRow` Q4_K
// path: 144-byte superblocks, d/dmin f16, 12 scale bytes, 128 quant bytes;
// per 256-block output order [g0_lo32, g0_hi32, g1_lo32, ...], value =
// (d*sc) * nibble - (dmin*m), scale sub-index = 2*group + half (low/high nibble).
// Grid: one block per superblock (K/256). Block: 256 threads, one output each.
struct EmbedPush { unsigned K; unsigned vocab; };

extern "C" __global__ void embed_lookup_q4k(const unsigned char* W,
                                            const unsigned* tok, float* out,
                                            EmbedPush pc) {
    unsigned t = tok[0];
    unsigned vmax = pc.vocab ? pc.vocab - 1u : 0u;
    if (t > vmax) t = vmax;
    unsigned nsb = pc.K >> 8;          // superblocks per row (K / 256)
    unsigned sb = blockIdx.x;
    if (sb >= nsb) return;
    const unsigned char* blk = W + ((size_t)t * nsb + sb) * 144u;
    float d = zinc_half_to_float(*(const unsigned short*)(blk + 0));
    float dmin = zinc_half_to_float(*(const unsigned short*)(blk + 2));
    const unsigned char* scales = blk + 4;
    const unsigned char* qs = blk + 16;
    unsigned idx = threadIdx.x;        // 0..255 output element within the superblock
    if (idx >= 256u) return;
    unsigned g = idx >> 6;             // group 0..3 (each 64 outputs)
    unsigned h = (idx >> 5) & 1u;      // 0 = low nibble half, 1 = high nibble half
    unsigned l = idx & 31u;
    unsigned char sc, m;
    zinc_q4k_scale_min((int)(2u * g + h), scales, &sc, &m);
    unsigned char qb = qs[g * 32u + l];
    unsigned nib = (h == 0u) ? (qb & 0xFu) : (unsigned)(qb >> 4);
    out[sb * 256u + idx] = (d * (float)sc) * (float)nib - (dmin * (float)m);
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

// ---- moe_weighted_acc_scaled (gemma4 batched MoE, GPU-side down scale) -------
// Same weighted combine as moe_weighted_acc, but folds the per-expert down scale
// (ffn_down_exps.scale[id]) into the weight GPU-side: w_j = weight_j * escale[id_j].
// id_j = routing[j], weight_j = routing[n_used+j]. This removes the per-layer host
// readback gemma previously used to fold the scale into the router weights, so the
// whole batched MoE block can run async on the stream (GPU stays at boost).
extern "C" __global__ void moe_weighted_acc_scaled(float* a, const float* b, const unsigned* routing, const float* escale, MoeAccPush pc) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pc.N) return;
    float sum = 0.0f;
    for (unsigned j = 0; j < pc.n_used; j++) {
        float w = __uint_as_float(routing[pc.n_used + j]) * escale[routing[j]];
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
// ---- dmmv_q4k_fast (perf research, 5090) — port of tuned Vulkan dmmv_q4k -----
// 16 threads per Q4_K superblock: header read once/thread (not 256x), qs read
// once total, x via float4. Block-reduce over the block = one output row.
// The per-row compute is factored into a __device__ helper so the fused dual
// kernel (dmmv_q4k_fast_dual) shares exactly this arithmetic path (bit-exact).
__device__ __forceinline__ float zinc_dmmv_q4k_fast_sum(const unsigned* a_u32, unsigned a_base, const float4* xv, unsigned bpr) {
    unsigned tid = threadIdx.x, itid = tid & 15u, grp = tid >> 4;
    unsigned il = itid >> 2, ir = itid & 3u, v_im = il >> 1, v_in = il & 1u;
    unsigned l0 = 4u * (2u * ir + v_in);
    unsigned q_off = 32u * v_im + l0, y_loc = 64u * v_im + l0, shift = v_im * 16u;
    unsigned ngrp = blockDim.x >> 4;
    float sum = 0.0f;
    for (unsigned i = grp; i < bpr; i += ngrp) {
        unsigned blk = a_base + i * 36u;
        unsigned dd = a_u32[blk];
        float d = zinc_half_to_float((unsigned short)(dd & 0xFFFF));
        float dm = zinc_half_to_float((unsigned short)(dd >> 16));
        unsigned sc0 = a_u32[blk + 1u], sc1 = a_u32[blk + 2u], sc2 = a_u32[blk + 3u];
        unsigned qs0 = a_u32[blk + 4u + (q_off >> 2)], qs1 = a_u32[blk + 4u + (q_off >> 2) + 16u];
        unsigned bidx = (i * 256u + y_loc) >> 2, bidx2 = (i * 256u + y_loc + 128u) >> 2;
        float4 by0 = xv[bidx], by1 = xv[bidx + 8u], by2 = xv[bidx2], by3 = xv[bidx2 + 8u];
        unsigned s0 = sc0 >> shift, s1 = sc1 >> shift, s2 = sc2 >> shift;
        float f0 = d * (float)(s0 & 0x3Fu), b0 = dm * (float)(s1 & 0x3Fu);
        float f1 = d * (float)((s0 >> 8) & 0x3Fu), b1 = dm * (float)((s1 >> 8) & 0x3Fu);
        float f2 = d * (float)((s2 & 0xFu) | ((s0 & 0xC0u) >> 2)), b2 = dm * (float)(((s2 & 0xF0u) >> 4) | ((s1 & 0xC0u) >> 2));
        float f3 = d * (float)(((s2 >> 8) & 0xFu) | (((s0 >> 8) & 0xC0u) >> 2)), b3 = dm * (float)((((s2 >> 8) & 0xF0u) >> 4) | (((s1 >> 8) & 0xC0u) >> 2));
        sum += (f0*(float)(qs0&0xFu)-b0)*by0.x + (f0*(float)((qs0>>8)&0xFu)-b0)*by0.y + (f0*(float)((qs0>>16)&0xFu)-b0)*by0.z + (f0*(float)((qs0>>24)&0xFu)-b0)*by0.w;
        sum += (f1*(float)((qs0>>4)&0xFu)-b1)*by1.x + (f1*(float)((qs0>>12)&0xFu)-b1)*by1.y + (f1*(float)((qs0>>20)&0xFu)-b1)*by1.z + (f1*(float)((qs0>>28)&0xFu)-b1)*by1.w;
        sum += (f2*(float)(qs1&0xFu)-b2)*by2.x + (f2*(float)((qs1>>8)&0xFu)-b2)*by2.y + (f2*(float)((qs1>>16)&0xFu)-b2)*by2.z + (f2*(float)((qs1>>24)&0xFu)-b2)*by2.w;
        sum += (f3*(float)((qs1>>4)&0xFu)-b3)*by3.x + (f3*(float)((qs1>>12)&0xFu)-b3)*by3.y + (f3*(float)((qs1>>20)&0xFu)-b3)*by3.z + (f3*(float)((qs1>>28)&0xFu)-b3)*by3.w;
    }
    return zinc_block_reduce_sum(sum);
}

extern "C" __global__ void dmmv_q4k_fast(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) {
    unsigned row = blockIdx.x;
    if (row >= pc.M) return;
    unsigned bpr = pc.K >> 8;
    unsigned a_base = (pc.a_offset >> 2) + row * bpr * 36u;
    const float4* xv = (const float4*)(x + (pc.x_offset >> 2));
    float sum = zinc_dmmv_q4k_fast_sum(a_u32, a_base, xv, bpr);
    if (threadIdx.x == 0) { unsigned yi = (pc.y_offset >> 2) + row; if (pc.acc_mode != 0u) y[yi] += sum; else y[yi] = sum; }
}

// ---- dmmv_q4k_fast_dual — fuse two same-input Q4_K matvecs into ONE launch ----
// Both weights (a0,a1) share input x and inner dim K; outputs go to y0,y1. Grid
// is M0+M1 blocks: block bx<M0 computes row bx of a0→y0, else row bx-M0 of a1→y1.
// Used for the gemma FFN gate/up pair and the attention Q/K pair (both Q4_K, same
// norm input) to remove one kernel-launch boundary per layer. Each block's work
// is bit-identical to the standalone dmmv_q4k_fast with zero offsets (no acc).
struct Dmmv2Push { unsigned M0, M1, K; };
extern "C" __global__ void dmmv_q4k_fast_dual(const unsigned* a0, const unsigned* a1, const float* x, float* y0, float* y1, Dmmv2Push pc) {
    unsigned bx = blockIdx.x;
    if (bx >= pc.M0 + pc.M1) return;
    unsigned bpr = pc.K >> 8;
    const unsigned* a; float* y; unsigned row;
    if (bx < pc.M0) { a = a0; y = y0; row = bx; }
    else { a = a1; y = y1; row = bx - pc.M0; }
    unsigned a_base = row * bpr * 36u;
    const float4* xv = (const float4*)x;
    float sum = zinc_dmmv_q4k_fast_sum(a, a_base, xv, bpr);
    if (threadIdx.x == 0) y[row] = sum;
}

// ---- dmmv_q6k_fast (perf research) — port of tuned Vulkan dmmv_q6k -----------
extern "C" __global__ void dmmv_q6k_fast(const unsigned char* a, const float* x, float* y, DmmvPush pc) {
    unsigned row = blockIdx.x;
    if (row >= pc.M) return;
    unsigned bpr = pc.K >> 8;
    const unsigned char* arow = a + pc.a_offset + (size_t)row * bpr * 210u;
    const float4* xv = (const float4*)(x + (pc.x_offset >> 2));
    unsigned tid = threadIdx.x, itid = tid & 15u, ix = tid >> 4;
    unsigned half_id = itid >> 3, local_id = itid & 7u, e_start = local_id * 4u, is = e_start >> 4;
    unsigned xvib = (half_id * 128u + e_start) >> 2, ngrp = blockDim.x >> 4;
    float sum = 0.0f;
    for (unsigned b = ix; b < bpr; b += ngrp) {
        const unsigned char* bb = arow + (size_t)b * 210u;
        float d = zinc_half_to_float((unsigned short)((unsigned)bb[208] | ((unsigned)bb[209] << 8)));
        const unsigned char* ql = bb + half_id * 64u;
        const unsigned char* qh = bb + 128u + half_id * 32u;
        const signed char* sc = (const signed char*)(bb + 192u + half_id * 8u);
        float ds0 = d * (float)sc[is], ds2 = d * (float)sc[is + 2], ds4 = d * (float)sc[is + 4], ds6 = d * (float)sc[is + 6];
        unsigned xb = (b * 256u) / 4u + xvib;
        float4 bx0 = xv[xb], bx32 = xv[xb + 8u], bx64 = xv[xb + 16u], bx96 = xv[xb + 24u];
        #pragma unroll
        for (unsigned li = 0; li < 4u; li++) {
            unsigned l = e_start + li, qllo = ql[l], qlhi = ql[l + 32u], qhv = qh[l];
            float q1 = (float)((qllo & 0xFu) | (((qhv >> 0) & 3u) << 4)) - 32.0f;
            float q2 = (float)((qlhi & 0xFu) | (((qhv >> 2) & 3u) << 4)) - 32.0f;
            float q3 = (float)((qllo >> 4) | (((qhv >> 4) & 3u) << 4)) - 32.0f;
            float q4 = (float)((qlhi >> 4) | (((qhv >> 6) & 3u) << 4)) - 32.0f;
            sum += ds0*q1*(&bx0.x)[li] + ds2*q2*(&bx32.x)[li] + ds4*q3*(&bx64.x)[li] + ds6*q4*(&bx96.x)[li];
        }
    }
    sum = zinc_block_reduce_sum(sum);
    if (tid == 0) { unsigned yi = (pc.y_offset >> 2) + row; if (pc.acc_mode != 0u) y[yi] += sum; else y[yi] = sum; }
}

// ---- dmmv_q5k_fast (perf research) — q4k_fast + Q5_K qh high-bit promote -----
extern "C" __global__ void dmmv_q5k_fast(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) {
    unsigned row = blockIdx.x; if (row >= pc.M) return;
    unsigned bpr = pc.K >> 8;
    unsigned a_base = (pc.a_offset >> 2) + row * bpr * 44u;       // 176 bytes/block
    const float4* xv = (const float4*)(x + (pc.x_offset >> 2));
    unsigned tid = threadIdx.x, itid = tid & 15u, grp = tid >> 4;
    unsigned il = itid >> 2, ir = itid & 3u, v_im = il >> 1, v_in = il & 1u;
    unsigned l0 = 4u * (2u * ir + v_in);
    unsigned q_off = 32u * v_im + l0, y_loc = 64u * v_im + l0, shift = v_im * 16u, ngrp = blockDim.x >> 4;
    unsigned sba = 2u*v_im, sbb = 2u*v_im+1u, sbc = 2u*v_im+4u, sbd = 2u*v_im+5u;
    float sum = 0.0f;
    for (unsigned i = grp; i < bpr; i += ngrp) {
        unsigned blk = a_base + i * 44u, dd = a_u32[blk];
        float d = zinc_half_to_float((unsigned short)(dd & 0xFFFF)), dm = zinc_half_to_float((unsigned short)(dd >> 16));
        unsigned sc0 = a_u32[blk+1u], sc1 = a_u32[blk+2u], sc2 = a_u32[blk+3u];
        unsigned qh = a_u32[blk + 4u + (l0 >> 2)];
        unsigned qs0 = a_u32[blk + 12u + (q_off >> 2)], qs1 = a_u32[blk + 12u + (q_off >> 2) + 16u];
        unsigned bidx = (i*256u + y_loc) >> 2, bidx2 = (i*256u + y_loc + 128u) >> 2;
        float4 by0=xv[bidx], by1=xv[bidx+8u], by2=xv[bidx2], by3=xv[bidx2+8u];
        unsigned s0=sc0>>shift, s1=sc1>>shift, s2=sc2>>shift;
        float f0=d*(float)(s0&0x3Fu), b0=dm*(float)(s1&0x3Fu);
        float f1=d*(float)((s0>>8)&0x3Fu), b1=dm*(float)((s1>>8)&0x3Fu);
        float f2=d*(float)((s2&0xFu)|((s0&0xC0u)>>2)), b2=dm*(float)(((s2&0xF0u)>>4)|((s1&0xC0u)>>2));
        float f3=d*(float)(((s2>>8)&0xFu)|(((s0>>8)&0xC0u)>>2)), b3=dm*(float)((((s2>>8)&0xF0u)>>4)|(((s1>>8)&0xC0u)>>2));
        #define Q5(q,sh,sb,j) ((float)(((q)>>(sh))&0xFu) + 16.0f*(float)(((qh)>>((sb)+(j)*8u))&1u))
        sum += (f0*Q5(qs0,0,sba,0)-b0)*by0.x + (f0*Q5(qs0,8,sba,1)-b0)*by0.y + (f0*Q5(qs0,16,sba,2)-b0)*by0.z + (f0*Q5(qs0,24,sba,3)-b0)*by0.w;
        sum += (f1*Q5(qs0,4,sbb,0)-b1)*by1.x + (f1*Q5(qs0,12,sbb,1)-b1)*by1.y + (f1*Q5(qs0,20,sbb,2)-b1)*by1.z + (f1*Q5(qs0,28,sbb,3)-b1)*by1.w;
        sum += (f2*Q5(qs1,0,sbc,0)-b2)*by2.x + (f2*Q5(qs1,8,sbc,1)-b2)*by2.y + (f2*Q5(qs1,16,sbc,2)-b2)*by2.z + (f2*Q5(qs1,24,sbc,3)-b2)*by2.w;
        sum += (f3*Q5(qs1,4,sbd,0)-b3)*by3.x + (f3*Q5(qs1,12,sbd,1)-b3)*by3.y + (f3*Q5(qs1,20,sbd,2)-b3)*by3.z + (f3*Q5(qs1,28,sbd,3)-b3)*by3.w;
        #undef Q5
    }
    sum = zinc_block_reduce_sum(sum);
    if (tid == 0) { unsigned yi = (pc.y_offset >> 2) + row; if (pc.acc_mode != 0u) y[yi] += sum; else y[yi] = sum; }
}
// ---- batched MoE expert matvecs ---------------------------------------------
// One launch over ALL n_used experts: block `g` handles (expert e=g/M, row=g%M).
// The chosen expert id is read GPU-side from `expert_ids` (router_out_buf), so
// there is NO host readback — the whole MoE block runs async and the GPU stays
// busy enough to leave its idle clock. Per-expert weight slice = id*slice; x is
// shared across experts for gate/up (x_stride=0) or per-expert for down
// (x_stride=K, i.e. swiglu[e*K..]). Output is slot-major: y[e*M + row].
struct ExpertsPush { unsigned M, K, slice, x_stride, n_used, base; };

extern "C" __global__ void dmmv_q4k_experts(const unsigned* a_u32, const float* x, float* y, const unsigned* expert_ids, ExpertsPush pc) {
    unsigned g = blockIdx.x;
    unsigned e = g / pc.M;
    if (e >= pc.n_used) return;
    unsigned row = g - e * pc.M;
    unsigned bpr = pc.K >> 8;
    unsigned a_base = ((expert_ids[e] * pc.slice + pc.base) >> 2) + row * bpr * 36u;
    const float4* xv = (const float4*)(x + (size_t)e * pc.x_stride);
    unsigned tid = threadIdx.x, itid = tid & 15u, grp = tid >> 4;
    unsigned il = itid >> 2, ir = itid & 3u, v_im = il >> 1, v_in = il & 1u;
    unsigned l0 = 4u * (2u * ir + v_in);
    unsigned q_off = 32u * v_im + l0, y_loc = 64u * v_im + l0, shift = v_im * 16u;
    unsigned ngrp = blockDim.x >> 4;
    float sum = 0.0f;
    for (unsigned i = grp; i < bpr; i += ngrp) {
        unsigned blk = a_base + i * 36u;
        unsigned dd = a_u32[blk];
        float d = zinc_half_to_float((unsigned short)(dd & 0xFFFF));
        float dm = zinc_half_to_float((unsigned short)(dd >> 16));
        unsigned sc0 = a_u32[blk + 1u], sc1 = a_u32[blk + 2u], sc2 = a_u32[blk + 3u];
        unsigned qs0 = a_u32[blk + 4u + (q_off >> 2)], qs1 = a_u32[blk + 4u + (q_off >> 2) + 16u];
        unsigned bidx = (i * 256u + y_loc) >> 2, bidx2 = (i * 256u + y_loc + 128u) >> 2;
        float4 by0 = xv[bidx], by1 = xv[bidx + 8u], by2 = xv[bidx2], by3 = xv[bidx2 + 8u];
        unsigned s0 = sc0 >> shift, s1 = sc1 >> shift, s2 = sc2 >> shift;
        float f0 = d * (float)(s0 & 0x3Fu), b0 = dm * (float)(s1 & 0x3Fu);
        float f1 = d * (float)((s0 >> 8) & 0x3Fu), b1 = dm * (float)((s1 >> 8) & 0x3Fu);
        float f2 = d * (float)((s2 & 0xFu) | ((s0 & 0xC0u) >> 2)), b2 = dm * (float)(((s2 & 0xF0u) >> 4) | ((s1 & 0xC0u) >> 2));
        float f3 = d * (float)(((s2 >> 8) & 0xFu) | (((s0 >> 8) & 0xC0u) >> 2)), b3 = dm * (float)((((s2 >> 8) & 0xF0u) >> 4) | (((s1 >> 8) & 0xC0u) >> 2));
        sum += (f0*(float)(qs0&0xFu)-b0)*by0.x + (f0*(float)((qs0>>8)&0xFu)-b0)*by0.y + (f0*(float)((qs0>>16)&0xFu)-b0)*by0.z + (f0*(float)((qs0>>24)&0xFu)-b0)*by0.w;
        sum += (f1*(float)((qs0>>4)&0xFu)-b1)*by1.x + (f1*(float)((qs0>>12)&0xFu)-b1)*by1.y + (f1*(float)((qs0>>20)&0xFu)-b1)*by1.z + (f1*(float)((qs0>>28)&0xFu)-b1)*by1.w;
        sum += (f2*(float)(qs1&0xFu)-b2)*by2.x + (f2*(float)((qs1>>8)&0xFu)-b2)*by2.y + (f2*(float)((qs1>>16)&0xFu)-b2)*by2.z + (f2*(float)((qs1>>24)&0xFu)-b2)*by2.w;
        sum += (f3*(float)((qs1>>4)&0xFu)-b3)*by3.x + (f3*(float)((qs1>>12)&0xFu)-b3)*by3.y + (f3*(float)((qs1>>20)&0xFu)-b3)*by3.z + (f3*(float)((qs1>>28)&0xFu)-b3)*by3.w;
    }
    sum = zinc_block_reduce_sum(sum);
    if (tid == 0) y[(size_t)e * pc.M + row] = sum;
}

extern "C" __global__ void dmmv_q5k_experts(const unsigned* a_u32, const float* x, float* y, const unsigned* expert_ids, ExpertsPush pc) {
    unsigned g = blockIdx.x;
    unsigned e = g / pc.M;
    if (e >= pc.n_used) return;
    unsigned row = g - e * pc.M;
    unsigned bpr = pc.K >> 8;
    unsigned a_base = (expert_ids[e] * pc.slice >> 2) + row * bpr * 44u;
    const float4* xv = (const float4*)(x + (size_t)e * pc.x_stride);
    unsigned tid = threadIdx.x, itid = tid & 15u, grp = tid >> 4;
    unsigned il = itid >> 2, ir = itid & 3u, v_im = il >> 1, v_in = il & 1u;
    unsigned l0 = 4u * (2u * ir + v_in);
    unsigned q_off = 32u * v_im + l0, y_loc = 64u * v_im + l0, shift = v_im * 16u, ngrp = blockDim.x >> 4;
    unsigned sba = 2u*v_im, sbb = 2u*v_im+1u, sbc = 2u*v_im+4u, sbd = 2u*v_im+5u;
    float sum = 0.0f;
    for (unsigned i = grp; i < bpr; i += ngrp) {
        unsigned blk = a_base + i * 44u, dd = a_u32[blk];
        float d = zinc_half_to_float((unsigned short)(dd & 0xFFFF)), dm = zinc_half_to_float((unsigned short)(dd >> 16));
        unsigned sc0 = a_u32[blk+1u], sc1 = a_u32[blk+2u], sc2 = a_u32[blk+3u];
        unsigned qh = a_u32[blk + 4u + (l0 >> 2)];
        unsigned qs0 = a_u32[blk + 12u + (q_off >> 2)], qs1 = a_u32[blk + 12u + (q_off >> 2) + 16u];
        unsigned bidx = (i*256u + y_loc) >> 2, bidx2 = (i*256u + y_loc + 128u) >> 2;
        float4 by0=xv[bidx], by1=xv[bidx+8u], by2=xv[bidx2], by3=xv[bidx2+8u];
        unsigned s0=sc0>>shift, s1=sc1>>shift, s2=sc2>>shift;
        float f0=d*(float)(s0&0x3Fu), b0=dm*(float)(s1&0x3Fu);
        float f1=d*(float)((s0>>8)&0x3Fu), b1=dm*(float)((s1>>8)&0x3Fu);
        float f2=d*(float)((s2&0xFu)|((s0&0xC0u)>>2)), b2=dm*(float)(((s2&0xF0u)>>4)|((s1&0xC0u)>>2));
        float f3=d*(float)(((s2>>8)&0xFu)|(((s0>>8)&0xC0u)>>2)), b3=dm*(float)((((s2>>8)&0xF0u)>>4)|(((s1>>8)&0xC0u)>>2));
        #define Q5E(q,sh,sb,j) ((float)(((q)>>(sh))&0xFu) + 16.0f*(float)(((qh)>>((sb)+(j)*8u))&1u))
        sum += (f0*Q5E(qs0,0,sba,0)-b0)*by0.x + (f0*Q5E(qs0,8,sba,1)-b0)*by0.y + (f0*Q5E(qs0,16,sba,2)-b0)*by0.z + (f0*Q5E(qs0,24,sba,3)-b0)*by0.w;
        sum += (f1*Q5E(qs0,4,sbb,0)-b1)*by1.x + (f1*Q5E(qs0,12,sbb,1)-b1)*by1.y + (f1*Q5E(qs0,20,sbb,2)-b1)*by1.z + (f1*Q5E(qs0,28,sbb,3)-b1)*by1.w;
        sum += (f2*Q5E(qs1,0,sbc,0)-b2)*by2.x + (f2*Q5E(qs1,8,sbc,1)-b2)*by2.y + (f2*Q5E(qs1,16,sbc,2)-b2)*by2.z + (f2*Q5E(qs1,24,sbc,3)-b2)*by2.w;
        sum += (f3*Q5E(qs1,4,sbd,0)-b3)*by3.x + (f3*Q5E(qs1,12,sbd,1)-b3)*by3.y + (f3*Q5E(qs1,20,sbd,2)-b3)*by3.z + (f3*Q5E(qs1,28,sbd,3)-b3)*by3.w;
        #undef Q5E
    }
    sum = zinc_block_reduce_sum(sum);
    if (tid == 0) y[(size_t)e * pc.M + row] = sum;
}

// Batched Q5_1 expert down-proj (gemma-26b MoE): one launch over all n_used
// experts, expert id read GPU-side. Same dequant as dmmv_q5_1; per-expert x is
// swiglu[e*x_stride..]; output slot-major y[e*M + row]. base unused (0).
extern "C" __global__ void dmmv_q5_1_experts(const unsigned char* a, const float* x, float* y, const unsigned* expert_ids, ExpertsPush pc) {
    unsigned g = blockIdx.x;
    unsigned e = g / pc.M;
    if (e >= pc.n_used) return;
    unsigned row = g - e * pc.M;
    unsigned bpr = pc.K >> 5;
    const unsigned char* arow = a + (size_t)expert_ids[e] * pc.slice + pc.base + (size_t)row * bpr * 24u;
    const float* xrow = x + (size_t)e * pc.x_stride;
    float sum = 0.0f;
    for (unsigned el = threadIdx.x; el < pc.K; el += blockDim.x) {
        unsigned blk = el >> 5, i = el & 31u;
        const unsigned char* blkp = arow + (size_t)blk * 24u;
        float d = zinc_half_to_float((unsigned short)((unsigned)blkp[0] | ((unsigned)blkp[1] << 8)));
        float m = zinc_half_to_float((unsigned short)((unsigned)blkp[2] | ((unsigned)blkp[3] << 8)));
        unsigned qh = (unsigned)blkp[4] | ((unsigned)blkp[5] << 8) | ((unsigned)blkp[6] << 16) | ((unsigned)blkp[7] << 24);
        const unsigned char* qs = blkp + 8;
        unsigned nib = (i < 16u) ? (unsigned)(qs[i] & 0xFu) : (unsigned)(qs[i - 16u] >> 4);
        unsigned q5 = nib | (((qh >> i) & 1u) << 4);
        sum += (d * (float)q5 + m) * xrow[el];
    }
    sum = zinc_block_reduce_sum(sum);
    if (threadIdx.x == 0) y[(size_t)e * pc.M + row] = sum;
}

// ---- dmmv_q8_0_fast — whole-block-per-thread, d once, float4 x --------------
extern "C" __global__ void dmmv_q8_0_fast(const unsigned char* a, const float* x, float* y, DmmvPush pc) {
    unsigned row = blockIdx.x; if (row >= pc.M) return;
    unsigned bpr = pc.K >> 5;
    const unsigned char* arow = a + pc.a_offset + (size_t)row * bpr * 34u;
    unsigned xb0 = pc.x_offset >> 2, tid = threadIdx.x;
    float sum = 0.0f;
    for (unsigned b = tid; b < bpr; b += blockDim.x) {
        const unsigned char* blk = arow + (size_t)b * 34u;
        float d = zinc_half_to_float((unsigned short)((unsigned)blk[0] | ((unsigned)blk[1] << 8)));
        const signed char* qs = (const signed char*)(blk + 2);
        const float4* xb = (const float4*)(x + xb0 + (size_t)b * 32u);
        #pragma unroll
        for (unsigned j = 0; j < 8u; j++) { float4 xx = xb[j]; sum += d * ((float)qs[j*4]*xx.x + (float)qs[j*4+1]*xx.y + (float)qs[j*4+2]*xx.z + (float)qs[j*4+3]*xx.w); }
    }
    sum = zinc_block_reduce_sum(sum);
    if (tid == 0) { unsigned yi = (pc.y_offset >> 2) + row; if (pc.acc_mode != 0u) y[yi] += sum; else y[yi] = sum; }
}

// ---- dmmv_q4k multi-row (perf research, agenda 1: small/mid-M occupancy) -----
// One block computes R output rows, loading each shared x-superblock ONCE and
// reusing it across all R rows -> amortizes x-loads, block launches, and the
// per-thread index math over R rows. Targets M=4096 (proj) / M=12288 (FFN)
// which are launch/occupancy-bound at 15-65% peak in the single-row *_fast.
// Same Q4_K dequant math as dmmv_q4k_fast (validated 1.96e-5 vs naive).
template<int R>
__device__ __forceinline__ void dmmv_q4k_mrow_impl(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) {
    unsigned row0 = blockIdx.x * (unsigned)R;
    if (row0 >= pc.M) return;
    unsigned bpr = pc.K >> 8;
    const float4* xv = (const float4*)(x + (pc.x_offset >> 2));
    unsigned tid = threadIdx.x, itid = tid & 15u, grp = tid >> 4;
    unsigned il = itid >> 2, ir = itid & 3u, v_im = il >> 1, v_in = il & 1u;
    unsigned l0 = 4u * (2u * ir + v_in);
    unsigned q_off = 32u * v_im + l0, y_loc = 64u * v_im + l0, shift = v_im * 16u;
    unsigned ngrp = blockDim.x >> 4;
    unsigned a0 = (pc.a_offset >> 2);
    float sum[R];
    #pragma unroll
    for (int r = 0; r < R; r++) sum[r] = 0.0f;
    for (unsigned i = grp; i < bpr; i += ngrp) {
        unsigned bidx = (i * 256u + y_loc) >> 2, bidx2 = (i * 256u + y_loc + 128u) >> 2;
        float4 by0 = xv[bidx], by1 = xv[bidx + 8u], by2 = xv[bidx2], by3 = xv[bidx2 + 8u];
        #pragma unroll
        for (int r = 0; r < R; r++) {
            unsigned row = row0 + (unsigned)r;
            if (row >= pc.M) continue;
            unsigned blk = a0 + row * bpr * 36u + i * 36u;
            unsigned dd = a_u32[blk];
            float d = zinc_half_to_float((unsigned short)(dd & 0xFFFF));
            float dm = zinc_half_to_float((unsigned short)(dd >> 16));
            unsigned sc0 = a_u32[blk + 1u], sc1 = a_u32[blk + 2u], sc2 = a_u32[blk + 3u];
            unsigned qs0 = a_u32[blk + 4u + (q_off >> 2)], qs1 = a_u32[blk + 4u + (q_off >> 2) + 16u];
            unsigned s0 = sc0 >> shift, s1 = sc1 >> shift, s2 = sc2 >> shift;
            float f0 = d*(float)(s0&0x3Fu), b0 = dm*(float)(s1&0x3Fu);
            float f1 = d*(float)((s0>>8)&0x3Fu), b1 = dm*(float)((s1>>8)&0x3Fu);
            float f2 = d*(float)((s2&0xFu)|((s0&0xC0u)>>2)), b2 = dm*(float)(((s2&0xF0u)>>4)|((s1&0xC0u)>>2));
            float f3 = d*(float)(((s2>>8)&0xFu)|(((s0>>8)&0xC0u)>>2)), b3 = dm*(float)((((s2>>8)&0xF0u)>>4)|(((s1>>8)&0xC0u)>>2));
            float s = 0.0f;
            s += (f0*(float)(qs0&0xFu)-b0)*by0.x + (f0*(float)((qs0>>8)&0xFu)-b0)*by0.y + (f0*(float)((qs0>>16)&0xFu)-b0)*by0.z + (f0*(float)((qs0>>24)&0xFu)-b0)*by0.w;
            s += (f1*(float)((qs0>>4)&0xFu)-b1)*by1.x + (f1*(float)((qs0>>12)&0xFu)-b1)*by1.y + (f1*(float)((qs0>>20)&0xFu)-b1)*by1.z + (f1*(float)((qs0>>28)&0xFu)-b1)*by1.w;
            s += (f2*(float)(qs1&0xFu)-b2)*by2.x + (f2*(float)((qs1>>8)&0xFu)-b2)*by2.y + (f2*(float)((qs1>>16)&0xFu)-b2)*by2.z + (f2*(float)((qs1>>24)&0xFu)-b2)*by2.w;
            s += (f3*(float)((qs1>>4)&0xFu)-b3)*by3.x + (f3*(float)((qs1>>12)&0xFu)-b3)*by3.y + (f3*(float)((qs1>>20)&0xFu)-b3)*by3.z + (f3*(float)((qs1>>28)&0xFu)-b3)*by3.w;
            sum[r] += s;
        }
    }
    #pragma unroll
    for (int r = 0; r < R; r++) {
        float t = zinc_block_reduce_sum(sum[r]);
        if (tid == 0) {
            unsigned row = row0 + (unsigned)r;
            if (row < pc.M) { unsigned yi = (pc.y_offset >> 2) + row; if (pc.acc_mode != 0u) y[yi] += t; else y[yi] = t; }
        }
        __syncthreads();
    }
}
extern "C" __global__ void dmmv_q4k_mr2(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q4k_mrow_impl<2>(a_u32, x, y, pc); }
extern "C" __global__ void dmmv_q4k_mr4(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) { dmmv_q4k_mrow_impl<4>(a_u32, x, y, pc); }

// ---- dmmv_q5k multi-row (perf research, agenda 1: extend mr2 to Q5_K) --------
// One block computes R=2 output rows; each shared x-superblock loaded once and
// reused across both rows. Same Q5_K dequant as dmmv_q5k_fast (qh 5th-bit
// promote). Targets Q5_K mid-M matvecs (ffn_down / attn_v in Q4_K_M mixes),
// stuck ~65% peak at M=12288 in the single-row fast kernel.
template<int R>
__device__ __forceinline__ void dmmv_q5k_mrow_impl(const unsigned* a_u32, const float* x, float* y, DmmvPush pc) {
    unsigned row0 = blockIdx.x * (unsigned)R;
    if (row0 >= pc.M) return;
    unsigned bpr = pc.K >> 8;
    const float4* xv = (const float4*)(x + (pc.x_offset >> 2));
    unsigned tid = threadIdx.x, itid = tid & 15u, grp = tid >> 4;
    unsigned il = itid >> 2, ir = itid & 3u, v_im = il >> 1, v_in = il & 1u;
    unsigned l0 = 4u * (2u * ir + v_in);
    unsigned q_off = 32u * v_im + l0, y_loc = 64u * v_im + l0, shift = v_im * 16u, ngrp = blockDim.x >> 4;
    unsigned sba = 2u*v_im, sbb = 2u*v_im+1u, sbc = 2u*v_im+4u, sbd = 2u*v_im+5u;
    unsigned a0 = (pc.a_offset >> 2);
    float sum[R];
    #pragma unroll
    for (int r=0;r<R;r++) sum[r]=0.0f;
    for (unsigned i = grp; i < bpr; i += ngrp) {
        unsigned bidx = (i*256u + y_loc) >> 2, bidx2 = (i*256u + y_loc + 128u) >> 2;
        float4 by0=xv[bidx], by1=xv[bidx+8u], by2=xv[bidx2], by3=xv[bidx2+8u];
        #pragma unroll
        for (int r=0;r<R;r++){
            unsigned row = row0 + (unsigned)r;
            if (row >= pc.M) continue;
            unsigned blk = a0 + row * bpr * 44u + i * 44u, dd = a_u32[blk];
            float d = zinc_half_to_float((unsigned short)(dd & 0xFFFF)), dm = zinc_half_to_float((unsigned short)(dd >> 16));
            unsigned sc0 = a_u32[blk+1u], sc1 = a_u32[blk+2u], sc2 = a_u32[blk+3u];
            unsigned qh = a_u32[blk + 4u + (l0 >> 2)];
            unsigned qs0 = a_u32[blk + 12u + (q_off >> 2)], qs1 = a_u32[blk + 12u + (q_off >> 2) + 16u];
            unsigned s0=sc0>>shift, s1=sc1>>shift, s2=sc2>>shift;
            float f0=d*(float)(s0&0x3Fu), b0=dm*(float)(s1&0x3Fu);
            float f1=d*(float)((s0>>8)&0x3Fu), b1=dm*(float)((s1>>8)&0x3Fu);
            float f2=d*(float)((s2&0xFu)|((s0&0xC0u)>>2)), b2=dm*(float)(((s2&0xF0u)>>4)|((s1&0xC0u)>>2));
            float f3=d*(float)(((s2>>8)&0xFu)|(((s0>>8)&0xC0u)>>2)), b3=dm*(float)((((s2>>8)&0xF0u)>>4)|(((s1>>8)&0xC0u)>>2));
            #define Q5MR(q,sh,sb,j) ((float)(((q)>>(sh))&0xFu) + 16.0f*(float)(((qh)>>((sb)+(j)*8u))&1u))
            float s = 0.0f;
            s += (f0*Q5MR(qs0,0,sba,0)-b0)*by0.x + (f0*Q5MR(qs0,8,sba,1)-b0)*by0.y + (f0*Q5MR(qs0,16,sba,2)-b0)*by0.z + (f0*Q5MR(qs0,24,sba,3)-b0)*by0.w;
            s += (f1*Q5MR(qs0,4,sbb,0)-b1)*by1.x + (f1*Q5MR(qs0,12,sbb,1)-b1)*by1.y + (f1*Q5MR(qs0,20,sbb,2)-b1)*by1.z + (f1*Q5MR(qs0,28,sbb,3)-b1)*by1.w;
            s += (f2*Q5MR(qs1,0,sbc,0)-b2)*by2.x + (f2*Q5MR(qs1,8,sbc,1)-b2)*by2.y + (f2*Q5MR(qs1,16,sbc,2)-b2)*by2.z + (f2*Q5MR(qs1,24,sbc,3)-b2)*by2.w;
            s += (f3*Q5MR(qs1,4,sbd,0)-b3)*by3.x + (f3*Q5MR(qs1,12,sbd,1)-b3)*by3.y + (f3*Q5MR(qs1,20,sbd,2)-b3)*by3.z + (f3*Q5MR(qs1,28,sbd,3)-b3)*by3.w;
            #undef Q5MR
            sum[r] += s;
        }
    }
    #pragma unroll
    for (int r=0;r<R;r++){
        float t = zinc_block_reduce_sum(sum[r]);
        if (tid==0){ unsigned row=row0+(unsigned)r; if(row<pc.M){ unsigned yi=(pc.y_offset>>2)+row; if(pc.acc_mode!=0u) y[yi]+=t; else y[yi]=t; } }
        __syncthreads();
    }
}
extern "C" __global__ void dmmv_q5k_mr2(const unsigned* a_u32, const float* x, float* y, DmmvPush pc){ dmmv_q5k_mrow_impl<2>(a_u32,x,y,pc); }

// ---- gemm_q4k_tiled (perf research, agenda 5: PREFILL) -----------------------
// Y[T,M] = A[T,K] x W[M,K]^T, W = Q4_K, A = f32. Block computes an 8-row x
// 32-token output tile. Per K-superblock (256 elems): dequant the 8 W-rows and
// stage 32 A-rows into shared (transposed: [e][row] so warp-lane reads are
// bank-conflict-free), __syncthreads, then a 256-thread tile-multiply (each
// thread owns one Y[tok,row], mm=warp, tt=lane). W (dequant) reused 32x across
// tokens, A reused 8x across rows -> both operands served from shared memory.
// fp32 accumulate (correctness-first; tensor-core inner product is the next step).
struct GemmPush { unsigned M, K, T, a_offset, x_offset, y_offset, acc_mode; };
extern "C" __global__ void gemm_q4k_tiled(const unsigned* a_u32, const float* A, float* Y, GemmPush pc) {
    __shared__ float Ws[256 * 8];   // Ws[e*8 + mm]  (8 rows)
    __shared__ float As[256 * 32];  // As[e*32 + tt] (32 tokens)
    const unsigned BM = 8u, BT = 32u;
    unsigned m0 = blockIdx.x * BM;
    unsigned t0 = blockIdx.y * BT;
    unsigned bpr = pc.K >> 8;
    unsigned tid = threadIdx.x;                 // 0..255
    unsigned warp = tid >> 5, lane = tid & 31u; // 8 warps x 32 lanes
    unsigned a0 = (pc.a_offset >> 2);
    const float* Abase = A + (pc.x_offset >> 2);
    unsigned mm = warp, tt = lane;              // this thread's output (row mm, token tt)
    float acc = 0.0f;
    for (unsigned sb = 0; sb < bpr; sb++) {
        // --- dequant W: warp `warp` dequants row (m0+warp)'s superblock sb ---
        unsigned row = m0 + warp;
        if (row < pc.M) {
            unsigned blk = a0 + row * bpr * 36u + sb * 36u;
            unsigned dd = a_u32[blk];
            float d = zinc_half_to_float((unsigned short)(dd & 0xFFFFu));
            float dmin = zinc_half_to_float((unsigned short)(dd >> 16));
            const unsigned char* scales = (const unsigned char*)(a_u32 + blk + 1u);
            const unsigned char* qs = (const unsigned char*)(a_u32 + blk + 4u);
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                unsigned e = lane + (unsigned)j * 32u;
                unsigned chunk = e >> 6, half_ = (e & 63u) >> 5, l = e & 31u;
                unsigned char sc, mn; zinc_q4k_scale_min((int)(chunk * 2u + half_), scales, &sc, &mn);
                unsigned char qb = qs[chunk * 32u + l];
                unsigned nib = (half_ == 0u) ? (qb & 0xFu) : (unsigned)(qb >> 4);
                Ws[e * 8u + warp] = d * (float)sc * (float)nib - dmin * (float)mn;
            }
        } else {
            #pragma unroll
            for (int j = 0; j < 8; j++) Ws[(lane + (unsigned)j * 32u) * 8u + warp] = 0.0f;
        }
        // --- load A tile: As[e*32 + t], 256 threads x 32 = 8192 elems ---
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            unsigned idx = tid + (unsigned)j * 256u;   // 0..8191
            unsigned t = idx >> 8, e = idx & 255u;     // token 0..31, elem 0..255
            unsigned tok = t0 + t;
            As[e * 32u + t] = (tok < pc.T) ? Abase[(size_t)tok * pc.K + sb * 256u + e] : 0.0f;
        }
        __syncthreads();
        // --- tile multiply: acc += sum_e Ws[mm] * As[tt] ---
        #pragma unroll 8
        for (int e = 0; e < 256; e++) acc += Ws[(unsigned)e * 8u + mm] * As[(unsigned)e * 32u + tt];
        __syncthreads();
    }
    unsigned tok = t0 + tt, row = m0 + mm;
    if (tok < pc.T && row < pc.M) {
        unsigned yi = (pc.y_offset >> 2) + (size_t)tok * pc.M + row;
        if (pc.acc_mode != 0u) Y[yi] += acc; else Y[yi] = acc;
    }
}

// ---- gemm_q4k_tiled_v2 — register-blocked prefill GEMM (perf research) -------
// 64-row x 64-token output tile, 256 threads (16x16) each computing a 4x4
// register micro-tile. BK=32 (one Q4_K sub-block) per chunk: dequant the 64
// W-rows' sub-block + stage 64 A-rows into shared, then a register-blocked
// multiply. W reused 64x, A reused 64x; 16 FMA per 8 shared loads (4x the
// arithmetic intensity of v1's 1-output-per-thread inner loop). fp32 accumulate.
extern "C" __global__ void gemm_q4k_tiled_v2(const unsigned* a_u32, const float* A, float* Y, GemmPush pc) {
    const unsigned BM=64u, BT=64u, BK=32u;
    __shared__ float Ws[BK * BM];   // Ws[kk*64 + r]
    __shared__ float As[BK * BT];   // As[kk*64 + t]
    unsigned m0 = blockIdx.x * BM, t0 = blockIdx.y * BT;
    unsigned bpr = pc.K >> 8;
    unsigned nchunk = pc.K >> 5;    // K/32 sub-blocks
    unsigned tid = threadIdx.x;
    unsigned tx = tid & 15u, ty = tid >> 4;   // 16x16 thread grid
    unsigned a0 = (pc.a_offset >> 2);
    const float* Abase = A + (pc.x_offset >> 2);
    float acc[4][4];
    #pragma unroll
    for (int i=0;i<4;i++) for (int j=0;j<4;j++) acc[i][j]=0.0f;
    for (unsigned c = 0; c < nchunk; c++) {
        unsigned sbk = c >> 3, sb8 = c & 7u;   // superblock, sub-block 0..7
        // dequant W tile: BM*BK = 2048 elems / 256 threads = 8 each
        #pragma unroll
        for (int u = 0; u < 8; u++) {
            unsigned idx = tid + (unsigned)u * 256u;   // 0..2047
            unsigned r = idx >> 5, l = idx & 31u;      // row 0..63, elem 0..31
            unsigned row = m0 + r;
            float wv = 0.0f;
            if (row < pc.M) {
                unsigned blk = a0 + row * bpr * 36u + sbk * 36u;
                unsigned dd = a_u32[blk];
                float d = zinc_half_to_float((unsigned short)(dd & 0xFFFFu));
                float dmin = zinc_half_to_float((unsigned short)(dd >> 16));
                const unsigned char* scales = (const unsigned char*)(a_u32 + blk + 1u);
                const unsigned char* qs = (const unsigned char*)(a_u32 + blk + 4u);
                unsigned char sc, mn; zinc_q4k_scale_min((int)sb8, scales, &sc, &mn);
                unsigned char qb = qs[(sb8 >> 1) * 32u + l];
                unsigned nib = (sb8 & 1u) == 0u ? (qb & 0xFu) : (unsigned)(qb >> 4);
                wv = d * (float)sc * (float)nib - dmin * (float)mn;
            }
            Ws[l * BM + r] = wv;
        }
        // load A tile: BT*BK = 2048 / 256 = 8 each
        #pragma unroll
        for (int u = 0; u < 8; u++) {
            unsigned idx = tid + (unsigned)u * 256u;
            unsigned t = idx >> 5, l = idx & 31u;
            unsigned tok = t0 + t;
            As[l * BT + t] = (tok < pc.T) ? Abase[(size_t)tok * pc.K + c * 32u + l] : 0.0f;
        }
        __syncthreads();
        // register-blocked multiply
        #pragma unroll
        for (unsigned kk = 0; kk < BK; kk++) {
            float wr[4], ar[4];
            #pragma unroll
            for (int i=0;i<4;i++) wr[i] = Ws[kk * BM + ty*4u + (unsigned)i];
            #pragma unroll
            for (int j=0;j<4;j++) ar[j] = As[kk * BT + tx*4u + (unsigned)j];
            #pragma unroll
            for (int i=0;i<4;i++)
                #pragma unroll
                for (int j=0;j<4;j++) acc[i][j] += wr[i]*ar[j];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i=0;i<4;i++) {
        unsigned row = m0 + ty*4u + (unsigned)i;
        #pragma unroll
        for (int j=0;j<4;j++) {
            unsigned tok = t0 + tx*4u + (unsigned)j;
            if (row < pc.M && tok < pc.T) {
                unsigned yi = (pc.y_offset >> 2) + (size_t)tok * pc.M + row;
                if (pc.acc_mode != 0u) Y[yi] += acc[i][j]; else Y[yi] = acc[i][j];
            }
        }
    }
}

// ---- gemm_q6k_tiled_v2 — register-blocked prefill GEMM for Q6_K weights ------
// Mirror of gemm_q4k_tiled_v2 (64x64 tile, 256 threads, 4x4 register micro-tile,
// BK=32) with the Q6_K dequant in the stage-to-shared step. Q6_K block = 210
// bytes/256 elems: ql[0..127] low4, qh[128..191] high2, scales[192..207] int8,
// d[208..209] f16; q = (ql_nibble | (qh_bits<<4)) 6-bit; value = d*sc*(q-32).
// Q6_K is BYTE-addressed -> param `const unsigned char* a`, pc.a_offset is BYTES.
extern "C" __global__ void gemm_q6k_tiled_v2(const unsigned char* a, const float* A, float* Y, GemmPush pc) {
    const unsigned BM=64u, BT=64u, BK=32u;
    __shared__ float Ws[BK*BM]; __shared__ float As[BK*BT];
    unsigned m0=blockIdx.x*BM, t0=blockIdx.y*BT, bpr=pc.K>>8, nchunk=pc.K>>5;
    unsigned tid=threadIdx.x, tx=tid&15u, ty=tid>>4;
    const float* Abase=A+(pc.x_offset>>2);
    float acc[4][4];
    #pragma unroll
    for(int i=0;i<4;i++) for(int j=0;j<4;j++) acc[i][j]=0.0f;
    for(unsigned c=0;c<nchunk;c++){
        #pragma unroll
        for(int u=0;u<8;u++){ unsigned idx=tid+(unsigned)u*256u, r=idx>>5, l=idx&31u, row=m0+r; float wv=0.0f;
            if(row<pc.M){ unsigned e=c*32u+l, within=e&255u, sb=e>>8;
                const unsigned char* blk=a+pc.a_offset+(size_t)row*bpr*210u+(size_t)sb*210u;
                float d=zinc_half_to_float((unsigned short)((unsigned)blk[208]|((unsigned)blk[209]<<8)));
                unsigned half_=within>>7, wh=within&127u, ll=wh&31u, group=wh>>5;
                const unsigned char* ql=blk+(size_t)half_*64u;
                const unsigned char* qh=blk+128u+(size_t)half_*32u;
                const signed char* sc=(const signed char*)(blk+192u+(size_t)half_*8u);
                unsigned is=ll>>4, qhb=qh[ll], q, sci;
                if(group==0u){ q=(ql[ll]&0xFu)|(((qhb>>0)&3u)<<4); sci=is+0u; }
                else if(group==1u){ q=(ql[ll+32u]&0xFu)|(((qhb>>2)&3u)<<4); sci=is+2u; }
                else if(group==2u){ q=(ql[ll]>>4)|(((qhb>>4)&3u)<<4); sci=is+4u; }
                else { q=(ql[ll+32u]>>4)|(((qhb>>6)&3u)<<4); sci=is+6u; }
                wv=d*(float)sc[sci]*((float)q-32.0f); }
            Ws[l*BM+r]=wv; }
        #pragma unroll
        for(int u=0;u<8;u++){ unsigned idx=tid+(unsigned)u*256u, t=idx>>5, l=idx&31u, tok=t0+t;
            As[l*BT+t]=(tok<pc.T)?Abase[(size_t)tok*pc.K+c*32u+l]:0.0f; }
        __syncthreads();
        #pragma unroll
        for(unsigned kk=0;kk<BK;kk++){ float wr[4],ar[4];
            #pragma unroll
            for(int i=0;i<4;i++) wr[i]=Ws[kk*BM+ty*4u+(unsigned)i];
            #pragma unroll
            for(int j=0;j<4;j++) ar[j]=As[kk*BT+tx*4u+(unsigned)j];
            #pragma unroll
            for(int i=0;i<4;i++)
                #pragma unroll
                for(int j=0;j<4;j++) acc[i][j]+=wr[i]*ar[j]; }
        __syncthreads();
    }
    #pragma unroll
    for(int i=0;i<4;i++){ unsigned row=m0+ty*4u+(unsigned)i;
        #pragma unroll
        for(int j=0;j<4;j++){ unsigned tok=t0+tx*4u+(unsigned)j;
            if(row<pc.M&&tok<pc.T){ unsigned yi=(pc.y_offset>>2)+(size_t)tok*pc.M+row; if(pc.acc_mode!=0u) Y[yi]+=acc[i][j]; else Y[yi]=acc[i][j]; } } }
}

// ---- gemm_q5k_tiled_v2 — register-blocked prefill GEMM for Q5_K weights ------
// Mirror of gemm_q4k_tiled_v2 with Q5_K dequant in the stage-to-shared step.
// Q5_K block = 176 B/256 elems: d,dmin f16 [0..3]; scales[4..15] (12B); qh[16..47]
// (32B 5th-bit); qs[48..175] (128B). q5 = nib + (qh_bit?16:0); value = d*sc*q5 - dmin*mn.
extern "C" __global__ void gemm_q5k_tiled_v2(const unsigned char* a, const float* A, float* Y, GemmPush pc) {
    const unsigned BM=64u, BT=64u, BK=32u;
    __shared__ float Ws[BK*BM]; __shared__ float As[BK*BT];
    unsigned m0=blockIdx.x*BM, t0=blockIdx.y*BT, bpr=pc.K>>8, nchunk=pc.K>>5;
    unsigned tid=threadIdx.x, tx=tid&15u, ty=tid>>4;
    const float* Abase=A+(pc.x_offset>>2);
    float acc[4][4];
    #pragma unroll
    for(int i=0;i<4;i++) for(int j=0;j<4;j++) acc[i][j]=0.0f;
    for(unsigned c=0;c<nchunk;c++){
        #pragma unroll
        for(int u=0;u<8;u++){ unsigned idx=tid+(unsigned)u*256u, r=idx>>5, l=idx&31u, row=m0+r; float wv=0.0f;
            if(row<pc.M){ unsigned e=c*32u+l, within=e&255u, sb=e>>8;
                const unsigned char* blk=a+pc.a_offset+(size_t)row*bpr*176u+(size_t)sb*176u;
                float d=zinc_half_to_float((unsigned short)((unsigned)blk[0]|((unsigned)blk[1]<<8)));
                float dmin=zinc_half_to_float((unsigned short)((unsigned)blk[2]|((unsigned)blk[3]<<8)));
                const unsigned char* scales=blk+4u; const unsigned char* qh=blk+16u; const unsigned char* qs=blk+48u;
                unsigned chunk=within>>6, half_=(within&63u)>>5, ll=within&31u;
                unsigned char qb=qs[chunk*32u+ll]; unsigned nib=half_==0u?(qb&0xFu):(unsigned)(qb>>4);
                unsigned bit=(qh[ll]>>(2u*chunk+half_))&1u; unsigned q5=nib+(bit?16u:0u);
                unsigned char sc,mn; zinc_q4k_scale_min((int)(chunk*2u+half_),scales,&sc,&mn);
                wv=d*(float)sc*(float)q5 - dmin*(float)mn; }
            Ws[l*BM+r]=wv; }
        #pragma unroll
        for(int u=0;u<8;u++){ unsigned idx=tid+(unsigned)u*256u, t=idx>>5, l=idx&31u, tok=t0+t;
            As[l*BT+t]=(tok<pc.T)?Abase[(size_t)tok*pc.K+c*32u+l]:0.0f; }
        __syncthreads();
        #pragma unroll
        for(unsigned kk=0;kk<BK;kk++){ float wr[4],ar[4];
            #pragma unroll
            for(int i=0;i<4;i++) wr[i]=Ws[kk*BM+ty*4u+(unsigned)i];
            #pragma unroll
            for(int j=0;j<4;j++) ar[j]=As[kk*BT+tx*4u+(unsigned)j];
            #pragma unroll
            for(int i=0;i<4;i++)
                #pragma unroll
                for(int j=0;j<4;j++) acc[i][j]+=wr[i]*ar[j]; }
        __syncthreads();
    }
    #pragma unroll
    for(int i=0;i<4;i++){ unsigned row=m0+ty*4u+(unsigned)i;
        #pragma unroll
        for(int j=0;j<4;j++){ unsigned tok=t0+tx*4u+(unsigned)j;
            if(row<pc.M&&tok<pc.T){ unsigned yi=(pc.y_offset>>2)+(size_t)tok*pc.M+row; if(pc.acc_mode!=0u) Y[yi]+=acc[i][j]; else Y[yi]=acc[i][j]; } } }
}

// ---- sigmoid_mul (qwen35 attention gate) — out[i] = a[i] * sigmoid(gate[i]) ---
// ABI: inputs first, output last (matches swiglu). In-place safe (out may alias a).
struct SigmoidMulPush { unsigned N; };
extern "C" __global__ void sigmoid_mul(const float* a, const float* gate, float* out, SigmoidMulPush pc) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pc.N) return;
    float g = gate[idx];
    out[idx] = a[idx] * (1.0f / (1.0f + expf(-g)));
}

// ===========================================================================
// Gemma 4 kernels (additive — never used by the qwen35/qwen36 path).
// ===========================================================================

// ---- rms_norm_noweight (gemma V normalization) -----------------------------
// y = x / sqrt(mean(x^2) + eps). Pure RMS normalize with NO learnable weight
// (gemma4 applies ggml_rms_norm to V per head before the KV write). One block
// per row of pc.N elements (grid = n_kv_heads, N = head_dim).
extern "C" __global__ void rms_norm_noweight(const float* x, float* y, RmsPush pc) {
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
    for (unsigned i = threadIdx.x; i < pc.N; i += blockDim.x)
        yt[i] = xt[i] * rinv;
}

// ---- rms_norm_kvwrite (gemma: fuse per-head V rms_norm-noweight + V KV write) -
// One block per KV head. Plain-normalizes V over head_dim (no weight, like
// rms_norm_noweight) and writes the result STRAIGHT into the V cache at
// dst_offset (= position*kv_dim), per head at dst_offset + head*head_dim.
// Replaces the rms_norm_noweight(→v_buf) + the V half of kv_cache_write pair on
// the gemma attention path — bit-equivalent (identical normalization, identical
// destination layout), one launch instead of two, no v_buf round-trip.
struct RmsKvWritePush { unsigned head_dim; float eps; unsigned dst_offset; };
extern "C" __global__ void rms_norm_kvwrite(const float* v_src, float* v_dst, RmsKvWritePush pc) {
    unsigned head = blockIdx.x;
    unsigned hd = pc.head_dim;
    const float* vh = v_src + (size_t)head * hd;
    float ss = 0.0f;
    for (unsigned i = threadIdx.x; i < hd; i += blockDim.x) {
        float v = vh[i];
        ss += v * v;
    }
    ss = zinc_block_reduce_sum(ss);
    __shared__ float rms_inv_sh;
    if (threadIdx.x == 0) rms_inv_sh = rsqrtf(ss / (float)hd + pc.eps);
    __syncthreads();
    float rinv = rms_inv_sh;
    size_t base = (size_t)pc.dst_offset + (size_t)head * hd;
    for (unsigned i = threadIdx.x; i < hd; i += blockDim.x)
        v_dst[base + i] = vh[i] * rinv;
}

// ---- rms_norm_rope_qkv (gemma: fuse the per-head V/Q/K norm launches) -------
// Collapses the THREE per-head norm launches on the gemma attention path into
// ONE: the V plain-normalize+KV-write (rms_norm_kvwrite), the Q norm+rope, and
// the K norm+rope (both rms_norm_rope). Grid is n_head + 2*n_kv_head blocks:
//   block <  n_head                       -> Q head: weighted norm + rope -> q_out (offset 0)
//   block <  n_head + n_kv_head           -> K head: weighted norm + rope -> k_out at kv_offset
//   else                                  -> V head: plain norm (no weight, no rope) -> v_out at kv_offset
// Each branch's arithmetic is COPIED verbatim from the standalone kernels
// (Q/K from rms_norm_rope, V from rms_norm_kvwrite) so the fused result is
// bit-identical. No cross-block hazard: K writes kv_k (never k_in), V writes
// kv_v (never v_in), Q writes q_out in-place per head — no block reads a buffer
// another block writes. Removes 2 tiny launch boundaries/layer on the gemma
// attention path (dense + MoE).
struct RmsRopeQkvPush {
    unsigned head_dim; float eps; unsigned rope_dim; unsigned position;
    unsigned n_head; unsigned n_kv_head; unsigned kv_offset;
};
extern "C" __global__ void rms_norm_rope_qkv(
    const float* q_in, const float* k_in, const float* v_in,
    const float* wq, const float* wk, const float* inv_freq,
    float* q_out, float* k_out, float* v_out, RmsRopeQkvPush pc)
{
    unsigned bx = blockIdx.x;
    unsigned hd = pc.head_dim;
    extern __shared__ float sh[]; // hd normalized values (Q/K rope staging)

    const float* xt;
    float* yt;
    const float* w;
    bool do_rope;
    if (bx < pc.n_head) {
        unsigned head = bx;
        xt = q_in + (size_t)head * hd;
        yt = q_out + (size_t)head * hd;                 // dst_offset 0
        w = wq; do_rope = true;
    } else if (bx < pc.n_head + pc.n_kv_head) {
        unsigned head = bx - pc.n_head;
        xt = k_in + (size_t)head * hd;
        yt = k_out + (size_t)pc.kv_offset + (size_t)head * hd;
        w = wk; do_rope = true;
    } else {
        unsigned head = bx - pc.n_head - pc.n_kv_head;
        xt = v_in + (size_t)head * hd;
        yt = v_out + (size_t)pc.kv_offset + (size_t)head * hd;
        w = nullptr; do_rope = false;
    }

    float ss = 0.0f;
    for (unsigned i = threadIdx.x; i < hd; i += blockDim.x) {
        float v = xt[i];
        ss += v * v;
    }
    ss = zinc_block_reduce_sum(ss);
    __shared__ float rms_inv_sh;
    if (threadIdx.x == 0) rms_inv_sh = rsqrtf(ss / (float)hd + pc.eps);
    __syncthreads();
    float rinv = rms_inv_sh;

    if (!do_rope) { // V: plain normalize, no weight (matches rms_norm_kvwrite)
        for (unsigned i = threadIdx.x; i < hd; i += blockDim.x)
            yt[i] = xt[i] * rinv;
        return;
    }

    // Q/K: weighted norm into shared, then NEOX partial rope (matches rms_norm_rope)
    for (unsigned i = threadIdx.x; i < hd; i += blockDim.x)
        sh[i] = w[i] * (xt[i] * rinv);
    __syncthreads();

    unsigned half_rot = pc.rope_dim >> 1;
    for (unsigned i = threadIdx.x; i < half_rot; i += blockDim.x) {
        float xi = sh[i];
        float xih = sh[i + half_rot];
        float theta = (float)pc.position * inv_freq[i];
        float ct = cosf(theta);
        float st = sinf(theta);
        yt[i] = xi * ct - xih * st;
        yt[i + half_rot] = xi * st + xih * ct;
    }
    for (unsigned i = pc.rope_dim + threadIdx.x; i < hd; i += blockDim.x)
        yt[i] = sh[i];
}

// ---- geglu (gemma FFN activation: gelu(gate) * up) -------------------------
// Matches ggml LLM_FFN_GELU (tanh approximation). gemma norm weights already
// carry the +1 offset (baked at GGUF conversion), so the surrounding norms use
// the standard rms_norm kernel.
extern "C" __global__ void geglu(const float* gate, const float* up, float* y, SwigluPush pc) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pc.N) return;
    const float k = 0.7978845608028654f; // sqrt(2/pi)
    float g = gate[idx];
    float gelu = 0.5f * g * (1.0f + tanhf(k * (g + 0.044715f * g * g * g)));
    y[idx] = gelu * up[idx];
}

// ---- scalar_mul (gemma per-layer output scale) ----------------------------
// a[i] *= s[0]. s is a device [1] buffer (blk.N.layer_output_scale.weight).
struct ScalarMulPush { unsigned N; };
extern "C" __global__ void scalar_mul(float* a, const float* s, ScalarMulPush pc) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pc.N) return;
    a[idx] *= s[0];
}

// ---- mul_vec_scaled (gemma4 MoE router pre-scale) --------------------------
// a[i] = a[i] * b[i] * scale. The gemma4 MoE router computes its logits from a
// plain-RMS-normed residual scaled by 1/sqrt(n_embd) and a per-channel weight
// (ffn_gate_inp.scale) before the gate projection.
struct MulVecPush { unsigned N; float scale; };
extern "C" __global__ void mul_vec_scaled(float* a, const float* b, MulVecPush pc) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pc.N) return;
    a[idx] = a[idx] * b[idx] * pc.scale;
}

// ---- zero_vec --------------------------------------------------------------
// a[i] = 0. Clears the MoE combine accumulator before moe_weighted_acc (which
// is a += kernel).
struct ZeroPush { unsigned N; };
extern "C" __global__ void zero_vec(float* a, ZeroPush pc) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pc.N) return;
    a[idx] = 0.0f;
}

// ---- gemma_attention (decode: softmax(scale*QK^T)V, GQA, sliding window) ----
// One block per query head. Causal + optional sliding-window mask: with window
// W>0 only the last W keys are attended (gemma4 SWA layers, W=1024). window==0
// is full attention. No attention sink. kq_scale is gemma's f_attention_scale
// (1.0); scale_bits==0 falls back to 1/sqrt(head_dim). KV cache layout matches
// kv_cache_write: contiguous [seq_len, n_kv_heads, head_dim].
struct GemmaAttnPush { unsigned head_dim, n_heads, n_kv_heads, seq_len, scale_bits, window; };
extern "C" __global__ void gemma_attention(const float* q, const float* k, const float* v,
                                           float* out, GemmaAttnPush pc) {
    extern __shared__ float s_scores[];           // size = seq_len floats (dynamic)
    __shared__ float s_m, s_inv;
    unsigned head = blockIdx.x;
    unsigned tid = threadIdx.x;
    unsigned hd = pc.head_dim;
    unsigned kv_head = head / (pc.n_heads / pc.n_kv_heads);
    const float* qh = q + (size_t)head * hd;
    float scale = pc.scale_bits != 0u ? __uint_as_float(pc.scale_bits) : rsqrtf((float)hd);

    // sliding-window start: decode query is the last position (seq_len-1).
    unsigned start = 0;
    if (pc.window != 0u && pc.seq_len > pc.window) start = pc.seq_len - pc.window;

    // Pass 1: scores = scale * (q . k_i), track max.
    float lmax = -3.4e38f;
    for (unsigned i = start + tid; i < pc.seq_len; i += blockDim.x) {
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
    for (unsigned i = start + tid; i < pc.seq_len; i += blockDim.x) {
        float e = expf(s_scores[i] - m);
        s_scores[i] = e;
        lsum += e;
    }
    lsum = zinc_block_reduce_sum(lsum);
    if (tid == 0) s_inv = (lsum > 0.0f) ? 1.0f / lsum : 0.0f;
    __syncthreads();
    float inv = s_inv;

    // Pass 3: out[d] = (sum_i e_i * V[i,d]) * inv.
    for (unsigned d = tid; d < hd; d += blockDim.x) {
        float acc = 0.0f;
        for (unsigned i = start; i < pc.seq_len; i++)
            acc += s_scores[i] * v[((size_t)i * pc.n_kv_heads + kv_head) * hd + d];
        out[(size_t)head * hd + d] = acc * inv;
    }
}

// ---- deinterleave_qgate (qwen35 packed Q+gate projection) ----
// wq outputs [2*head_dim] per head, laid out as [Q(head_dim) | gate(head_dim)]
// interleaved across heads: [Q0,g0,Q1,g1,...]. Split into contiguous q_out and
// gate_out (each [n_head*head_dim]). One thread per (head,dim) element.
struct DeintPush { unsigned head_dim, n_head; };
extern "C" __global__ void deinterleave_qgate(const float* qfull, float* q_out, float* gate_out, DeintPush pc) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total = pc.n_head * pc.head_dim;
    if (i >= total) return;
    unsigned h = i / pc.head_dim;
    unsigned d = i % pc.head_dim;
    unsigned src = h * 2u * pc.head_dim + d;
    q_out[i]    = qfull[src];
    gate_out[i] = qfull[src + pc.head_dim];
}
