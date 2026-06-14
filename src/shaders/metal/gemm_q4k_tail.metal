#include <metal_stdlib>
using namespace metal;

// Q4_K small-batch GEMM tail for dense Qwen3.5 9B prefill.
//
// This adapts llama.cpp's Metal `kernel_mul_mv_ext_q4x4_f32_impl` for the
// 4-8 prompt-row remainder after a 32-token simdgroup-MM tile. Output remains
// token-major `[token][row]`, matching gemm_q4k.metal.

struct GemmTailPush {
    uint32_t K;
    uint32_t M;
    uint32_t N;
    uint32_t token_offset;
    uint64_t nb01;
    uint64_t nb10;
    uint64_t nb11;
    uint32_t src0_off;
    uint32_t flags;
};

struct block_q4_K {
    half d;
    half dmin;
    uchar scales[12];
    uchar qs[128];
};

#define QK_K 256
#define QK_NL 16
#define NXPSG 8
#define NYPSG 4
#define NSG 2
#define R1PTG 4

static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j + 0 + k] & 63), uchar(q[j + 4 + k] & 63)}
                 : uchar2{uchar((q[j + 4 + k] & 0xF) | ((q[j - 4 + k] & 0xc0) >> 2)),
                           uchar((q[j + 4 + k] >> 4) | ((q[j - 0 + k] & 0xc0) >> 2))};
}

static void dequantize_q4_K_tail(device const block_q4_K * xb, short il, thread float4x4 & reg) {
    device const uchar * q = xb->qs;
    short is = (il / 4) * 2;
    q = q + (il / 4) * 32 + 16 * (il & 1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il / 2, xb->scales);
    const float d = il < 2 ? float(xb->d) : float(xb->d) / 16.0f;
    const float mn = float(xb->dmin);
    const float dl = d * sc[0];
    const float ml = mn * sc[1];
    const ushort mask = il < 2 ? 0x0F : 0xF0;
    for (int i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = dl * float(q[i] & mask) - ml;
    }
}

kernel void main0(
    constant GemmTailPush & args [[buffer(0)]],
    device const char * src0 [[buffer(1)]],
    device const char * src1 [[buffer(2)]],
    device       char * dst  [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort lane  [[thread_index_in_simdgroup]],
    ushort sg    [[simdgroup_index_in_threadgroup]]
) {
    const short tx = lane % NXPSG;
    const short ty = lane / NXPSG;
    const uint row = tgpig.x * (NYPSG * NSG) + uint(NYPSG * sg + ty);
    const uint token_base = tgpig.y * R1PTG;

    short cch = tx;
    device const block_q4_K * xq = (row < args.M)
        ? (device const block_q4_K *)(src0 + args.src0_off + (uint64_t)row * args.nb01) + tx / QK_NL
        : (device const block_q4_K *)src0;

    device const float4x4 * y4[R1PTG];
    for (uint t = 0; t < R1PTG; ++t) {
        const uint token = token_base + t;
        y4[t] = (token < args.N)
            ? (device const float4x4 *)(src1 + (uint64_t)(args.token_offset + token) * args.nb11) + tx
            : (device const float4x4 *)src1;
    }

    float sum[R1PTG];
    for (uint t = 0; t < R1PTG; ++t) sum[t] = 0.0f;

    for (uint ich = tx; 16u * ich < args.K; ich += NXPSG) {
        float4x4 lx;
        dequantize_q4_K_tail(xq, cch, lx);

        for (uint t = 0; t < R1PTG; ++t) {
            const float4x4 yy = y4[t][0];
            sum[t] += dot(lx[0], yy[0]) +
                      dot(lx[1], yy[1]) +
                      dot(lx[2], yy[2]) +
                      dot(lx[3], yy[3]);
            y4[t] += NXPSG;
        }

        cch += NXPSG;
        if (cch >= QK_NL) {
            xq += cch / QK_NL;
            cch %= QK_NL;
        }
    }

    for (uint t = 0; t < R1PTG; ++t) {
        sum[t] += simd_shuffle_down(sum[t], 4);
        sum[t] += simd_shuffle_down(sum[t], 2);
        sum[t] += simd_shuffle_down(sum[t], 1);
    }

    if (tx == 0 && row < args.M) {
        const bool accumulate = (args.flags & 1u) != 0u;
        for (uint t = 0; t < R1PTG; ++t) {
            const uint token = token_base + t;
            if (token < args.N) {
                device float * out = (device float *)dst + (uint64_t)(args.token_offset + token) * args.M + row;
                if (accumulate) {
                    *out += sum[t];
                } else {
                    *out = sum[t];
                }
            }
        }
    }
}
