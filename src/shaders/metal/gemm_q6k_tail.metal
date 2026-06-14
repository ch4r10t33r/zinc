#include <metal_stdlib>
using namespace metal;

// Q6_K small-batch GEMM tail for dense Qwen3.5 9B prefill.
// Adapted from llama.cpp's Metal `kernel_mul_mv_ext_q4x4_f32_impl`.

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

struct block_q6_K {
    uchar ql[128];
    uchar qh[64];
    char scales[16];
    half d;
};

#define QK_K 256
#define QK_NL 16
#define NXPSG 8
#define NYPSG 4
#define NSG 2
#define R1PTG 4

static void dequantize_q6_K_tail(device const block_q6_K * xb, short il, thread float4x4 & reg) {
    const half d_all = xb->d;
    device const ushort * ql = (device const ushort *)xb->ql;
    device const ushort * qh = (device const ushort *)xb->qh;
    device const char * scales = (device const char *)xb->scales;

    ql = ql + 32 * (il / 8) + 16 * ((il / 2) & 1) + 8 * (il & 1);
    qh = qh + 16 * (il / 8) + 8 * (il & 1);
    const float sc = (float)scales[(il % 2) + 2 * (il / 2)];
    il = (il / 2) & 3;

    const uint kmask1 = il > 1 ? (il > 2 ? 0xC0C0C0C0 : 0x30303030) : (il > 0 ? 0x0C0C0C0C : 0x03030303);
    const uint kmask2 = il > 1 ? 0xF0F0F0F0 : 0x0F0F0F0F;
    const float ml = float(d_all) * sc * 32.0f;
    const float dl0 = float(d_all) * sc;
    const float dl1 = dl0 / 256.0f;
    const float dl2 = dl0 / (256.0f * 256.0f);
    const float dl3 = dl0 / (256.0f * 256.0f * 256.0f);
    const uchar shr_h = il > 2 ? 2 : 0;
    const uchar shl_h = il > 1 ? 0 : (il > 0 ? 2 : 4);
    const uchar shr_l = il > 1 ? 4 : 0;
    for (int i = 0; i < 4; ++i) {
        const uint low = (ql[2 * i] | ((uint)ql[2 * i + 1] << 16)) & kmask2;
        const uint high = (qh[2 * i] | ((uint)qh[2 * i + 1] << 16)) & kmask1;
        const uint q = ((high << shl_h) >> shr_h) | (low >> shr_l);
        reg[i][0] = dl0 * float(q & 0xFF) - ml;
        reg[i][1] = dl1 * float(q & 0xFF00) - ml;
        reg[i][2] = dl2 * float(q & 0xFF0000) - ml;
        reg[i][3] = dl3 * float(q & 0xFF000000) - ml;
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
    device const block_q6_K * xq = (row < args.M)
        ? (device const block_q6_K *)(src0 + args.src0_off + (uint64_t)row * args.nb01) + tx / QK_NL
        : (device const block_q6_K *)src0;

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
        dequantize_q6_K_tail(xq, cch, lx);

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
        for (uint t = 0; t < R1PTG; ++t) {
            const uint token = token_base + t;
            if (token < args.N) {
                device float * out = (device float *)dst + (uint64_t)(args.token_offset + token) * args.M + row;
                *out = sum[t];
            }
        }
    }
}
