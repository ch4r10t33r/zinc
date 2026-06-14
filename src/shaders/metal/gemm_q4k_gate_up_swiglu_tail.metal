#include <metal_stdlib>
using namespace metal;

// Q4_K gate/up small-batch tail fused with SwiGLU for dense Qwen3.5 9B
// prefill. This is the fused counterpart of gemm_q4k_tail.metal.

struct GemmGateUpTailPush {
    uint32_t K;
    uint32_t M;
    uint32_t N;
    uint32_t token_offset;
    uint64_t nb01;
    uint64_t nb10;
    uint64_t nb11;
    uint32_t src0_off;
    uint32_t src1_off;
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

static inline float swiglu(float gate, float up) {
    return up * gate * fast::divide(1.0f, 1.0f + fast::exp(-gate));
}

kernel void main0(
    constant GemmGateUpTailPush & args [[buffer(0)]],
    device const char * gate_w [[buffer(1)]],
    device const char * up_w [[buffer(2)]],
    device const char * src1 [[buffer(3)]],
    device       char * dst [[buffer(4)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort lane  [[thread_index_in_simdgroup]],
    ushort sg    [[simdgroup_index_in_threadgroup]]
) {
    const short tx = lane % NXPSG;
    const short ty = lane / NXPSG;
    const uint row = tgpig.x * (NYPSG * NSG) + uint(NYPSG * sg + ty);
    const uint token_base = tgpig.y * R1PTG;

    short cch_gate = tx;
    short cch_up = tx;
    device const block_q4_K * gate_x = (row < args.M)
        ? (device const block_q4_K *)(gate_w + args.src0_off + (uint64_t)row * args.nb01) + tx / QK_NL
        : (device const block_q4_K *)gate_w;
    device const block_q4_K * up_x = (row < args.M)
        ? (device const block_q4_K *)(up_w + args.src1_off + (uint64_t)row * args.nb01) + tx / QK_NL
        : (device const block_q4_K *)up_w;

    device const float4x4 * y4[R1PTG];
    for (uint t = 0; t < R1PTG; ++t) {
        const uint token = token_base + t;
        y4[t] = (token < args.N)
            ? (device const float4x4 *)(src1 + (uint64_t)(args.token_offset + token) * args.nb11) + tx
            : (device const float4x4 *)src1;
    }

    float gate_sum[R1PTG];
    float up_sum[R1PTG];
    for (uint t = 0; t < R1PTG; ++t) {
        gate_sum[t] = 0.0f;
        up_sum[t] = 0.0f;
    }

    for (uint ich = tx; 16u * ich < args.K; ich += NXPSG) {
        float4x4 gate_lx;
        float4x4 up_lx;
        dequantize_q4_K_tail(gate_x, cch_gate, gate_lx);
        dequantize_q4_K_tail(up_x, cch_up, up_lx);

        for (uint t = 0; t < R1PTG; ++t) {
            const float4x4 yy = y4[t][0];
            gate_sum[t] += dot(gate_lx[0], yy[0]) +
                           dot(gate_lx[1], yy[1]) +
                           dot(gate_lx[2], yy[2]) +
                           dot(gate_lx[3], yy[3]);
            up_sum[t] += dot(up_lx[0], yy[0]) +
                         dot(up_lx[1], yy[1]) +
                         dot(up_lx[2], yy[2]) +
                         dot(up_lx[3], yy[3]);
            y4[t] += NXPSG;
        }

        cch_gate += NXPSG;
        if (cch_gate >= QK_NL) {
            gate_x += cch_gate / QK_NL;
            cch_gate %= QK_NL;
        }
        cch_up += NXPSG;
        if (cch_up >= QK_NL) {
            up_x += cch_up / QK_NL;
            cch_up %= QK_NL;
        }
    }

    for (uint t = 0; t < R1PTG; ++t) {
        gate_sum[t] += simd_shuffle_down(gate_sum[t], 4);
        gate_sum[t] += simd_shuffle_down(gate_sum[t], 2);
        gate_sum[t] += simd_shuffle_down(gate_sum[t], 1);
        up_sum[t] += simd_shuffle_down(up_sum[t], 4);
        up_sum[t] += simd_shuffle_down(up_sum[t], 2);
        up_sum[t] += simd_shuffle_down(up_sum[t], 1);
    }

    if (tx == 0 && row < args.M) {
        for (uint t = 0; t < R1PTG; ++t) {
            const uint token = token_base + t;
            if (token < args.N) {
                device float * out = (device float *)dst + (uint64_t)(args.token_offset + token) * args.M + row;
                *out = swiglu(gate_sum[t], up_sum[t]);
            }
        }
    }
}
