#include <metal_stdlib>
using namespace metal;

struct DmmvPush {
    uint M;
    uint K;
    uint a_offset;
    uint x_offset;
    uint y_offset;
};

inline float fp16_to_fp32(uint h) {
    return float(as_type<half>(ushort(h)));
}

inline float s8_to_f32(uint x) {
    return float((x < 128u) ? int(x) : (int(x) - 256));
}

// Port of llama.cpp's kernel_mul_mv_q6_K_f32 for dense single-token decode.
// Thread organization matches llama.cpp N_SG_Q6_K=2, N_R0_Q6_K=2:
// 64 threads = 2 simdgroups, each simdgroup computes 2 rows.
#define NSG 2
#define NR0 2
#define QK_K 256
#define BLOCK_SIZE 210
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

kernel void main0(
    device const uchar* W [[buffer(0)]],
    constant DmmvPush& p [[buffer(1)]],
    device const float* X [[buffer(2)]],
    device float* Y [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    const uint nb = p.K / QK_K;
    const uint first_row = (tgpig.x * NSG + uint(sgitg)) * NR0;
    const uint row_bytes = nb * BLOCK_SIZE;

    device const uchar* src0 = W + p.a_offset;
    device const float* src1 = X + (p.x_offset / 4u);

    constexpr uint kmask1 = 0x03u;
    constexpr uint kmask2 = 0x0Cu;
    constexpr uint kmask3 = 0x30u;
    constexpr uint kmask4 = 0xC0u;

    const ushort tid = tiisg / 2u;
    const ushort ix = tiisg % 2u;
    const ushort ip = tid / 8u;
    const ushort il = tid % 8u;
    const ushort l0 = 4u * il;
    const ushort is = 8u * ip + l0 / 16u;

    const uint y_offset = 128u * uint(ip) + uint(l0);
    const uint q_offset_l = 64u * uint(ip) + uint(l0);
    const uint q_offset_h = 32u * uint(ip) + uint(l0);

    float sumf[NR0] = {0.0f, 0.0f};

    for (uint bi = ix; bi < nb; bi += 2u) {
        device const float* y = src1 + bi * QK_K + y_offset;

        // Cycle 39: explicit FOR_UNROLL on the yl-fill loop. Mirrors cycle 37's
        // pragma added to the inner 32-FMA loop. Without unroll the yl[]
        // register array is indexed by a runtime loop variable, which can
        // force the compiler to spill yl[] to stack instead of promoting to
        // registers. Unrolling makes all 16 indices compile-time constants so
        // yl can live in SSA values for the duration of the FMA loop.
        float yl[16];
        FOR_UNROLL (ushort l = 0u; l < 4u; ++l) {
            yl[4u * l + 0u] = y[l + 0u];
            yl[4u * l + 1u] = y[l + 32u];
            yl[4u * l + 2u] = y[l + 64u];
            yl[4u * l + 3u] = y[l + 96u];
        }

        // Cycle 36: interleave NR0=2 row0/row1 — load both rows' q1v4/q2v4/qhv4/
        // sc/d up front, then run a single FOR_UNROLL l=0..3 that updates both
        // rows' sums alternately. Removes the per-row serialization chain (row 1's
        // loads previously waited on row 0's accumulator chain). Mirrors cycle 33's
        // row-interleave pattern that landed in dmmv_q4k.metal (+0.3 tok/s on dense
        // Q4_K). Bounds check moved to output simd_sum loop (matches dmmv_q4k.metal
        // pattern); dispatched M for ffn_down (4096) and lm_head (151936) on
        // Qwen3-8B are both divisible by NSG*NR0=4.
        device const uchar* block0 = src0 + ulong(first_row) * ulong(row_bytes) + ulong(bi) * BLOCK_SIZE;
        device const uchar* block1 = block0 + row_bytes;

        const uchar4 q1v4_0 = *((device const uchar4*)(block0 + q_offset_l));
        const uchar4 q2v4_0 = *((device const uchar4*)(block0 + q_offset_l + 32u));
        const uchar4 qhv4_0 = *((device const uchar4*)(block0 + 128u + q_offset_h));
        device const uchar* sc_0 = block0 + 192u + uint(is);
        // Cycle 38: replace 2 scalar uchar reads + bit-shift + as_type<half> with a
        // single 2-byte aligned half load. block+208 is 2-byte aligned (BLOCK_SIZE=210
        // is even, row_bytes = nb*210 is even, base buffer is ≥16-byte aligned by Metal
        // convention). Saves the uint construction (block[208] | (block[209] << 8))
        // and the ushort→half bit-cast, leaving a direct half→float promotion.
        const float d_0 = float(*((device const half*)(block0 + 208)));

        const uchar4 q1v4_1 = *((device const uchar4*)(block1 + q_offset_l));
        const uchar4 q2v4_1 = *((device const uchar4*)(block1 + q_offset_l + 32u));
        const uchar4 qhv4_1 = *((device const uchar4*)(block1 + 128u + q_offset_h));
        device const uchar* sc_1 = block1 + 192u + uint(is);
        const float d_1 = float(*((device const half*)(block1 + 208)));

        float4 sums_0 = float4(0.0f);
        float4 sums_1 = float4(0.0f);
        FOR_UNROLL (ushort l = 0u; l < 4u; ++l) {
            const uint h0 = uint(qhv4_0[l]);
            const uint q1b0 = uint(q1v4_0[l]);
            const uint q2b0 = uint(q2v4_0[l]);
            const float q00 = float(int((q1b0 & 0x0Fu) | ((h0 & kmask1) << 4u)) - 32);
            const float q10 = float(int((q2b0 & 0x0Fu) | ((h0 & kmask2) << 2u)) - 32);
            const float q20 = float(int((q1b0 >> 4u) | ((h0 & kmask3) << 0u)) - 32);
            const float q30 = float(int((q2b0 >> 4u) | ((h0 & kmask4) >> 2u)) - 32);

            const uint h1 = uint(qhv4_1[l]);
            const uint q1b1 = uint(q1v4_1[l]);
            const uint q2b1 = uint(q2v4_1[l]);
            const float q01 = float(int((q1b1 & 0x0Fu) | ((h1 & kmask1) << 4u)) - 32);
            const float q11 = float(int((q2b1 & 0x0Fu) | ((h1 & kmask2) << 2u)) - 32);
            const float q21 = float(int((q1b1 >> 4u) | ((h1 & kmask3) << 0u)) - 32);
            const float q31 = float(int((q2b1 >> 4u) | ((h1 & kmask4) >> 2u)) - 32);

            const float ylv0 = yl[4u * l + 0u];
            const float ylv1 = yl[4u * l + 1u];
            const float ylv2 = yl[4u * l + 2u];
            const float ylv3 = yl[4u * l + 3u];

            sums_0[0] += ylv0 * q00;
            sums_1[0] += ylv0 * q01;
            sums_0[1] += ylv1 * q10;
            sums_1[1] += ylv1 * q11;
            sums_0[2] += ylv2 * q20;
            sums_1[2] += ylv2 * q21;
            sums_0[3] += ylv3 * q30;
            sums_1[3] += ylv3 * q31;
        }

        sumf[0] += d_0 * (
            sums_0[0] * s8_to_f32(uint(sc_0[0])) +
            sums_0[1] * s8_to_f32(uint(sc_0[2])) +
            sums_0[2] * s8_to_f32(uint(sc_0[4])) +
            sums_0[3] * s8_to_f32(uint(sc_0[6]))
        );
        sumf[1] += d_1 * (
            sums_1[0] * s8_to_f32(uint(sc_1[0])) +
            sums_1[1] * s8_to_f32(uint(sc_1[2])) +
            sums_1[2] * s8_to_f32(uint(sc_1[4])) +
            sums_1[3] * s8_to_f32(uint(sc_1[6]))
        );
    }

    device float* out = Y + (p.y_offset / 4u);
    for (ushort row = 0u; row < NR0; ++row) {
        const uint dst_row = first_row + uint(row);
        if (dst_row >= p.M) {
            continue;
        }

        const float total = simd_sum(sumf[row]);
        if (tiisg == 0u) {
            out[dst_row] = total;
        }
    }
}
