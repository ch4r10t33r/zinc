#include <metal_stdlib>
using namespace metal;

struct DmmvPush {
    uint M;
    uint K;
    uint a_offset;
    uint x_offset;
    uint y_offset;
};

// Q8_0 DMMV specialization for Qwen3.6 shared-down (M=2048, K=512).
//
// The generic quad-row Q8 path assigns one Q8 block to one lane. At K=512 that
// leaves lanes 16..31 idle because there are only 16 blocks per row. Split each
// block across two lanes instead, keeping four output rows per simdgroup while
// making every lane contribute to the dot product.
kernel void main0(
    constant DmmvPush& p [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device const float* X [[buffer(2)]],
    device float* Y [[buffer(3)]],
    uint tg_id [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroups_per_tg [[simdgroups_per_threadgroup]]
) {
    const uint base_row = (tg_id * simdgroups_per_tg + sg_idx) * 4u;
    if (base_row >= p.M) return;

    device const float* input = X + (p.x_offset >> 2);
    device float* output = Y + (p.y_offset >> 2);

    const uint blocks_per_row = 16u; // K=512, Q8_0 block size=32
    const ulong row_bytes = ulong(blocks_per_row) * 34ull;
    const bool has1 = base_row + 1u < p.M;
    const bool has2 = base_row + 2u < p.M;
    const bool has3 = base_row + 3u < p.M;

    device const uchar* row0 = W + p.a_offset + ulong(base_row) * row_bytes;
    device const uchar* row1 = has1 ? (row0 + row_bytes) : row0;
    device const uchar* row2 = has2 ? (row0 + 2ull * row_bytes) : row0;
    device const uchar* row3 = has3 ? (row0 + 3ull * row_bytes) : row0;

    const uint block_idx = lane >> 1;
    const uint half_sel = lane & 1u;
    const uint half_base = half_sel << 4; // 0 or 16 elements within the Q8_0 block
    const uint x_base = (block_idx << 5) + half_base;

    device const uchar* blk0 = row0 + block_idx * 34u;
    device const uchar* blk1 = row1 + block_idx * 34u;
    device const uchar* blk2 = row2 + block_idx * 34u;
    device const uchar* blk3 = row3 + block_idx * 34u;

    const float s0 = float(as_type<half>(*(device const ushort*)(blk0)));
    const float s1 = float(as_type<half>(*(device const ushort*)(blk1)));
    const float s2 = float(as_type<half>(*(device const ushort*)(blk2)));
    const float s3 = float(as_type<half>(*(device const ushort*)(blk3)));

    device const packed_char4* q0 = (device const packed_char4*)(blk0 + 2u + half_base);
    device const packed_char4* q1 = (device const packed_char4*)(blk1 + 2u + half_base);
    device const packed_char4* q2 = (device const packed_char4*)(blk2 + 2u + half_base);
    device const packed_char4* q3 = (device const packed_char4*)(blk3 + 2u + half_base);

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    #pragma unroll
    for (uint vi = 0u; vi < 4u; ++vi) {
        const float4 x = *(device const float4*)(input + x_base + (vi << 2));
        acc0 = fma(s0, dot(float4(char4(q0[vi])), x), acc0);
        acc1 = fma(s1, dot(float4(char4(q1[vi])), x), acc1);
        acc2 = fma(s2, dot(float4(char4(q2[vi])), x), acc2);
        acc3 = fma(s3, dot(float4(char4(q3[vi])), x), acc3);
    }

    const float sum0 = simd_sum(acc0);
    const float sum1 = simd_sum(acc1);
    const float sum2 = simd_sum(acc2);
    const float sum3 = simd_sum(acc3);
    if (lane == 0u) {
        output[base_row] = sum0;
        if (has1) output[base_row + 1u] = sum1;
        if (has2) output[base_row + 2u] = sum2;
        if (has3) output[base_row + 3u] = sum3;
    }
}
