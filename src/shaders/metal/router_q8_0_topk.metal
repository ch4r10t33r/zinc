#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

// Fused Q8_0 router for Qwen3.6 MoE decode/prefill.
//
// This is the single-token analogue of vLLM's topk -> selected-experts device
// path, with the Q8_0 row walk adapted from llama.cpp's
// ggml-metal.metal::kernel_mul_mv_q8_0_f32_impl. It writes the compact routing
// row consumed by ZINC's MoE DMMV kernels: [k expert ids][k f32 weights as u32].

struct RouterQ8TopkPush {
    uint n_experts;
    uint K;
    uint k;
    uint a_offset;
    uint x_offset;
};

#define MAX_EXPERTS 256
#define MAX_K_USED 16

kernel void main0(
    constant RouterQ8TopkPush& p [[buffer(0)]],
    device const uchar* W [[buffer(1)]],
    device const float* X [[buffer(2)]],
    device uint* output_data [[buffer(3)]],
    uint local_id [[thread_position_in_threadgroup]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroups_per_tg [[simdgroups_per_threadgroup]]
) {
    threadgroup float values[MAX_EXPERTS];
    threadgroup float selected_val[MAX_K_USED];

    device const float* input = X + (p.x_offset >> 2);
    const uint blocks_per_row = p.K >> 5;
    const ulong row_bytes = ulong(blocks_per_row) * 34ull;
    const uint row_pairs = (p.n_experts + 1u) >> 1;

    for (uint pair = sg_idx; pair < row_pairs; pair += simdgroups_per_tg) {
        const uint base_row = pair << 1;
        device const uchar* row0 = W + p.a_offset + ulong(base_row) * row_bytes;
        device const uchar* row1 = row0 + row_bytes;

        float acc0 = 0.0f;
        float acc1 = 0.0f;

        for (uint bi = lane; bi < blocks_per_row; bi += 32u) {
            device const uchar* blk0 = row0 + bi * 34u;
            device const uchar* blk1 = row1 + bi * 34u;
            const float s0 = float(as_type<half>(*(device const ushort*)(blk0)));
            const float s1 = float(as_type<half>(*(device const ushort*)(blk1)));
            device const packed_char4* q0 = (device const packed_char4*)(blk0 + 2u);
            device const packed_char4* q1 = (device const packed_char4*)(blk1 + 2u);
            const uint x_base = bi << 5;

            #pragma unroll
            for (uint vi = 0u; vi < 8u; ++vi) {
                const float4 x = *(device const float4*)(input + x_base + (vi << 2));
                acc0 = fma(s0, dot(float4(char4(q0[vi])), x), acc0);
                if (base_row + 1u < p.n_experts) {
                    acc1 = fma(s1, dot(float4(char4(q1[vi])), x), acc1);
                }
            }
        }

        const float sum0 = simd_sum(acc0);
        const float sum1 = simd_sum(acc1);
        if (lane == 0u) {
            values[base_row] = sum0;
            if (base_row + 1u < p.n_experts) {
                values[base_row + 1u] = sum1;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (local_id == 0u) {
        if (p.n_experts == 256u && p.k == 8u) {
            #pragma unroll
            for (uint slot = 0u; slot < 8u; slot++) {
                float best_val = -INFINITY;
                uint best_idx = 0u;
                for (uint expert = 0u; expert < 256u; expert++) {
                    const float v = values[expert];
                    if (v > best_val) {
                        best_val = v;
                        best_idx = expert;
                    }
                }
                output_data[slot] = best_idx;
                selected_val[slot] = best_val;
                values[best_idx] = -INFINITY;
            }

            float max_sel = -INFINITY;
            #pragma unroll
            for (uint slot = 0u; slot < 8u; slot++) {
                max_sel = max(max_sel, selected_val[slot]);
            }

            float sum = 0.0f;
            #pragma unroll
            for (uint slot = 0u; slot < 8u; slot++) {
                const float e = fast::exp(selected_val[slot] - max_sel);
                selected_val[slot] = e;
                sum += e;
            }

            const float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
            #pragma unroll
            for (uint slot = 0u; slot < 8u; slot++) {
                output_data[8u + slot] = as_type<uint>(selected_val[slot] * inv_sum);
            }
            return;
        }

        const uint k = min(p.k, uint(MAX_K_USED));
        for (uint slot = 0u; slot < k; slot++) {
            float best_val = -INFINITY;
            uint best_idx = 0u;
            for (uint expert = 0u; expert < p.n_experts; expert++) {
                const float v = values[expert];
                if (v > best_val) {
                    best_val = v;
                    best_idx = expert;
                }
            }
            output_data[slot] = best_idx;
            selected_val[slot] = best_val;
            values[best_idx] = -INFINITY;
        }

        float max_sel = -INFINITY;
        for (uint slot = 0u; slot < k; slot++) {
            max_sel = max(max_sel, selected_val[slot]);
        }

        float sum = 0.0f;
        for (uint slot = 0u; slot < k; slot++) {
            const float e = fast::exp(selected_val[slot] - max_sel);
            selected_val[slot] = e;
            sum += e;
        }

        const float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
        for (uint slot = 0u; slot < k; slot++) {
            output_data[p.k + slot] = as_type<uint>(selected_val[slot] * inv_sum);
        }
    }
}
