#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

// Qwen3.6 F32-router MoE boundary fusion.
//
// Fuses residual_rms_norm.metal with router_f32_topk_batched.metal for the
// single-token prefill path: hidden += residual, materialize ffn_norm for the
// expert DMMVs, then route the F32 router weights from the cached normalized
// vector in the same dispatch.

struct Params {
    uint n;
    float eps;
    float scale;
    uint residual_offset;
    uint n_experts;
    uint K;
    uint k;
    uint a_offset;
};

#define TG_SIZE 512
#define SIMD_WIDTH 32
#define N_SIMDGROUPS (TG_SIZE / SIMD_WIDTH)
#define ROWS_PER_TG (N_SIMDGROUPS * 2)
#define MAX_EXPERTS 256
#define MAX_K_USED 16
#define MAX_K_VEC4 1024

kernel void main0(
    constant Params& p [[buffer(0)]],
    device float* hidden [[buffer(1)]],
    device const float* residual [[buffer(2)]],
    device float* norm_out [[buffer(3)]],
    device const float* norm_weight [[buffer(4)]],
    device const float* W [[buffer(5)]],
    device uint* output_data [[buffer(6)]],
    uint local_id [[thread_position_in_threadgroup]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    if (p.n == 0u || p.K != p.n || (p.K & 3u) != 0u ||
        (p.K >> 2) > MAX_K_VEC4 ||
        p.n_experts == 0u || p.n_experts > MAX_EXPERTS ||
        p.k == 0u || p.k > MAX_K_USED) {
        return;
    }

    threadgroup float4 x_cache4[MAX_K_VEC4];
    threadgroup float partial_sums[N_SIMDGROUPS];
    threadgroup float values[MAX_EXPERTS];
    threadgroup float selected_val[MAX_K_USED];

    const uint k_vec4 = p.K >> 2;
    float sum_sq = 0.0f;
    for (uint vi = local_id; vi < k_vec4; vi += TG_SIZE) {
        const uint idx = vi << 2;
        const float4 h = *(device const float4*)(hidden + idx);
        const float4 r = *(device const float4*)(residual + p.residual_offset + idx);
        const float4 updated = fma(float4(p.scale), r, h);
        *(device float4*)(hidden + idx) = updated;
        x_cache4[vi] = updated;
        sum_sq += dot(updated, updated);
    }

    const float sg_sum = simd_sum(sum_sq);
    if (lane == 0u) {
        partial_sums[sg_idx] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float partial = (lane < N_SIMDGROUPS) ? partial_sums[lane] : 0.0f;
    const float total_sq = simd_sum(partial);
    const float rms_inv = fast::rsqrt(fast::divide(total_sq, float(p.n)) + p.eps);

    for (uint vi = local_id; vi < k_vec4; vi += TG_SIZE) {
        const uint idx = vi << 2;
        const float4 w = *(device const float4*)(norm_weight + idx);
        const float4 nval = w * (x_cache4[vi] * rms_inv);
        x_cache4[vi] = nval;
        *(device float4*)(norm_out + idx) = nval;
    }

    if (local_id < MAX_EXPERTS) {
        values[local_id] = -INFINITY;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint weight_base = p.a_offset >> 2;
    for (uint row_block = 0u; row_block < p.n_experts; row_block += ROWS_PER_TG) {
        const uint base_row = row_block + sg_idx * 2u;
        float acc0 = 0.0f;
        float acc1 = 0.0f;

        if (base_row < p.n_experts) {
            device const float* row0 = W + weight_base + base_row * p.K;
            device const float* row1 = row0 + p.K;

            for (uint vi = lane; vi < k_vec4; vi += SIMD_WIDTH) {
                const float4 x = x_cache4[vi];
                acc0 += dot(*(device const float4*)(row0 + (vi << 2)), x);
                if (base_row + 1u < p.n_experts) {
                    acc1 += dot(*(device const float4*)(row1 + (vi << 2)), x);
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
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (p.n_experts == 256u && p.k == 8u) {
        if (sg_idx == 0u) {
            float selected_score[8];
            uint selected_mask = 0u;
            #pragma unroll
            for (uint slot = 0u; slot < 8u; ++slot) {
                selected_score[slot] = -INFINITY;
            }

            #pragma unroll
            for (uint slot = 0u; slot < 8u; ++slot) {
                float lane_best = -INFINITY;
                uint lane_best_idx = 0xffffffffu;
                #pragma unroll
                for (uint lane_row = 0u; lane_row < 8u; ++lane_row) {
                    const uint expert = lane + (lane_row << 5);
                    const float score = ((selected_mask & (1u << lane_row)) == 0u) ? values[expert] : -INFINITY;
                    if (score > lane_best) {
                        lane_best = score;
                        lane_best_idx = expert;
                    }
                }
                const float best_val = simd_max(lane_best);
                const uint best_idx = simd_min((lane_best == best_val) ? lane_best_idx : 0xffffffffu);
                selected_score[slot] = best_val;
                if ((best_idx & 31u) == lane) {
                    selected_mask |= 1u << (best_idx >> 5);
                }
                if (lane == 0u) {
                    output_data[slot] = best_idx;
                }
            }

            const bool weight_lane = lane < 8u;
            const float score = weight_lane ? selected_score[lane] : -INFINITY;
            const float max_sel = simd_max(score);
            const float exp_score = weight_lane ? fast::exp(score - max_sel) : 0.0f;
            const float sum = simd_sum(exp_score);
            const float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
            if (weight_lane) {
                output_data[8u + lane] = as_type<uint>(exp_score * inv_sum);
            }
        }
        return;
    }

    if (local_id == 0u) {
        const uint k = min(p.k, uint(MAX_K_USED));
        for (uint slot = 0u; slot < k; ++slot) {
            float best_val = -INFINITY;
            uint best_idx = 0u;
            for (uint expert = 0u; expert < p.n_experts; ++expert) {
                const float score = values[expert];
                if (score > best_val) {
                    best_val = score;
                    best_idx = expert;
                }
            }
            output_data[slot] = best_idx;
            selected_val[slot] = best_val;
            values[best_idx] = -INFINITY;
        }

        float max_sel = -INFINITY;
        for (uint slot = 0u; slot < k; ++slot) {
            max_sel = max(max_sel, selected_val[slot]);
        }

        float sum = 0.0f;
        for (uint slot = 0u; slot < k; ++slot) {
            const float e = fast::exp(selected_val[slot] - max_sel);
            selected_val[slot] = e;
            sum += e;
        }

        const float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 0.0f;
        for (uint slot = 0u; slot < k; ++slot) {
            output_data[p.k + slot] = as_type<uint>(selected_val[slot] * inv_sum);
        }
    }
}
