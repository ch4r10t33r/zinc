#include <metal_stdlib>
using namespace metal;

struct FlashAttnPush {
    uint head_dim;
    uint n_heads;
    uint n_kv_heads;
    uint seq_len;
    uint sliding_window_size;
    uint page_size;
    uint attn_scale_bits;
    uint kv_head_stride_bytes;
    uint kv_token_stride_bytes;
};

constant uint FLASH_TG_SIZE = 64;
constant uint FLASH_BLOCK_TOKENS = 256;
// Max head_dim supported. Gemma 4 global attention layers use head_dim=512;
// SWA layers use 256. Sized for the larger value so a single shader handles both.
constant uint FLASH_MAX_HEAD_DIM = 512;
constant uint FLASH_MAX_HEAD_VEC4 = FLASH_MAX_HEAD_DIM / 4;

// Simdgroup-redundant-reduction pattern: each simdgroup lane-0 writes its
// wave-reduced partial into scratch, then after a single threadgroup barrier
// every thread re-reads all partials and merges them locally. Saves the
// second broadcast barrier vs the prior "tid==0 merges, barrier, all read"
// idiom. Same pattern that won in cycles 15/16/17 for the rms_norm shaders.
inline float reduceThreadgroupMax(
    float local_value,
    threadgroup float* scratch,
    uint tid,
    uint subgroup_size,
    uint simd_lane,
    uint simd_group
) {
    const float wave_max = simd_max(local_value);
    if (subgroup_size < FLASH_TG_SIZE) {
        if (simd_lane == 0u) {
            scratch[simd_group] = wave_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint n_groups = (FLASH_TG_SIZE + subgroup_size - 1u) / subgroup_size;
        float merged = -INFINITY;
        for (uint sg = 0u; sg < n_groups; ++sg) {
            merged = fast::max(merged, scratch[sg]);
        }
        return merged;
    }
    return wave_max;
}

inline float reduceThreadgroupSum(
    float local_value,
    threadgroup float* scratch,
    uint tid,
    uint subgroup_size,
    uint simd_lane,
    uint simd_group
) {
    const float wave_sum = simd_sum(local_value);
    if (subgroup_size < FLASH_TG_SIZE) {
        if (simd_lane == 0u) {
            scratch[simd_group] = wave_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint n_groups = (FLASH_TG_SIZE + subgroup_size - 1u) / subgroup_size;
        float merged = 0.0f;
        for (uint sg = 0u; sg < n_groups; ++sg) {
            merged += scratch[sg];
        }
        return merged;
    }
    return wave_sum;
}

inline uint kvBaseForToken(
    device const uint* page_table,
    constant FlashAttnPush& p,
    uint kv_head,
    uint token_idx
) {
    const uint page_size = max(p.page_size, 1u);
    const uint page = token_idx / page_size;
    const uint page_offset = token_idx % page_size;
    const uint physical_token = page_table[page] * page_size + page_offset;
    return (physical_token * p.n_kv_heads + kv_head) * p.head_dim;
}

kernel void main0(
    constant FlashAttnPush& p [[buffer(0)]],
    device const uint* page_table [[buffer(1)]],
    device const float* q [[buffer(2)]],
    device const float* k_cache [[buffer(3)]],
    device const float* v_cache [[buffer(4)]],
    device float* out [[buffer(5)]],
    device const float* sinks [[buffer(6)]],  // per-head attention sink values
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint subgroup_size [[thread_execution_width]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]]
) {
    const uint q_per_kv = max(p.n_heads / max(p.n_kv_heads, 1u), 1u);
    const uint kv_head = head / q_per_kv;
    const uint q_base = head * p.head_dim;
    const uint vec4_dim = p.head_dim >> 2;
    const float scale = p.attn_scale_bits != 0u ? as_type<float>(p.attn_scale_bits) : rsqrt((float)p.head_dim);
    const bool contiguous_kv = p.page_size == 0u;
    const uint token_stride = contiguous_kv ? (p.kv_token_stride_bytes / uint(sizeof(float))) : (p.n_kv_heads * p.head_dim);
    const uint kv_head_stride = contiguous_kv ? (p.kv_head_stride_bytes / uint(sizeof(float))) : p.head_dim;
    const bool use_sliding_window = p.sliding_window_size > 0u && p.sliding_window_size < p.seq_len;
    const uint sliding_start = use_sliding_window ? (p.seq_len - p.sliding_window_size) : 0u;

    threadgroup float4 q_cache4[FLASH_MAX_HEAD_VEC4];
    threadgroup float4 acc_cache4[FLASH_MAX_HEAD_VEC4];
    threadgroup float scores[FLASH_BLOCK_TOKENS];
    threadgroup float reduce[FLASH_TG_SIZE];
    // running_max/running_sum are per-thread registers (every thread maintains
    // the identical running state in lockstep since rescale/block_max/block_sum
    // are produced by full-threadgroup reductions). Saves the broadcast barrier
    // that previously protected tid==0's update to threadgroup-scoped versions.
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    // Strided loop: vec4_dim may exceed FLASH_TG_SIZE when head_dim > 256
    // (e.g. Gemma 4 global attention layers use head_dim=512 → vec4_dim=128).
    for (uint i = tid; i < vec4_dim; i += FLASH_TG_SIZE) {
        q_cache4[i] = *(device const float4*)(q + q_base + (i << 2));
        acc_cache4[i] = float4(0.0f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint block_start = 0u; block_start < p.seq_len; block_start += FLASH_BLOCK_TOKENS) {
        const uint block_tokens = min(FLASH_BLOCK_TOKENS, p.seq_len - block_start);
        const uint block_base = (block_start * token_stride) + kv_head * kv_head_stride;
        float local_max = -INFINITY;

        for (uint token_offset = tid; token_offset < block_tokens; token_offset += FLASH_TG_SIZE) {
            const uint token_idx = block_start + token_offset;
            if (use_sliding_window && token_idx < sliding_start) {
                scores[token_offset] = -INFINITY;
                continue;
            }
            const uint kv_base = contiguous_kv
                ? (block_base + token_offset * token_stride)
                : kvBaseForToken(page_table, p, kv_head, token_idx);

            float score = 0.0f;
            for (uint i = 0; i < vec4_dim; i++) {
                const float4 qv = q_cache4[i];
                const float4 kv = *(device const float4*)(k_cache + kv_base + (i << 2));
                score += dot(qv, kv);
            }
            score *= scale;
            scores[token_offset] = score;
            local_max = fast::max(local_max, score);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float block_max = reduceThreadgroupMax(local_max, reduce, tid, subgroup_size, simd_lane, simd_group);
        const float next_max = fast::max(running_max, block_max);

        float local_sum = 0.0f;
        for (uint token_offset = tid; token_offset < block_tokens; token_offset += FLASH_TG_SIZE) {
            const float weight = fast::exp(scores[token_offset] - next_max);
            scores[token_offset] = weight;
            local_sum += weight;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float block_sum = reduceThreadgroupSum(local_sum, reduce, tid, subgroup_size, simd_lane, simd_group);
        const float rescale = running_sum > 0.0f ? fast::exp(running_max - next_max) : 0.0f;

        // Strided loop over head_dim slices when vec4_dim > FLASH_TG_SIZE.
        for (uint vi = tid; vi < vec4_dim; vi += FLASH_TG_SIZE) {
            float4 acc = acc_cache4[vi] * rescale;
            const uint dim_base = vi << 2;

            if (contiguous_kv) {
                uint kv_base = block_base + dim_base;
                for (uint token_offset = 0; token_offset < block_tokens; token_offset++) {
                    acc += *(device const float4*)(v_cache + kv_base) * scores[token_offset];
                    kv_base += token_stride;
                }
            } else {
                for (uint token_offset = 0; token_offset < block_tokens; token_offset++) {
                    const uint kv_base = kvBaseForToken(page_table, p, kv_head, block_start + token_offset);
                    acc += *(device const float4*)(v_cache + kv_base + dim_base) * scores[token_offset];
                }
            }

            acc_cache4[vi] = acc;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Every thread maintains the identical running state in registers — no
        // broadcast needed because rescale/block_sum/next_max are all derived
        // from full-threadgroup reductions executed above.
        running_sum = running_sum * rescale + block_sum;
        running_max = next_max;
    }

    // Attention sink: per-head learned scalar acts as virtual token in softmax.
    // Absorbs excess attention weight (no V contribution). sinks[head] = NaN means disabled.
    float final_sum = running_sum;
    const float sink_val = sinks[head];
    if (!isnan(sink_val)) {
        const float sink_max = fast::max(running_max, sink_val);
        const float rescale_s = running_sum > 0.0f ? fast::exp(running_max - sink_max) : 0.0f;
        final_sum = running_sum * rescale_s + fast::exp(sink_val - sink_max);
        for (uint vi = tid; vi < vec4_dim; vi += FLASH_TG_SIZE) {
            acc_cache4[vi] *= rescale_s;
        }
    }

    // fast::divide maps to Apple GPU hardware reciprocal+mul (vs precise IEEE
    // division), saving ~10 cycles per call. Fires once per Q head per layer
    // (~1152 calls/token on Qwen3-8B: 32 heads × 36 layers). Companion to the
    // fast::exp uses above in the running softmax; precision stays within the
    // already-accepted FA approximation budget.
    const float inv_sum = final_sum > 0.0f ? fast::divide(1.0f, final_sum) : 0.0f;
    for (uint vi = tid; vi < vec4_dim; vi += FLASH_TG_SIZE) {
        *(device float4*)(out + q_base + (vi << 2)) = acc_cache4[vi] * inv_sum;
    }
}
