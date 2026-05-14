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
    threadgroup float scores[FLASH_BLOCK_TOKENS];
    threadgroup float reduce[FLASH_TG_SIZE];
    // running_max/running_sum are per-thread registers (every thread maintains
    // the identical running state in lockstep since rescale/block_max/block_sum
    // are produced by full-threadgroup reductions). Saves the broadcast barrier
    // that previously protected tid==0's update to threadgroup-scoped versions.
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    // Cycle 94: keep per-thread V accumulators in registers instead of the
    // threadgroup `acc_cache4` buffer. Because the V loop reads/writes
    // `acc_cache4[vi]` with the strided pattern `vi = tid; vi += FLASH_TG_SIZE`,
    // each thread *always* owns the same set of vi values across every
    // block_start iter (and across the sink-scale + output-writeback passes).
    // So the buffer was acting as expensive per-thread state instead of cross-
    // thread shared state. With FLASH_MAX_HEAD_VEC4=128 and FLASH_TG_SIZE=64,
    // each thread owns at most ceil(128/64)=2 vi values → `float4 acc_local[2]`
    // = 8 floats = 32B per thread (fits easily in registers). Removes a 2 KiB
    // threadgroup allocation, the per-block-iter TG read+write of acc_cache4,
    // and the sink-rescale + output-write TG roundtrips. Mirrors cycle 92's
    // "kill dead-after-use threadgroup roundtrips" philosophy applied to the
    // V accumulator, in the same kernel.
    float4 acc_local[(FLASH_MAX_HEAD_VEC4 + FLASH_TG_SIZE - 1u) / FLASH_TG_SIZE];
    for (uint li = 0u; li < (FLASH_MAX_HEAD_VEC4 + FLASH_TG_SIZE - 1u) / FLASH_TG_SIZE; ++li) {
        acc_local[li] = float4(0.0f);
    }

    // Strided loop: vec4_dim may exceed FLASH_TG_SIZE when head_dim > 256
    // (e.g. Gemma 4 global attention layers use head_dim=512 → vec4_dim=128).
    for (uint i = tid; i < vec4_dim; i += FLASH_TG_SIZE) {
        q_cache4[i] = *(device const float4*)(q + q_base + (i << 2));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Cycle 92: keep per-thread scores in a register array across the
    // QK→softmax pipeline. The two strided loops below visit the same
    // per-thread indices {tid, tid+64, …}; the legacy form wrote each
    // `score` into `scores[token_offset]` (TG memory) at end-of-QK and
    // re-read it as `scores[token_offset]` at top-of-softmax — a pure
    // intra-thread roundtrip through TG memory. With FLASH_BLOCK_TOKENS=256
    // and FLASH_TG_SIZE=64, every thread owns at most 4 tokens per block,
    // so a stack-resident `float local_scores[4]` fits comfortably (16B).
    // The softmax write of `weight` back to `scores[]` is preserved
    // because the V loop reads weights cross-thread (guarded by the
    // existing barrier after softmax). Net: 1 TG-write + 1 TG-read removed
    // per (token,thread) per block per (head×layer). Mirrors the philosophy
    // of cycle 91's "kill dead-after-use threadgroup roundtrips" cleanups.
    float local_scores[FLASH_BLOCK_TOKENS / FLASH_TG_SIZE];

    for (uint block_start = 0u; block_start < p.seq_len; block_start += FLASH_BLOCK_TOKENS) {
        const uint block_tokens = min(FLASH_BLOCK_TOKENS, p.seq_len - block_start);
        const uint block_base = (block_start * token_stride) + kv_head * kv_head_stride;
        float local_max = -INFINITY;

        uint local_idx = 0u;
        for (uint token_offset = tid; token_offset < block_tokens; token_offset += FLASH_TG_SIZE) {
            const uint token_idx = block_start + token_offset;
            if (use_sliding_window && token_idx < sliding_start) {
                local_scores[local_idx++] = -INFINITY;
                continue;
            }
            const uint kv_base = contiguous_kv
                ? (block_base + token_offset * token_stride)
                : kvBaseForToken(page_table, p, kv_head, token_idx);

            // Cycle 90: vectorize the QK score reduction. The legacy form
            // `score += dot(qv, kv)` collapses each 4-wide qv*kv to a scalar
            // *inside* the loop, forcing a serial scalar accumulator chain
            // vec4_dim-deep (32 for Qwen3-8B head_dim=128). Replace with a
            // 4-wide FMA accumulator `score4 = fma(qv, kv, score4)` that
            // builds 4 independent parallel accumulator lanes (one per lane
            // of qv*kv), then collapse with `dot(score4, 1.f)` after the
            // loop — reduction depth vec4_dim → vec4_dim/4. Same compiler-
            // hint philosophy as cycles 44-89 on Q4_K matvec kernels: tell
            // the compiler the 4-wide ALU shape explicitly instead of relying
            // on auto-vectorization across a scalar reduction. Sum-of-
            // products = sum-of-sums by reassociation; the lane order of
            // additions differs but with fast:: math already enabled (see
            // fast::exp/max/divide in this kernel) the result is within
            // existing tolerance. flash_attn is the second-largest dispatch
            // bucket since cycle 30 (per EFFORT_14_NOTES.md priorities) and
            // fires ~1152 times per token on Qwen3-8B (32 Q-heads ×
            // 36 layers).
            float4 score4 = float4(0.0f);
            for (uint i = 0; i < vec4_dim; i++) {
                const float4 qv = q_cache4[i];
                const float4 kv = *(device const float4*)(k_cache + kv_base + (i << 2));
                score4 = fma(qv, kv, score4);
            }
            float score = dot(score4, float4(1.0f));
            score *= scale;
            local_scores[local_idx++] = score;
            local_max = fast::max(local_max, score);
        }
        // No threadgroup barrier here: scores writes above are at per-thread
        // indices {tid, tid+64, ...} and the softmax loop below reads the
        // same per-thread indices — pure intra-thread dependency. The cross-
        // thread scores read happens in the V loop, which is guarded by the
        // barrier after the softmax loop. reduceThreadgroupMax uses thread-
        // local local_max and has its own internal barrier on the scratch
        // array (separate shmem region from scores).

        const float block_max = reduceThreadgroupMax(local_max, reduce, tid, subgroup_size, simd_lane, simd_group);
        const float next_max = fast::max(running_max, block_max);

        float local_sum = 0.0f;
        local_idx = 0u;
        for (uint token_offset = tid; token_offset < block_tokens; token_offset += FLASH_TG_SIZE) {
            const float weight = fast::exp(local_scores[local_idx++] - next_max);
            scores[token_offset] = weight;
            local_sum += weight;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float block_sum = reduceThreadgroupSum(local_sum, reduce, tid, subgroup_size, simd_lane, simd_group);
        const float rescale = running_sum > 0.0f ? fast::exp(running_max - next_max) : 0.0f;

        // Strided loop over head_dim slices when vec4_dim > FLASH_TG_SIZE.
        // Cycle 93: extend cycle 91's 4-wide V accumulator to 8-wide
        // (acc0..acc7 over 8 staggered tokens). With ~133 V·scores FMAs per
        // (vi, block) for Qwen3-8B decode at n=128, the 4-wide form had 4
        // serial chains of depth ~33 FMAs each; the 8-wide form has 8 chains
        // of depth ~17 — halving the critical-path FMA latency if the loop
        // is latency-bound (which it is at chain-depth ~33 × FMA latency
        // ≫ throughput-limit). 8 float4 accumulators + 8 transient float4
        // V loads = 64 32-bit regs of pressure, well within Apple GPU per-
        // thread register budget. Tail at 4-wide → 1-wide preserves
        // correctness for residual tokens. Same associativity argument as
        // cycles 90/91: sum-of-(v*s) = sum-of-sums by reassociation, within
        // existing fast::math tolerance. Collapse with a balanced 3-level
        // tree `((a0+a1)+(a2+a3)) + ((a4+a5)+(a6+a7))` to keep the final
        // reduction depth at log2(8)=3 instead of serial.
        uint li = 0u;
        for (uint vi = tid; vi < vec4_dim; vi += FLASH_TG_SIZE) {
            float4 acc0 = acc_local[li] * rescale;
            float4 acc1 = float4(0.0f);
            float4 acc2 = float4(0.0f);
            float4 acc3 = float4(0.0f);
            float4 acc4 = float4(0.0f);
            float4 acc5 = float4(0.0f);
            float4 acc6 = float4(0.0f);
            float4 acc7 = float4(0.0f);
            const uint dim_base = vi << 2;

            if (contiguous_kv) {
                uint kv_base = block_base + dim_base;
                const uint stride2 = token_stride << 1;
                const uint stride3 = stride2 + token_stride;
                const uint stride4 = token_stride << 2;
                const uint stride5 = stride4 + token_stride;
                const uint stride6 = stride4 + stride2;
                const uint stride7 = stride4 + stride3;
                const uint stride8 = token_stride << 3;
                uint t = 0;
                for (; t + 8u <= block_tokens; t += 8u) {
                    const float4 v0 = *(device const float4*)(v_cache + kv_base);
                    const float4 v1 = *(device const float4*)(v_cache + kv_base + token_stride);
                    const float4 v2 = *(device const float4*)(v_cache + kv_base + stride2);
                    const float4 v3 = *(device const float4*)(v_cache + kv_base + stride3);
                    const float4 v4 = *(device const float4*)(v_cache + kv_base + stride4);
                    const float4 v5 = *(device const float4*)(v_cache + kv_base + stride5);
                    const float4 v6 = *(device const float4*)(v_cache + kv_base + stride6);
                    const float4 v7 = *(device const float4*)(v_cache + kv_base + stride7);
                    acc0 += v0 * scores[t + 0u];
                    acc1 += v1 * scores[t + 1u];
                    acc2 += v2 * scores[t + 2u];
                    acc3 += v3 * scores[t + 3u];
                    acc4 += v4 * scores[t + 4u];
                    acc5 += v5 * scores[t + 5u];
                    acc6 += v6 * scores[t + 6u];
                    acc7 += v7 * scores[t + 7u];
                    kv_base += stride8;
                }
                for (; t + 4u <= block_tokens; t += 4u) {
                    const float4 v0 = *(device const float4*)(v_cache + kv_base);
                    const float4 v1 = *(device const float4*)(v_cache + kv_base + token_stride);
                    const float4 v2 = *(device const float4*)(v_cache + kv_base + stride2);
                    const float4 v3 = *(device const float4*)(v_cache + kv_base + stride3);
                    acc0 += v0 * scores[t + 0u];
                    acc1 += v1 * scores[t + 1u];
                    acc2 += v2 * scores[t + 2u];
                    acc3 += v3 * scores[t + 3u];
                    kv_base += stride4;
                }
                for (; t < block_tokens; t++) {
                    acc0 += *(device const float4*)(v_cache + kv_base) * scores[t];
                    kv_base += token_stride;
                }
            } else {
                uint t = 0;
                for (; t + 8u <= block_tokens; t += 8u) {
                    const uint kb0 = kvBaseForToken(page_table, p, kv_head, block_start + t + 0u);
                    const uint kb1 = kvBaseForToken(page_table, p, kv_head, block_start + t + 1u);
                    const uint kb2 = kvBaseForToken(page_table, p, kv_head, block_start + t + 2u);
                    const uint kb3 = kvBaseForToken(page_table, p, kv_head, block_start + t + 3u);
                    const uint kb4 = kvBaseForToken(page_table, p, kv_head, block_start + t + 4u);
                    const uint kb5 = kvBaseForToken(page_table, p, kv_head, block_start + t + 5u);
                    const uint kb6 = kvBaseForToken(page_table, p, kv_head, block_start + t + 6u);
                    const uint kb7 = kvBaseForToken(page_table, p, kv_head, block_start + t + 7u);
                    acc0 += *(device const float4*)(v_cache + kb0 + dim_base) * scores[t + 0u];
                    acc1 += *(device const float4*)(v_cache + kb1 + dim_base) * scores[t + 1u];
                    acc2 += *(device const float4*)(v_cache + kb2 + dim_base) * scores[t + 2u];
                    acc3 += *(device const float4*)(v_cache + kb3 + dim_base) * scores[t + 3u];
                    acc4 += *(device const float4*)(v_cache + kb4 + dim_base) * scores[t + 4u];
                    acc5 += *(device const float4*)(v_cache + kb5 + dim_base) * scores[t + 5u];
                    acc6 += *(device const float4*)(v_cache + kb6 + dim_base) * scores[t + 6u];
                    acc7 += *(device const float4*)(v_cache + kb7 + dim_base) * scores[t + 7u];
                }
                for (; t + 4u <= block_tokens; t += 4u) {
                    const uint kb0 = kvBaseForToken(page_table, p, kv_head, block_start + t + 0u);
                    const uint kb1 = kvBaseForToken(page_table, p, kv_head, block_start + t + 1u);
                    const uint kb2 = kvBaseForToken(page_table, p, kv_head, block_start + t + 2u);
                    const uint kb3 = kvBaseForToken(page_table, p, kv_head, block_start + t + 3u);
                    acc0 += *(device const float4*)(v_cache + kb0 + dim_base) * scores[t + 0u];
                    acc1 += *(device const float4*)(v_cache + kb1 + dim_base) * scores[t + 1u];
                    acc2 += *(device const float4*)(v_cache + kb2 + dim_base) * scores[t + 2u];
                    acc3 += *(device const float4*)(v_cache + kb3 + dim_base) * scores[t + 3u];
                }
                for (; t < block_tokens; t++) {
                    const uint kv_base = kvBaseForToken(page_table, p, kv_head, block_start + t);
                    acc0 += *(device const float4*)(v_cache + kv_base + dim_base) * scores[t];
                }
            }

            acc_local[li] = ((acc0 + acc1) + (acc2 + acc3)) + ((acc4 + acc5) + (acc6 + acc7));
            ++li;
        }
        // No threadgroup_barrier here: acc_cache4 is accessed exclusively via
        // per-thread strided indexing (vi = tid; vi += FLASH_TG_SIZE), so each
        // thread reads back only what it wrote — pure intra-thread dependency.
        // The next iteration's first cross-thread threadgroup-memory read is
        // scores[token_offset] in the V loop, which is protected by the
        // softmax barrier. Saves ~1152 barriers/token on Qwen3-8B decode
        // (32 Q-heads × 36 layers × 1 block_start iter at avg ctx ≤256).
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
        uint li = 0u;
        for (uint vi = tid; vi < vec4_dim; vi += FLASH_TG_SIZE) {
            acc_local[li++] *= rescale_s;
        }
    }

    // fast::divide maps to Apple GPU hardware reciprocal+mul (vs precise IEEE
    // division), saving ~10 cycles per call. Fires once per Q head per layer
    // (~1152 calls/token on Qwen3-8B: 32 heads × 36 layers). Companion to the
    // fast::exp uses above in the running softmax; precision stays within the
    // already-accepted FA approximation budget.
    const float inv_sum = final_sum > 0.0f ? fast::divide(1.0f, final_sum) : 0.0f;
    uint li_out = 0u;
    for (uint vi = tid; vi < vec4_dim; vi += FLASH_TG_SIZE) {
        *(device float4*)(out + q_base + (vi << 2)) = acc_local[li_out++] * inv_sum;
    }
}
