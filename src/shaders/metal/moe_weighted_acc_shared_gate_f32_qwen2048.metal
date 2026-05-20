#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n;
    uint n_used;
    uint src_stride;
    uint gate_weight_offset;
    uint norm_offset;
};

// Exact Qwen3.6 token-major MoE finalize for hidden_dim=2048.
//
// The generic kernel splits the hidden row into 512-wide tiles, so it
// recomputes sigmoid(dot(norm, shared_gate_weight)) once per tile. This variant
// runs one 1024-thread group, computes the gate dot once, and lets each thread
// update two hidden dimensions.
kernel void main0(
    device float* accum [[buffer(0)]],
    device const float* src [[buffer(1)]],
    device const uint* routing [[buffer(2)]],
    constant Params& p [[buffer(3)]],
    device const float* shared_src [[buffer(4)]],
    device const float* norm_src [[buffer(5)]],
    device const char* gate_weight_bytes [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint simdgroups_per_tg [[simdgroups_per_threadgroup]]
) {
    if (p.n == 0u) {
        return;
    }

    threadgroup float partials[32];
    threadgroup float gate_value;

    device const float* gate_weight = (device const float*)(gate_weight_bytes + p.gate_weight_offset);
    device const float* norm = norm_src + (p.norm_offset >> 2);
    const uint threads_per_tg = simdgroups_per_tg * 32u;

    float dot = 0.0f;
    for (uint dim = tid; dim < p.n; dim += threads_per_tg) {
        dot = fma(gate_weight[dim], norm[dim], dot);
    }

    const float simd_sum_dot = simd_sum(dot);
    if (lane == 0u) {
        partials[simdgroup] = simd_sum_dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0u) {
        const float part = (lane < simdgroups_per_tg) ? partials[lane] : 0.0f;
        const float group_sum = simd_sum(part);
        if (lane == 0u) {
            gate_value = 1.0f / (1.0f + exp(-group_sum));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint id = tid; id < p.n; id += threads_per_tg) {
        float sum = 0.0f;
        for (uint expert = 0u; expert < p.n_used; expert++) {
            const float weight = as_type<float>(routing[p.n_used + expert]);
            sum += weight * src[expert * p.src_stride + id];
        }
        accum[id] += sum + gate_value * shared_src[id];
    }
}
