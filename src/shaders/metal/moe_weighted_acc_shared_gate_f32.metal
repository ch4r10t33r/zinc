#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n;
    uint n_used;
    uint src_stride;
    uint gate_weight_offset;
};

// Token-major Qwen MoE finalize with one-row F32 shared gate.
//
// This mirrors vLLM's final top-k weight+reduce discipline for one token: first
// compute the shared-gate scalar once, then sweep the hidden vector and fold the
// routed expert sum plus gated shared expert into hidden.
kernel void main0(
    device float* accum [[buffer(0)]],
    device const float* src [[buffer(1)]],
    device const uint* routing [[buffer(2)]],
    constant Params& p [[buffer(3)]],
    device const float* shared_src [[buffer(4)]],
    device const float* norm_src [[buffer(5)]],
    device const char* gate_weight_bytes [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]]
) {
    if (p.n == 0u) {
        return;
    }

    threadgroup float partials[8];
    threadgroup float gate_value;

    device const float* gate_weight = (device const float*)(gate_weight_bytes + p.gate_weight_offset);

    float dot = 0.0f;
    for (uint dim = tid; dim < p.n; dim += 256u) {
        dot = fma(gate_weight[dim], norm_src[dim], dot);
    }

    const float simd_sum_dot = simd_sum(dot);
    if (lane == 0) {
        partials[simdgroup] = simd_sum_dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup == 0) {
        const float part = (lane < 8) ? partials[lane] : 0.0f;
        const float group_sum = simd_sum(part);
        if (lane == 0) {
            gate_value = 1.0f / (1.0f + exp(-group_sum));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint id = tid; id < p.n; id += 256u) {
        float sum = 0.0f;
        for (uint expert = 0u; expert < p.n_used; expert++) {
            const float weight = as_type<float>(routing[p.n_used + expert]);
            sum += weight * src[expert * p.src_stride + id];
        }

        accum[id] += sum + gate_value * shared_src[id];
    }
}
