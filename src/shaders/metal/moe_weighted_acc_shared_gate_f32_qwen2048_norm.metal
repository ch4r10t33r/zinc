#include <metal_stdlib>
using namespace metal;

struct Params {
    uint n;
    uint n_used;
    uint src_stride;
    uint gate_weight_offset;
    uint norm_offset;
    float eps;
};

// Exact Qwen3.6 token-major MoE finalize plus next-layer RMSNorm.
//
// Adapts llama.cpp's Metal graph-tail fusion discipline: once the MoE reduce
// has the final hidden row in registers, immediately materialize the next
// layer's normalized input and avoid a separate RMSNorm dispatch/barrier.
kernel void main0(
    device float* hidden [[buffer(0)]],
    device const float* src [[buffer(1)]],
    device const uint* routing [[buffer(2)]],
    constant Params& p [[buffer(3)]],
    device const float* shared_src [[buffer(4)]],
    device const float* norm_src [[buffer(5)]],
    device const char* gate_weight_bytes [[buffer(6)]],
    device float* next_norm [[buffer(7)]],
    device const float* next_norm_weight [[buffer(8)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]],
    uint simdgroups_per_tg [[simdgroups_per_threadgroup]]
) {
    if (p.n != 2048u) {
        return;
    }

    threadgroup float partials[32];
    device const float* gate_weight = (device const float*)(gate_weight_bytes + p.gate_weight_offset);
    device const float* norm = norm_src + (p.norm_offset >> 2);
    const uint threads_per_tg = simdgroups_per_tg * 32u;

    float gate_dot = 0.0f;
    for (uint dim = tid; dim < p.n; dim += threads_per_tg) {
        gate_dot = fma(gate_weight[dim], norm[dim], gate_dot);
    }

    float sg_sum = simd_sum(gate_dot);
    if (lane == 0u) {
        partials[simdgroup] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float part = (lane < simdgroups_per_tg) ? partials[lane] : 0.0f;
    const float gate = fast::divide(1.0f, 1.0f + fast::exp(-simd_sum(part)));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float h_vals[2];
    uint h_idxs[2];
    uint h_count = 0u;
    float hidden_sq = 0.0f;
    for (uint id = tid; id < p.n; id += threads_per_tg) {
        float sum = 0.0f;
        for (uint expert = 0u; expert < p.n_used; expert++) {
            const float weight = as_type<float>(routing[p.n_used + expert]);
            sum += weight * src[expert * p.src_stride + id];
        }

        const float h = hidden[id] + sum + gate * shared_src[id];
        hidden[id] = h;
        h_vals[h_count] = h;
        h_idxs[h_count] = id;
        h_count++;
        hidden_sq = fma(h, h, hidden_sq);
    }

    sg_sum = simd_sum(hidden_sq);
    if (lane == 0u) {
        partials[simdgroup] = sg_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    part = (lane < simdgroups_per_tg) ? partials[lane] : 0.0f;
    const float rms_inv = fast::rsqrt(fast::divide(simd_sum(part), float(p.n)) + p.eps);

    for (uint i = 0u; i < h_count; i++) {
        const uint id = h_idxs[i];
        next_norm[id] = next_norm_weight[id] * (h_vals[i] * rms_inv);
    }
}
