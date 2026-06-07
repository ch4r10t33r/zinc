#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct Params {
    uint n;
    float eps;
    float hidden_scale;
};

// Shape-specific sibling of post_norm_residual_rms_norm.metal for dense
// Gemma 31B's hidden_dim=5376 decode tail. Same math, wider 512-thread
// group to expose more parallelism on the two reductions.
#define N_SIMDGROUPS 16
#define SIMD_WIDTH 32
#define TG_SIZE (N_SIMDGROUPS * SIMD_WIDTH)
// This variant is only dispatched for n=5376 with TG_SIZE=512, so each thread
// caches at most ceil(5376 / 512) = 11 hidden values between reductions.
#define MAX_PER_THREAD 12

kernel void main0(
    constant Params& p [[buffer(0)]],
    device float* hidden [[buffer(1)]],
    device const float* residual [[buffer(2)]],
    device const float* residual_w [[buffer(3)]],
    device float* norm_out [[buffer(4)]],
    device const float* output_w [[buffer(5)]],
    uint tid [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    uint sg_idx [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]
) {
    threadgroup float partial_sums_r[N_SIMDGROUPS];
    threadgroup float partial_sums_h[N_SIMDGROUPS];

    const uint base = group_id * p.n;

    float sum_sq_r = 0.0f;
    for (uint i = tid; i < p.n; i += TG_SIZE) {
        const float r = residual[base + i];
        sum_sq_r += r * r;
    }

    float sg_sum = simd_sum(sum_sq_r);
    if (lane == 0) partial_sums_r[sg_idx] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float part_r = (lane < N_SIMDGROUPS) ? partial_sums_r[lane] : 0.0f;
    const float total_sq_r = simd_sum(part_r);
    const float rms_inv_r = fast::rsqrt(fast::divide(total_sq_r, float(p.n)) + p.eps);

    float h_vals[MAX_PER_THREAD];
    float sum_sq_h = 0.0f;
    uint count = 0;
    const float hidden_scale = p.hidden_scale;
    const bool apply_hidden_scale = hidden_scale != 1.0f;
    for (uint i = tid; i < p.n; i += TG_SIZE) {
        const float r = residual[base + i];
        const float r_normed = residual_w[i] * (r * rms_inv_r);
        const float h = hidden[base + i] + r_normed;
        hidden[base + i] = apply_hidden_scale ? (h * hidden_scale) : h;
        h_vals[count++] = h;
        sum_sq_h += h * h;
    }

    sg_sum = simd_sum(sum_sq_h);
    if (lane == 0) partial_sums_h[sg_idx] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float part_h = (lane < N_SIMDGROUPS) ? partial_sums_h[lane] : 0.0f;
    const float total_sq_h = simd_sum(part_h);
    const float rms_inv_h = fast::rsqrt(fast::divide(total_sq_h, float(p.n)) + p.eps);

    count = 0;
    for (uint i = tid; i < p.n; i += TG_SIZE) {
        norm_out[base + i] = output_w[i] * (h_vals[count++] * rms_inv_h);
    }
}
