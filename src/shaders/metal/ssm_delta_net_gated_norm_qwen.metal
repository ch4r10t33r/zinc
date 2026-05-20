#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct Params {
    uint d_inner;
    uint dt_rank;
    uint head_v_dim;
    uint d_state;
    uint n_group;
    uint ssm_a_is_f16;
    uint dt_bias_is_f16;
    uint has_dt_bias;
    uint has_ssm_a;
    uint alpha_offset;
    uint beta_offset;
    uint z_offset;
    uint output_offset;
    uint norm_per_head;
};

kernel void main0(
    constant Params& p [[buffer(0)]],
    device const float* conv_out [[buffer(1)]],
    device const float* alpha [[buffer(2)]],
    device const float* dt_bias [[buffer(3)]],
    device const float* ssm_a [[buffer(4)]],
    device const float* beta [[buffer(5)]],
    device float* state [[buffer(6)]],
    device const float* z_gate [[buffer(7)]],
    device const float* norm_weight [[buffer(8)]],
    device float* output [[buffer(9)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint simd_width [[thread_execution_width]],
    uint simdgroups_per_tg [[simdgroups_per_threadgroup]]
) {
    if (head >= 32u) {
        return;
    }

    constexpr uint head_v_dim = 128u;
    constexpr uint d_state = 128u;
    constexpr uint qk_dim = 2048u;
    constexpr uint v_base0 = 4096u;
    constexpr float inv_sqrt_d_state = 0.08838834764831845f;

    threadgroup float q[128];
    threadgroup float k[128];
    threadgroup float delta_out[128];
    threadgroup float partial_q[4];
    threadgroup float partial_k[4];
    threadgroup float partial_sq[4];

    const uint tg_threads = simd_width * simdgroups_per_tg;
    const uint simd_lane = tid & 31u;
    const uint simd_idx = tid >> 5u;
    const uint group = head & 15u;
    const uint q_base = group * d_state;
    const uint k_base = qk_dim + group * d_state;
    const uint v_base = v_base0 + head * head_v_dim;
    const uint head_state_base = head * head_v_dim * head_v_dim;

    float q_ss = 0.0f;
    float k_ss = 0.0f;
    for (uint i = tid; i < head_v_dim; i += tg_threads) {
        const float qv = conv_out[q_base + i];
        const float kv = conv_out[k_base + i];
        q[i] = qv;
        k[i] = kv;
        q_ss = fma(qv, qv, q_ss);
        k_ss = fma(kv, kv, k_ss);
    }

    const float q_sum = simd_sum(q_ss);
    const float k_sum = simd_sum(k_ss);
    if (simd_lane == 0u) {
        partial_q[simd_idx] = q_sum;
        partial_k[simd_idx] = k_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Every simdgroup folds the tiny cross-simdgroup partial arrays itself,
    // mirroring the final RMS reduction below. Lane 0 computes the scalar
    // setup once per simdgroup and broadcasts within the simdgroup, avoiding a
    // second threadgroup barrier before the 128 row workers enter the state loop.
    const float q_partial = (simd_lane < simdgroups_per_tg) ? partial_q[simd_lane] : 0.0f;
    const float k_partial = (simd_lane < simdgroups_per_tg) ? partial_k[simd_lane] : 0.0f;
    const float q_norm_sq = simd_sum(q_partial);
    const float k_norm_sq = simd_sum(k_partial);

    float q_scale_lane = 0.0f;
    float k_scale_lane = 0.0f;
    float decay_lane = 0.0f;
    float beta_lane = 0.0f;
    if (simd_lane == 0u) {
        const float alpha_raw = alpha[p.alpha_offset + head] + dt_bias[head];
        const float softplus_alpha = log(1.0f + fast::exp(alpha_raw));
        q_scale_lane = fast::rsqrt(fast::max(q_norm_sq, 1.0e-13f)) * inv_sqrt_d_state;
        k_scale_lane = fast::rsqrt(fast::max(k_norm_sq, 1.0e-13f));
        decay_lane = fast::exp(softplus_alpha * ssm_a[head]);
        beta_lane = fast::divide(1.0f, 1.0f + fast::exp(-beta[p.beta_offset + head]));
    }

    const float q_scale = simd_broadcast(q_scale_lane, 0u);
    const float k_scale = simd_broadcast(k_scale_lane, 0u);
    const float decay = simd_broadcast(decay_lane, 0u);
    const float beta_val = simd_broadcast(beta_lane, 0u);

    float local_sq = 0.0f;
    for (uint row = tid; row < head_v_dim; row += tg_threads) {
        const uint row_base = head_state_base + row * head_v_dim;
        device float4* state_vec = (device float4*)(state + row_base);
        threadgroup const float4* k_vec = (threadgroup const float4*)k;
        threadgroup const float4* q_vec = (threadgroup const float4*)q;
        float sk_raw = 0.0f;
        #pragma unroll
        for (uint col4 = 0u; col4 < 32u; ++col4) {
            const float4 decayed = state_vec[col4] * decay;
            state_vec[col4] = decayed;
            const float4 kv = k_vec[col4];
            sk_raw = fma(decayed.x, kv.x, sk_raw);
            sk_raw = fma(decayed.y, kv.y, sk_raw);
            sk_raw = fma(decayed.z, kv.z, sk_raw);
            sk_raw = fma(decayed.w, kv.w, sk_raw);
        }

        const float v = conv_out[v_base + row];
        const float delta = beta_val * (v - sk_raw * k_scale);
        const float scaled_delta = delta * k_scale;
        float out_v = 0.0f;
        #pragma unroll
        for (uint col4 = 0u; col4 < 32u; ++col4) {
            const float4 kv = k_vec[col4];
            const float4 qv = q_vec[col4];
            const float4 updated = fma(kv, scaled_delta, state_vec[col4]);
            state_vec[col4] = updated;
            out_v = fma(updated.x, qv.x, out_v);
            out_v = fma(updated.y, qv.y, out_v);
            out_v = fma(updated.z, qv.z, out_v);
            out_v = fma(updated.w, qv.w, out_v);
        }
        out_v *= q_scale;
        delta_out[row] = out_v;
        local_sq = fma(out_v, out_v, local_sq);
    }

    const float sq_sum = simd_sum(local_sq);
    if (simd_lane == 0u) {
        partial_sq[simd_idx] = sq_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Mirror rms_norm_mul.metal: each simdgroup redundantly folds the tiny
    // partial array, avoiding a second threadgroup barrier just to broadcast
    // the final RMS scalar.
    const float partial = (simd_lane < simdgroups_per_tg) ? partial_sq[simd_lane] : 0.0f;
    const float total_sq = simd_sum(partial);
    const float rms = fast::rsqrt(fast::divide(total_sq, float(head_v_dim)) + 1.0e-6f);
    for (uint row = tid; row < head_v_dim; row += tg_threads) {
        const uint idx = head * head_v_dim + row;
        const uint weight_idx = (p.norm_per_head != 0u) ? idx : row;
        const float z = z_gate[p.z_offset + idx];
        const float silu_z = z * fast::divide(1.0f, 1.0f + fast::exp(-z));
        output[p.output_offset + idx] = delta_out[row] * rms * norm_weight[weight_idx] * silu_z;
    }
}
