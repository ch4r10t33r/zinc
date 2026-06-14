#include <metal_stdlib>
using namespace metal;

// Batched K-RoPE + Q8 KV-cache write for Qwen 3.5 9B prefill.
//
// This folds the existing `rope_batched` K pass and `kv_cache_write_q8`
// pass into one dispatch. The rotated K row has no consumer besides the Q8
// cache on this path, so it can stay in registers and be quantized directly.

struct Params {
    uint stride;           // head_dim
    uint rope_dim;         // rotary dimensions per head, even and <= stride
    uint n_kv_heads;
    uint position_base;
    uint freq_base_bits;   // RoPE base-frequency bits (0 = use inv_freq buffer)
    uint dst_offset_bytes;
};

inline char quantizeQ8(float value, float inv_scale) {
    if (inv_scale == 0.0f) return char(0);
    const int q = clamp(int(rint(value * inv_scale)), -127, 127);
    return char(q);
}

inline float rotatedKValue(
    device const float* src,
    uint elem,
    uint half_rot,
    uint rope_dim,
    uint position,
    device const float* inv_freq,
    float freq_base,
    bool use_freq_buf
) {
    if (elem >= rope_dim) {
        return src[elem];
    }

    const uint pair_i = elem < half_rot ? elem : elem - half_rot;
    const float x0 = src[pair_i];
    const float x1 = src[pair_i + half_rot];

    float freq_i;
    if (use_freq_buf) {
        freq_i = inv_freq[pair_i];
    } else {
        const float exponent = float(2u * pair_i) / float(rope_dim);
        freq_i = 1.0f / pow(freq_base, exponent);
    }

    const float theta = float(position) * freq_i;
    const float cos_t = cos(theta);
    const float sin_t = sin(theta);
    return elem < half_rot ? (x0 * cos_t - x1 * sin_t) : (x0 * sin_t + x1 * cos_t);
}

kernel void main0(
    constant Params& p [[buffer(0)]],
    device const float* src_k [[buffer(1)]],
    device const float* src_v [[buffer(2)]],
    device const float* inv_freq [[buffer(3)]],
    device uchar* dst_k [[buffer(4)]],
    device uchar* dst_v [[buffer(5)]],
    uint block [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    if (lane >= 32u || p.n_kv_heads == 0u || p.stride == 0u || p.rope_dim == 0u) return;

    const uint blocks_per_head = p.stride / 32u;
    if (blocks_per_head == 0u) return;

    const uint block_in_head = block % blocks_per_head;
    const uint head = (block / blocks_per_head) % p.n_kv_heads;
    const uint tok = block / (blocks_per_head * p.n_kv_heads);
    const uint elem = block_in_head * 32u + lane;
    const uint position = p.position_base + tok;
    const uint src_base = (tok * p.n_kv_heads + head) * p.stride;
    const uint half_rot = p.rope_dim / 2u;
    const float freq_base = as_type<float>(p.freq_base_bits);
    const bool use_freq_buf = (p.freq_base_bits == 0u);

    const float k_value = rotatedKValue(
        src_k + src_base,
        elem,
        half_rot,
        p.rope_dim,
        position,
        inv_freq,
        freq_base,
        use_freq_buf
    );
    const float v_value = src_v[src_base + elem];

    const float k_abs_max = simd_max(fast::abs(k_value));
    const float v_abs_max = simd_max(fast::abs(v_value));
    const float k_scale = k_abs_max > 0.0f ? k_abs_max / 127.0f : 0.0f;
    const float v_scale = v_abs_max > 0.0f ? v_abs_max / 127.0f : 0.0f;
    const float k_inv_scale = k_scale > 0.0f ? 1.0f / k_scale : 0.0f;
    const float v_inv_scale = v_scale > 0.0f ? 1.0f / v_scale : 0.0f;

    device uchar* k_block = dst_k + p.dst_offset_bytes + block * 34u;
    device uchar* v_block = dst_v + p.dst_offset_bytes + block * 34u;

    if (lane == 0u) {
        *(device ushort*)(k_block) = as_type<ushort>(half(k_scale));
        *(device ushort*)(v_block) = as_type<ushort>(half(v_scale));
    }

    k_block[2u + lane] = as_type<uchar>(quantizeQ8(k_value, k_inv_scale));
    v_block[2u + lane] = as_type<uchar>(quantizeQ8(v_value, v_inv_scale));
}
