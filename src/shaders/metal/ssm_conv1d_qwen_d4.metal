#include <metal_stdlib>

using namespace metal;

struct Params {
    uint conv_channels;
    uint d_conv;
    uint kernel_is_f16;
    uint input_offset;
};

kernel void main0(
    constant Params& p [[buffer(0)]],
    device const float* kernel_w [[buffer(1)]],
    device float* state [[buffer(2)]],
    device const float* input [[buffer(3)]],
    device float* output [[buffer(4)]],
    uint ch [[thread_position_in_grid]]
) {
    if (ch >= p.conv_channels || p.d_conv != 4u || p.kernel_is_f16 != 0u) {
        return;
    }

    const uint c = p.conv_channels;
    const float x0 = state[ch];
    const float x1 = state[c + ch];
    const float x2 = state[2u * c + ch];
    const float x3 = input[p.input_offset + ch];
    const uint k = ch * 4u;

    float sum = 0.0f;
    sum += kernel_w[k] * x0;
    sum += kernel_w[k + 1u] * x1;
    sum += kernel_w[k + 2u] * x2;
    sum += kernel_w[k + 3u] * x3;

    output[ch] = sum / (1.0f + exp(-sum));
    state[ch] = x1;
    state[c + ch] = x2;
    state[2u * c + ch] = x3;
}
