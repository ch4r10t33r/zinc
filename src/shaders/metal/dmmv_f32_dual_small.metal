#include <metal_stdlib>
using namespace metal;

struct DualF32DmmvPush {
    uint M0;
    uint M1;
    uint K;
    uint a0_offset;
    uint a1_offset;
    uint x_offset;
    uint y0_offset;
    uint y1_offset;
};

// Two small F32 matvecs in one launch.
//
// Qwen3.6 SSM alpha/beta tails are both 32x2048 F32 projections. The previous
// path launched dmmv_f32 twice; this keeps the same one-thread-per-row,
// sequential K reduction, but uses grid.y to select alpha vs beta.
kernel void main0(
    constant DualF32DmmvPush& p [[buffer(0)]],
    device const char* W0 [[buffer(1)]],
    device const char* W1 [[buffer(2)]],
    device const float* X [[buffer(3)]],
    device float* Y0 [[buffer(4)]],
    device float* Y1 [[buffer(5)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]
) {
    const bool second = tg_pos.y != 0u;
    const uint M = second ? p.M1 : p.M0;
    const uint row = tg_pos.x * 64u + tid;
    if (row >= M || tg_pos.y > 1u) {
        return;
    }

    device const char* Wc = second ? W1 : W0;
    const uint a_offset = second ? p.a1_offset : p.a0_offset;
    const uint y_offset = second ? p.y1_offset : p.y0_offset;
    device float* Y = second ? Y1 : Y0;

    device const float* W = (device const float*)(Wc + a_offset);
    device const float* x = X + (p.x_offset >> 2);
    device const float* w = W + row * p.K;

    float acc = 0.0f;
    for (uint k = 0u; k < p.K; ++k) {
        acc = fma(w[k], x[k], acc);
    }

    Y[(y_offset >> 2) + row] = acc;
}
