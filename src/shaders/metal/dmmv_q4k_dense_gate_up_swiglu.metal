#include <metal_stdlib>
using namespace metal;

// Dense Qwen3 single-token gate/up Q4_K matvec fused with SwiGLU.
//
// Sibling of dmmv_q4k_dense_gate_up_geglu.metal — same row layout
// (NSG=2, NR0=2, 64 threads/threadgroup), same q4_K block dot product,
// only the activation differs:
//   SwiGLU(gate, up) = (gate * sigmoid(gate)) * up
// vs the GeGLU variant used by Gemma.
//
// Saves 2 dispatches + 1 barrier per Qwen3 dense FFN layer compared to
// the un-fused gate / up / swiglu sequence, and skips the DRAM round
// trip through gate_buf and up_buf for the intermediate FFN width.
//
// Cycle 5 (reverted): NR0 bump 2→4 (8 rows/TG) measured 43.66 vs 44.5
// baseline median (-2%) on Qwen3-8B M1 Max. The doubled per-TG arithmetic
// hurt more than the halved TG count helped — the GPU was already
// saturating on the 3072-TG dispatch and TG launch overhead is not the
// lever on M1 Max. Reverted to NR0=2.

struct DualQ4KDmmvPush {
    uint M0;
    uint M1;
    uint K;
    uint a0_offset;
    uint a1_offset;
    uint x_offset;
    uint y0_offset;
    uint y1_offset;
};

#define NSG 2
#define NR0 2
#define QK_K 256
#define BLOCK_SIZE 144
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

// Cycle 32: fused gate+up block-dot pair. Loads both blocks' q1/q2/sc/dh
// up front and interleaves the FOR_UNROLL inner accumulator updates so the
// compiler can schedule independent gate-vs-up FMAs simultaneously. Same
// algorithm as two sequential q4k_block_dot calls but the interleaving
// removes a serialization point across the inline-helper boundary, which
// the Metal compiler does not always reorder aggressively across.
inline float2 q4k_block_dot_pair(
    device const uchar* block_g,
    device const uchar* block_u,
    thread const float* yl,
    thread const float* yh,
    float4 sumy,
    ushort iq,
    ushort ir
) {
    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;

    ushort sc16_g[4];
    ushort sc16_u[4];
    thread const uchar* sc8_g = (thread const uchar*)sc16_g;
    thread const uchar* sc8_u = (thread const uchar*)sc16_u;

    device const ushort* sc_g = (device const ushort*)(block_g + 4) + iq;
    device const ushort* sc_u = (device const ushort*)(block_u + 4) + iq;
    device const ushort* q1_g = (device const ushort*)(block_g + 16) + 16 * iq + 4 * ir;
    device const ushort* q1_u = (device const ushort*)(block_u + 16) + 16 * iq + 4 * ir;
    device const half* dh_g = (device const half*)block_g;
    device const half* dh_u = (device const half*)block_u;

    sc16_g[0] = sc_g[0] & kmask1;
    sc16_g[1] = sc_g[2] & kmask1;
    sc16_g[2] = ((sc_g[4] >> 0) & kmask2) | ((sc_g[0] & kmask3) >> 2);
    sc16_g[3] = ((sc_g[4] >> 4) & kmask2) | ((sc_g[2] & kmask3) >> 2);

    sc16_u[0] = sc_u[0] & kmask1;
    sc16_u[1] = sc_u[2] & kmask1;
    sc16_u[2] = ((sc_u[4] >> 0) & kmask2) | ((sc_u[0] & kmask3) >> 2);
    sc16_u[3] = ((sc_u[4] >> 4) & kmask2) | ((sc_u[2] & kmask3) >> 2);

    const ushort4 q1v_g = *((device const ushort4*)q1_g);
    const ushort4 q2v_g = *((device const ushort4*)(q1_g + 32));
    const ushort4 q1v_u = *((device const ushort4*)q1_u);
    const ushort4 q2v_u = *((device const ushort4*)(q1_u + 32));

    float4 acc1_g = {0.f, 0.f, 0.f, 0.f};
    float4 acc2_g = {0.f, 0.f, 0.f, 0.f};
    float4 acc1_u = {0.f, 0.f, 0.f, 0.f};
    float4 acc2_u = {0.f, 0.f, 0.f, 0.f};

    FOR_UNROLL (short i = 0; i < 4; ++i) {
        const float yl0 = yl[2 * i + 0];
        const float yl1 = yl[2 * i + 1];
        const float yl8 = yl[2 * i + 8];
        const float yl9 = yl[2 * i + 9];
        const float yh0 = yh[2 * i + 0];
        const float yh1 = yh[2 * i + 1];
        const float yh8 = yh[2 * i + 8];
        const float yh9 = yh[2 * i + 9];
        acc1_g[0] += yl0 * (q1v_g[i] & 0x000F);
        acc1_u[0] += yl0 * (q1v_u[i] & 0x000F);
        acc1_g[1] += yl1 * (q1v_g[i] & 0x0F00);
        acc1_u[1] += yl1 * (q1v_u[i] & 0x0F00);
        acc1_g[2] += yl8 * (q1v_g[i] & 0x00F0);
        acc1_u[2] += yl8 * (q1v_u[i] & 0x00F0);
        acc1_g[3] += yl9 * (q1v_g[i] & 0xF000);
        acc1_u[3] += yl9 * (q1v_u[i] & 0xF000);
        acc2_g[0] += yh0 * (q2v_g[i] & 0x000F);
        acc2_u[0] += yh0 * (q2v_u[i] & 0x000F);
        acc2_g[1] += yh1 * (q2v_g[i] & 0x0F00);
        acc2_u[1] += yh1 * (q2v_u[i] & 0x0F00);
        acc2_g[2] += yh8 * (q2v_g[i] & 0x00F0);
        acc2_u[2] += yh8 * (q2v_u[i] & 0x00F0);
        acc2_g[3] += yh9 * (q2v_g[i] & 0xF000);
        acc2_u[3] += yh9 * (q2v_u[i] & 0xF000);
    }

    const float g = dh_g[0] * ((acc1_g[0] + 1.f / 256.f * acc1_g[1]) * sc8_g[0] +
            (acc1_g[2] + 1.f / 256.f * acc1_g[3]) * sc8_g[1] * 1.f / 16.f +
            (acc2_g[0] + 1.f / 256.f * acc2_g[1]) * sc8_g[4] +
            (acc2_g[2] + 1.f / 256.f * acc2_g[3]) * sc8_g[5] * 1.f / 16.f) -
        dh_g[1] * (sumy[0] * sc8_g[2] + sumy[1] * sc8_g[3] + sumy[2] * sc8_g[6] + sumy[3] * sc8_g[7]);
    const float u = dh_u[0] * ((acc1_u[0] + 1.f / 256.f * acc1_u[1]) * sc8_u[0] +
            (acc1_u[2] + 1.f / 256.f * acc1_u[3]) * sc8_u[1] * 1.f / 16.f +
            (acc2_u[0] + 1.f / 256.f * acc2_u[1]) * sc8_u[4] +
            (acc2_u[2] + 1.f / 256.f * acc2_u[3]) * sc8_u[5] * 1.f / 16.f) -
        dh_u[1] * (sumy[0] * sc8_u[2] + sumy[1] * sc8_u[3] + sumy[2] * sc8_u[6] + sumy[3] * sc8_u[7]);
    return float2(g, u);
}

inline float swiglu(float gate, float up) {
    // SiLU(gate) * up = (gate / (1 + exp(-gate))) * up
    // fast::exp maps to Apple GPU hardware exp2 (vs precise::exp polynomial).
    // fast::divide maps to Apple GPU hardware reciprocal+mul (vs precise IEEE
    // division), saving ~10 cycles per call. Fires inter_dim × n_layers
    // times per token (~442K calls/token on Qwen3-8B).
    return up * gate * fast::divide(1.0f, 1.0f + fast::exp(-gate));
}

kernel void main0(
    device const uchar* W0 [[buffer(0)]],
    device const uchar* W1 [[buffer(1)]],
    constant DualQ4KDmmvPush& p [[buffer(2)]],
    device const float* X [[buffer(3)]],
    device float* activatedY [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    const int nb = p.K / QK_K;
    const int first_row = (tgpig.x * NSG + sgitg) * NR0;
    const int row_bytes = nb * BLOCK_SIZE;

    device const uchar* gate_src = W0 + p.a0_offset;
    device const uchar* up_src = W1 + p.a1_offset;
    device const float* x = X + (p.x_offset / 4);
    device float* out = activatedY + (p.y0_offset / 4);

    float yl[16];
    float yh[16];
    float gate_sum[NR0] = {0.f, 0.f};
    float up_sum[NR0] = {0.f, 0.f};

    device const float* y4 = x + ix * QK_K + 64 * iq + 8 * ir;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy;

        // Explicit float4 loads of the four 8-float slices (offsets 0,32,128,160
        // from y4 — all 32-byte aligned by construction of ix,iq,ir). This
        // forces 8×16-byte coalesced loads instead of relying on the compiler
        // to vectorize 32 scalar `y4[i]` reads across the `q4k_block_dot`
        // helper-function boundary that consumes yl/yh next. dot(v, 1) gives
        // the sumy partials in a single fused-mul-add chain per slice.
        const device float4* y4v = (const device float4*)y4;
        const float4 a0 = y4v[0];   const float4 a1 = y4v[1];
        const float4 b0 = y4v[8];   const float4 b1 = y4v[9];
        const float4 c0 = y4v[32];  const float4 c1 = y4v[33];
        const float4 d0 = y4v[40];  const float4 d1 = y4v[41];

        yl[ 0] = a0[0]; yl[ 1] = a0[1]; yl[ 2] = a0[2]; yl[ 3] = a0[3];
        yl[ 4] = a1[0]; yl[ 5] = a1[1]; yl[ 6] = a1[2]; yl[ 7] = a1[3];
        yl[ 8] = b0[0]; yl[ 9] = b0[1]; yl[10] = b0[2]; yl[11] = b0[3];
        yl[12] = b1[0]; yl[13] = b1[1]; yl[14] = b1[2]; yl[15] = b1[3];
        yh[ 0] = c0[0]; yh[ 1] = c0[1]; yh[ 2] = c0[2]; yh[ 3] = c0[3];
        yh[ 4] = c1[0]; yh[ 5] = c1[1]; yh[ 6] = c1[2]; yh[ 7] = c1[3];
        yh[ 8] = d0[0]; yh[ 9] = d0[1]; yh[10] = d0[2]; yh[11] = d0[3];
        yh[12] = d1[0]; yh[13] = d1[1]; yh[14] = d1[2]; yh[15] = d1[3];

        const float4 ones = float4(1.0f);
        sumy[0] = dot(a0, ones) + dot(a1, ones);
        sumy[1] = dot(b0, ones) + dot(b1, ones);
        sumy[2] = dot(c0, ones) + dot(c1, ones);
        sumy[3] = dot(d0, ones) + dot(d1, ones);

        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            const int dst_row = first_row + row;
            if (dst_row < int(p.M0)) {
                const ulong row_off = ulong(dst_row) * ulong(row_bytes) + ulong(ib) * BLOCK_SIZE;
                const float2 gu = q4k_block_dot_pair(gate_src + row_off, up_src + row_off, yl, yh, sumy, iq, ir);
                gate_sum[row] += gu.x;
                up_sum[row] += gu.y;
            }
        }

        y4 += 4 * QK_K;
    }

    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const int dst_row = first_row + row;
        if (dst_row < int(p.M0)) {
            const float gate_total = simd_sum(gate_sum[row]);
            const float up_total = simd_sum(up_sum[row]);
            if (tiisg == 0) {
                out[dst_row] = swiglu(gate_total, up_total);
            }
        }
    }
}
