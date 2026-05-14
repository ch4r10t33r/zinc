#include <metal_stdlib>
using namespace metal;

// Push constants for DMMV dispatch (matches Zig DmmvPush layout).
struct DmmvPush {
    uint M;        // rows
    uint K;        // cols
    uint a_offset; // byte offset into weight matrix
    uint x_offset; // byte offset into input vector
    uint y_offset; // byte offset into output vector
};

// Port of llama.cpp's kernel_mul_mv_q4_K_f32 (non-ext variant).
// Matches the exact floating-point accumulation pattern for bit-identical results.
//
// Thread organization (matches llama.cpp with N_SG_Q4_K=2, N_R0_Q4_K=2):
//   64 threads per threadgroup = 2 simdgroups x 32 threads
//   Each simdgroup processes 2 rows => 4 rows per threadgroup
//
// Q4_K block layout (144 bytes, 256 elements):
//   [0..1]   d    (float16)
//   [2..3]   dmin (float16)
//   [4..15]  scales (12 bytes, packed 6-bit scale/min pairs)
//   [16..143] qs  (128 bytes, 256 x 4-bit quants)

#define NSG   2
#define NR0   2
#define QK_K  256
#define BLOCK_SIZE 144
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

kernel void main0(
    device const uchar* W [[buffer(0)]],
    constant DmmvPush& p [[buffer(1)]],
    device const float* X [[buffer(2)]],
    device float* Y [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;

    const short ix = tiisg / 8;  // 0..3
    const short it = tiisg % 8;  // 0..7
    const short iq = it / 4;     // 0 or 1
    const short ir = it % 4;     // 0..3

    const int nb = p.K / QK_K;   // blocks per row

    const int r0 = tgpig.x;

    const int first_row = (r0 * NSG + sgitg) * NR0;

    // nb01 in llama.cpp is the byte stride per row = nb * sizeof(block_q4_K)
    const int nb01 = nb * BLOCK_SIZE;

    device const uchar* src0 = W + p.a_offset;
    device const float* src1 = X + (p.x_offset / 4);

    device const uchar* x_base = src0 + (uint64_t)first_row * nb01;
    device const float* y = src1;

    float yl[16];
    float yh[16];

    float sumf[NR0] = {0.f, 0.f};

    device const float* y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    ushort sc16[4];
    thread const uchar* sc8 = (thread const uchar*)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy;

        // Explicit float4 loads of the four 8-float slices (offsets 0,32,128,160
        // from y4 — all 32-byte aligned by construction of ix,iq,ir). Forces
        // 8×16-byte coalesced loads instead of 32 scalar `y4[i]` reads, and folds
        // the sumy partials into 4 fused dot(v,1) chains. Mirrors the cycle 23
        // pattern that landed in dmmv_q4k_dense_gate_up_swiglu.metal; this kernel
        // handles attn_qkv/attn_o/ffn_down on the Qwen3 dense path (~53% of all
        // Q4_K bytes/token, complementing the ~50% FFN gate/up traffic already
        // covered by the swiglu variant).
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

        // Pointer to scales for this block, offset by iq (uint16 stride)
        device const ushort* sc = (device const ushort*)(x_base + (uint64_t)ib * BLOCK_SIZE + 4) + iq;
        // Pointer to quants for this block
        device const ushort* q1 = (device const ushort*)(x_base + (uint64_t)ib * BLOCK_SIZE + 16) + 16 * iq + 4 * ir;
        // Pointer to d/dmin halves for this block
        device const half* dh = (device const half*)(x_base + (uint64_t)ib * BLOCK_SIZE);

        FOR_UNROLL (short row = 0; row < NR0; row++) {
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            device const ushort* q2 = q1 + 32;

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};

            // ushort4 vectorized loads of the 8 q-quant ushorts: q1[0..3] and
            // q2[0..3]. Pointers are 8-byte aligned by construction (block
            // base is 144-byte stride, +16 byte qs offset, +16*iq + 4*ir
            // ushort offsets all preserve 8B alignment; q2 = q1 + 64 bytes).
            // Replaces 8 scalar ushort loads with 2 ushort4 loads — same
            // coalescing pattern that cycle 25 applied to the y4[] floats.
            const ushort4 q1v = *((device const ushort4*)q1);
            const ushort4 q2v = *((device const ushort4*)q2);

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2 * i + 0] * (q1v[i] & 0x000F);
                acc1[1] += yl[2 * i + 1] * (q1v[i] & 0x0F00);
                acc1[2] += yl[2 * i + 8] * (q1v[i] & 0x00F0);
                acc1[3] += yl[2 * i + 9] * (q1v[i] & 0xF000);
                acc2[0] += yh[2 * i + 0] * (q2v[i] & 0x000F);
                acc2[1] += yh[2 * i + 1] * (q2v[i] & 0x0F00);
                acc2[2] += yh[2 * i + 8] * (q2v[i] & 0x00F0);
                acc2[3] += yh[2 * i + 9] * (q2v[i] & 0xF000);
            }

            sumf[row] += dh[0] * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                    (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                    (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                    (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                dh[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            // Advance to next row: nb01/2 is stride in uint16 units
            q1 += nb01 / 2;
            sc += nb01 / 2;
            dh += nb01 / 2;
        }

        y4 += 4 * QK_K;
    }

    device float* dst_f32 = Y + (p.y_offset / 4);

    for (int row = 0; row < NR0 && first_row + row < (int)p.M; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}
