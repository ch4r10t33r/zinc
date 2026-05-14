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

        // Cycle 34: inline the q4k_block_dot_pair helper and interleave a
        // 4-way row0/row1 × gate/up FOR_UNROLL. Loads all 4 blocks' q1v/q2v/
        // sc16/dh up front and folds 64 independent FMAs into a single i=0..3
        // unrolled loop — extends cycle 33's row0/row1 inline pattern (which
        // landed in dmmv_q4k.metal) by also interleaving the gate/up axis on
        // top, matching the cross-axis interleaving cycle 32 applied within
        // the helper. Frees the compiler to schedule 4 independent FMA chains
        // simultaneously, which the helper-function boundary previously
        // blocked. Covers ffn_gate+ffn_up on Qwen3-8B dense path (~50% of
        // Q4_K bytes/token).
        constexpr ushort kmask1 = 0x3f3f;
        constexpr ushort kmask2 = 0x0f0f;
        constexpr ushort kmask3 = 0xc0c0;

        const int dst_row_0 = first_row + 0;
        const int dst_row_1 = first_row + 1;
        const ulong row_off_0 = ulong(dst_row_0) * ulong(row_bytes) + ulong(ib) * BLOCK_SIZE;
        const ulong row_off_1 = ulong(dst_row_1) * ulong(row_bytes) + ulong(ib) * BLOCK_SIZE;

        device const uchar* block_g0 = gate_src + row_off_0;
        device const uchar* block_u0 = up_src + row_off_0;
        device const uchar* block_g1 = gate_src + row_off_1;
        device const uchar* block_u1 = up_src + row_off_1;

        device const ushort* sc_g0 = (device const ushort*)(block_g0 + 4) + iq;
        device const ushort* sc_u0 = (device const ushort*)(block_u0 + 4) + iq;
        device const ushort* sc_g1 = (device const ushort*)(block_g1 + 4) + iq;
        device const ushort* sc_u1 = (device const ushort*)(block_u1 + 4) + iq;
        device const ushort* q1_g0 = (device const ushort*)(block_g0 + 16) + 16 * iq + 4 * ir;
        device const ushort* q1_u0 = (device const ushort*)(block_u0 + 16) + 16 * iq + 4 * ir;
        device const ushort* q1_g1 = (device const ushort*)(block_g1 + 16) + 16 * iq + 4 * ir;
        device const ushort* q1_u1 = (device const ushort*)(block_u1 + 16) + 16 * iq + 4 * ir;
        device const half* dh_g0 = (device const half*)block_g0;
        device const half* dh_u0 = (device const half*)block_u0;
        device const half* dh_g1 = (device const half*)block_g1;
        device const half* dh_u1 = (device const half*)block_u1;

        // Cycle 70: store sc16 as `ushort4` register vectors instead of
        // stack-allocated `ushort[4]` arrays accessed via `(uchar*)` byte
        // alias. The previous form forced the compiler to materialize
        // sc16 in thread-private memory so the byte alias could read
        // individual lanes; the new form keeps the four packed scales in
        // SSA-eligible registers and lets the per-ib sc_pos / sc_neg
        // byte gathers compile to vector AND + vector shift rather than
        // 8 scalar uchar loads from spilled stack memory. Port of cycle
        // 69 (same change in dmmv_q4k.metal, 2 sc16 sites); this shader
        // has 4 sc16 sites (g0, u0, g1, u1) so the impact area is ~2×:
        // 32 scalar uchar loads per ib → 16 packed byte-extractions.
        // dmmv_q4k_dense_gate_up_swiglu.metal is the hottest Q4_K
        // shader (~50% of Q4_K bytes/token = ffn_gate+ffn_up on
        // Qwen3-8B dense; Q4_K = 71.6% of decode bytes/token).
        const ushort4 sc16_g0 = ushort4(
            sc_g0[0] & kmask1,
            sc_g0[2] & kmask1,
            ((sc_g0[4] >> 0) & kmask2) | ((sc_g0[0] & kmask3) >> 2),
            ((sc_g0[4] >> 4) & kmask2) | ((sc_g0[2] & kmask3) >> 2));

        const ushort4 sc16_u0 = ushort4(
            sc_u0[0] & kmask1,
            sc_u0[2] & kmask1,
            ((sc_u0[4] >> 0) & kmask2) | ((sc_u0[0] & kmask3) >> 2),
            ((sc_u0[4] >> 4) & kmask2) | ((sc_u0[2] & kmask3) >> 2));

        const ushort4 sc16_g1 = ushort4(
            sc_g1[0] & kmask1,
            sc_g1[2] & kmask1,
            ((sc_g1[4] >> 0) & kmask2) | ((sc_g1[0] & kmask3) >> 2),
            ((sc_g1[4] >> 4) & kmask2) | ((sc_g1[2] & kmask3) >> 2));

        const ushort4 sc16_u1 = ushort4(
            sc_u1[0] & kmask1,
            sc_u1[2] & kmask1,
            ((sc_u1[4] >> 0) & kmask2) | ((sc_u1[0] & kmask3) >> 2),
            ((sc_u1[4] >> 4) & kmask2) | ((sc_u1[2] & kmask3) >> 2));

        const ushort4 q1v_g0 = *((device const ushort4*)q1_g0);
        const ushort4 q2v_g0 = *((device const ushort4*)(q1_g0 + 32));
        const ushort4 q1v_u0 = *((device const ushort4*)q1_u0);
        const ushort4 q2v_u0 = *((device const ushort4*)(q1_u0 + 32));
        const ushort4 q1v_g1 = *((device const ushort4*)q1_g1);
        const ushort4 q2v_g1 = *((device const ushort4*)(q1_g1 + 32));
        const ushort4 q1v_u1 = *((device const ushort4*)q1_u1);
        const ushort4 q2v_u1 = *((device const ushort4*)(q1_u1 + 32));

        float4 acc1_g0 = {0.f, 0.f, 0.f, 0.f};
        float4 acc2_g0 = {0.f, 0.f, 0.f, 0.f};
        float4 acc1_u0 = {0.f, 0.f, 0.f, 0.f};
        float4 acc2_u0 = {0.f, 0.f, 0.f, 0.f};
        float4 acc1_g1 = {0.f, 0.f, 0.f, 0.f};
        float4 acc2_g1 = {0.f, 0.f, 0.f, 0.f};
        float4 acc1_u1 = {0.f, 0.f, 0.f, 0.f};
        float4 acc2_u1 = {0.f, 0.f, 0.f, 0.f};

        constexpr ushort4 nibble_mask = ushort4(0x000F, 0x0F00, 0x00F0, 0xF000);

        FOR_UNROLL (short i = 0; i < 4; ++i) {
            // Cycle 49: vectorize the per-quant nibble-mask expansion. Cycle 46
            // already packed the FMAs into explicit float4 form, but each
            // `float4 q1m_X = float4(qi & 0x000F, qi & 0x0F00, qi & 0x00F0, qi & 0xF000)`
            // still expressed 4 scalar AND ops + 4 scalar int→float conversions
            // per nibble-set. Replace with `float4(ushort4(qi) & nibble_mask)` —
            // explicit broadcast-then-vector-AND-then-vector-convert. Each
            // ushort4(qi) is a register splat, the AND lowers to a single 4-wide
            // vector AND against the constexpr mask, and the float4(ushort4)
            // conversion is a single vector instruction on Apple7 instead of 4
            // separate lane converts. 8 mask expansions × 4 unrolled i iterations
            // = 32 expansions per ib iteration in this hottest Q4_K shader
            // (~50% of Q4_K bytes/token = ffn_gate+ffn_up on Qwen3-8B). Mirrors
            // the cycle 44-48 philosophy of telling the compiler the SIMD shape
            // explicitly instead of relying on it to lift lane-by-lane forms.
            const float4 yl4 = float4(yl[2 * i + 0], yl[2 * i + 1], yl[2 * i + 8], yl[2 * i + 9]);
            const float4 yh4 = float4(yh[2 * i + 0], yh[2 * i + 1], yh[2 * i + 8], yh[2 * i + 9]);
            const ushort q1_g0i = q1v_g0[i];
            const ushort q1_u0i = q1v_u0[i];
            const ushort q1_g1i = q1v_g1[i];
            const ushort q1_u1i = q1v_u1[i];
            const ushort q2_g0i = q2v_g0[i];
            const ushort q2_u0i = q2v_u0[i];
            const ushort q2_g1i = q2v_g1[i];
            const ushort q2_u1i = q2v_u1[i];
            const float4 q1m_g0 = float4(ushort4(q1_g0i) & nibble_mask);
            const float4 q1m_u0 = float4(ushort4(q1_u0i) & nibble_mask);
            const float4 q1m_g1 = float4(ushort4(q1_g1i) & nibble_mask);
            const float4 q1m_u1 = float4(ushort4(q1_u1i) & nibble_mask);
            const float4 q2m_g0 = float4(ushort4(q2_g0i) & nibble_mask);
            const float4 q2m_u0 = float4(ushort4(q2_u0i) & nibble_mask);
            const float4 q2m_g1 = float4(ushort4(q2_g1i) & nibble_mask);
            const float4 q2m_u1 = float4(ushort4(q2_u1i) & nibble_mask);
            acc1_g0 = fma(yl4, q1m_g0, acc1_g0);
            acc1_u0 = fma(yl4, q1m_u0, acc1_u0);
            acc1_g1 = fma(yl4, q1m_g1, acc1_g1);
            acc1_u1 = fma(yl4, q1m_u1, acc1_u1);
            acc2_g0 = fma(yh4, q2m_g0, acc2_g0);
            acc2_u0 = fma(yh4, q2m_u0, acc2_u0);
            acc2_g1 = fma(yh4, q2m_g1, acc2_g1);
            acc2_u1 = fma(yh4, q2m_u1, acc2_u1);
        }

        // Cycle 52: port cycle 51's vectorized per-ib reduction from
        // dmmv_q4k.metal here. Replace each row's 4-term head sum
        // `(acc[even] + 1/256*acc[odd]) * sc8[*]` and 4-term tail
        // `sumy[k]*sc8[*]` with one `fma`-built `head_pair` float4 plus a
        // `dot(head_pair, sc_pos4)` and a `dot(sumy, sc_neg4)`. Two rows ×
        // (gate + up) = 4 independent reductions per ib here vs 2 in the
        // base kernel, so ~2× the impact area of cycle 51. Maps the per-
        // block accumulator collapse onto Apple7's 4-wide ALU shape. The
        // swiglu kernel handles ffn_gate + ffn_up on Qwen3-8B dense
        // (~50% of Q4_K bytes/token), the hottest Q4_K kernel.
        const float4 head_pair_g0 = fma(
            float4(acc1_g0[1], acc1_g0[3], acc2_g0[1], acc2_g0[3]),
            float4(1.f / 256.f),
            float4(acc1_g0[0], acc1_g0[2], acc2_g0[0], acc2_g0[2]));
        const float4 head_pair_u0 = fma(
            float4(acc1_u0[1], acc1_u0[3], acc2_u0[1], acc2_u0[3]),
            float4(1.f / 256.f),
            float4(acc1_u0[0], acc1_u0[2], acc2_u0[0], acc2_u0[2]));
        const float4 head_pair_g1 = fma(
            float4(acc1_g1[1], acc1_g1[3], acc2_g1[1], acc2_g1[3]),
            float4(1.f / 256.f),
            float4(acc1_g1[0], acc1_g1[2], acc2_g1[0], acc2_g1[2]));
        const float4 head_pair_u1 = fma(
            float4(acc1_u1[1], acc1_u1[3], acc2_u1[1], acc2_u1[3]),
            float4(1.f / 256.f),
            float4(acc1_u1[0], acc1_u1[2], acc2_u1[0], acc2_u1[2]));
        // Cycle 70: derive sc_pos / sc_neg via vector byte-extraction
        // from the ushort4 sc16. sc8_X[0..7] (the old uchar* alias)
        // maps to {sc16.x.lo, sc16.x.hi, sc16.y.lo, sc16.y.hi,
        // sc16.z.lo, sc16.z.hi, sc16.w.lo, sc16.w.hi}, so:
        //   sc_pos = (sc16.x.lo, sc16.x.hi/16, sc16.z.lo, sc16.z.hi/16)
        //   sc_neg = (sc16.y.lo, sc16.y.hi,    sc16.w.lo, sc16.w.hi)
        // Builds each float4 in 1 vector AND + 1 ushort4→float4 widen
        // (+ 1 vector mul for sc_pos), replacing 4 scalar byte loads +
        // 2 scalar muls. × 4 sc16 sites per ib (g0, u0, g1, u1).
        constexpr ushort4 lo_mask = ushort4(0x00FFu);
        constexpr float4 sc_pos_scale = float4(1.f, 1.f / 16.f, 1.f, 1.f / 16.f);
        const float4 sc_pos_g0 = float4(ushort4(sc16_g0.x, sc16_g0.x >> 8, sc16_g0.z, sc16_g0.z >> 8) & lo_mask) * sc_pos_scale;
        const float4 sc_pos_u0 = float4(ushort4(sc16_u0.x, sc16_u0.x >> 8, sc16_u0.z, sc16_u0.z >> 8) & lo_mask) * sc_pos_scale;
        const float4 sc_pos_g1 = float4(ushort4(sc16_g1.x, sc16_g1.x >> 8, sc16_g1.z, sc16_g1.z >> 8) & lo_mask) * sc_pos_scale;
        const float4 sc_pos_u1 = float4(ushort4(sc16_u1.x, sc16_u1.x >> 8, sc16_u1.z, sc16_u1.z >> 8) & lo_mask) * sc_pos_scale;
        const float4 sc_neg_g0 = float4(ushort4(sc16_g0.y, sc16_g0.y >> 8, sc16_g0.w, sc16_g0.w >> 8) & lo_mask);
        const float4 sc_neg_u0 = float4(ushort4(sc16_u0.y, sc16_u0.y >> 8, sc16_u0.w, sc16_u0.w >> 8) & lo_mask);
        const float4 sc_neg_g1 = float4(ushort4(sc16_g1.y, sc16_g1.y >> 8, sc16_g1.w, sc16_g1.w >> 8) & lo_mask);
        const float4 sc_neg_u1 = float4(ushort4(sc16_u1.y, sc16_u1.y >> 8, sc16_u1.w, sc16_u1.w >> 8) & lo_mask);
        // Cycle 60: vectorize the cross-(gate,up) × cross-row final reduction.
        // The 4 independent per-block updates here (gate0, up0, gate1, up1)
        // each compute `dh[0] * dot(head, sc_pos) - dh[1] * dot(sumy, sc_neg)`
        // — 4 scalar `a*b - c*d` chains in lane-by-lane form. Pack the 4
        // (dh_d, dh_dmin, head_dot, tail_dot) tuples into float4s and fold
        // the 4 reductions into one vector mul + one vector fnma. Mirrors
        // cycle 51-53's "vectorize the per-ib reduction" pattern, but
        // applied at the outer (row × projection) axis instead of the
        // inner head/tail axis. Apple7 ALU is 4-wide; the indexed scalar
        // form was 4 separate scalar mul + 4 scalar mul + 4 scalar sub per
        // ib (12 scalar ops); the new form is 1 vec mul + 1 vec fnma + 4
        // scalar accumulator adds (~6 ops). This is the hottest Q4_K shader
        // (~50% of Q4_K bytes/token = ffn_gate+ffn_up on Qwen3-8B dense).
        //
        // Cycle 66: replace 8 scalar half loads (dh_X[0], dh_X[1] × 4 rows)
        // with 4 explicit half2 loads. Q4_K's block layout starts with
        // [d (half), dmin (half)] at offsets 0..3 — exactly a half2 pair
        // — so a half2 load matches the natural alignment and tells the
        // compiler the pair is intended to be read together. Subsequent
        // `float(h2.x)` / `float(h2.y)` lowers to a single half2→float2
        // widen per row instead of 2 scalar half→float casts. Same
        // compiler-hint philosophy as cycles 38 (single-half Q6_K load),
        // 49/50 (nibble-mask vectorize), 51-53 (per-ib reduction), and
        // 60-62 (cross-row reduction): tell the compiler the SIMD shape
        // explicitly. 8 scalar half loads → 4 half2 loads per ib, × 64
        // threads/TG × 3072 TGs per layer × 36 layers per token.
        const half2 dh_g0_h2 = *((device const half2*)dh_g0);
        const half2 dh_u0_h2 = *((device const half2*)dh_u0);
        const half2 dh_g1_h2 = *((device const half2*)dh_g1);
        const half2 dh_u1_h2 = *((device const half2*)dh_u1);
        const float4 dh_d = float4(float(dh_g0_h2.x), float(dh_u0_h2.x), float(dh_g1_h2.x), float(dh_u1_h2.x));
        const float4 dh_dmin = float4(float(dh_g0_h2.y), float(dh_u0_h2.y), float(dh_g1_h2.y), float(dh_u1_h2.y));
        const float4 head_dots = float4(
            dot(head_pair_g0, sc_pos_g0),
            dot(head_pair_u0, sc_pos_u0),
            dot(head_pair_g1, sc_pos_g1),
            dot(head_pair_u1, sc_pos_u1));
        const float4 tail_dots = float4(
            dot(sumy, sc_neg_g0),
            dot(sumy, sc_neg_u0),
            dot(sumy, sc_neg_g1),
            dot(sumy, sc_neg_u1));
        const float4 delta = dh_d * head_dots - dh_dmin * tail_dots;
        gate_sum[0] += delta[0];
        up_sum[0]   += delta[1];
        gate_sum[1] += delta[2];
        up_sum[1]   += delta[3];

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
