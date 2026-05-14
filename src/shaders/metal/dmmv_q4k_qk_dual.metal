#include <metal_stdlib>
using namespace metal;

// Dense Gemma single-token Q+K dual Q4_K matvec.
//
// Pairs the attention Q and K projections (same K dimension, both Q4_K, but
// distinct M_q and M_k) into a single dispatch. Adapts llama.cpp's
// kernel_mul_mv_q4_K_f32 row layout (NSG=2, NR0=2, 64 threads). Unlike the
// existing dmmv_q4k_dual_llama kernel — which dispatches the gate/up pair
// over grid.y={0,1} with grid.x = max(M_gate, M_up) and so wastes
// threadgroups when M0 != M1 — this kernel uses a single-axis row layout
// where rows 0..M_q-1 select the Q weight/output and rows M_q..M_q+M_k-1
// select the K weight/output. No threadgroup is launched for rows past
// M_q + M_k, so dense Gemma 31B (M_q=8192, M_k=4096) sees the same total
// threadgroups (3072) as the two separate single-projection dispatches —
// just one dispatch instead of two.
//
// Requires p.M_q % (NSG * NR0) == 0 so each threadgroup is wholly Q or
// wholly K (the dispatcher checks this; otherwise it falls back to the two
// single-projection dispatches).

struct QKDualPush {
    uint M_q;
    uint M_k;
    uint K;
    uint a_q_offset;
    uint a_k_offset;
    uint x_offset;
    uint y_q_offset;
    uint y_k_offset;
};

#define NSG 2
#define NR0 2
#define QK_K 256
#define BLOCK_SIZE 144
#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

// Cycle 64: return per-row partials (head_dot, tail_dot, d, dmin) instead
// of a final scalar so the caller can fold the 2 cross-row
// `d*head - dmin*tail` chains into a single vec2 fma + vec2 mul. Mirrors
// cycle 61's 2-wide cross-row pattern on dmmv_q4k.metal — the qk_dual
// kernel's helper was hiding the cross-row vectorization opportunity by
// finalizing each row's reduction to scalar inside the helper.
inline float4 q4k_block_dot_parts(
    device const uchar* block,
    thread const float* yl,
    thread const float* yh,
    float4 sumy,
    ushort iq,
    ushort ir
) {
    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;

    // Cycle 75: port cycles 73/74's `packed_uint3` sc_u load pattern to
    // qk_dual. The Q4_K block's 12-byte packed-scales field sits at byte
    // offsets 4..15 — contiguous and 4-byte aligned — matching
    // packed_uint3 (12 bytes, 4-byte alignment). The previous form's
    // `+ iq` ushort-pointer offset (iq ∈ {0,1}) left the base 2- or
    // 4-byte aligned depending on iq, forcing the compiler to treat each
    // sc[k] as an independent strided ushort load it had to prove
    // contiguous via alias analysis before coalescing. Replacing with
    // packed_uint3 + a runtime `sc_shift = iq*16` to select the lower
    // or upper ushort lane of each uint guarantees one 12-byte coalesced
    // load per row per ib. The helper is called twice per ib (once per
    // NR0=2 row), so per-ib this folds 6 strided ushort loads → 2
    // packed_uint3 loads. dmmv_q4k_qk_dual.metal handles Qwen3-8B
    // attn_qkv Q+K paired (~18.5% of decode bytes/token via 25.08 GiB
    // attn path).
    device const packed_uint3* sc_u3 = (device const packed_uint3*)(block + 4);
    const uint sc_shift = uint(iq) * 16u;
    device const ushort* q1 = (device const ushort*)(block + 16) + 16 * iq + 4 * ir;
    // Cycle 68: port cycles 66/67's half2 dh-load pattern to qk_dual.
    // Block layout starts with `[d (half), dmin (half)]` at byte offsets
    // 0..3, exactly a half2 pair matching 4-byte block alignment. Replaces
    // 2 scalar half reads (dh[0] = d, dh[1] = dmin) with one packed half2
    // load + .x/.y swizzles. The helper is called twice per ib (once per
    // row), so per-ib this folds 4 scalar half loads → 2 half2 loads.
    // dmmv_q4k_qk_dual.metal handles Qwen3-8B attn_qkv Q+K paired
    // (~18.5% of decode bytes/token via the 25.08 GiB attn path).
    const half2 dh = *((device const half2*)block);

    // Cycle 71: port cycle 69's ushort4-sc16 register-vector pattern to
    // qk_dual. Replace the stack-allocated `ushort sc16[4]` + uchar*
    // alias with a register-resident `ushort4`; the previous form
    // forced the compiler to materialize sc16 in thread-private memory
    // so the byte alias could address individual lanes. The new form
    // keeps the four packed scales in SSA-eligible registers and lowers
    // the per-ib sc_pos / sc_neg byte gathers to vector AND + vector
    // shift over packed lanes. The helper is called twice per ib (once
    // per row), so per-ib this folds 8 scalar uchar loads → 4 packed
    // ushort4 byte-extractions. Same compiler-hint philosophy as cycles
    // 49/50 (nibble-mask vectorize), 51 (per-ib reduction), 61
    // (cross-row reduction), 69 (dmmv_q4k.metal), 70 (swiglu).
    const uint3 sc_u3v = uint3(*sc_u3);
    const ushort sc_0 = ushort((sc_u3v.x >> sc_shift) & 0xFFFFu);
    const ushort sc_2 = ushort((sc_u3v.y >> sc_shift) & 0xFFFFu);
    const ushort sc_4 = ushort((sc_u3v.z >> sc_shift) & 0xFFFFu);
    const ushort4 sc16 = ushort4(
        sc_0 & kmask1,
        sc_2 & kmask1,
        ((sc_4 >> 0) & kmask2) | ((sc_0 & kmask3) >> 2),
        ((sc_4 >> 4) & kmask2) | ((sc_2 & kmask3) >> 2));

    const ushort4 q1v = *((device const ushort4*)q1);
    const ushort4 q2v = *((device const ushort4*)(q1 + 32));

    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
    float4 acc2 = {0.f, 0.f, 0.f, 0.f};

    FOR_UNROLL (short i = 0; i < 4; ++i) {
        const float4 yl4 = float4(yl[2 * i + 0], yl[2 * i + 1], yl[2 * i + 8], yl[2 * i + 9]);
        const float4 yh4 = float4(yh[2 * i + 0], yh[2 * i + 1], yh[2 * i + 8], yh[2 * i + 9]);
        const ushort q1i = q1v[i];
        const ushort q2i = q2v[i];
        const float4 q1m = float4(q1i & 0x000F, q1i & 0x0F00, q1i & 0x00F0, q1i & 0xF000);
        const float4 q2m = float4(q2i & 0x000F, q2i & 0x0F00, q2i & 0x00F0, q2i & 0xF000);
        acc1 = fma(yl4, q1m, acc1);
        acc2 = fma(yh4, q2m, acc2);
    }

    const float4 head_pair = fma(
        float4(acc1[1], acc1[3], acc2[1], acc2[3]),
        float4(1.f / 256.f),
        float4(acc1[0], acc1[2], acc2[0], acc2[2]));
    // Cycle 71: derive sc_pos / sc_neg via vector byte-extraction from
    // the ushort4 sc16. sc8[0..7] (the old uchar* alias) maps to
    // {sc16.x.lo, sc16.x.hi, sc16.y.lo, sc16.y.hi, sc16.z.lo, sc16.z.hi,
    // sc16.w.lo, sc16.w.hi}, so:
    //   sc_pos = (sc16.x.lo, sc16.x.hi/16, sc16.z.lo, sc16.z.hi/16)
    //   sc_neg = (sc16.y.lo, sc16.y.hi,    sc16.w.lo, sc16.w.hi)
    constexpr ushort4 lo_mask = ushort4(0x00FFu);
    const ushort4 sc_pos_bytes = ushort4(sc16.x, sc16.x >> 8, sc16.z, sc16.z >> 8) & lo_mask;
    constexpr float4 sc_pos_scale = float4(1.f, 1.f / 16.f, 1.f, 1.f / 16.f);
    const float4 sc_pos = float4(sc_pos_bytes) * sc_pos_scale;
    const float4 sc_neg = float4(ushort4(sc16.y, sc16.y >> 8, sc16.w, sc16.w >> 8) & lo_mask);
    return float4(dot(head_pair, sc_pos), dot(sumy, sc_neg), float(dh.x), float(dh.y));
}

kernel void main0(
    device const uchar* W_q [[buffer(0)]],
    device const uchar* W_k [[buffer(1)]],
    constant QKDualPush& p [[buffer(2)]],
    device const float* X [[buffer(3)]],
    device float* Y_q [[buffer(4)]],
    device float* Y_k [[buffer(5)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    const short ix = tiisg / 8;
    const short it = tiisg % 8;
    const short iq = it / 4;
    const short ir = it % 4;

    const int nb = int(p.K) / QK_K;
    const int linear_row = (int(tgpig.x) * NSG + sgitg) * NR0;
    const int row_bytes = nb * BLOCK_SIZE;

    // Decide projection from the simdgroup's first row. Caller guarantees
    // M_q is a multiple of NSG*NR0 so a TG never straddles the boundary.
    const bool is_k = linear_row >= int(p.M_q);
    device const uchar* src = is_k ? (W_k + p.a_k_offset) : (W_q + p.a_q_offset);
    device float* out = is_k ? (Y_k + (p.y_k_offset / 4)) : (Y_q + (p.y_q_offset / 4));
    const int dst_row_base = is_k ? (linear_row - int(p.M_q)) : linear_row;
    const int M = is_k ? int(p.M_k) : int(p.M_q);

    device const float* x = X + (p.x_offset / 4);

    float yl[16];
    float yh[16];
    float sumf[NR0] = {0.f, 0.f};

    device const float* y4 = x + ix * QK_K + 64 * iq + 8 * ir;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy;

        // Cycle 57: port the explicit float4 y-load + dot4 sumy reduction
        // pattern from dmmv_q4k.metal. The previous scalar `sumy[k] += yl[i]`
        // accumulator chain inside the i=0..7 FOR_UNROLL had per-lane writes
        // into a float4 indexed by a compile-time constant — exactly the
        // shape cycles 44-51 found the Apple7 compiler does not always lift
        // to a single vector op. Replacing with 8×16-byte coalesced float4
        // loads of the four 8-float slices (offsets 0,32,128,160 — all
        // 32-byte aligned by construction of ix,iq,ir) folds the sumy
        // partials into 4 fused dot(v, 1) chains per ib instead of 32
        // serial scalar adds. y4 alignment: base = x + ix*QK_K + 64*iq +
        // 8*ir, all 32-byte boundaries. Mirrors cycle 23's win on
        // dmmv_q4k.metal. This kernel handles Qwen3-8B attn_qkv (Q+K
        // paired) — 25.08 GiB/step attn path = 18.5% of decode bytes/token.
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

        // Cycle 64: cross-row 2-wide reduction. Call the parts helper for
        // both rows so head_dot/tail_dot/d/dmin from both rows are in scope
        // together, then fold the 2 scalar `d*head - dmin*tail` chains into
        // one vec2 fma + one vec2 mul producing a float2 `delta` of the 2
        // per-row increments. Mirrors cycle 61's pattern on dmmv_q4k.metal.
        // dmmv_q4k_qk_dual.metal serves Qwen3-8B attn_qkv (Q+K paired into
        // one dispatch). Dispatcher guarantees M_q % (NSG*NR0)==0, and for
        // Qwen3-8B both M_q=4096 and M_k=1024 are divisible by NSG*NR0=4
        // so both rows are always valid when this kernel fires; the per-
        // row `dst_row < M` guard is preserved by zeroing the parts when
        // a row is past M so the cross-row fold contributes 0 to sumf for
        // that row, matching the original per-row skip.
        const int dst_row_0 = dst_row_base + 0;
        const int dst_row_1 = dst_row_base + 1;
        const ulong row_off_0 = ulong(dst_row_0) * ulong(row_bytes) + ulong(ib) * BLOCK_SIZE;
        const ulong row_off_1 = ulong(dst_row_1) * ulong(row_bytes) + ulong(ib) * BLOCK_SIZE;
        const float4 parts_0 = (dst_row_0 < M)
            ? q4k_block_dot_parts(src + row_off_0, yl, yh, sumy, iq, ir)
            : float4(0.0f);
        const float4 parts_1 = (dst_row_1 < M)
            ? q4k_block_dot_parts(src + row_off_1, yl, yh, sumy, iq, ir)
            : float4(0.0f);
        const float2 dh_d = float2(parts_0[2], parts_1[2]);
        const float2 dh_dmin = float2(parts_0[3], parts_1[3]);
        const float2 head_dots = float2(parts_0[0], parts_1[0]);
        const float2 tail_dots = float2(parts_0[1], parts_1[1]);
        const float2 delta = fma(dh_d, head_dots, -dh_dmin * tail_dots);
        sumf[0] += delta[0];
        sumf[1] += delta[1];

        y4 += 4 * QK_K;
    }

    for (short row = 0; row < NR0; ++row) {
        const int dst_row = dst_row_base + row;
        const float total = simd_sum(sumf[row]);
        if (tiisg == 0) {
            if (dst_row < M) out[dst_row] = total;
        }
    }
}
