//! CUDA forward pass for the dense gemma4 transformer (Gemma 4 31B-it).
//!
//! Effort 22 — completes the 5/5 catalog on the 4090. Separate from
//! forward_cuda.zig (qwen35/qwen36 hybrid-SSM family) because gemma4 is a plain
//! transformer with a different per-layer geometry: sliding-window attention on
//! a period-6 pattern (5 SWA + 1 full), per-layer head dims (256 SWA / 512 full)
//! and KV-head counts (16 SWA / 4 full), per-head Q/K RMS norm + per-head V RMS
//! normalize, four norms per layer (pre/post attn + pre/post ffn), GeGLU FFN, a
//! learned per-layer output scale, scaled token embeddings, and a tied LM head.
//!
//! Norm convention: the gemma RMSNorm `(1 + weight)` offset is baked into the
//! GGUF weights at conversion (confirmed: attn_q_norm ≈ 1.0234), so every gemma
//! norm reuses the standard `rms_norm` kernel; V uses `rms_norm_noweight`.
//!
//! Attention scale: gemma4 sets f_attention_scale = 1.0 (no 1/sqrt(d) scaling).
//! Final-logit soft-cap is monotonic, so it does not change the greedy argmax
//! and is intentionally skipped here (correctness-first bring-up).
//!
//! @section Inference Runtime
const std = @import("std");
const buffer = @import("../cuda/buffer.zig");
const pipeline = @import("../cuda/pipeline.zig");
const command = @import("../cuda/command.zig");
const shim = @import("../cuda/c.zig").shim;
const gguf = @import("../model/gguf.zig");
const loader = @import("../model/loader_cuda.zig");

const log = std.log.scoped(.cuda_fwd_gemma);
const CudaBuffer = buffer.CudaBuffer;
const CudaPipeline = pipeline.CudaPipeline;
const LoadedTensor = loader.LoadedTensor;

const KERNELS_CU = @embedFile("../shaders/cuda/kernels.cu");

// ---- kernel push-constant structs (must byte-match kernels.cu) --------------
const RmsPush = extern struct { N: u32, eps: f32 };
const DmmvPush = extern struct {
    M: u32,
    K: u32,
    a_offset: u32 = 0,
    x_offset: u32 = 0,
    y_offset: u32 = 0,
    acc_mode: u32 = 0,
};
const RopePush = extern struct {
    stride: u32,
    rope_dim: u32,
    n_heads: u32,
    position: u32,
    freq_base_bits: u32,
    attn_scale_bits: u32,
};
const GemmaAttnPush = extern struct {
    head_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    seq_len: u32,
    scale_bits: u32,
    window: u32,
};
const GemmaAttnBatchPush = extern struct {
    head_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    T: u32,
    scale_bits: u32,
    window: u32,
};
const RmsRopePush = extern struct { head_dim: u32, eps: f32, rope_dim: u32, position: u32, dst_offset: u32 };
const RmsKvWritePush = extern struct { head_dim: u32, eps: f32, dst_offset: u32 };
// Batched-prefill twins (grid.y = T): explicit per-token src/dst strides.
const RmsRopeBatchPush = extern struct { head_dim: u32, eps: f32, rope_dim: u32, base_position: u32, src_stride: u32, dst_stride: u32 };
const RmsKvWriteBatchPush = extern struct { head_dim: u32, eps: f32, src_stride: u32, dst_stride: u32 };
// Decode fusion: per-head Q/K rms_norm + RoPE + KV-write in one launch.
const RmsRopeQkvPush = extern struct { head_dim: u32, eps: f32, rope_dim: u32, position: u32, n_head: u32, n_kv_head: u32, kv_offset: u32 };
const SwigluPush = extern struct { N: u32 };
const F32ToF16Push = extern struct { N: u32 }; // cycle 12: activation downcast for the TC f16-A GEMM
const DequantQ4KPush = extern struct { M: u32, K: u32, a_offset: u32 = 0 }; // e26 c9: Q4_K weight → fp16 for the cuBLAS prefill GEMM
const ScaleAccPush = extern struct { N: u32, scale: f32 };
const ScalarMulPush = extern struct { N: u32 };
const ArgmaxPush = extern struct { N: u32 };
// MoE router/combine kernels (byte-match kernels.cu).
const TopkPush = extern struct { n_experts: u32, k: u32 };
const MoeAccPush = extern struct { N: u32, n_used: u32, src_stride: u32 };
// Token-batched MoE combine (Effort 24 cycle 9): per-token strides so one launch
// (grid.y = T) does the weighted accumulate for all prompt tokens.
const MoeAccBatchPush = extern struct { N: u32, n_used: u32, src_stride: u32, a_tok_stride: u32, b_tok_stride: u32, routing_stride: u32 };
const MulVecPush = extern struct { N: u32, scale: f32 };
const MulVecBatchPush = extern struct { row: u32, total: u32, scale: f32 };
const ZeroPush = extern struct { N: u32 };
// Batched MoE expert matvec (one launch over all experts; ids read GPU-side).
const ExpertsPush = extern struct { M: u32, K: u32, slice: u32, x_stride: u32, n_used: u32, base: u32 = 0 };
// Token-batched routed-expert matvec (Effort 24 cycle 8): adds per-token strides
// so one launch (grid.y = T) covers all prompt tokens' routed experts.
const ExpertsBatchPush = extern struct { M: u32, K: u32, slice: u32, x_stride: u32, n_used: u32, base: u32 = 0, routing_stride: u32, x_tok_stride: u32, y_tok_stride: u32 };
// Effort 24 cycle 18: builds the expert-sorted (token,slot) work list for the
// grouped routed-expert matvecs (single-block counting sort over T*n_used items).
const BuildOrderPush = extern struct { T: u32, n_used: u32, n_experts: u32, routing_stride: u32 };
// Batched prefill GEMM (Effort 24): Y[T,M] = A[T,K]·W[M,K]^T over all T prompt
// tokens at once (the gemm_*_tiled_v2 kernels). Must byte-match `struct GemmPush`
// in kernels.cu. Offsets are in BYTES (the kernels shift them internally).
const GemmPush = extern struct {
    M: u32,
    K: u32,
    T: u32,
    a_offset: u32 = 0,
    x_offset: u32 = 0,
    y_offset: u32 = 0,
    acc_mode: u32 = 0,
};
// Decode fusion: fused dual Q4_K matvec (two same-input weights → two outputs in one launch).
const Dmmv2Push = extern struct { M0: u32, M1: u32, K: u32 };

fn dmmvIdx(t: gguf.GGMLType) usize {
    return switch (t) {
        .q4_k => 0,
        .q5_k => 1,
        .q6_k => 2,
        .q8_0 => 3,
        .f32 => 4,
        .q5_1 => 5,
        else => 0,
    };
}

/// Per-layer geometry. gemma4 alternates 5 sliding-window layers (head_dim 256,
/// 16 KV heads, rope freq_base 1e4) with 1 full-attention layer (head_dim 512,
/// 4 KV heads, rope freq_base 1e6 + proportional rope_freqs) on a period of 6.
const LayerGeom = struct {
    is_swa: bool,
    head_dim: u32,
    n_kv_head: u32,
    q_dim: u32, // n_head * head_dim
    kv_dim: u32, // n_kv_head * head_dim
    rope_dim: u32, // == head_dim
};

const Derived = struct {
    n_embd: u32,
    n_ff: u32,
    n_head: u32,
    vocab: u32,
    rms_eps: f32,
    n_layers: u32,
    sliding_window: u32,
    // buffer-sizing maxima across layer types
    q_dim_max: u32,
    kv_dim_max: u32,
    head_dim_max: u32,
    // MoE (0 for the dense gemma4-31b; >0 for the gemma4-26b-a4b)
    n_experts: u32, // total routed experts (128)
    n_experts_used: u32, // top-k active experts per token (8)
    shexp_ff: u32, // shared-expert intermediate dim (2112)
    ff_buf_max: u32, // max(n_ff, shexp_ff) for FFN scratch sizing
};

const Pipelines = struct {
    rms_norm: CudaPipeline,
    rms_norm_noweight: CudaPipeline,
    rms_norm_residual: CudaPipeline,
    rms_norm_residual_scale: CudaPipeline,
    rms_norm_residual_norm: CudaPipeline,
    rms_norm_residual_scale_norm: CudaPipeline,
    rms_norm_rope: CudaPipeline,
    rms_norm_rope_qkv: CudaPipeline,
    rms_norm_kvwrite: CudaPipeline,
    rms_norm_rope_batched: CudaPipeline,
    rms_norm_kvwrite_batched: CudaPipeline,
    dmmv: [6]CudaPipeline,
    dmmv_fast: [4]CudaPipeline,
    dmmv_q4k_fast_dual: CudaPipeline, // fuse gate/up & Q/K same-input Q4_K matvecs
    rope: CudaPipeline,
    gemma_attention: CudaPipeline,
    gemma_attention_batched: CudaPipeline,
    geglu: CudaPipeline,
    scale_accumulate: CudaPipeline,
    scalar_mul: CudaPipeline,
    argmax: CudaPipeline,
    // MoE (compiled unconditionally; dispatched only when n_experts>0)
    softmax_topk: CudaPipeline,
    softmax_topk_batched: CudaPipeline, // gemma4-MoE prefill: top-k over all T tokens
    moe_weighted_acc: CudaPipeline,
    moe_weighted_acc_scaled: CudaPipeline, // batched MoE: folds down scale GPU-side
    moe_weighted_acc_scaled_batched: CudaPipeline, // gemma4-MoE prefill combine (all T)
    mul_vec_scaled: CudaPipeline,
    mul_vec_scaled_batched: CudaPipeline, // gemma4-MoE prefill router pre-scale (all T)
    zero_vec: CudaPipeline,
    dmmv_q4k_experts: CudaPipeline, // batched fused gate/up over all experts
    dmmv_q5_1_experts: CudaPipeline, // batched down over all experts
    dmmv_q4k_experts_batched: CudaPipeline, // token-batched gate/up (all T prompt tokens)
    dmmv_q5_1_experts_batched: CudaPipeline, // token-batched down (all T prompt tokens)
    // Effort 24 cycle 18: token-GROUPED routed-expert matvecs — same per-block math
    // as the _batched kernels but grid.y indexes a precomputed expert-sorted work
    // list (build_expert_order) so each expert's weight stays L2-resident across all
    // its tokens. Byte-identical output; opt-in via ZINC_BATCHED_EXPERTS_GROUPED.
    dmmv_q4k_experts_grouped: CudaPipeline,
    dmmv_q5_1_experts_grouped: CudaPipeline,
    build_expert_order: CudaPipeline, // single-block counting sort of (token,slot) by expert
    // Effort 24: register-blocked prefill GEMMs (Q4_K / Q5_K / Q6_K / Q8_0 weights).
    gemm: [4]CudaPipeline,
    gemm_f32: CudaPipeline, // f32-weight prefill GEMM (gemma4-MoE batched router)
    // Effort 24 cycle 11: tensor-core (wmma) fp16 GEMM for Q4_K weights — the
    // dense prefill GEMMs' +2.2× lever, opt-in via ZINC_BATCHED_TC (NOT byte-
    // identical → its own token-correctness gate, never the default path).
    gemm_q4k_tc: CudaPipeline,
    // Effort 24 cycle 12: TC Q4_K GEMM reading a PRE-CONVERTED fp16 activation
    // (f32_to_f16 downcasts the activation once → halves the dominant f32-A read
    // traffic). Output byte-identical to gemm_q4k_tc. Opt-in with the TC path.
    gemm_q4k_tc_f16a: CudaPipeline,
    // Effort 24 cycle 13: same f16-A TC pattern extended to Q6_K weights (dense
    // gemma-31b's ffn_down etc.), which cycles 11/12 left on the f32 fallback.
    gemm_q6k_tc_f16a: CudaPipeline,
    // Effort 24 cycle 14: wider 128x64 M-tile variant of gemm_q4k_tc_f16a. The
    // f16-A activation is the dominant traffic and is re-read once per output
    // M-block (grid.x = M/BM); BM=128 halves grid.x → halves that read. Output is
    // byte-identical to gemm_q4k_tc_f16a (verified). NEGATIVE RESULT: the 44 KB
    // static shared caps occupancy at 1 block/SM (vs m64's 24 KB → 2 blocks/SM),
    // so the lost latency-hiding outweighs the saved A traffic — measured -11.8%
    // on gemma-31b (ABBA x2, 4090). Kept OPT-IN behind ZINC_BATCHED_TC_M128 as a
    // documented experiment; the TC path DEFAULTS to the proven 64x64 m64 kernel.
    gemm_q4k_tc_f16a_m128: CudaPipeline,
    // Effort 24 cycle 15: 8 KB-shared variant of gemm_q4k_tc_f16a — now the DEFAULT
    // Q4_K TC kernel. The prior m64 kernel's 24 KB static shared (dominated by the
    // 16 KB float Cs output stage) caps occupancy at 2 blocks/SM. This kernel reuses
    // the dead Ws+As region for a two-phase Cs output store → 8 KB total → ~3x
    // occupancy → +11.6% on gemma-31b (ABBA x2, 4090). Byte-identical to the m64
    // kernel (same wmma math; phases only reorder writes; GEN_IDS verified identical).
    // ZINC_BATCHED_TC_M64 is the A/B kill-switch back to the 24 KB m64 kernel.
    gemm_q4k_tc_f16a_lowsmem: CudaPipeline,
    // Effort 24 cycle 16: 8 KB-shared variant of gemm_q6k_tc_f16a (dense gemma-31b
    // ffn_down, idx 2) — the cycle-15 Q4_K two-phase-Cs occupancy trick extended to the
    // Q6_K dequant (prior 24 KB m64 kernel caps occupancy at 2 blocks/SM; this reuses the
    // dead Ws+As region for a two-phase Cs store → 8 KB → ~3x occupancy). Byte-identical
    // to gemm_q6k_tc_f16a (same wmma math; phases only reorder writes), but perf-NEUTRAL
    // (Q6_K is ~1/7 of the dense GEMM → below the boost floor) → kept OPT-IN via
    // ZINC_BATCHED_TC_Q6_LOWSMEM; the proven m64 gemm_q6k_tc_f16a stays the default.
    gemm_q6k_tc_f16a_lowsmem: CudaPipeline,
    // Effort 24 cycle 17: the SYNTHESIS of cycle 14 (wider 128x64 M-tile → grid.x = M/128
    // halves the dominant f16-A re-read) and cycle 15 (low-shared two-phase Cs → high
    // occupancy). Cycle 14's plain m128 was -11.8% ONLY because its 44 KB static shared
    // capped occupancy at 1 block/SM; this kernel writes the 128x64 tile in FOUR phases
    // reusing the dead Ws+As region → 12 KB static shared (vs 44 KB) → ~6 blocks/SM (same
    // as the lowsmem default), so the halved A read should now pay off. Byte-identical to
    // the m128/m64/lowsmem kernels (same wmma math; phases only reorder writes).
    gemm_q4k_tc_f16a_m128_lowsmem: CudaPipeline,
    f32_to_f16: CudaPipeline, // element-wise activation downcast for the TC f16-A path
    dequant_q4k_to_f16: CudaPipeline, // e26 c9: full Q4_K weight → fp16 for the cuBLAS prefill GEMM
    // Cycle 21: fp16-EMITTING producers for the TC path — write the normalized /
    // GeGLU activation directly as half into act_f16 (byte-for-byte f32_to_f16 of
    // their f32 twins), dropping the per-GEMM recast launch entirely.
    rms_norm_f16: CudaPipeline,
    geglu_f16: CudaPipeline,
};

/// Per-prompt batched activation scratch (Effort 24 batched prefill). Allocated
/// lazily on the first `prefillBatched` call, sized to the prompt length T, and
/// laid out token-major ([T, dim] contiguous) so the gemm_*_tiled_v2 kernels can
/// read each weight once for all T tokens. Independent of the single-token decode
/// scratch (`hidden`/`q_buf`/… on ForwardGemma) — additive, never aliases it.
const BatchScratch = struct {
    t_cap: u32,
    hidden: CudaBuffer, // [T, n_embd] residual stream
    norm: CudaBuffer, // [T, n_embd] pre-attn / pre-ffn norm output
    q: CudaBuffer, // [T, q_dim_max]
    k: CudaBuffer, // [T, kv_dim_max]
    v: CudaBuffer, // [T, kv_dim_max]
    attn_out: CudaBuffer, // [T, q_dim_max]
    o: CudaBuffer, // [T, n_embd] O-projection
    ffn_norm: CudaBuffer, // [T, n_embd]
    gate: CudaBuffer, // [T, ff_buf_max]
    up: CudaBuffer, // [T, ff_buf_max]
    geglu: CudaBuffer, // [T, ff_buf_max]
    down: CudaBuffer, // [T, n_embd]
    shared: CudaBuffer, // [T, n_embd] gemma4-MoE shared-expert output (post_ffw_norm_1)
    // gemma4-MoE batched router (n_experts>0; size 1 on the dense model)
    router_in: CudaBuffer, // [T, n_embd] plain-RMS-normed residual × ffn_gate_inp.scale
    router_logits: CudaBuffer, // [T, n_experts] f32 router logits
    router_table: CudaBuffer, // [T, 2*n_used] u32: per-token ids then weight-bits
    // gemma4-MoE batched routed-expert FFN scratch (cycle 8; size 1 on the dense model)
    moe_norm_e: CudaBuffer, // [T, n_embd] pre_ffw_norm_2 of the residual, per token
    gate_e: CudaBuffer, // [T, n_used*ef] routed gate projection (slot-major per token)
    up_e: CudaBuffer, // [T, n_used*ef] routed up projection
    geglu_e: CudaBuffer, // [T, n_used*ef] GeGLU(gate,up)
    down_e: CudaBuffer, // [T, n_used*n_embd] routed down projection (slot-major per token)
    moe_out_e: CudaBuffer, // [T, n_embd] routed-expert weighted sum (post_ffw_norm_2), cycle 9
    expert_order: CudaBuffer, // [T*n_used] u32: (token<<16|slot) sorted by expert (cycle 18 grouped path; size 1 on dense)
    // Effort 24 cycle 12: fp16 activation scratch for the TC f16-A GEMM path
    // ([T, ff_buf_max] halves; sized to the largest activation; TC opt-in only).
    act_f16: CudaBuffer,
    // Effort 26 cycle 9: fp16 dense-weight scratch for the cuBLAS prefill GEMM
    // (dequant Q4_K [M,K] → here, then cublasGemmEx). Sized to the largest dense
    // Q4_K weight (max(ff,q_dim,n_embd)·max(n_embd,q_dim) halves). cuBLAS opt-in.
    w_f16: CudaBuffer,
};

pub const ForwardGemma = struct {
    allocator: std.mem.Allocator,
    ctx: ?*shim.CudaCtx,
    model: *loader.Model,
    d: Derived,
    max_ctx: u32,
    geom: []LayerGeom,

    pipes: Pipelines,

    // activation scratch (device, f32)
    hidden: CudaBuffer,
    norm_buf: CudaBuffer,
    q_buf: CudaBuffer,
    k_buf: CudaBuffer,
    v_buf: CudaBuffer,
    attn_out_buf: CudaBuffer, // [q_dim_max]
    o_buf: CudaBuffer, // [n_embd] O-projection / post-attn-norm output
    ffn_norm_buf: CudaBuffer,
    gate_buf: CudaBuffer, // [ff_buf_max]
    up_buf: CudaBuffer, // [ff_buf_max]
    geglu_buf: CudaBuffer, // [ff_buf_max]
    down_buf: CudaBuffer, // [n_embd] dense; [n_used*n_embd] slot-major (MoE)
    logits_buf: CudaBuffer,
    argmax_buf: CudaBuffer,
    host_embed: []f32,
    // async decode command ring (dense path, n_experts==0): each per-block
    // command commitAsync's on the shared auto-ordered CUstream and stashes
    // here; the tail commitAndWait drains the stream and drainPending frees the
    // events. Sized for gemma-31b's ~180 ops/token (3 blocks × 60 layers). The
    // 26b MoE keeps the sync path (its router reads ids back mid-block).
    pending: [1024]command.CudaCommand = undefined,
    n_pending: u32 = 0,

    // MoE scratch (only used when n_experts > 0)
    shared_buf: CudaBuffer, // [n_embd] shared-expert output (post_ffw_norm_1)
    moe_norm_buf: CudaBuffer, // [n_embd] pre_ffw_norm_2 (expert input)
    moe_out_buf: CudaBuffer, // [n_embd] routed-expert weighted sum (post_ffw_norm_2)
    router_logits_buf: CudaBuffer, // [n_experts] f32 router logits
    router_out_buf: CudaBuffer, // [2*n_used] u32: ids then weight-bits
    host_router: []u32, // [2*n_used] downloaded ids + weight bits
    down_scales: []f32, // [n_layers*n_experts] per-expert ffn_down_exps scale

    // per-layer-type rope tables (host-precomputed effective inv_freq)
    inv_freq_swa: CudaBuffer, // [rope_dim_swa/2]
    inv_freq_full: CudaBuffer, // [rope_dim_full/2] (folds in rope_freqs)

    // KV cache per layer (sized by that layer's kv_dim)
    kv_k: []CudaBuffer,
    kv_v: []CudaBuffer,

    // Effort 24: lazily-allocated batched-prefill scratch (null until the first
    // ZINC_BATCHED_PREFILL run; freed in deinit).
    batch: ?BatchScratch = null,
    // Effort 24 cycle 11: route the dense batched Q4_K GEMMs through the fp16
    // tensor-core kernel (gemm_q4k_tc) instead of the f32 register-tiled GEMM.
    // Opt-in (ZINC_BATCHED_TC, read once per prefillBatched); off by default so
    // the proven byte-identical path is unchanged. NOT byte-identical when on.
    use_tc: bool = false,
    use_cublas: bool = false, // e26 c9: dense Q4_K prefill GEMMs via cuBLAS fp16 TC (dequant W→fp16 + cublasGemmEx). DEFAULT-ON (opt out ZINC_BATCHED_CUBLAS=0/off); supersedes the use_tc Q4_K (idx==0) branch when T >= cublas_min_t.
    cublas_min_t: u32 = 128, // e26 c9: only route Q4_K GEMMs through cuBLAS when the token batch T >= this (the dequant→fp16 round-trip is a fixed per-weight cost; cuBLAS wins +76% @T=512 / +15% @T=128 but is break-even @T=64). Below it, fall back to gemm_q4k_tc.
    use_tc_plain: bool = false, // cycle 12 A/B: force cycle-11 plain TC (no f16-A pre-convert)
    use_tc_q6: bool = true, // cycle 13 A/B: ZINC_BATCHED_TC_NOQ6 forces Q6_K back to f32 TC-off
    use_tc_m128: bool = false, // cycle 14 A/B: ZINC_BATCHED_TC_M128 opts into the wider 128x64 Q4_K TC kernel (NEGATIVE: -11.8%, off by default)
    use_tc_m64: bool = false, // cycle 15 A/B: ZINC_BATCHED_TC_M64 kill-switch forces the prior 24 KB-shared Q4_K TC kernel (cycle 12 default); the new default is the 8 KB-shared lowsmem kernel (+11.6%, byte-identical)
    use_tc_q6_lowsmem: bool = false, // cycle 16 A/B: ZINC_BATCHED_TC_Q6_LOWSMEM opts INTO the 8 KB-shared lowsmem Q6_K TC kernel (gemm_q6k_tc_f16a_lowsmem). Byte-identical to the default 24 KB m64 Q6_K kernel but in-noise on perf (Q6_K is ~1/7 of the dense GEMM → its occupancy win is below the box's boost floor; 2 ABBA runs nominally -1/-5%) → kept OPT-IN, the proven m64 kernel stays the default.
    use_grouped: bool = false, // cycle 18: ZINC_BATCHED_EXPERTS_GROUPED opts into token-GROUPED routed experts (build_expert_order + grouped matvecs → expert weight L2-resident across its tokens). Byte-identical to the cycle-8 _batched path; opt-in pending a measured win.
    use_tc_m128_lowsmem: bool = false, // cycle 17 A/B: ZINC_BATCHED_TC_M128_LOWSMEM opts INTO the 12 KB-shared wider 128x64 M-tile Q4_K TC kernel (gemm_q4k_tc_f16a_m128_lowsmem) — synthesis of cycle 14's wider tile (halves the dominant f16-A read) + cycle 15's two-phase Cs (12 KB shared → ~6 blocks/SM, NOT m128's 44 KB→1 block/SM that lost -11.8%). Byte-identical to the m64/lowsmem default; measured this cycle to decide if it becomes the default.
    use_tc_sharea: bool = false, // cycle 19: ZINC_BATCHED_TC_SHAREA shares ONE f32→f16 activation recast across GEMMs that read the SAME input (attn Q/K/V from b.norm; FFN gate/up from b.ffn_norm) — skips the redundant per-GEMM f32_to_f16 launch + read for the 2nd/3rd GEMM of each group. Byte-identical (same __float2half bits, same act_f16 contents reused stream-ordered). Off → each GEMM recasts independently (cycle 12 behavior).
    use_tc_normf16: bool = false, // cycle 21: ZINC_BATCHED_TC_NORMF16 has the norm/GeGLU PRODUCERS emit fp16 directly into act_f16 (rms_norm_f16/geglu_f16) so ALL the dense TC GEMMs reading a produced activation (attn Q/K/V from the pre-attn norm; FFN gate/up from the pre-FFN norm; ffn_down from GeGLU) skip their per-GEMM f32→fp16 recast launch ENTIRELY — not just the shared-A dedup. Byte-identical to the per-GEMM-recast TC path (the producer __float2half's the SAME f32 value f32_to_f16 would). Off → cycle-12 per-GEMM recast.

    pub fn init(allocator: std.mem.Allocator, model: *loader.Model, max_ctx: u32) !ForwardGemma {
        const ctx = model.ctx;
        if (model.config.architecture != .gemma) return error.UnsupportedArchitecture;
        const c = model.config;

        // ---- per-layer geometry from the GGUF arrays ------------------------
        const arch_str = model.gguf_file.getString("general.architecture") orelse "gemma4";
        const n_layers = c.n_layers;
        const n_head = c.n_heads;
        const geom = try allocator.alloc(LayerGeom, n_layers);
        errdefer allocator.free(geom);

        const swa_pattern = try readBoolArray(allocator, &model.gguf_file, arch_str, "attention.sliding_window_pattern", n_layers);
        defer allocator.free(swa_pattern);
        const kv_heads = try readU32Array(allocator, &model.gguf_file, arch_str, "attention.head_count_kv", n_layers);
        defer allocator.free(kv_heads);

        const hd_full = c.head_dim; // attention.key_length (512)
        const hd_swa = readArchU32(&model.gguf_file, arch_str, "attention.key_length_swa") orelse hd_full;

        var q_dim_max: u32 = 0;
        var kv_dim_max: u32 = 0;
        var head_dim_max: u32 = 0;
        for (0..n_layers) |i| {
            const is_swa = swa_pattern[i];
            const hd: u32 = if (is_swa) hd_swa else hd_full;
            const nkv: u32 = kv_heads[i];
            const g = LayerGeom{
                .is_swa = is_swa,
                .head_dim = hd,
                .n_kv_head = nkv,
                .q_dim = n_head * hd,
                .kv_dim = nkv * hd,
                .rope_dim = hd,
            };
            geom[i] = g;
            q_dim_max = @max(q_dim_max, g.q_dim);
            kv_dim_max = @max(kv_dim_max, g.kv_dim);
            head_dim_max = @max(head_dim_max, hd);
        }

        const d = Derived{
            .n_embd = c.hidden_dim,
            .n_ff = c.intermediate_dim,
            .n_head = n_head,
            .vocab = c.vocab_size,
            .rms_eps = c.rms_norm_eps,
            .n_layers = n_layers,
            .sliding_window = c.sliding_window_size,
            .q_dim_max = q_dim_max,
            .kv_dim_max = kv_dim_max,
            .head_dim_max = head_dim_max,
            .n_experts = c.n_experts,
            .n_experts_used = c.n_experts_used,
            .shexp_ff = c.shared_expert_intermediate_dim,
            .ff_buf_max = @max(c.intermediate_dim, c.shared_expert_intermediate_dim),
        };

        // ---- compile kernels -----------------------------------------------
        const src = try allocator.dupeZ(u8, KERNELS_CU);
        defer allocator.free(src);
        var pipes: Pipelines = undefined;
        pipes.rms_norm = try pipeline.createPipeline(ctx, src.ptr, "rms_norm");
        pipes.rms_norm_noweight = try pipeline.createPipeline(ctx, src.ptr, "rms_norm_noweight");
        pipes.rms_norm_residual = try pipeline.createPipeline(ctx, src.ptr, "rms_norm_residual");
        pipes.rms_norm_residual_scale = try pipeline.createPipeline(ctx, src.ptr, "rms_norm_residual_scale");
        pipes.rms_norm_residual_norm = try pipeline.createPipeline(ctx, src.ptr, "rms_norm_residual_norm");
        pipes.rms_norm_residual_scale_norm = try pipeline.createPipeline(ctx, src.ptr, "rms_norm_residual_scale_norm");
        pipes.rms_norm_rope = try pipeline.createPipeline(ctx, src.ptr, "rms_norm_rope");
        pipes.rms_norm_rope_qkv = try pipeline.createPipeline(ctx, src.ptr, "rms_norm_rope_qkv");
        pipes.rms_norm_kvwrite = try pipeline.createPipeline(ctx, src.ptr, "rms_norm_kvwrite");
        pipes.rms_norm_rope_batched = try pipeline.createPipeline(ctx, src.ptr, "rms_norm_rope_batched");
        pipes.rms_norm_kvwrite_batched = try pipeline.createPipeline(ctx, src.ptr, "rms_norm_kvwrite_batched");
        pipes.dmmv[0] = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q4k");
        pipes.dmmv[1] = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q5k");
        pipes.dmmv[2] = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q6k");
        pipes.dmmv[3] = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q8_0");
        pipes.dmmv[4] = try pipeline.createPipeline(ctx, src.ptr, "dmmv_f32");
        pipes.dmmv[5] = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q5_1");
        pipes.dmmv_fast[0] = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q4k_fast");
        pipes.dmmv_fast[1] = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q5k_fast");
        pipes.dmmv_fast[2] = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q6k_fast");
        pipes.dmmv_fast[3] = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q8_0_fast");
        pipes.dmmv_q4k_fast_dual = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q4k_fast_dual");
        pipes.rope = try pipeline.createPipeline(ctx, src.ptr, "rope");
        pipes.gemma_attention = try pipeline.createPipeline(ctx, src.ptr, "gemma_attention");
        pipes.gemma_attention_batched = try pipeline.createPipeline(ctx, src.ptr, "gemma_attention_batched");
        pipes.geglu = try pipeline.createPipeline(ctx, src.ptr, "geglu");
        pipes.scale_accumulate = try pipeline.createPipeline(ctx, src.ptr, "scale_accumulate");
        pipes.scalar_mul = try pipeline.createPipeline(ctx, src.ptr, "scalar_mul");
        pipes.argmax = try pipeline.createPipeline(ctx, src.ptr, "argmax");
        pipes.softmax_topk = try pipeline.createPipeline(ctx, src.ptr, "softmax_topk");
        pipes.softmax_topk_batched = try pipeline.createPipeline(ctx, src.ptr, "softmax_topk_batched");
        pipes.moe_weighted_acc = try pipeline.createPipeline(ctx, src.ptr, "moe_weighted_acc");
        pipes.moe_weighted_acc_scaled = try pipeline.createPipeline(ctx, src.ptr, "moe_weighted_acc_scaled");
        pipes.moe_weighted_acc_scaled_batched = try pipeline.createPipeline(ctx, src.ptr, "moe_weighted_acc_scaled_batched");
        pipes.mul_vec_scaled = try pipeline.createPipeline(ctx, src.ptr, "mul_vec_scaled");
        pipes.mul_vec_scaled_batched = try pipeline.createPipeline(ctx, src.ptr, "mul_vec_scaled_batched");
        pipes.zero_vec = try pipeline.createPipeline(ctx, src.ptr, "zero_vec");
        pipes.dmmv_q4k_experts = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q4k_experts");
        pipes.dmmv_q5_1_experts = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q5_1_experts");
        pipes.dmmv_q4k_experts_batched = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q4k_experts_batched");
        pipes.dmmv_q5_1_experts_batched = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q5_1_experts_batched");
        pipes.dmmv_q4k_experts_grouped = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q4k_experts_grouped");
        pipes.dmmv_q5_1_experts_grouped = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q5_1_experts_grouped");
        pipes.build_expert_order = try pipeline.createPipeline(ctx, src.ptr, "build_expert_order");
        // Effort 24: batched-prefill GEMMs (Q4_K / Q5_K / Q6_K).
        pipes.gemm[0] = try pipeline.createPipeline(ctx, src.ptr, "gemm_q4k_tiled_v2");
        pipes.gemm[1] = try pipeline.createPipeline(ctx, src.ptr, "gemm_q5k_tiled_v2");
        pipes.gemm[2] = try pipeline.createPipeline(ctx, src.ptr, "gemm_q6k_tiled_v2");
        pipes.gemm[3] = try pipeline.createPipeline(ctx, src.ptr, "gemm_q8_0_tiled_v2");
        pipes.gemm_f32 = try pipeline.createPipeline(ctx, src.ptr, "gemm_f32_tiled_v2");
        pipes.gemm_q4k_tc = try pipeline.createPipeline(ctx, src.ptr, "gemm_q4k_tc");
        pipes.gemm_q4k_tc_f16a = try pipeline.createPipeline(ctx, src.ptr, "gemm_q4k_tc_f16a");
        pipes.gemm_q6k_tc_f16a = try pipeline.createPipeline(ctx, src.ptr, "gemm_q6k_tc_f16a");
        pipes.gemm_q4k_tc_f16a_m128 = try pipeline.createPipeline(ctx, src.ptr, "gemm_q4k_tc_f16a_m128");
        pipes.gemm_q4k_tc_f16a_lowsmem = try pipeline.createPipeline(ctx, src.ptr, "gemm_q4k_tc_f16a_lowsmem");
        pipes.gemm_q6k_tc_f16a_lowsmem = try pipeline.createPipeline(ctx, src.ptr, "gemm_q6k_tc_f16a_lowsmem");
        pipes.gemm_q4k_tc_f16a_m128_lowsmem = try pipeline.createPipeline(ctx, src.ptr, "gemm_q4k_tc_f16a_m128_lowsmem");
        pipes.f32_to_f16 = try pipeline.createPipeline(ctx, src.ptr, "f32_to_f16");
        pipes.dequant_q4k_to_f16 = try pipeline.createPipeline(ctx, src.ptr, "dequant_q4k_to_f16");
        pipes.rms_norm_f16 = try pipeline.createPipeline(ctx, src.ptr, "rms_norm_f16");
        pipes.geglu_f16 = try pipeline.createPipeline(ctx, src.ptr, "geglu_f16");
        log.info("nvrtc: compiled gemma4 kernel pipelines", .{});

        const f4 = @sizeOf(f32);
        var self = ForwardGemma{
            .allocator = allocator,
            .ctx = ctx,
            .model = model,
            .d = d,
            .max_ctx = max_ctx,
            .geom = geom,
            .pipes = pipes,
            .hidden = try buffer.createBuffer(ctx, d.n_embd * f4),
            .norm_buf = try buffer.createBuffer(ctx, d.n_embd * f4),
            .q_buf = try buffer.createBuffer(ctx, q_dim_max * f4),
            .k_buf = try buffer.createBuffer(ctx, kv_dim_max * f4),
            .v_buf = try buffer.createBuffer(ctx, kv_dim_max * f4),
            .attn_out_buf = try buffer.createBuffer(ctx, q_dim_max * f4),
            .o_buf = try buffer.createBuffer(ctx, d.n_embd * f4),
            .ffn_norm_buf = try buffer.createBuffer(ctx, d.n_embd * f4),
            .gate_buf = try buffer.createBuffer(ctx, @max(d.ff_buf_max, d.n_experts_used * d.n_ff) * f4),
            .up_buf = try buffer.createBuffer(ctx, @max(d.ff_buf_max, d.n_experts_used * d.n_ff) * f4),
            .geglu_buf = try buffer.createBuffer(ctx, @max(d.ff_buf_max, d.n_experts_used * d.n_ff) * f4),
            .down_buf = try buffer.createBuffer(ctx, @max(d.n_embd, d.n_experts_used * d.n_embd) * f4),
            .logits_buf = try buffer.createBuffer(ctx, d.vocab * f4),
            .argmax_buf = try buffer.createBuffer(ctx, @sizeOf(u32)),
            .host_embed = try allocator.alloc(f32, d.n_embd),
            // MoE scratch (tiny-but-nonzero stubs keep the dense path uniform).
            .shared_buf = try buffer.createBuffer(ctx, d.n_embd * f4),
            .moe_norm_buf = try buffer.createBuffer(ctx, d.n_embd * f4),
            .moe_out_buf = try buffer.createBuffer(ctx, d.n_embd * f4),
            .router_logits_buf = try buffer.createBuffer(ctx, @max(@as(u32, 1), d.n_experts) * f4),
            .router_out_buf = try buffer.createBuffer(ctx, @max(@as(u32, 1), 2 * d.n_experts_used) * @sizeOf(u32)),
            .host_router = try allocator.alloc(u32, @max(@as(u32, 1), 2 * d.n_experts_used)),
            .down_scales = try allocator.alloc(f32, @max(@as(u32, 1), d.n_layers * d.n_experts)),
            .inv_freq_swa = try buffer.createBuffer(ctx, @max(@as(u32, 1), hd_swa / 2) * f4),
            .inv_freq_full = try buffer.createBuffer(ctx, @max(@as(u32, 1), hd_full / 2) * f4),
            .kv_k = try allocator.alloc(CudaBuffer, n_layers),
            .kv_v = try allocator.alloc(CudaBuffer, n_layers),
        };

        // ---- per-layer KV cache --------------------------------------------
        for (0..n_layers) |li| {
            self.kv_k[li] = try buffer.createBuffer(ctx, max_ctx * geom[li].kv_dim * f4);
            self.kv_v[li] = try buffer.createBuffer(ctx, max_ctx * geom[li].kv_dim * f4);
        }

        // ---- rope tables ----------------------------------------------------
        // SWA layers: inv_freq[i] = 1 / freq_base_swa^(2i/rope_dim_swa).
        {
            const half = hd_swa / 2;
            const hf = try allocator.alloc(f32, half);
            defer allocator.free(hf);
            const fb = c.rope_freq_base_swa; // 1e4
            for (0..half) |k| {
                const exp = @as(f32, @floatFromInt(2 * k)) / @as(f32, @floatFromInt(hd_swa));
                hf[k] = 1.0 / std.math.pow(f32, fb, exp);
            }
            buffer.upload(ctx, &self.inv_freq_swa, std.mem.sliceAsBytes(hf));
        }
        // Full layers: inv_freq[i] = (1 / freq_base^(2i/rope_dim)) / rope_freqs[i].
        {
            const half = hd_full / 2;
            const hf = try allocator.alloc(f32, half);
            defer allocator.free(hf);
            const fb = c.rope_freq_base; // 1e6
            const rf = try allocator.alloc(f32, half);
            defer allocator.free(rf);
            @memset(rf, 1.0);
            // rope_freqs is stored PER-LAYER as `blk.{i}.rope_freqs.weight` on the
            // full-attention layers (llama.cpp `tn(LLM_TENSOR_ROPE_FREQS,"weight",i)`),
            // all sharing one copy (TENSOR_DUPLICATED) — NOT as a global tensor. The
            // old global `model.get("rope_freqs.weight")` returned null, so rf stayed
            // 1.0 and proportional rope was silently skipped on full layers, drifting
            // the argmax with position. Prefer the global name (some converters use
            // it) then fall back to the first full-attention layer's per-layer tensor.
            var rope_freqs_t: ?*const LoadedTensor = model.get("rope_freqs.weight");
            if (rope_freqs_t == null) {
                for (0..n_layers) |i| {
                    if (geom[i].is_swa) continue;
                    if (model.getLayer(@intCast(i), "rope_freqs.weight")) |t| {
                        rope_freqs_t = t;
                        break;
                    }
                }
            }
            if (rope_freqs_t) |t| {
                if (t.info.numElements() == half and t.info.type_ == .f32) {
                    buffer.download(ctx, &t.gpu_buffer, std.mem.sliceAsBytes(rf));
                }
            }
            for (0..half) |k| {
                const exp = @as(f32, @floatFromInt(2 * k)) / @as(f32, @floatFromInt(hd_full));
                const base = 1.0 / std.math.pow(f32, fb, exp);
                const ff = if (rf[k] != 0) rf[k] else 1.0;
                hf[k] = base / ff;
            }
            buffer.upload(ctx, &self.inv_freq_full, std.mem.sliceAsBytes(hf));
        }

        // ---- per-expert down scales (MoE) ----------------------------------
        // gemma4 MoE multiplies each routed expert's down output by a per-expert
        // scalar (ffn_down_exps.scale). Pre-download to the host so the routed
        // combine can fold it into the router weights.
        if (d.n_experts > 0) {
            @memset(self.down_scales, 1.0);
            for (0..n_layers) |li| {
                const ts = model.getLayer(@intCast(li), "ffn_down_exps.scale") orelse continue;
                if (ts.info.numElements() != d.n_experts) continue;
                buffer.download(ctx, &ts.gpu_buffer, std.mem.sliceAsBytes(self.down_scales[li * d.n_experts ..][0..d.n_experts]));
            }
        }

        return self;
    }

    pub fn deinit(self: *ForwardGemma) void {
        const a = self.allocator;
        inline for (.{ &self.hidden, &self.norm_buf, &self.q_buf, &self.k_buf, &self.v_buf, &self.attn_out_buf, &self.o_buf, &self.ffn_norm_buf, &self.gate_buf, &self.up_buf, &self.geglu_buf, &self.down_buf, &self.logits_buf, &self.argmax_buf, &self.inv_freq_swa, &self.inv_freq_full, &self.shared_buf, &self.moe_norm_buf, &self.moe_out_buf, &self.router_logits_buf, &self.router_out_buf }) |b| {
            buffer.freeBuffer(b);
        }
        for (self.kv_k) |*b| buffer.freeBuffer(b);
        for (self.kv_v) |*b| buffer.freeBuffer(b);
        a.free(self.kv_k);
        a.free(self.kv_v);
        a.free(self.geom);
        a.free(self.host_embed);
        a.free(self.host_router);
        a.free(self.down_scales);
        self.freeBatch();
        inline for (std.meta.fields(Pipelines)) |f| {
            if (comptime std.mem.eql(u8, f.name, "dmmv")) {
                for (&self.pipes.dmmv) |*p| pipeline.freePipeline(p);
            } else if (comptime std.mem.eql(u8, f.name, "dmmv_fast")) {
                for (&self.pipes.dmmv_fast) |*p| pipeline.freePipeline(p);
            } else if (comptime std.mem.eql(u8, f.name, "gemm")) {
                for (&self.pipes.gemm) |*p| pipeline.freePipeline(p);
            } else {
                pipeline.freePipeline(&@field(self.pipes, f.name));
            }
        }
        self.* = undefined;
    }

    /// One greedy decode step for `token` at sequence position `pos`.
    pub fn decodeStep(self: *ForwardGemma, token: u32, pos: u32, run_layers: bool) !u32 {
        const d = self.d;
        const ctx = self.ctx;

        // EMBED: dequant token row on the CPU, scale by sqrt(n_embd), upload.
        self.model.dequantEmbeddingRow(token, self.host_embed);
        const embd_scale = std.math.sqrt(@as(f32, @floatFromInt(d.n_embd)));
        for (self.host_embed) |*v| v.* *= embd_scale;
        buffer.upload(ctx, &self.hidden, std.mem.sliceAsBytes(self.host_embed));

        if (run_layers) {
            var L: u32 = 0;
            while (L < d.n_layers) : (L += 1) {
                try self.attentionLayer(L, pos);
                if (d.n_experts > 0 and self.model.getLayer(L, "ffn_gate_inp.weight") != null) {
                    try self.moeFfnBlock(L);
                } else {
                    try self.ffnBlock(L);
                }
                try self.layerOutScale(L);
            }
        }

        // TAIL: final rms_norm → LM head (tied token_embd) → argmax. The final
        // logit soft-cap is monotonic and omitted (greedy argmax unaffected).
        const out_norm = self.model.get("output_norm.weight") orelse return error.MissingTensor;
        const lm_head = self.model.get("output.weight") orelse self.model.get("token_embd.weight") orelse return error.MissingTensor;

        var cmd = try command.beginCommand(ctx);
        const rms = RmsPush{ .N = d.n_embd, .eps = d.rms_eps };
        cmd.dispatch(&self.pipes.rms_norm, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.hidden, &out_norm.gpu_buffer, &self.norm_buf }, &rms, @sizeOf(RmsPush), 0);
        const lm = DmmvPush{ .M = d.vocab, .K = d.n_embd };
        const lm_idx = dmmvIdx(lm_head.info.type_);
        if (lm_idx < 4) {
            cmd.dispatch(&self.pipes.dmmv_fast[lm_idx], .{ d.vocab, 1, 1 }, .{ 64, 1, 1 }, &.{ &lm_head.gpu_buffer, &self.norm_buf, &self.logits_buf }, &lm, @sizeOf(DmmvPush), 0);
        } else {
            cmd.dispatch(&self.pipes.dmmv[lm_idx], .{ d.vocab, 1, 1 }, .{ 256, 1, 1 }, &.{ &lm_head.gpu_buffer, &self.norm_buf, &self.logits_buf }, &lm, @sizeOf(DmmvPush), 0);
        }
        const am = ArgmaxPush{ .N = d.vocab };
        cmd.dispatch(&self.pipes.argmax, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.logits_buf, &self.argmax_buf }, &am, @sizeOf(ArgmaxPush), 0);
        cmd.commitAndWait(); // drains the shared stream incl. the async layer ops
        self.drainPending(); // free the stashed async commands (completion guaranteed)

        var tok: u32 = 0;
        buffer.download(ctx, &self.argmax_buf, std.mem.asBytes(&tok));
        return tok;
    }

    /// Prefill helper (mirrors ForwardCuda.prefillStep): run every layer to
    /// build the KV cache, but SKIP the tail rms_norm + LM head + argmax — a
    /// prompt-internal token's logits are never read. Saves the vocab-sized
    /// head matvec on T-1 of the T prompt tokens; the MoE gemma model (small
    /// active forward, full head) benefits most. Async layer ops drained here;
    /// bit-identical generation.
    pub fn prefillStep(self: *ForwardGemma, token: u32, pos: u32) !void {
        const d = self.d;
        const ctx = self.ctx;
        self.model.dequantEmbeddingRow(token, self.host_embed);
        const embd_scale = std.math.sqrt(@as(f32, @floatFromInt(d.n_embd)));
        for (self.host_embed) |*v| v.* *= embd_scale;
        buffer.upload(ctx, &self.hidden, std.mem.sliceAsBytes(self.host_embed));
        var L: u32 = 0;
        while (L < d.n_layers) : (L += 1) {
            try self.attentionLayer(L, pos);
            if (d.n_experts > 0 and self.model.getLayer(L, "ffn_gate_inp.weight") != null) {
                try self.moeFfnBlock(L);
            } else {
                try self.ffnBlock(L);
            }
            try self.layerOutScale(L);
        }
        self.waitPending(); // drain async layer ops; no logits for prompt-internal tokens
    }

    // ---- Effort 24: batched-GEMM prefill ------------------------------------

    /// Batched dense-gemma prefill: process ALL T prompt tokens at once, reading
    /// each weight ONCE for all tokens via the gemm_*_tiled_v2 register-blocked
    /// GEMMs (5.9× over the per-token matvec). Returns the last token's argmax
    /// (= the first generated token), exactly like running prefillStep on tokens
    /// [0..T-1] then decodeStep on the last. ADDITIVE: builds its own token-major
    /// scratch and never touches the single-token decode path.
    ///
    /// Phase-1 scope is the DENSE gemma-31b (n_experts==0). Phase-2 cycle 1 adds
    /// the 26b MoE: its ATTENTION block is the SAME structure as the dense model,
    /// so it shares the batched attention path (GEMM Q/K/V/O + batched causal attn
    /// + batched norm/RoPE/KV-write) — bit-identical to the per-token attentionLayer.
    /// Its routed-expert FFN is still LOOPED per token (the existing single-token
    /// moeFfnBlock, fed each token's hidden slice via an alias-swap) because the
    /// FFN is position-independent → looping it is output-identical; full
    /// batched-expert routing (group T tokens by expert) is a later cycle.
    pub fn prefillBatched(self: *ForwardGemma, tokens: []const u32) !u32 {
        const d = self.d;
        const ctx = self.ctx;
        const T: u32 = @intCast(tokens.len);
        const f4 = @sizeOf(f32);
        const moe = d.n_experts > 0; // gemma-26b-a4b: batched attn + per-token MoE FFN
        // Cycle 11: fp16 tensor-core GEMM for the dense Q4_K projections/FFN.
        // Read once here so gemmDispatch can pick the kernel per weight without a
        // getenv per launch.
        // Effort 26 cycle 1: DEFAULT ON. Re-profiling on the RTX 5090 (Blackwell)
        // showed gemma-31b dense prefill is COMPUTE-bound (~100% util at full
        // 2850 MHz boost), not launch-bound — so the tensor-core GEMM is a real
        // end-to-end win here (+24% prefill, ABBA x3, 5090; catalog 5/5
        // token-correct). Effort 24 found it neutral on the 4090 (weaker fp16 TC),
        // never negative, so defaulting on is safe there. Opt out with
        // ZINC_BATCHED_TC=0/off/false/no (the A/B kill-switch back to the
        // f32 register-tiled GEMM).
        self.use_tc = tcDefaultOn();
        // Effort 26 cycle 9: dense Q4_K prefill GEMMs run on cuBLAS fp16 tensor
        // cores (dequant W→fp16 + cublasGemmEx). DEFAULT-ON (opt out
        // ZINC_BATCHED_CUBLAS=0/off/false/no) — validated catalog 5/5 token-correct
        // and +76% on gemma-31b dense prefill @T=512 (the effort's #1 gap row),
        // neutral on gemma-26b. Gated on T >= cublas_min_t in gemmDispatchA so
        // short prompts keep the proven gemm_q4k_tc path.
        self.use_cublas = cublasDefaultOn();
        // Cycle 12 A/B knob: ZINC_BATCHED_TC_PLAIN forces the cycle-11 plain TC
        // GEMM (f32 activation re-read per M-block) instead of the cycle-12 f16-A
        // path (activation pre-converted to fp16 once). Lets us measure the f16-A
        // memory-traffic win in isolation. Unset → the f16-A path (cycle 12 default).
        self.use_tc_plain = std.posix.getenv("ZINC_BATCHED_TC_PLAIN") != null;
        // Cycle 13 A/B knob: ZINC_BATCHED_TC_NOQ6 forces the dense Q6_K GEMMs
        // (ffn_down etc.) back onto the f32 register-tiled gemm_q6k_tiled_v2 even
        // when the TC path is on — lets us measure the Q6_K-on-TC increment in
        // isolation (= cycle-12 behavior: Q4_K on TC, Q6_K on f32). Unset → Q6_K
        // also runs the fp16 TC f16-A kernel (cycle 13 default).
        self.use_tc_q6 = std.posix.getenv("ZINC_BATCHED_TC_NOQ6") == null;
        // Cycle 14 A/B knob: ZINC_BATCHED_TC_M128 opts INTO the wider 128x64 Q4_K
        // TC kernel (gemm_q4k_tc_f16a_m128). It halves the dominant f16-A re-read
        // (grid.x = M/128 vs M/64) but its 44 KB static shared caps occupancy at 1
        // block/SM → NEGATIVE: -11.8% on gemma-31b (ABBA x2, 4090). Default unset →
        // the proven 64x64 gemm_q4k_tc_f16a (cycle 12). plain-TC (A/B above) overrides.
        self.use_tc_m128 = std.posix.getenv("ZINC_BATCHED_TC_M128") != null;
        // Cycle 15: the 8 KB-shared Q4_K TC kernel (gemm_q4k_tc_f16a_lowsmem) is now
        // the DEFAULT. Cycle 14's m128 result proved this GEMM is occupancy-bound; the
        // prior m64 kernel's 24 KB shared (dominated by the 16 KB float Cs output stage)
        // caps it at 2 blocks/SM. The lowsmem kernel reuses the dead Ws+As region for a
        // two-phase Cs store → 8 KB → ~3x occupancy → +11.6% on gemma-31b (ABBA x2, 4090),
        // byte-identical output (verified: GEN_IDS identical to m64 on collapsed + varied
        // prompts). ZINC_BATCHED_TC_M64 is the A/B kill-switch back to the prior 24 KB
        // kernel (gemm_q4k_tc_f16a, cycle 12 default). m128/plain-TC (above) override it.
        self.use_tc_m64 = std.posix.getenv("ZINC_BATCHED_TC_M64") != null;
        // Cycle 16: the 8 KB-shared Q6_K TC kernel (gemm_q6k_tc_f16a_lowsmem) applies the
        // cycle-15 two-phase-Cs occupancy trick (24 KB→8 KB shared, 2→~8 blocks/SM) to the
        // dense gemma-31b ffn_down Q6_K GEMM (idx 2). Byte-identical to the default 24 KB m64
        // Q6_K kernel (gemm_q6k_tc_f16a) — but unlike the Q4_K lowsmem win it is perf-NEUTRAL
        // (in-noise): the Q6_K GEMM is only ~1/7 of the dense work, so its occupancy win is
        // below the box's ±10% boost floor (2 ABBA runs nominally -1%/-5%, ranges overlapping).
        // → kept OPT-IN; ZINC_BATCHED_TC_Q6_LOWSMEM opts into it, the proven m64 kernel stays default.
        self.use_tc_q6_lowsmem = std.posix.getenv("ZINC_BATCHED_TC_Q6_LOWSMEM") != null;
        // Cycle 17: ZINC_BATCHED_TC_M128_LOWSMEM opts into the wider 128x64 M-tile Q4_K TC
        // kernel that ALSO uses the low-shared two-phase Cs trick — the synthesis of cycle 14
        // (halve the dominant f16-A re-read via grid.x=M/128) and cycle 15 (12 KB shared →
        // ~6 blocks/SM, avoiding the 44 KB→1 block/SM occupancy collapse that made plain m128
        // -11.8%). Byte-identical to the lowsmem default; if measured faster it becomes default.
        self.use_tc_m128_lowsmem = std.posix.getenv("ZINC_BATCHED_TC_M128_LOWSMEM") != null;
        // Cycle 18: ZINC_BATCHED_EXPERTS_GROUPED routes the gemma-26b MoE routed-expert
        // matvecs through the GROUPED kernels (build_expert_order sorts the T*n_used
        // (token,slot) work-items by expert id so each expert's Q4_K/Q5_1 weight stays
        // L2-resident across all the tokens routed to it — a memory-traffic win beyond
        // the cycle-8 launch batching). Byte-identical output (same per-block math; each
        // output computed once). Off → the proven cycle-8 _batched matvecs.
        self.use_grouped = std.posix.getenv("ZINC_BATCHED_EXPERTS_GROUPED") != null;
        // Cycle 19: ZINC_BATCHED_TC_SHAREA shares one f32→f16 activation recast across
        // the GEMMs that read the SAME input on the TC path (attn Q/K/V all read b.norm;
        // FFN/shared-expert gate+up both read b.ffn_norm). With it on, only the FIRST GEMM
        // of each group runs f32_to_f16 into act_f16; the rest reuse it (a_preconv=true).
        // Byte-identical (same downcast bits, act_f16 untouched between the group's GEMMs,
        // stream-ordered reuse) — removes 2 recast launches/attn layer + 1/FFN + 1/shared.
        self.use_tc_sharea = std.posix.getenv("ZINC_BATCHED_TC_SHAREA") != null;
        // Cycle 21: ZINC_BATCHED_TC_NORMF16 — the heavier half of the activation-fp16
        // lever: the norm/GeGLU producers EMIT fp16 (rms_norm_f16/geglu_f16) into act_f16
        // so every dense TC GEMM reading a produced activation skips its f32→fp16 recast
        // entirely (the O projection, whose input is the f32 attention output, still
        // recasts). Byte-identical to the per-GEMM-recast TC path; only meaningful with
        // ZINC_BATCHED_TC. Implies the shared-A reuse for the consumer GEMMs.
        self.use_tc_normf16 = std.posix.getenv("ZINC_BATCHED_TC_NORMF16") != null;

        const b = try self.ensureBatch(T);

        // EMBED all T tokens into hidden [T, n_embd] (dequant row, scale, upload).
        const embd_scale = std.math.sqrt(@as(f32, @floatFromInt(d.n_embd)));
        const host = try self.allocator.alloc(f32, T * d.n_embd);
        defer self.allocator.free(host);
        for (0..T) |t| {
            const row = host[t * d.n_embd ..][0..d.n_embd];
            self.model.dequantEmbeddingRow(tokens[t], row);
            for (row) |*v| v.* *= embd_scale;
        }
        buffer.upload(ctx, &b.hidden, std.mem.sliceAsBytes(host));

        var L: u32 = 0;
        while (L < d.n_layers) : (L += 1) {
            try self.attentionLayerBatched(L, T, b);
            // Cycle 10: every batched-prefill block now COMMITS ASYNC on the single
            // shared CUstream (attention/shared/ffn no longer commitAndWait per layer),
            // so the CPU never blocks between layers — the same ~0.4ms WSL2 sync round-
            // trips (and the boost-starvation their idle gaps cause) that the decode
            // async ring removes are gone from prefill too. The stream still serializes
            // the GPU in submission order, so cross-layer buffer reuse is byte-identical;
            // the per-token MoE FALLBACK path (non-`pre`) still commitAndWaits internally
            // around its host id readback. All stashed commands are freed by the single
            // waitPending() before the tail (ring depth = blocks/layer × n_layers ≈ 5×48
            // for gemma-26b MoE, well under the 1024 ring; submit() syncs if it ever fills).
            // FFN type is per LAYER (a MoE model may carry dense layers): exactly
            // mirror the per-token path's `n_experts>0 && ffn_gate_inp present` test.
            const layer_is_moe = moe and self.model.getLayer(L, "ffn_gate_inp.weight") != null;
            if (layer_is_moe) {
                // The gemma4-MoE FFN is batched in stages over all T tokens, each a
                // bit-identical twin of the per-token path: the Q8_0 shared expert
                // (cycle 6 → b.shared), the F32 router (cycle 7 → b.router_table), the
                // routed gate/up/down expert matvecs (cycle 8 → b.down_e), and — cycle
                // 9 — the accumulate + post_ffw_norm + residual combine + output scale
                // (`moeRoutedCombineBatched`, the last per-token launches). With all
                // four batched, the prefill MoE FFN has NO per-token loop on the GPU-
                // side async expert path: each stage reads the batched streams in place.
                try self.sharedExpertBatched(L, T, b);
                const wgu = self.layer(L, "ffn_gate_up_exps.weight");
                const wde = self.layer(L, "ffn_down_exps.weight");
                // The batched router + routed-expert matvecs + combine run only on the
                // GPU-side async expert path (Q4_K gate_up + Q5_1 down); the host-
                // readback fallback keeps its per-token router/experts/combine loop.
                const pre = dmmvIdx(wgu.info.type_) == 0 and dmmvIdx(wde.info.type_) == 5;
                if (pre) {
                    try self.routerBatched(L, T, b);
                    if (self.use_grouped) {
                        try self.moeRoutedExpertsGrouped(L, T, b);
                    } else {
                        try self.moeRoutedExpertsBatched(L, T, b);
                    }
                    try self.moeRoutedCombineBatched(L, T, b);
                } else {
                    // Fallback (non-Q4_K/Q5_1 experts): the router + routed matvecs are
                    // NOT batched, so loop the per-token combine, aliasing self.hidden /
                    // self.shared_buf to this token's batched slices (b.shared holds the
                    // already-batched shared expert; moeRoutedCombine computes this
                    // token's router + experts + combine into the single-token scratch).
                    const saved_hidden = self.hidden;
                    const saved_shared = self.shared_buf;
                    var t: u32 = 0;
                    while (t < T) : (t += 1) {
                        self.hidden = try buffer.aliasBuffer(&b.hidden, t * d.n_embd * f4, d.n_embd * f4);
                        self.shared_buf = try buffer.aliasBuffer(&b.shared, t * d.n_embd * f4, d.n_embd * f4);
                        try self.moeRoutedCombine(L, false, false);
                        try self.layerOutScale(L); // MoE's final write is scale_accumulate → standalone scale
                        buffer.freeBuffer(&self.hidden);
                        buffer.freeBuffer(&self.shared_buf);
                    }
                    self.hidden = saved_hidden;
                    self.shared_buf = saved_shared;
                }
            } else {
                try self.ffnBlockBatched(L, T, b);
                // dense layer_output_scale is folded into the post-ffn norm+residual.
            }
        }
        // Drain every layer's stashed async commands (attention/shared/ffn/MoE) before
        // the (synchronous) tail — the dense path now uses the ring too (cycle 10).
        self.waitPending();

        // TAIL on the last token only: rms_norm → LM head → argmax. Reuse the
        // single-token decode scratch (norm_buf/logits_buf/argmax_buf) on the
        // last token's slice of the batched hidden stream.
        const last = T - 1;
        const out_norm = self.model.get("output_norm.weight") orelse return error.MissingTensor;
        const lm_head = self.model.get("output.weight") orelse self.model.get("token_embd.weight") orelse return error.MissingTensor;
        var hid_last = try buffer.aliasBuffer(&b.hidden, last * d.n_embd * f4, d.n_embd * f4);
        defer buffer.freeBuffer(&hid_last);

        var cmd = try command.beginCommand(ctx);
        const rms = RmsPush{ .N = d.n_embd, .eps = d.rms_eps };
        cmd.dispatch(&self.pipes.rms_norm, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ &hid_last, &out_norm.gpu_buffer, &self.norm_buf }, &rms, @sizeOf(RmsPush), 0);
        const lm = DmmvPush{ .M = d.vocab, .K = d.n_embd };
        const lm_idx = dmmvIdx(lm_head.info.type_);
        if (lm_idx < 4) {
            cmd.dispatch(&self.pipes.dmmv_fast[lm_idx], .{ d.vocab, 1, 1 }, .{ 64, 1, 1 }, &.{ &lm_head.gpu_buffer, &self.norm_buf, &self.logits_buf }, &lm, @sizeOf(DmmvPush), 0);
        } else {
            cmd.dispatch(&self.pipes.dmmv[lm_idx], .{ d.vocab, 1, 1 }, .{ 256, 1, 1 }, &.{ &lm_head.gpu_buffer, &self.norm_buf, &self.logits_buf }, &lm, @sizeOf(DmmvPush), 0);
        }
        const am = ArgmaxPush{ .N = d.vocab };
        cmd.dispatch(&self.pipes.argmax, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.logits_buf, &self.argmax_buf }, &am, @sizeOf(ArgmaxPush), 0);
        cmd.commitAndWait();

        var tok: u32 = 0;
        buffer.download(ctx, &self.argmax_buf, std.mem.asBytes(&tok));
        return tok;
    }

    /// Batched attention block: pre-norm + Q/K/V projections + O projection via
    /// GEMM over all T tokens; per-head Q·K·V normalize, RoPE, KV write and the
    /// causal softmax LOOPED per token (reusing the single-token kernels through
    /// token-major aliases). Mirrors `attentionLayer` op-for-op so the output is
    /// the same residual stream, batched. One stream-ordered command per layer.
    fn attentionLayerBatched(self: *ForwardGemma, L: u32, T: u32, b: *BatchScratch) !void {
        const d = self.d;
        const ctx = self.ctx;
        const g = self.geom[L];
        const f4 = @sizeOf(f32);
        const wan = self.layer(L, "attn_norm.weight");
        const wq = self.layer(L, "attn_q.weight");
        const wk = self.layer(L, "attn_k.weight");
        const wv_opt = self.model.getLayer(L, "attn_v.weight");
        const wqn = self.layer(L, "attn_q_norm.weight");
        const wkn = self.layer(L, "attn_k_norm.weight");
        const wo = self.layer(L, "attn_output.weight");
        const wpan = self.layer(L, "post_attention_norm.weight");

        var cmd = try command.beginCommand(ctx);
        const rms = RmsPush{ .N = d.n_embd, .eps = d.rms_eps };
        // Batched pre-attention norm: one block per token.
        // Batched Q/K projections; V from Wv (SWA layers) else the raw K projection.
        // Cycle 19: Q/K/V all read b.norm — Q recasts it to fp16 (act_f16) on the TC
        // path; K/V reuse that recast (a_preconv) when ZINC_BATCHED_TC_SHAREA is set.
        // Cycle 21 (normf16): emit the pre-attn norm as fp16 DIRECTLY into act_f16
        // (rms_norm_f16) so Q/K/V (all Q4_K) all skip their recast (a_preconv=true).
        if (self.use_tc and self.use_tc_normf16) {
            cmd.dispatch(&self.pipes.rms_norm_f16, .{ T, 1, 1 }, .{ 256, 1, 1 }, &.{ &b.hidden, &wan.gpu_buffer, &b.act_f16 }, &rms, @sizeOf(RmsPush), 0);
            self.gemmDispatchA(&cmd, wq, &b.norm, &b.q, g.q_dim, d.n_embd, T, true);
            self.gemmDispatchA(&cmd, wk, &b.norm, &b.k, g.kv_dim, d.n_embd, T, true);
            if (wv_opt) |wv| self.gemmDispatchA(&cmd, wv, &b.norm, &b.v, g.kv_dim, d.n_embd, T, true);
        } else {
            cmd.dispatch(&self.pipes.rms_norm, .{ T, 1, 1 }, .{ 256, 1, 1 }, &.{ &b.hidden, &wan.gpu_buffer, &b.norm }, &rms, @sizeOf(RmsPush), 0);
            self.gemmDispatch(&cmd, wq, &b.norm, &b.q, g.q_dim, d.n_embd, T);
            self.gemmDispatchA(&cmd, wk, &b.norm, &b.k, g.kv_dim, d.n_embd, T, true);
            if (wv_opt) |wv| self.gemmDispatchA(&cmd, wv, &b.norm, &b.v, g.kv_dim, d.n_embd, T, true);
        }

        // Batched (grid.y = T) V normalize+KV-write and Q/K per-head norm+RoPE:
        // ONE launch each over all T tokens (token t at sequence position t),
        // replacing the per-token loop. Each (head,t) block does exactly the
        // single-token kernel's math (per-block reduction order unchanged), so
        // this is bit-identical to the per-token launches. No aliasing needed —
        // the kernels index the token-major scratch directly via t*stride.
        const inv_freq = if (g.is_swa) &self.inv_freq_swa else &self.inv_freq_full;
        const nr_sh = g.head_dim * f4;
        const v_base = if (wv_opt != null) &b.v else &b.k;
        // V per-head plain-normalize fused with the V KV-cache write.
        const kvw = RmsKvWriteBatchPush{ .head_dim = g.head_dim, .eps = d.rms_eps, .src_stride = g.kv_dim, .dst_stride = g.kv_dim };
        cmd.dispatch(&self.pipes.rms_norm_kvwrite_batched, .{ g.n_kv_head, T, 1 }, .{ 256, 1, 1 }, &.{ v_base, &self.kv_v[L] }, &kvw, @sizeOf(RmsKvWriteBatchPush), 0);
        // Q/K per-head rms_norm fused with NEOX RoPE; K writes into kv_k.
        const nr_q = RmsRopeBatchPush{ .head_dim = g.head_dim, .eps = d.rms_eps, .rope_dim = g.rope_dim, .base_position = 0, .src_stride = g.q_dim, .dst_stride = g.q_dim };
        const nr_k = RmsRopeBatchPush{ .head_dim = g.head_dim, .eps = d.rms_eps, .rope_dim = g.rope_dim, .base_position = 0, .src_stride = g.kv_dim, .dst_stride = g.kv_dim };
        cmd.dispatch(&self.pipes.rms_norm_rope_batched, .{ d.n_head, T, 1 }, .{ 256, 1, 1 }, &.{ &b.q, &wqn.gpu_buffer, inv_freq, &b.q }, &nr_q, @sizeOf(RmsRopeBatchPush), nr_sh);
        cmd.dispatch(&self.pipes.rms_norm_rope_batched, .{ g.n_kv_head, T, 1 }, .{ 256, 1, 1 }, &.{ &b.k, &wkn.gpu_buffer, inv_freq, &self.kv_k[L] }, &nr_k, @sizeOf(RmsRopeBatchPush), nr_sh);

        // Single batched causal (sliding-window on SWA) softmax attention over all
        // T queries: grid=(n_head, T). Reads RoPE'd Q from b.q (token-major) and the
        // prompt region [0..T) of the KV cache; writes b.attn_out (token-major).
        // Replaces the T per-token gemma_attention launches; bit-identical math.
        const window: u32 = if (g.is_swa) d.sliding_window else 0;
        const attn = GemmaAttnBatchPush{
            .head_dim = g.head_dim,
            .n_heads = d.n_head,
            .n_kv_heads = g.n_kv_head,
            .T = T,
            .scale_bits = @bitCast(@as(f32, 1.0)),
            .window = window,
        };
        cmd.dispatch(&self.pipes.gemma_attention_batched, .{ d.n_head, T, 1 }, .{ 256, 1, 1 }, &.{ &b.q, &self.kv_k[L], &self.kv_v[L], &b.attn_out }, &attn, @sizeOf(GemmaAttnBatchPush), T * 4);

        // Batched O projection then the fused post-attention norm + residual add.
        self.gemmDispatch(&cmd, wo, &b.attn_out, &b.o, d.n_embd, g.q_dim, T);
        cmd.dispatch(&self.pipes.rms_norm_residual, .{ T, 1, 1 }, .{ 256, 1, 1 }, &.{ &b.o, &wpan.gpu_buffer, &b.hidden }, &rms, @sizeOf(RmsPush), 0);
        // Async on the shared stream (cycle 10): the FFN block + next layer chain after
        // this in submission order; the single tail waitPending() frees it. No host sync.
        self.submit(cmd);
    }

    /// Batched dense GeGLU FFN block: pre-norm + gate/up/down projections via
    /// GEMM over all T tokens, element-wise GeGLU across [T, n_ff], and the fused
    /// post-ffn norm + residual (folding the per-layer output scale when present).
    /// Mirrors `ffnBlock`, batched.
    fn ffnBlockBatched(self: *ForwardGemma, L: u32, T: u32, b: *BatchScratch) !void {
        const d = self.d;
        const ctx = self.ctx;
        const wfn = self.layer(L, "ffn_norm.weight");
        const wgate = self.layer(L, "ffn_gate.weight");
        const wup = self.layer(L, "ffn_up.weight");
        const wdown = self.layer(L, "ffn_down.weight");
        const wpfn = self.layer(L, "post_ffw_norm.weight");
        const wlos = self.model.getLayer(L, "layer_output_scale.weight");

        var cmd = try command.beginCommand(ctx);
        const rms = RmsPush{ .N = d.n_embd, .eps = d.rms_eps };
        // GeGLU is element-wise over the whole [T, n_ff] tile.
        const sg = SwigluPush{ .N = T * d.n_ff };
        // Cycle 21 (normf16): emit the pre-FFN norm as fp16 DIRECTLY into act_f16 so
        // gate/up (Q4_K) skip their recast; and (when ffn_down takes the act_f16 TC
        // path — Q4_K always, Q6_K only when use_tc_q6) emit GeGLU as fp16 so down
        // skips its recast too. Byte-identical to the per-GEMM-recast TC path.
        const ffn_normf16 = self.use_tc and self.use_tc_normf16;
        const down_act_f16 = ffn_normf16 and switch (wdown.info.type_) {
            .q4_k => true,
            .q6_k => self.use_tc_q6,
            else => false,
        };
        if (ffn_normf16) {
            cmd.dispatch(&self.pipes.rms_norm_f16, .{ T, 1, 1 }, .{ 256, 1, 1 }, &.{ &b.hidden, &wfn.gpu_buffer, &b.act_f16 }, &rms, @sizeOf(RmsPush), 0);
            self.gemmDispatchA(&cmd, wgate, &b.ffn_norm, &b.gate, d.n_ff, d.n_embd, T, true);
            self.gemmDispatchA(&cmd, wup, &b.ffn_norm, &b.up, d.n_ff, d.n_embd, T, true);
        } else {
            cmd.dispatch(&self.pipes.rms_norm, .{ T, 1, 1 }, .{ 256, 1, 1 }, &.{ &b.hidden, &wfn.gpu_buffer, &b.ffn_norm }, &rms, @sizeOf(RmsPush), 0);
            // Cycle 19: gate+up both read b.ffn_norm — up reuses gate's fp16 recast (shared-A).
            self.gemmDispatch(&cmd, wgate, &b.ffn_norm, &b.gate, d.n_ff, d.n_embd, T);
            self.gemmDispatchA(&cmd, wup, &b.ffn_norm, &b.up, d.n_ff, d.n_embd, T, true);
        }
        if (down_act_f16) {
            cmd.dispatch(&self.pipes.geglu_f16, .{ ceilDiv(T * d.n_ff, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &b.gate, &b.up, &b.act_f16 }, &sg, @sizeOf(SwigluPush), 0);
            self.gemmDispatchA(&cmd, wdown, &b.geglu, &b.down, d.n_embd, d.n_ff, T, true);
        } else {
            cmd.dispatch(&self.pipes.geglu, .{ ceilDiv(T * d.n_ff, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &b.gate, &b.up, &b.geglu }, &sg, @sizeOf(SwigluPush), 0);
            self.gemmDispatch(&cmd, wdown, &b.geglu, &b.down, d.n_embd, d.n_ff, T);
        }
        if (wlos) |ws| {
            cmd.dispatch(&self.pipes.rms_norm_residual_scale, .{ T, 1, 1 }, .{ 256, 1, 1 }, &.{ &b.down, &wpfn.gpu_buffer, &b.hidden, &ws.gpu_buffer }, &rms, @sizeOf(RmsPush), 0);
        } else {
            cmd.dispatch(&self.pipes.rms_norm_residual, .{ T, 1, 1 }, .{ 256, 1, 1 }, &.{ &b.down, &wpfn.gpu_buffer, &b.hidden }, &rms, @sizeOf(RmsPush), 0);
        }
        // Async on the shared stream (cycle 10): chains before the next layer's attention
        // in submission order; freed by the single tail waitPending(). No per-layer sync.
        self.submit(cmd);
    }

    /// Batched prefill GEMM dispatch: Y[T,M] = A[T,K]·W[M,K]^T. Q4_K/Q5_K/Q6_K
    /// weights take the register-blocked gemm_*_tiled_v2 tile; any other quant
    /// (q8_0/f32) falls back to a per-token dmmv loop (correctness-first). Buffers
    /// are token-major [T,K] (x) and [T,M] (y), contiguous (a_offset/x/y == 0).
    fn gemmDispatch(self: *ForwardGemma, cmd: *command.CudaCommand, w: *const LoadedTensor, x: *const CudaBuffer, y: *const CudaBuffer, M: u32, K: u32, T: u32) void {
        self.gemmDispatchA(cmd, w, x, y, M, K, T, false);
    }

    /// As `gemmDispatch`, but `a_preconv=true` (cycle 19, only honored when
    /// `use_tc_sharea`) signals that act_f16 ALREADY holds this GEMM's input x
    /// downcast to fp16 — set by a PRECEDING gemmDispatch on the same command that
    /// read the SAME x (e.g. attn K/V after Q; FFN up after gate). The redundant
    /// f32_to_f16 recast is then skipped and the TC kernel reads the existing
    /// act_f16. Byte-identical: x is unchanged and nothing writes act_f16 between
    /// the group's GEMMs, so the staged half bits are bit-for-bit Q's/gate's.
    fn gemmDispatchA(self: *ForwardGemma, cmd: *command.CudaCommand, w: *const LoadedTensor, x: *const CudaBuffer, y: *const CudaBuffer, M: u32, K: u32, T: u32, a_preconv: bool) void {
        const gi: ?usize = switch (w.info.type_) {
            .q4_k => 0,
            .q5_k => 1,
            .q6_k => 2,
            .q8_0 => 3,
            else => null,
        };
        if (gi) |idx| {
            const push = GemmPush{ .M = M, .K = K, .T = T };
            // Effort 26 cycle 9: dense Q4_K GEMM (idx 0) on cuBLAS fp16 tensor
            // cores — ~6× gemm_q4k_tc in isolation. (1) dequant the Q4_K weight
            // [M,K] → fp16 (b.w_f16); (2) downcast the f32 activation [T,K] → fp16
            // (b.act_f16) once; (3) cublasGemmEx fp16→fp32. All three run on the
            // ctx stream (dequant/convert via cmd, cuBLAS via cublasSetStream) so
            // they are correctly ordered. fp16-rounded → token-correctness gate
            // (same as the TC path), NOT byte-identical. Gated on T >= cublas_min_t:
            // the full-weight dequant→fp16 round-trip is a fixed per-GEMM cost, so
            // cuBLAS only wins once it amortizes over enough tokens (+76% @T=512,
            // +15% @T=128, break-even @T=64) — below that, fall through to gemm_q4k_tc.
            if (idx == 0 and self.use_cublas and T >= self.cublas_min_t) {
                const b = &self.batch.?;
                const dq = DequantQ4KPush{ .M = M, .K = K };
                cmd.dispatch(&self.pipes.dequant_q4k_to_f16, .{ ceilDiv(M * K, 256), 1, 1 }, .{ 256, 1, 1 }, &.{ &w.gpu_buffer, &b.w_f16 }, &dq, @sizeOf(DequantQ4KPush), 0);
                const a16 = &b.act_f16;
                if (!(a_preconv and (self.use_tc_sharea or self.use_tc_normf16))) {
                    const cvt = F32ToF16Push{ .N = T * K };
                    cmd.dispatch(&self.pipes.f32_to_f16, .{ ceilDiv(T * K, 256), 1, 1 }, .{ 256, 1, 1 }, &.{ x, a16 }, &cvt, @sizeOf(F32ToF16Push), 0);
                }
                shim.cuda_cublas_hgemm(self.ctx, @intCast(M), @intCast(T), @intCast(K), b.w_f16.handle, a16.handle, y.handle);
                return;
            }
            // Cycle 11: when ZINC_BATCHED_TC is set, Q4_K GEMMs (idx 0 — the bulk of
            // the dense FLOPs: gate/up + attn Q/K/V/O) run on the fp16 tensor cores.
            if (self.use_tc and idx == 0 and self.use_tc_plain) {
                // Cycle 11 plain TC (A/B baseline): the kernel re-reads f32 A from
                // global once per output M-block. Same GemmPush/grid/block.
                cmd.dispatch(&self.pipes.gemm_q4k_tc, .{ ceilDiv(M, 64), ceilDiv(T, 64), 1 }, .{ 256, 1, 1 }, &.{ &w.gpu_buffer, x, y }, &push, @sizeOf(GemmPush), 0);
                return;
            }
            if (self.use_tc and idx == 0) {
                // Cycle 12: pre-convert the f32 activation [T,K] to fp16 ONCE
                // (f32_to_f16) so the TC GEMM reads half-width A — the TC kernel
                // otherwise re-reads f32 A once per output M-block, and that f32
                // activation traffic (~7× the Q4_K weight traffic for a 64×64 tile)
                // is what makes the dense GEMM memory-bound. The downcast uses the
                // SAME __float2half the TC kernel applied in shared → byte-for-byte
                // identical output, just half the dominant A read traffic.
                const a16 = &self.batch.?.act_f16;
                // Cycle 19: skip the recast when a preceding same-x GEMM already
                // filled act_f16 (shared-A) — byte-identical, one fewer launch+read.
                // Cycle 21: also skip when the norm/GeGLU producer already emitted the
                // fp16 activation into act_f16 (normf16) — the recast is fully gone.
                if (!(a_preconv and (self.use_tc_sharea or self.use_tc_normf16))) {
                    const cvt = F32ToF16Push{ .N = T * K };
                    cmd.dispatch(&self.pipes.f32_to_f16, .{ ceilDiv(T * K, 256), 1, 1 }, .{ 256, 1, 1 }, &.{ x, a16 }, &cvt, @sizeOf(F32ToF16Push), 0);
                }
                // Cycle 15: DEFAULT to the 8 KB-shared lowsmem kernel (same 64×64
                // grid/block as m64 but ~3x occupancy → +11.6%, byte-identical). The
                // 24 KB m64 kernel (gemm_q4k_tc_f16a, cycle 12 default) is the A/B
                // kill-switch via ZINC_BATCHED_TC_M64. The wider 128×64 M-tile kernel
                // (gemm_q4k_tc_f16a_m128) halves the dominant f16-A re-read but its
                // 44 KB shared caps occupancy at 1 block/SM → -11.8%; kept opt-in via
                // ZINC_BATCHED_TC_M128. Both byte-identical to the lowsmem default.
                if (self.use_tc_m128_lowsmem) {
                    // Cycle 17: wider 128×64 M-tile (grid.x = M/128 → halved f16-A read)
                    // WITH the low-shared two-phase Cs (12 KB → ~6 blocks/SM, not m128's
                    // 1 block/SM). Byte-identical to the lowsmem default.
                    cmd.dispatch(&self.pipes.gemm_q4k_tc_f16a_m128_lowsmem, .{ ceilDiv(M, 128), ceilDiv(T, 64), 1 }, .{ 256, 1, 1 }, &.{ &w.gpu_buffer, a16, y }, &push, @sizeOf(GemmPush), 0);
                } else if (self.use_tc_m128) {
                    cmd.dispatch(&self.pipes.gemm_q4k_tc_f16a_m128, .{ ceilDiv(M, 128), ceilDiv(T, 64), 1 }, .{ 256, 1, 1 }, &.{ &w.gpu_buffer, a16, y }, &push, @sizeOf(GemmPush), 0);
                } else if (self.use_tc_m64) {
                    cmd.dispatch(&self.pipes.gemm_q4k_tc_f16a, .{ ceilDiv(M, 64), ceilDiv(T, 64), 1 }, .{ 256, 1, 1 }, &.{ &w.gpu_buffer, a16, y }, &push, @sizeOf(GemmPush), 0);
                } else {
                    cmd.dispatch(&self.pipes.gemm_q4k_tc_f16a_lowsmem, .{ ceilDiv(M, 64), ceilDiv(T, 64), 1 }, .{ 256, 1, 1 }, &.{ &w.gpu_buffer, a16, y }, &push, @sizeOf(GemmPush), 0);
                }
                return;
            }
            if (self.use_tc and idx == 2 and self.use_tc_q6) {
                // Cycle 13: Q6_K weights (dense gemma-31b's ffn_down etc.) on the
                // fp16 tensor cores, same pre-converted-fp16-A pattern as Q4_K above
                // (f32_to_f16 once → half-width A read). gemm_q6k_tc_f16a mirrors the
                // f32 gemm_q6k_tiled_v2 dequant; fp16 rounding → token-correctness gate.
                const a16 = &self.batch.?.act_f16;
                if (!(a_preconv and (self.use_tc_sharea or self.use_tc_normf16))) { // cycle 19 shared-A + cycle 21 normf16 (see Q4_K branch)
                    const cvt = F32ToF16Push{ .N = T * K };
                    cmd.dispatch(&self.pipes.f32_to_f16, .{ ceilDiv(T * K, 256), 1, 1 }, .{ 256, 1, 1 }, &.{ x, a16 }, &cvt, @sizeOf(F32ToF16Push), 0);
                }
                // Cycle 16: default to the proven 24 KB m64 Q6_K kernel (gemm_q6k_tc_f16a,
                // cycle 13); ZINC_BATCHED_TC_Q6_LOWSMEM opts into the byte-identical 8 KB-shared
                // lowsmem kernel (perf-neutral here — Q6_K is ~1/7 of the dense GEMM, below the
                // boost floor — so kept opt-in rather than promoted to default).
                if (self.use_tc_q6_lowsmem) {
                    cmd.dispatch(&self.pipes.gemm_q6k_tc_f16a_lowsmem, .{ ceilDiv(M, 64), ceilDiv(T, 64), 1 }, .{ 256, 1, 1 }, &.{ &w.gpu_buffer, a16, y }, &push, @sizeOf(GemmPush), 0);
                } else {
                    cmd.dispatch(&self.pipes.gemm_q6k_tc_f16a, .{ ceilDiv(M, 64), ceilDiv(T, 64), 1 }, .{ 256, 1, 1 }, &.{ &w.gpu_buffer, a16, y }, &push, @sizeOf(GemmPush), 0);
                }
                return;
            }
            // Same GemmPush / grid / block; gemm uses static shared only.
            cmd.dispatch(&self.pipes.gemm[idx], .{ ceilDiv(M, 64), ceilDiv(T, 64), 1 }, .{ 256, 1, 1 }, &.{ &w.gpu_buffer, x, y }, &push, @sizeOf(GemmPush), 0);
            return;
        }
        // Fallback: loop the per-token matvec over the token-major buffers.
        const didx = dmmvIdx(w.info.type_);
        var t: u32 = 0;
        while (t < T) : (t += 1) {
            const push = DmmvPush{ .M = M, .K = K, .x_offset = t * K * 4, .y_offset = t * M * 4 };
            if (didx < 4) {
                cmd.dispatch(&self.pipes.dmmv_fast[didx], .{ M, 1, 1 }, .{ 64, 1, 1 }, &.{ &w.gpu_buffer, x, y }, &push, @sizeOf(DmmvPush), 0);
            } else {
                cmd.dispatch(&self.pipes.dmmv[didx], .{ M, 1, 1 }, .{ 256, 1, 1 }, &.{ &w.gpu_buffer, x, y }, &push, @sizeOf(DmmvPush), 0);
            }
        }
    }

    /// Allocate (or reuse) the token-major batched scratch for T tokens.
    fn ensureBatch(self: *ForwardGemma, T: u32) !*BatchScratch {
        if (self.batch) |*bb| {
            if (bb.t_cap >= T) return bb;
            self.freeBatch();
        }
        const d = self.d;
        const ctx = self.ctx;
        const f4 = @sizeOf(f32);
        const ff = d.ff_buf_max;
        self.batch = BatchScratch{
            .t_cap = T,
            .hidden = try buffer.createBuffer(ctx, T * d.n_embd * f4),
            .norm = try buffer.createBuffer(ctx, T * d.n_embd * f4),
            .q = try buffer.createBuffer(ctx, T * d.q_dim_max * f4),
            .k = try buffer.createBuffer(ctx, T * d.kv_dim_max * f4),
            .v = try buffer.createBuffer(ctx, T * d.kv_dim_max * f4),
            .attn_out = try buffer.createBuffer(ctx, T * d.q_dim_max * f4),
            .o = try buffer.createBuffer(ctx, T * d.n_embd * f4),
            .ffn_norm = try buffer.createBuffer(ctx, T * d.n_embd * f4),
            .gate = try buffer.createBuffer(ctx, T * ff * f4),
            .up = try buffer.createBuffer(ctx, T * ff * f4),
            .geglu = try buffer.createBuffer(ctx, T * ff * f4),
            .down = try buffer.createBuffer(ctx, T * d.n_embd * f4),
            .shared = try buffer.createBuffer(ctx, T * d.n_embd * f4),
            .router_in = try buffer.createBuffer(ctx, T * d.n_embd * f4),
            .router_logits = try buffer.createBuffer(ctx, T * @max(@as(u32, 1), d.n_experts) * f4),
            .router_table = try buffer.createBuffer(ctx, T * @max(@as(u32, 1), 2 * d.n_experts_used) * @sizeOf(u32)),
            // Routed-expert FFN scratch: n_used routed experts × intermediate ef per token.
            .moe_norm_e = try buffer.createBuffer(ctx, T * d.n_embd * f4),
            .gate_e = try buffer.createBuffer(ctx, T * @max(@as(u32, 1), d.n_experts_used * d.n_ff) * f4),
            .up_e = try buffer.createBuffer(ctx, T * @max(@as(u32, 1), d.n_experts_used * d.n_ff) * f4),
            .geglu_e = try buffer.createBuffer(ctx, T * @max(@as(u32, 1), d.n_experts_used * d.n_ff) * f4),
            .down_e = try buffer.createBuffer(ctx, T * @max(@as(u32, 1), d.n_experts_used * d.n_embd) * f4),
            .moe_out_e = try buffer.createBuffer(ctx, T * d.n_embd * f4),
            .expert_order = try buffer.createBuffer(ctx, T * @max(@as(u32, 1), d.n_experts_used) * @sizeOf(u32)),
            // fp16 activation scratch: T × largest-activation halves (2 bytes each).
            // TC Q4_K GEMMs read A with K ∈ {n_embd (gate/up,Q/K/V), q_dim (O)};
            // size to the max of those and ff for headroom.
            .act_f16 = try buffer.createBuffer(ctx, T * @max(ff, @max(d.q_dim_max, d.n_embd)) * @sizeOf(u16)),
            // Largest dense Q4_K weight: gate/up (ff·n_embd), O (n_embd·q_dim),
            // Q (q_dim·n_embd). max(M)·max(K) is a safe upper bound on M·K.
            .w_f16 = try buffer.createBuffer(ctx, @max(ff, @max(d.q_dim_max, d.n_embd)) * @max(d.n_embd, d.q_dim_max) * @sizeOf(u16)),
        };
        return &self.batch.?;
    }

    fn freeBatch(self: *ForwardGemma) void {
        if (self.batch) |*bb| {
            inline for (.{ &bb.hidden, &bb.norm, &bb.q, &bb.k, &bb.v, &bb.attn_out, &bb.o, &bb.ffn_norm, &bb.gate, &bb.up, &bb.geglu, &bb.down, &bb.shared, &bb.router_in, &bb.router_logits, &bb.router_table, &bb.moe_norm_e, &bb.gate_e, &bb.up_e, &bb.geglu_e, &bb.down_e, &bb.moe_out_e, &bb.expert_order, &bb.act_f16, &bb.w_f16 }) |buf| {
                buffer.freeBuffer(buf);
            }
            self.batch = null;
        }
    }

    // ---- per-block builders -------------------------------------------------

    fn attentionLayer(self: *ForwardGemma, L: u32, pos: u32) !void {
        const d = self.d;
        const ctx = self.ctx;
        const g = self.geom[L];
        const wan = self.layer(L, "attn_norm.weight");
        const wq = self.layer(L, "attn_q.weight");
        const wk = self.layer(L, "attn_k.weight");
        // gemma4 "alternative attention": full-attention layers omit attn_v and
        // reuse the raw K projection (pre-norm, pre-rope) as V.
        const wv_opt = self.model.getLayer(L, "attn_v.weight");
        const wqn = self.layer(L, "attn_q_norm.weight");
        const wkn = self.layer(L, "attn_k_norm.weight");
        const wo = self.layer(L, "attn_output.weight");
        const wpan = self.layer(L, "post_attention_norm.weight");

        // Dense gemma folds each block's INPUT norm into the PRECEDING block's
        // output norm+residual (see rms_norm_residual_norm). When folding, the
        // pre-attn norm (norm_buf) is produced by the previous layer's fused
        // post-ffn kernel — only layer 0 needs the standalone pre-attn norm.
        const fold = d.n_experts == 0;

        var cmd = try command.beginCommand(ctx);
        // pre-attention norm (gemma rms, +1 baked in)
        const rms = RmsPush{ .N = d.n_embd, .eps = d.rms_eps };
        if (!fold or L == 0) {
            cmd.dispatch(&self.pipes.rms_norm, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.hidden, &wan.gpu_buffer, &self.norm_buf }, &rms, @sizeOf(RmsPush), 0);
        }
        // Q, K projections; V from Wv if present, else the raw K projection. Q & K
        // share the pre-attention norm input — when both are Q4_K, fuse the two
        // matvecs into one launch (q_buf gets the Q rows, k_buf the K rows).
        if (dmmvIdx(wq.info.type_) == 0 and dmmvIdx(wk.info.type_) == 0) {
            self.dmmvDualQ4k(&cmd, wq, wk, &self.norm_buf, &self.q_buf, &self.k_buf, g.q_dim, g.kv_dim, d.n_embd);
        } else {
            self.dmmvDispatch(&cmd, wq, &self.norm_buf, &self.q_buf, g.q_dim, d.n_embd, 0, 0);
            self.dmmvDispatch(&cmd, wk, &self.norm_buf, &self.k_buf, g.kv_dim, d.n_embd, 0, 0);
        }
        const v_src: *const CudaBuffer = if (wv_opt) |wv| blk: {
            self.dmmvDispatch(&cmd, wv, &self.norm_buf, &self.v_buf, g.kv_dim, d.n_embd, 0, 0);
            break :blk &self.v_buf;
        } else &self.k_buf;
        // Per-head V/Q/K norm FUSED into ONE launch (was 3): V plain-normalize +
        // KV-write (rms_norm_kvwrite), Q norm+rope, K norm+rope (rms_norm_rope ×2).
        // Grid = n_head + 2*n_kv_head blocks: Q heads first (norm+rope → q_buf,
        // offset 0), then K heads (norm+rope → kv_k at pos*kv_dim), then V heads
        // (plain norm → kv_v at pos*kv_dim). Bit-identical per-branch arithmetic;
        // no cross-block hazard (K→kv_k, V→kv_v, Q in-place; nobody reads another
        // block's destination), so v_src aliasing k_buf on full-attention layers is
        // safe (k_buf is read-only here — K writes straight to its cache).
        const kv_off = pos * g.kv_dim;
        const inv_freq = if (g.is_swa) &self.inv_freq_swa else &self.inv_freq_full;
        const qkv = RmsRopeQkvPush{ .head_dim = g.head_dim, .eps = d.rms_eps, .rope_dim = g.rope_dim, .position = pos, .n_head = d.n_head, .n_kv_head = g.n_kv_head, .kv_offset = kv_off };
        const nr_sh = g.head_dim * @sizeOf(f32);
        cmd.dispatch(&self.pipes.rms_norm_rope_qkv, .{ d.n_head + 2 * g.n_kv_head, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.q_buf, &self.k_buf, v_src, &wqn.gpu_buffer, &wkn.gpu_buffer, inv_freq, &self.q_buf, &self.kv_k[L], &self.kv_v[L] }, &qkv, @sizeOf(RmsRopeQkvPush), nr_sh);
        // attention (scale=1.0, sliding window on SWA layers) → attn_out_buf
        const seq_len = pos + 1;
        const window: u32 = if (g.is_swa) d.sliding_window else 0;
        const attn = GemmaAttnPush{
            .head_dim = g.head_dim,
            .n_heads = d.n_head,
            .n_kv_heads = g.n_kv_head,
            .seq_len = seq_len,
            .scale_bits = @bitCast(@as(f32, 1.0)),
            .window = window,
        };
        cmd.dispatch(&self.pipes.gemma_attention, .{ d.n_head, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.q_buf, &self.kv_k[L], &self.kv_v[L], &self.attn_out_buf }, &attn, @sizeOf(GemmaAttnPush), seq_len * 4);
        // O projection → o_buf (NOT accumulated; post-norm happens first)
        self.dmmvDispatch(&cmd, wo, &self.attn_out_buf, &self.o_buf, d.n_embd, g.q_dim, 0, 0);
        // post-attention norm (gemma rms) on the attention output, fused with the
        // residual add into `hidden` (scale 1.0) — one launch, no o_buf round-trip.
        // When folding, the SAME launch also produces the pre-ffn norm
        // (ffn_norm_buf), so ffnBlock skips its standalone pre-ffn norm.
        if (fold) {
            const wfn = self.layer(L, "ffn_norm.weight");
            cmd.dispatch(&self.pipes.rms_norm_residual_norm, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.o_buf, &wpan.gpu_buffer, &self.hidden, &wfn.gpu_buffer, &self.ffn_norm_buf }, &rms, @sizeOf(RmsPush), 0);
        } else {
            cmd.dispatch(&self.pipes.rms_norm_residual, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.o_buf, &wpan.gpu_buffer, &self.hidden }, &rms, @sizeOf(RmsPush), 0);
        }
        self.submit(cmd);
    }

    fn ffnBlock(self: *ForwardGemma, L: u32) !void {
        const d = self.d;
        const ctx = self.ctx;
        const wfn = self.layer(L, "ffn_norm.weight");
        const wgate = self.layer(L, "ffn_gate.weight");
        const wup = self.layer(L, "ffn_up.weight");
        const wdown = self.layer(L, "ffn_down.weight");
        const wpfn = self.layer(L, "post_ffw_norm.weight");
        // gemma's per-layer output scale (optional). On the dense path it is the
        // LAST write to `hidden` in the layer, so fold it into the post-ffn
        // norm+residual instead of a standalone scalar_mul command (layerOutScale
        // self-skips dense layers). Absent → plain rms_norm_residual.
        const wlos = self.model.getLayer(L, "layer_output_scale.weight");

        // Dense gemma folds each block's INPUT norm into the PRECEDING block's
        // output norm+residual: the pre-ffn norm (ffn_norm_buf) was produced by
        // this layer's fused post-attn kernel, and this block's post-ffn kernel
        // produces the NEXT layer's pre-attn norm (norm_buf).
        const fold = d.n_experts == 0;

        var cmd = try command.beginCommand(ctx);
        const rms = RmsPush{ .N = d.n_embd, .eps = d.rms_eps };
        // pre-ffn norm (skipped when folded — ffn_norm_buf already filled)
        if (!fold) {
            cmd.dispatch(&self.pipes.rms_norm, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.hidden, &wfn.gpu_buffer, &self.ffn_norm_buf }, &rms, @sizeOf(RmsPush), 0);
        }
        // GeGLU FFN: gelu(gate) * up → down. gate & up share the pre-ffn norm
        // input — when both are Q4_K, fuse the two matvecs into one launch.
        if (dmmvIdx(wgate.info.type_) == 0 and dmmvIdx(wup.info.type_) == 0) {
            self.dmmvDualQ4k(&cmd, wgate, wup, &self.ffn_norm_buf, &self.gate_buf, &self.up_buf, d.n_ff, d.n_ff, d.n_embd);
        } else {
            self.dmmvDispatch(&cmd, wgate, &self.ffn_norm_buf, &self.gate_buf, d.n_ff, d.n_embd, 0, 0);
            self.dmmvDispatch(&cmd, wup, &self.ffn_norm_buf, &self.up_buf, d.n_ff, d.n_embd, 0, 0);
        }
        const sg = SwigluPush{ .N = d.n_ff };
        cmd.dispatch(&self.pipes.geglu, .{ ceilDiv(d.n_ff, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.gate_buf, &self.up_buf, &self.geglu_buf }, &sg, @sizeOf(SwigluPush), 0);
        self.dmmvDispatch(&cmd, wdown, &self.geglu_buf, &self.down_buf, d.n_embd, d.n_ff, 0, 0);
        // post-ffn norm (gemma rms) on the FFN output, fused with the residual add
        // into `hidden` (scale 1.0) — one launch, no down_buf round-trip. When the
        // per-layer output scale is present it is folded in here too (one launch).
        // When folding and not the last layer, the SAME launch also produces the
        // NEXT layer's pre-attn norm (norm_buf), so attentionLayer(L+1) skips it.
        const fold_next = fold and (L + 1 < d.n_layers);
        if (fold_next) {
            const wan_next = self.layer(L + 1, "attn_norm.weight");
            if (wlos) |ws| {
                cmd.dispatch(&self.pipes.rms_norm_residual_scale_norm, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.down_buf, &wpfn.gpu_buffer, &self.hidden, &ws.gpu_buffer, &wan_next.gpu_buffer, &self.norm_buf }, &rms, @sizeOf(RmsPush), 0);
            } else {
                cmd.dispatch(&self.pipes.rms_norm_residual_norm, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.down_buf, &wpfn.gpu_buffer, &self.hidden, &wan_next.gpu_buffer, &self.norm_buf }, &rms, @sizeOf(RmsPush), 0);
            }
        } else if (wlos) |ws| {
            cmd.dispatch(&self.pipes.rms_norm_residual_scale, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.down_buf, &wpfn.gpu_buffer, &self.hidden, &ws.gpu_buffer }, &rms, @sizeOf(RmsPush), 0);
        } else {
            cmd.dispatch(&self.pipes.rms_norm_residual, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.down_buf, &wpfn.gpu_buffer, &self.hidden }, &rms, @sizeOf(RmsPush), 0);
        }
        self.submit(cmd);
    }

    /// MoE FFN block (gemma4-26b-a4b). On entry `hidden` holds attn_out, the
    /// shared input to the dense shared expert, the routed experts, AND the
    /// router; it stays untouched until the final residual add. Mirrors the
    /// llama.cpp gemma4.cpp build graph:
    ///   shared = post_ffw_norm_1( geglu_ffn( ffn_norm(attn_out) ) )
    ///   logits = ffn_gate_inp · ( rms(attn_out)/sqrt(n_embd) * gate_inp_s )
    ///   moe    = post_ffw_norm_2( Σ_j w_j·downᵉ( geglu( gate_upᵉ( pre_ffw_norm_2(attn_out) ) ) ) )
    ///   cur    = post_ffw_norm( shared + moe );  hidden += cur
    /// The per-expert down scale (ffn_down_exps.scale) is folded into the router
    /// weights on the host before the weighted combine.
    fn moeFfnBlock(self: *ForwardGemma, L: u32) !void {
        const d = self.d;
        const ctx = self.ctx;
        const n_used = d.n_experts_used;
        const ef = d.n_ff; // routed-expert intermediate (704)
        const sf = d.shexp_ff; // shared-expert intermediate (2112)

        const wfn = self.layer(L, "ffn_norm.weight");
        const wgate = self.layer(L, "ffn_gate.weight");
        const wup = self.layer(L, "ffn_up.weight");
        const wdown = self.layer(L, "ffn_down.weight");
        const wpn1 = self.layer(L, "post_ffw_norm_1.weight");
        const wpre2 = self.layer(L, "pre_ffw_norm_2.weight");
        const wpn2 = self.layer(L, "post_ffw_norm_2.weight");
        const wpost = self.layer(L, "post_ffw_norm.weight");
        const wrouter = self.layer(L, "ffn_gate_inp.weight"); // [n_embd, n_experts] F32
        const wrscale = self.layer(L, "ffn_gate_inp.scale"); // [n_embd] F32
        const wgu = self.layer(L, "ffn_gate_up_exps.weight"); // [n_embd, 2*ef, n_experts]
        const wde = self.layer(L, "ffn_down_exps.weight"); // [ef, n_embd, n_experts]

        const rms = RmsPush{ .N = d.n_embd, .eps = d.rms_eps };

        // Batched path (fused gate_up Q4_K + down Q5_1): one launch over all experts,
        // ids read GPU-side, the down scale folded GPU-side in the weighted combine —
        // so the whole block runs async with NO host readback. Other expert quants
        // (e.g. a Q8_0 down layer) take the per-slot fallback, which reads ids back.
        const batched = dmmvIdx(wgu.info.type_) == 0 and dmmvIdx(wde.info.type_) == 5;

        // --- shared expert → shared_buf -------------------------------------
        {
            var cmd = try command.beginCommand(ctx);
            cmd.dispatch(&self.pipes.rms_norm, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.hidden, &wfn.gpu_buffer, &self.ffn_norm_buf }, &rms, @sizeOf(RmsPush), 0);
            self.dmmvDispatch(&cmd, wgate, &self.ffn_norm_buf, &self.gate_buf, sf, d.n_embd, 0, 0);
            self.dmmvDispatch(&cmd, wup, &self.ffn_norm_buf, &self.up_buf, sf, d.n_embd, 0, 0);
            const sg = SwigluPush{ .N = sf };
            cmd.dispatch(&self.pipes.geglu, .{ ceilDiv(sf, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.gate_buf, &self.up_buf, &self.geglu_buf }, &sg, @sizeOf(SwigluPush), 0);
            self.dmmvDispatch(&cmd, wdown, &self.geglu_buf, &self.shared_buf, d.n_embd, sf, 0, 0);
            cmd.dispatch(&self.pipes.rms_norm, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.shared_buf, &wpn1.gpu_buffer, &self.shared_buf }, &rms, @sizeOf(RmsPush), 0);
            self.submit(cmd);
        }

        // --- router logits + top-k softmax (computed from attn_out) ----------
        {
            var cmd = try command.beginCommand(ctx);
            cmd.dispatch(&self.pipes.rms_norm_noweight, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.hidden, &self.norm_buf }, &rms, @sizeOf(RmsPush), 0);
            const mv = MulVecPush{ .N = d.n_embd, .scale = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(d.n_embd))) };
            cmd.dispatch(&self.pipes.mul_vec_scaled, .{ ceilDiv(d.n_embd, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.norm_buf, &wrscale.gpu_buffer }, &mv, @sizeOf(MulVecPush), 0);
            self.dmmvDispatch(&cmd, wrouter, &self.norm_buf, &self.router_logits_buf, d.n_experts, d.n_embd, 0, 0);
            const tk = TopkPush{ .n_experts = d.n_experts, .k = n_used };
            cmd.dispatch(&self.pipes.softmax_topk, .{ 1, 1, 1 }, .{ 64, 1, 1 }, &.{ &self.router_logits_buf, &self.router_out_buf }, &tk, @sizeOf(TopkPush), 0);
            if (batched) {
                self.submit(cmd); // async: experts read ids GPU-side, scale folded GPU-side
            } else {
                cmd.commitAndWait(); // sync: the fallback host-gathers ids + folds the scale next
                self.drainPending();
            }
        }

        // Fallback only: download ids+weights and fold the per-expert down scale into
        // the weights host-side. The batched path folds it GPU-side (moe_weighted_acc_scaled).
        if (!batched) {
            buffer.download(ctx, &self.router_out_buf, std.mem.sliceAsBytes(self.host_router[0 .. 2 * n_used]));
            const scales = self.down_scales[L * d.n_experts ..][0..d.n_experts];
            var j: u32 = 0;
            while (j < n_used) : (j += 1) {
                const id = self.host_router[j];
                const w: f32 = @bitCast(self.host_router[n_used + j]);
                self.host_router[n_used + j] = @bitCast(w * scales[id]);
            }
            buffer.upload(ctx, &self.router_out_buf, std.mem.sliceAsBytes(self.host_router[0 .. 2 * n_used]));
        }

        // --- routed experts → moe_out_buf -----------------------------------
        {
            // Per-expert byte strides into the fused gate_up / stacked down.
            const gu_half = expertSliceBytes(wgu.info.type_, ef, d.n_embd); // ef rows
            const gu_full = gu_half * 2; // 2*ef rows per expert
            const down_slice = expertSliceBytes(wde.info.type_, d.n_embd, ef);

            var cmd = try command.beginCommand(ctx);
            cmd.dispatch(&self.pipes.rms_norm, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.hidden, &wpre2.gpu_buffer, &self.moe_norm_buf }, &rms, @sizeOf(RmsPush), 0);
            // Batched path (gate_up Q4_K + down Q5_1): one launch over all
            // experts, ids read GPU-side from router_out_buf, slot-major output.
            // The fused gate_up reuses dmmv_q4k_experts with base=gu_half for the
            // up half. Falls back to the per-slot loop for other expert quants.
            if (batched) {
                const nrows = n_used * ef;
                const pg = ExpertsPush{ .M = ef, .K = d.n_embd, .slice = gu_full, .x_stride = 0, .n_used = n_used, .base = 0 };
                cmd.dispatch(&self.pipes.dmmv_q4k_experts, .{ nrows, 1, 1 }, .{ 64, 1, 1 }, &.{ &wgu.gpu_buffer, &self.moe_norm_buf, &self.gate_buf, &self.router_out_buf }, &pg, @sizeOf(ExpertsPush), 0);
                const pu = ExpertsPush{ .M = ef, .K = d.n_embd, .slice = gu_full, .x_stride = 0, .n_used = n_used, .base = gu_half };
                cmd.dispatch(&self.pipes.dmmv_q4k_experts, .{ nrows, 1, 1 }, .{ 64, 1, 1 }, &.{ &wgu.gpu_buffer, &self.moe_norm_buf, &self.up_buf, &self.router_out_buf }, &pu, @sizeOf(ExpertsPush), 0);
                const sgb = SwigluPush{ .N = nrows };
                cmd.dispatch(&self.pipes.geglu, .{ ceilDiv(nrows, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.gate_buf, &self.up_buf, &self.geglu_buf }, &sgb, @sizeOf(SwigluPush), 0);
                const pd = ExpertsPush{ .M = d.n_embd, .K = ef, .slice = down_slice, .x_stride = ef, .n_used = n_used, .base = 0 };
                cmd.dispatch(&self.pipes.dmmv_q5_1_experts, .{ n_used * d.n_embd, 1, 1 }, .{ 64, 1, 1 }, &.{ &wde.gpu_buffer, &self.geglu_buf, &self.down_buf, &self.router_out_buf }, &pd, @sizeOf(ExpertsPush), 0);
            } else {
                const sg = SwigluPush{ .N = ef };
                var j: u32 = 0;
                while (j < n_used) : (j += 1) {
                    const id = self.host_router[j];
                    // fused gate_up: gate = rows[0..ef], up = rows[ef..2ef].
                    self.dmmvDispatch(&cmd, wgu, &self.moe_norm_buf, &self.gate_buf, ef, d.n_embd, 0, id * gu_full);
                    self.dmmvDispatch(&cmd, wgu, &self.moe_norm_buf, &self.up_buf, ef, d.n_embd, 0, id * gu_full + gu_half);
                    cmd.dispatch(&self.pipes.geglu, .{ ceilDiv(ef, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.gate_buf, &self.up_buf, &self.geglu_buf }, &sg, @sizeOf(SwigluPush), 0);
                    const down_push = DmmvPush{ .M = d.n_embd, .K = ef, .acc_mode = 0, .a_offset = id * down_slice, .y_offset = j * d.n_embd * @sizeOf(f32) };
                    const didx = dmmvIdx(wde.info.type_);
                    if (didx < 4) {
                        cmd.dispatch(&self.pipes.dmmv_fast[didx], .{ d.n_embd, 1, 1 }, .{ 64, 1, 1 }, &.{ &wde.gpu_buffer, &self.geglu_buf, &self.down_buf }, &down_push, @sizeOf(DmmvPush), 0);
                    } else {
                        cmd.dispatch(&self.pipes.dmmv[didx], .{ d.n_embd, 1, 1 }, .{ 256, 1, 1 }, &.{ &wde.gpu_buffer, &self.geglu_buf, &self.down_buf }, &down_push, @sizeOf(DmmvPush), 0);
                    }
                }
            }
            // zero accumulator → weighted combine of the k slots → post_ffw_norm_2.
            const zp = ZeroPush{ .N = d.n_embd };
            cmd.dispatch(&self.pipes.zero_vec, .{ ceilDiv(d.n_embd, 64), 1, 1 }, .{ 64, 1, 1 }, &.{&self.moe_out_buf}, &zp, @sizeOf(ZeroPush), 0);
            const ma = MoeAccPush{ .N = d.n_embd, .n_used = n_used, .src_stride = d.n_embd };
            if (batched) {
                // Fold the per-expert down scale GPU-side (no host readback).
                const wdscale = self.layer(L, "ffn_down_exps.scale"); // [n_experts] F32
                cmd.dispatch(&self.pipes.moe_weighted_acc_scaled, .{ ceilDiv(d.n_embd, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.moe_out_buf, &self.down_buf, &self.router_out_buf, &wdscale.gpu_buffer }, &ma, @sizeOf(MoeAccPush), 0);
            } else {
                // Fallback: the down scale was already folded into the router weights host-side.
                cmd.dispatch(&self.pipes.moe_weighted_acc, .{ ceilDiv(d.n_embd, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.moe_out_buf, &self.down_buf, &self.router_out_buf }, &ma, @sizeOf(MoeAccPush), 0);
            }
            cmd.dispatch(&self.pipes.rms_norm, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.moe_out_buf, &wpn2.gpu_buffer, &self.moe_out_buf }, &rms, @sizeOf(RmsPush), 0);
            if (batched) self.submit(cmd) else cmd.commitAndWait();
        }

        // --- combine: cur = post_ffw_norm(shared + moe); hidden += cur. ------
        {
            var cmd = try command.beginCommand(ctx);
            const acc = ScaleAccPush{ .N = d.n_embd, .scale = 1.0 };
            cmd.dispatch(&self.pipes.scale_accumulate, .{ ceilDiv(d.n_embd, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.shared_buf, &self.moe_out_buf }, &acc, @sizeOf(ScaleAccPush), 0);
            cmd.dispatch(&self.pipes.rms_norm, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.shared_buf, &wpost.gpu_buffer, &self.shared_buf }, &rms, @sizeOf(RmsPush), 0);
            cmd.dispatch(&self.pipes.scale_accumulate, .{ ceilDiv(d.n_embd, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.hidden, &self.shared_buf }, &acc, @sizeOf(ScaleAccPush), 0);
            if (batched) self.submit(cmd) else cmd.commitAndWait();
        }
    }

    /// Batched gemma4-MoE shared-expert FFN over all T tokens → b.shared[T,n_embd].
    /// Pre-norm + gate/up/down projections via GEMM (the Q8_0 shared weights now
    /// take gemm_q8_0_tiled_v2, read ONCE for all T tokens), element-wise GeGLU
    /// across [T, sf], then post_ffw_norm_1. Mirrors the shared-expert sub-block of
    /// `moeFfnBlock` op-for-op — the only change is the per-token dmmv → batched
    /// GEMM swap, the same one the proven dense `ffnBlockBatched` makes (token-level
    /// output-identical). The per-token `moeRoutedCombine` then reads b.shared[t].
    fn sharedExpertBatched(self: *ForwardGemma, L: u32, T: u32, b: *BatchScratch) !void {
        const d = self.d;
        const ctx = self.ctx;
        const sf = d.shexp_ff; // shared-expert intermediate (2112)
        const wfn = self.layer(L, "ffn_norm.weight");
        const wgate = self.layer(L, "ffn_gate.weight");
        const wup = self.layer(L, "ffn_up.weight");
        const wdown = self.layer(L, "ffn_down.weight");
        const wpn1 = self.layer(L, "post_ffw_norm_1.weight");

        var cmd = try command.beginCommand(ctx);
        const rms = RmsPush{ .N = d.n_embd, .eps = d.rms_eps };
        cmd.dispatch(&self.pipes.rms_norm, .{ T, 1, 1 }, .{ 256, 1, 1 }, &.{ &b.hidden, &wfn.gpu_buffer, &b.ffn_norm }, &rms, @sizeOf(RmsPush), 0);
        // Cycle 19: shared-expert gate+up both read b.ffn_norm — up reuses gate's recast.
        self.gemmDispatch(&cmd, wgate, &b.ffn_norm, &b.gate, sf, d.n_embd, T);
        self.gemmDispatchA(&cmd, wup, &b.ffn_norm, &b.up, sf, d.n_embd, T, true);
        const sg = SwigluPush{ .N = T * sf };
        cmd.dispatch(&self.pipes.geglu, .{ ceilDiv(T * sf, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &b.gate, &b.up, &b.geglu }, &sg, @sizeOf(SwigluPush), 0);
        self.gemmDispatch(&cmd, wdown, &b.geglu, &b.shared, d.n_embd, sf, T);
        cmd.dispatch(&self.pipes.rms_norm, .{ T, 1, 1 }, .{ 256, 1, 1 }, &.{ &b.shared, &wpn1.gpu_buffer, &b.shared }, &rms, @sizeOf(RmsPush), 0);
        // Async on the shared stream (cycle 10): the router/experts/combine stages chain
        // after this in submission order; freed by the single tail waitPending(). On the
        // per-token MoE fallback path the following moeRoutedCombine commitAndWaits, which
        // drains this safely. No host sync per MoE layer.
        self.submit(cmd);
    }

    /// Batched routed-expert matvecs (gemma4-MoE prefill, cycle 8). The pre_ffw_norm_2,
    /// the gate/up/down routed-expert matvecs and the GeGLU for ALL T prompt tokens run
    /// in single token-batched launches (grid.y = T) using the per-token routing already
    /// in b.router_table → b.down_e [T, n_used*n_embd] (slot-major per token). Each token's
    /// accumulate+combine tail (`moeRoutedCombine(preexperts=true)`) then reads its slice
    /// of b.down_e — so the only change vs the per-token path is that the heavy expert
    /// matvecs are issued ONCE over all T tokens instead of looped. Every launch is a
    /// bit-identical twin of the per-token kernel (same dequant + zinc_block_reduce_sum
    /// order), so the result is byte-for-byte the per-token path's. Valid only on the
    /// GPU-side async expert path (Q4_K gate_up + Q5_1 down); async on the shared stream.
    fn moeRoutedExpertsBatched(self: *ForwardGemma, L: u32, T: u32, b: *BatchScratch) !void {
        const d = self.d;
        const ctx = self.ctx;
        const n_used = d.n_experts_used;
        const ef = d.n_ff; // routed-expert intermediate (704)
        const wpre2 = self.layer(L, "pre_ffw_norm_2.weight");
        const wgu = self.layer(L, "ffn_gate_up_exps.weight");
        const wde = self.layer(L, "ffn_down_exps.weight");
        const gu_half = expertSliceBytes(wgu.info.type_, ef, d.n_embd); // ef rows
        const gu_full = gu_half * 2; // 2*ef rows per expert
        const down_slice = expertSliceBytes(wde.info.type_, d.n_embd, ef);
        const rt_stride = 2 * n_used;

        var cmd = try command.beginCommand(ctx);
        const rms = RmsPush{ .N = d.n_embd, .eps = d.rms_eps };
        // Batched pre_ffw_norm_2 of each token's residual → b.moe_norm_e.
        cmd.dispatch(&self.pipes.rms_norm, .{ T, 1, 1 }, .{ 256, 1, 1 }, &.{ &b.hidden, &wpre2.gpu_buffer, &b.moe_norm_e }, &rms, @sizeOf(RmsPush), 0);
        // gate (base 0) and up (base gu_half) routed-expert matvecs over all T tokens.
        const pg = ExpertsBatchPush{ .M = ef, .K = d.n_embd, .slice = gu_full, .x_stride = 0, .n_used = n_used, .base = 0, .routing_stride = rt_stride, .x_tok_stride = d.n_embd, .y_tok_stride = n_used * ef };
        cmd.dispatch(&self.pipes.dmmv_q4k_experts_batched, .{ n_used * ef, T, 1 }, .{ 64, 1, 1 }, &.{ &wgu.gpu_buffer, &b.moe_norm_e, &b.gate_e, &b.router_table }, &pg, @sizeOf(ExpertsBatchPush), 0);
        const pu = ExpertsBatchPush{ .M = ef, .K = d.n_embd, .slice = gu_full, .x_stride = 0, .n_used = n_used, .base = gu_half, .routing_stride = rt_stride, .x_tok_stride = d.n_embd, .y_tok_stride = n_used * ef };
        cmd.dispatch(&self.pipes.dmmv_q4k_experts_batched, .{ n_used * ef, T, 1 }, .{ 64, 1, 1 }, &.{ &wgu.gpu_buffer, &b.moe_norm_e, &b.up_e, &b.router_table }, &pu, @sizeOf(ExpertsBatchPush), 0);
        // GeGLU element-wise over the whole [T, n_used*ef] tile.
        const sg = SwigluPush{ .N = T * n_used * ef };
        cmd.dispatch(&self.pipes.geglu, .{ ceilDiv(T * n_used * ef, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &b.gate_e, &b.up_e, &b.geglu_e }, &sg, @sizeOf(SwigluPush), 0);
        // Routed-expert down matvec over all T tokens → b.down_e (slot-major per token).
        const pd = ExpertsBatchPush{ .M = d.n_embd, .K = ef, .slice = down_slice, .x_stride = ef, .n_used = n_used, .base = 0, .routing_stride = rt_stride, .x_tok_stride = n_used * ef, .y_tok_stride = n_used * d.n_embd };
        cmd.dispatch(&self.pipes.dmmv_q5_1_experts_batched, .{ n_used * d.n_embd, T, 1 }, .{ 64, 1, 1 }, &.{ &wde.gpu_buffer, &b.geglu_e, &b.down_e, &b.router_table }, &pd, @sizeOf(ExpertsBatchPush), 0);
        self.submit(cmd);
    }

    /// Effort 24 cycle 18: token-GROUPED routed-expert matvecs. Identical to
    /// `moeRoutedExpertsBatched` (same pre_ffw_norm_2, gate/up Q4_K + down Q5_1
    /// matvecs, GeGLU, same push params, same output buffers/layout) EXCEPT the
    /// heavy matvecs run the GROUPED kernels: `build_expert_order` first sorts the
    /// T*n_used (token,slot) work-items by expert id into b.expert_order, then each
    /// grouped matvec launches grid = (M output rows, P = T*n_used work-items) and
    /// reads order[blockIdx.y] for its (token,slot) — so consecutive work-items share
    /// the same expert weight, keeping it L2-resident across all tokens routed to it
    /// (a memory-traffic win beyond the cycle-8 launch batching). The per-block
    /// dequant + reduction + the y write location are byte-for-byte the _batched
    /// kernel's, and every output is computed exactly once → byte-identical result
    /// regardless of the order permutation. Async on the shared stream (order is
    /// written before the matvecs read it; both after routerBatched's router_table).
    fn moeRoutedExpertsGrouped(self: *ForwardGemma, L: u32, T: u32, b: *BatchScratch) !void {
        const d = self.d;
        const ctx = self.ctx;
        const n_used = d.n_experts_used;
        const ef = d.n_ff;
        const P = n_used * T; // total (token,slot) work-items
        const wpre2 = self.layer(L, "pre_ffw_norm_2.weight");
        const wgu = self.layer(L, "ffn_gate_up_exps.weight");
        const wde = self.layer(L, "ffn_down_exps.weight");
        const gu_half = expertSliceBytes(wgu.info.type_, ef, d.n_embd);
        const gu_full = gu_half * 2;
        const down_slice = expertSliceBytes(wde.info.type_, d.n_embd, ef);
        const rt_stride = 2 * n_used;

        var cmd = try command.beginCommand(ctx);
        const rms = RmsPush{ .N = d.n_embd, .eps = d.rms_eps };
        cmd.dispatch(&self.pipes.rms_norm, .{ T, 1, 1 }, .{ 256, 1, 1 }, &.{ &b.hidden, &wpre2.gpu_buffer, &b.moe_norm_e }, &rms, @sizeOf(RmsPush), 0);
        // Sort the (token,slot) work-items by expert id → b.expert_order (single block).
        const bo = BuildOrderPush{ .T = T, .n_used = n_used, .n_experts = d.n_experts, .routing_stride = rt_stride };
        cmd.dispatch(&self.pipes.build_expert_order, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ &b.router_table, &b.expert_order }, &bo, @sizeOf(BuildOrderPush), 0);
        // gate (base 0) and up (base gu_half) — grid = (ef rows, P work-items).
        const pg = ExpertsBatchPush{ .M = ef, .K = d.n_embd, .slice = gu_full, .x_stride = 0, .n_used = n_used, .base = 0, .routing_stride = rt_stride, .x_tok_stride = d.n_embd, .y_tok_stride = n_used * ef };
        cmd.dispatch(&self.pipes.dmmv_q4k_experts_grouped, .{ ef, P, 1 }, .{ 64, 1, 1 }, &.{ &wgu.gpu_buffer, &b.moe_norm_e, &b.gate_e, &b.router_table, &b.expert_order }, &pg, @sizeOf(ExpertsBatchPush), 0);
        const pu = ExpertsBatchPush{ .M = ef, .K = d.n_embd, .slice = gu_full, .x_stride = 0, .n_used = n_used, .base = gu_half, .routing_stride = rt_stride, .x_tok_stride = d.n_embd, .y_tok_stride = n_used * ef };
        cmd.dispatch(&self.pipes.dmmv_q4k_experts_grouped, .{ ef, P, 1 }, .{ 64, 1, 1 }, &.{ &wgu.gpu_buffer, &b.moe_norm_e, &b.up_e, &b.router_table, &b.expert_order }, &pu, @sizeOf(ExpertsBatchPush), 0);
        const sg = SwigluPush{ .N = T * n_used * ef };
        cmd.dispatch(&self.pipes.geglu, .{ ceilDiv(T * n_used * ef, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &b.gate_e, &b.up_e, &b.geglu_e }, &sg, @sizeOf(SwigluPush), 0);
        // down — grid = (n_embd rows, P work-items).
        const pd = ExpertsBatchPush{ .M = d.n_embd, .K = ef, .slice = down_slice, .x_stride = ef, .n_used = n_used, .base = 0, .routing_stride = rt_stride, .x_tok_stride = n_used * ef, .y_tok_stride = n_used * d.n_embd };
        cmd.dispatch(&self.pipes.dmmv_q5_1_experts_grouped, .{ d.n_embd, P, 1 }, .{ 64, 1, 1 }, &.{ &wde.gpu_buffer, &b.geglu_e, &b.down_e, &b.router_table, &b.expert_order }, &pd, @sizeOf(ExpertsBatchPush), 0);
        self.submit(cmd);
    }

    /// Batched gemma4-MoE routed-expert accumulate + combine over all T tokens
    /// (Effort 24 cycle 9) — the last per-token cost on the prefill MoE FFN path.
    /// Replaces the per-token `moeRoutedCombine(prerouted, preexperts)` loop (one
    /// launch per token of zero/acc/norm/scale-acc/norm/scale-acc/output-scale) with
    /// ~7 batched launches/layer that read the already-batched b.down_e / b.shared /
    /// b.router_table / b.hidden streams in place. Every op is a bit-identical twin
    /// of the per-token tail:
    ///   - zero_vec / scale_accumulate / scalar_mul are element-wise → run over the
    ///     whole [T, n_embd] tile (N = T*n_embd); each element's result is exactly
    ///     the per-token launch's (contiguous, token-major layout).
    ///   - rms_norm (post_ffw_norm_2, post_ffw_norm) already indexes token=blockIdx.x
    ///     → grid.x = T reproduces the per-token reduction order block-for-block.
    ///   - moe_weighted_acc_scaled_batched is the per-token kernel with grid.y = T +
    ///     per-token strides, so the j-loop FMA order + GPU-side down scale are
    ///     unchanged. The combined output is byte-for-byte the per-token loop's.
    /// Async on the shared stream (stream order: experts/router write the buffers
    /// this reads). The standalone per-token `layerOutScale` is folded in as the
    /// final scalar_mul (self-skipping when the layer has no output scale).
    fn moeRoutedCombineBatched(self: *ForwardGemma, L: u32, T: u32, b: *BatchScratch) !void {
        const d = self.d;
        const ctx = self.ctx;
        const n_used = d.n_experts_used;
        const wpn2 = self.layer(L, "post_ffw_norm_2.weight");
        const wpost = self.layer(L, "post_ffw_norm.weight");
        const wdscale = self.layer(L, "ffn_down_exps.scale"); // [n_experts] F32
        const ws = self.model.getLayer(L, "layer_output_scale.weight"); // optional scalar
        const rms = RmsPush{ .N = d.n_embd, .eps = d.rms_eps };
        const total = T * d.n_embd;

        var cmd = try command.beginCommand(ctx);
        // zero the [T, n_embd] accumulator (moe_weighted_acc_scaled_batched is a +=).
        const zp = ZeroPush{ .N = total };
        cmd.dispatch(&self.pipes.zero_vec, .{ ceilDiv(total, 64), 1, 1 }, .{ 64, 1, 1 }, &.{&b.moe_out_e}, &zp, @sizeOf(ZeroPush), 0);
        // Weighted combine of each token's n_used routed-down slices (down scale
        // folded GPU-side) → b.moe_out_e[t]. grid.y = T, per-token strides.
        const ma = MoeAccBatchPush{ .N = d.n_embd, .n_used = n_used, .src_stride = d.n_embd, .a_tok_stride = d.n_embd, .b_tok_stride = n_used * d.n_embd, .routing_stride = 2 * n_used };
        cmd.dispatch(&self.pipes.moe_weighted_acc_scaled_batched, .{ ceilDiv(d.n_embd, 64), T, 1 }, .{ 64, 1, 1 }, &.{ &b.moe_out_e, &b.down_e, &b.router_table, &wdscale.gpu_buffer }, &ma, @sizeOf(MoeAccBatchPush), 0);
        // post_ffw_norm_2 over each token's combined routed output (grid.x = T).
        cmd.dispatch(&self.pipes.rms_norm, .{ T, 1, 1 }, .{ 256, 1, 1 }, &.{ &b.moe_out_e, &wpn2.gpu_buffer, &b.moe_out_e }, &rms, @sizeOf(RmsPush), 0);
        // shared += moe (element-wise over the whole tile).
        const acc = ScaleAccPush{ .N = total, .scale = 1.0 };
        cmd.dispatch(&self.pipes.scale_accumulate, .{ ceilDiv(total, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &b.shared, &b.moe_out_e }, &acc, @sizeOf(ScaleAccPush), 0);
        // post_ffw_norm(shared + moe) (grid.x = T).
        cmd.dispatch(&self.pipes.rms_norm, .{ T, 1, 1 }, .{ 256, 1, 1 }, &.{ &b.shared, &wpost.gpu_buffer, &b.shared }, &rms, @sizeOf(RmsPush), 0);
        // hidden += cur (element-wise over the whole tile).
        cmd.dispatch(&self.pipes.scale_accumulate, .{ ceilDiv(total, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &b.hidden, &b.shared }, &acc, @sizeOf(ScaleAccPush), 0);
        // layer_output_scale (folded-in per-token layerOutScale; scalar broadcast).
        if (ws) |wscale| {
            const sm = ScalarMulPush{ .N = total };
            cmd.dispatch(&self.pipes.scalar_mul, .{ ceilDiv(total, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &b.hidden, &wscale.gpu_buffer }, &sm, @sizeOf(ScalarMulPush), 0);
        }
        self.submit(cmd);
    }

    /// Batched gemma4-MoE router over all T tokens → b.router_table[T, 2*n_used]
    /// (per-token expert ids then renorm-softmax weight-bits). Computes the router
    /// input (plain-RMS-norm of the residual × ffn_gate_inp.scale × 1/sqrt(n_embd))
    /// for all T tokens, the F32 router logits via gemm_f32_tiled_v2 (the F32 router
    /// weight read ONCE instead of T times), and the per-token top-k softmax in one
    /// batched launch. Mirrors the per-token router sub-block of `moeRoutedCombine`
    /// op-for-op — the only change is batching, so the routing it produces is the
    /// per-token path's (token-correct; the F32 GEMM is the batched twin of looping
    /// dmmv_f32, same class as the proven dense quant GEMMs). The per-token
    /// `moeRoutedCombine(prerouted=true)` then reads its row of b.router_table.
    fn routerBatched(self: *ForwardGemma, L: u32, T: u32, b: *BatchScratch) !void {
        const d = self.d;
        const ctx = self.ctx;
        const wrouter = self.layer(L, "ffn_gate_inp.weight"); // [n_embd, n_experts] F32
        const wrscale = self.layer(L, "ffn_gate_inp.scale"); // [n_embd] F32

        var cmd = try command.beginCommand(ctx);
        const rms = RmsPush{ .N = d.n_embd, .eps = d.rms_eps };
        // Plain-RMS-norm of each token's residual (no learnable weight), batched.
        cmd.dispatch(&self.pipes.rms_norm_noweight, .{ T, 1, 1 }, .{ 256, 1, 1 }, &.{ &b.hidden, &b.router_in }, &rms, @sizeOf(RmsPush), 0);
        // Per-channel ffn_gate_inp.scale × 1/sqrt(n_embd), broadcast across tokens.
        const mv = MulVecBatchPush{ .row = d.n_embd, .total = T * d.n_embd, .scale = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(d.n_embd))) };
        cmd.dispatch(&self.pipes.mul_vec_scaled_batched, .{ ceilDiv(T * d.n_embd, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &b.router_in, &wrscale.gpu_buffer }, &mv, @sizeOf(MulVecBatchPush), 0);
        // Router logits [T, n_experts] = router_in[T, n_embd] · wrouter[n_experts, n_embd]^T.
        const gp = GemmPush{ .M = d.n_experts, .K = d.n_embd, .T = T };
        cmd.dispatch(&self.pipes.gemm_f32, .{ ceilDiv(d.n_experts, 64), ceilDiv(T, 64), 1 }, .{ 256, 1, 1 }, &.{ &wrouter.gpu_buffer, &b.router_in, &b.router_logits }, &gp, @sizeOf(GemmPush), 0);
        // Per-token top-k softmax → routing table (one block per token).
        const tk = TopkPush{ .n_experts = d.n_experts, .k = d.n_experts_used };
        cmd.dispatch(&self.pipes.softmax_topk_batched, .{ T, 1, 1 }, .{ 64, 1, 1 }, &.{ &b.router_logits, &b.router_table }, &tk, @sizeOf(TopkPush), 0);
        // Async on the shared stream: the per-token expert launches that follow read
        // the finished table by stream order (no host sync needed).
        self.submit(cmd);
    }

    /// gemma4-MoE routed-expert FFN + combine for ONE token, reading a pre-computed
    /// shared-expert output from `self.shared_buf`. This is exactly `moeFfnBlock`
    /// MINUS its shared-expert sub-block (now computed once for all T tokens by
    /// `sharedExpertBatched`): router top-k, routed experts, and the
    /// post_ffw_norm(shared+moe)+residual combine, all on `self.hidden` /
    /// `self.shared_buf` (aliased by the caller to this token's batched slices).
    /// The router/routed/combine kernels + push constants are identical to
    /// moeFfnBlock, so the per-token math is byte-for-byte the per-token path's.
    ///
    /// When `prerouted` is set (the batched prefill path), the per-token router
    /// sub-block is SKIPPED: `routerBatched` has already computed all T tokens'
    /// routing in one pass and the caller has aliased `self.router_out_buf` to this
    /// token's row of the table, so the routed-expert launches read it as before.
    ///
    /// When `preexperts` is set (cycle 8), the pre_ffw_norm_2 + the gate/up/down
    /// routed-expert matvecs + GeGLU are ALSO skipped: `moeRoutedExpertsBatched` has
    /// already produced this token's routed-down output and the caller has aliased
    /// `self.down_buf` to its slice of b.down_e, so only the per-token accumulate +
    /// post_ffw_norm + residual combine remain (byte-identical). Implies prerouted.
    fn moeRoutedCombine(self: *ForwardGemma, L: u32, prerouted: bool, preexperts: bool) !void {
        const d = self.d;
        const ctx = self.ctx;
        const n_used = d.n_experts_used;
        const ef = d.n_ff; // routed-expert intermediate (704)

        const wpre2 = self.layer(L, "pre_ffw_norm_2.weight");
        const wpn2 = self.layer(L, "post_ffw_norm_2.weight");
        const wpost = self.layer(L, "post_ffw_norm.weight");
        const wgu = self.layer(L, "ffn_gate_up_exps.weight"); // [n_embd, 2*ef, n_experts]
        const wde = self.layer(L, "ffn_down_exps.weight"); // [ef, n_embd, n_experts]

        const rms = RmsPush{ .N = d.n_embd, .eps = d.rms_eps };
        const batched = dmmvIdx(wgu.info.type_) == 0 and dmmvIdx(wde.info.type_) == 5;

        // --- router logits + top-k softmax (computed from this token's hidden) ---
        // Skipped when prerouted: routerBatched filled self.router_out_buf's row.
        if (!prerouted) {
            const wrouter = self.layer(L, "ffn_gate_inp.weight"); // [n_embd, n_experts] F32
            const wrscale = self.layer(L, "ffn_gate_inp.scale"); // [n_embd] F32
            var cmd = try command.beginCommand(ctx);
            cmd.dispatch(&self.pipes.rms_norm_noweight, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.hidden, &self.norm_buf }, &rms, @sizeOf(RmsPush), 0);
            const mv = MulVecPush{ .N = d.n_embd, .scale = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(d.n_embd))) };
            cmd.dispatch(&self.pipes.mul_vec_scaled, .{ ceilDiv(d.n_embd, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.norm_buf, &wrscale.gpu_buffer }, &mv, @sizeOf(MulVecPush), 0);
            self.dmmvDispatch(&cmd, wrouter, &self.norm_buf, &self.router_logits_buf, d.n_experts, d.n_embd, 0, 0);
            const tk = TopkPush{ .n_experts = d.n_experts, .k = n_used };
            cmd.dispatch(&self.pipes.softmax_topk, .{ 1, 1, 1 }, .{ 64, 1, 1 }, &.{ &self.router_logits_buf, &self.router_out_buf }, &tk, @sizeOf(TopkPush), 0);
            if (batched) {
                self.submit(cmd); // async: experts read ids GPU-side, scale folded GPU-side
            } else {
                cmd.commitAndWait(); // sync: the fallback host-gathers ids + folds the scale next
                self.drainPending();
            }

            // Fallback only: download ids+weights and fold the per-expert down scale
            // into the weights host-side. The batched path folds it GPU-side
            // (moe_weighted_acc_scaled). prerouted implies batched, so this never runs.
            if (!batched) {
                buffer.download(ctx, &self.router_out_buf, std.mem.sliceAsBytes(self.host_router[0 .. 2 * n_used]));
                const scales = self.down_scales[L * d.n_experts ..][0..d.n_experts];
                var j: u32 = 0;
                while (j < n_used) : (j += 1) {
                    const id = self.host_router[j];
                    const w: f32 = @bitCast(self.host_router[n_used + j]);
                    self.host_router[n_used + j] = @bitCast(w * scales[id]);
                }
                buffer.upload(ctx, &self.router_out_buf, std.mem.sliceAsBytes(self.host_router[0 .. 2 * n_used]));
            }
        }

        // --- routed experts → moe_out_buf -----------------------------------
        {
            var cmd = try command.beginCommand(ctx);
            // When preexperts: the matvecs ran batched over all T tokens already and
            // self.down_buf aliases this token's b.down_e slice → skip straight to the
            // accumulate. Otherwise compute this token's routed experts in place.
            if (!preexperts) {
                const gu_half = expertSliceBytes(wgu.info.type_, ef, d.n_embd); // ef rows
                const gu_full = gu_half * 2; // 2*ef rows per expert
                const down_slice = expertSliceBytes(wde.info.type_, d.n_embd, ef);
                cmd.dispatch(&self.pipes.rms_norm, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.hidden, &wpre2.gpu_buffer, &self.moe_norm_buf }, &rms, @sizeOf(RmsPush), 0);
                if (batched) {
                    const nrows = n_used * ef;
                    const pg = ExpertsPush{ .M = ef, .K = d.n_embd, .slice = gu_full, .x_stride = 0, .n_used = n_used, .base = 0 };
                    cmd.dispatch(&self.pipes.dmmv_q4k_experts, .{ nrows, 1, 1 }, .{ 64, 1, 1 }, &.{ &wgu.gpu_buffer, &self.moe_norm_buf, &self.gate_buf, &self.router_out_buf }, &pg, @sizeOf(ExpertsPush), 0);
                    const pu = ExpertsPush{ .M = ef, .K = d.n_embd, .slice = gu_full, .x_stride = 0, .n_used = n_used, .base = gu_half };
                    cmd.dispatch(&self.pipes.dmmv_q4k_experts, .{ nrows, 1, 1 }, .{ 64, 1, 1 }, &.{ &wgu.gpu_buffer, &self.moe_norm_buf, &self.up_buf, &self.router_out_buf }, &pu, @sizeOf(ExpertsPush), 0);
                    const sgb = SwigluPush{ .N = nrows };
                    cmd.dispatch(&self.pipes.geglu, .{ ceilDiv(nrows, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.gate_buf, &self.up_buf, &self.geglu_buf }, &sgb, @sizeOf(SwigluPush), 0);
                    const pd = ExpertsPush{ .M = d.n_embd, .K = ef, .slice = down_slice, .x_stride = ef, .n_used = n_used, .base = 0 };
                    cmd.dispatch(&self.pipes.dmmv_q5_1_experts, .{ n_used * d.n_embd, 1, 1 }, .{ 64, 1, 1 }, &.{ &wde.gpu_buffer, &self.geglu_buf, &self.down_buf, &self.router_out_buf }, &pd, @sizeOf(ExpertsPush), 0);
                } else {
                    const sg = SwigluPush{ .N = ef };
                    var j: u32 = 0;
                    while (j < n_used) : (j += 1) {
                        const id = self.host_router[j];
                        self.dmmvDispatch(&cmd, wgu, &self.moe_norm_buf, &self.gate_buf, ef, d.n_embd, 0, id * gu_full);
                        self.dmmvDispatch(&cmd, wgu, &self.moe_norm_buf, &self.up_buf, ef, d.n_embd, 0, id * gu_full + gu_half);
                        cmd.dispatch(&self.pipes.geglu, .{ ceilDiv(ef, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.gate_buf, &self.up_buf, &self.geglu_buf }, &sg, @sizeOf(SwigluPush), 0);
                        const down_push = DmmvPush{ .M = d.n_embd, .K = ef, .acc_mode = 0, .a_offset = id * down_slice, .y_offset = j * d.n_embd * @sizeOf(f32) };
                        const didx = dmmvIdx(wde.info.type_);
                        if (didx < 4) {
                            cmd.dispatch(&self.pipes.dmmv_fast[didx], .{ d.n_embd, 1, 1 }, .{ 64, 1, 1 }, &.{ &wde.gpu_buffer, &self.geglu_buf, &self.down_buf }, &down_push, @sizeOf(DmmvPush), 0);
                        } else {
                            cmd.dispatch(&self.pipes.dmmv[didx], .{ d.n_embd, 1, 1 }, .{ 256, 1, 1 }, &.{ &wde.gpu_buffer, &self.geglu_buf, &self.down_buf }, &down_push, @sizeOf(DmmvPush), 0);
                        }
                    }
                }
            } // end if (!preexperts)
            const zp = ZeroPush{ .N = d.n_embd };
            cmd.dispatch(&self.pipes.zero_vec, .{ ceilDiv(d.n_embd, 64), 1, 1 }, .{ 64, 1, 1 }, &.{&self.moe_out_buf}, &zp, @sizeOf(ZeroPush), 0);
            const ma = MoeAccPush{ .N = d.n_embd, .n_used = n_used, .src_stride = d.n_embd };
            if (batched) {
                const wdscale = self.layer(L, "ffn_down_exps.scale"); // [n_experts] F32
                cmd.dispatch(&self.pipes.moe_weighted_acc_scaled, .{ ceilDiv(d.n_embd, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.moe_out_buf, &self.down_buf, &self.router_out_buf, &wdscale.gpu_buffer }, &ma, @sizeOf(MoeAccPush), 0);
            } else {
                cmd.dispatch(&self.pipes.moe_weighted_acc, .{ ceilDiv(d.n_embd, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.moe_out_buf, &self.down_buf, &self.router_out_buf }, &ma, @sizeOf(MoeAccPush), 0);
            }
            cmd.dispatch(&self.pipes.rms_norm, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.moe_out_buf, &wpn2.gpu_buffer, &self.moe_out_buf }, &rms, @sizeOf(RmsPush), 0);
            if (batched) self.submit(cmd) else cmd.commitAndWait();
        }

        // --- combine: cur = post_ffw_norm(shared + moe); hidden += cur. ------
        {
            var cmd = try command.beginCommand(ctx);
            const acc = ScaleAccPush{ .N = d.n_embd, .scale = 1.0 };
            cmd.dispatch(&self.pipes.scale_accumulate, .{ ceilDiv(d.n_embd, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.shared_buf, &self.moe_out_buf }, &acc, @sizeOf(ScaleAccPush), 0);
            cmd.dispatch(&self.pipes.rms_norm, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.shared_buf, &wpost.gpu_buffer, &self.shared_buf }, &rms, @sizeOf(RmsPush), 0);
            cmd.dispatch(&self.pipes.scale_accumulate, .{ ceilDiv(d.n_embd, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.hidden, &self.shared_buf }, &acc, @sizeOf(ScaleAccPush), 0);
            if (batched) self.submit(cmd) else cmd.commitAndWait();
        }
    }

    /// Multiply the residual stream by the learned per-layer output scale.
    /// Dense layers fold this scale into the post-ffn rms_norm_residual_scale
    /// (the layer's last `hidden` write), so this self-skips them; only the MoE
    /// path — whose final write is a scale_accumulate — needs the standalone op.
    fn layerOutScale(self: *ForwardGemma, L: u32) !void {
        const d = self.d;
        const ctx = self.ctx;
        const is_moe = d.n_experts > 0 and self.model.getLayer(L, "ffn_gate_inp.weight") != null;
        if (!is_moe) return; // dense: folded into the post-ffn norm+residual
        const ws = self.model.getLayer(L, "layer_output_scale.weight") orelse return; // optional
        var cmd = try command.beginCommand(ctx);
        const sm = ScalarMulPush{ .N = d.n_embd };
        cmd.dispatch(&self.pipes.scalar_mul, .{ ceilDiv(d.n_embd, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.hidden, &ws.gpu_buffer }, &sm, @sizeOf(ScalarMulPush), 0);
        self.submit(cmd);
    }

    // ---- public per-block hooks (dbg_cuda per-layer residual diff) ----------
    // Mirror ForwardCuda's *Pub hooks so dbg_cuda can dump the residual stream
    // after each gemma layer block and diff it against llama.cpp `l_out-N`.
    pub fn attentionLayerPub(self: *ForwardGemma, L: u32, pos: u32) !void {
        try self.attentionLayer(L, pos);
        self.waitPending(); // block may be async in-flight; readHidden needs it done
    }
    /// FFN block dispatched exactly as decodeStep: routed MoE when this layer
    /// carries a router, dense GeGLU otherwise.
    pub fn ffnLayerPub(self: *ForwardGemma, L: u32) !void {
        if (self.d.n_experts > 0 and self.model.getLayer(L, "ffn_gate_inp.weight") != null) {
            try self.moeFfnBlock(L);
        } else {
            try self.ffnBlock(L);
        }
        self.waitPending();
    }
    pub fn layerOutScalePub(self: *ForwardGemma, L: u32) !void {
        try self.layerOutScale(L);
        self.waitPending();
    }

    pub fn readHidden(self: *ForwardGemma, out: []f32) void {
        buffer.download(self.ctx, &self.hidden, std.mem.sliceAsBytes(out[0..self.d.n_embd]));
    }
    pub fn readLogits(self: *ForwardGemma, out: []f32) void {
        buffer.download(self.ctx, &self.logits_buf, std.mem.sliceAsBytes(out[0..@min(out.len, self.d.vocab)]));
    }

    // ---- helpers ------------------------------------------------------------

    fn layer(self: *ForwardGemma, L: u32, suffix: []const u8) *const LoadedTensor {
        return self.model.getLayer(L, suffix) orelse {
            log.err("missing tensor blk.{d}.{s}", .{ L, suffix });
            @panic("missing tensor");
        };
    }

    fn dmmvDispatch(self: *ForwardGemma, cmd: *command.CudaCommand, w: *const LoadedTensor, x: *const CudaBuffer, y: *const CudaBuffer, M: u32, K: u32, acc_mode: u32, a_offset: u32) void {
        const push = DmmvPush{ .M = M, .K = K, .acc_mode = acc_mode, .a_offset = a_offset };
        const idx = dmmvIdx(w.info.type_);
        if (idx < 4) {
            cmd.dispatch(&self.pipes.dmmv_fast[idx], .{ M, 1, 1 }, .{ 64, 1, 1 }, &.{ &w.gpu_buffer, x, y }, &push, @sizeOf(DmmvPush), 0);
        } else {
            cmd.dispatch(&self.pipes.dmmv[idx], .{ M, 1, 1 }, .{ 256, 1, 1 }, &.{ &w.gpu_buffer, x, y }, &push, @sizeOf(DmmvPush), 0);
        }
    }

    /// Fuse two same-input Q4_K matvecs (w0→y0 [M0 rows], w1→y1 [M1 rows], shared
    /// input x of inner dim K) into ONE launch over M0+M1 blocks — removes a
    /// kernel-launch boundary. Both weights MUST be Q4_K (caller-checked); each
    /// block's compute is bit-identical to dmmvDispatch's fast path. Used for the
    /// gemma FFN gate/up and attention Q/K pairs.
    fn dmmvDualQ4k(self: *ForwardGemma, cmd: *command.CudaCommand, w0: *const LoadedTensor, w1: *const LoadedTensor, x: *const CudaBuffer, y0: *const CudaBuffer, y1: *const CudaBuffer, M0: u32, M1: u32, K: u32) void {
        const push = Dmmv2Push{ .M0 = M0, .M1 = M1, .K = K };
        cmd.dispatch(&self.pipes.dmmv_q4k_fast_dual, .{ M0 + M1, 1, 1 }, .{ 64, 1, 1 }, &.{ &w0.gpu_buffer, &w1.gpu_buffer, x, y0, y1 }, &push, @sizeOf(Dmmv2Push), 0);
    }

    // ---- async decode command ring (mirror ForwardCuda) ---------------------
    /// Dense path (n_experts==0): commit the per-block command asynchronously on
    /// the shared auto-ordered CUstream and stash it — the CPU never blocks per
    /// block. The stream still serializes the GPU, so cross-block buffer reuse is
    /// safe (only the ~0.4 ms WSL2 CPU↔GPU sync round-trips are removed, which
    /// also stops the boost-starvation those idle gaps cause). The batched MoE path
    /// is async too (down scale folded GPU-side, no host id readback); the per-slot
    /// MoE fallback keeps explicit commitAndWait around its readback. Falls back to
    /// sync if the ring fills.
    fn submit(self: *ForwardGemma, cmd: command.CudaCommand) void {
        var c = cmd;
        // Async whenever the ring has room. The batched MoE path (gate_up Q4_K +
        // down Q5_1) folds the down scale GPU-side, so it no longer reads ids back
        // mid-block — its commands chain on the same auto-ordered stream like the
        // dense path. The per-slot fallback still uses explicit commitAndWait around
        // its host id readback, so it never relies on this going async.
        if (self.n_pending < self.pending.len) {
            c.commitAsync();
            self.pending[self.n_pending] = c;
            self.n_pending += 1;
        } else {
            c.commitAndWait();
        }
    }

    /// Free the stashed async commands. Safe once a later same-stream
    /// commitAndWait (the tail) has drained the stream — completion guaranteed.
    fn drainPending(self: *ForwardGemma) void {
        var i: u32 = 0;
        while (i < self.n_pending) : (i += 1) self.pending[i].releaseCompleted();
        self.n_pending = 0;
    }

    /// Wait on + free the stashed async commands, for callers that read a GPU
    /// result before any tail sync (the per-block Pub wrappers).
    fn waitPending(self: *ForwardGemma) void {
        var i: u32 = 0;
        while (i < self.n_pending) : (i += 1) self.pending[i].wait();
        self.n_pending = 0;
    }
};

fn ceilDiv(a: u32, b: u32) u32 {
    return (a + b - 1) / b;
}

/// Effort 26 cycle 1: the fp16 tensor-core dense GEMM is now ON by default for
/// gemma prefill (a real +24% on the RTX 5090, neutral on the 4090). True unless
/// ZINC_BATCHED_TC is explicitly off (0/off/false/no) — the A/B kill-switch back
/// to the f32 register-tiled GEMM. Mirrors batchedPrefillDefaultOn's parsing.
fn tcDefaultOn() bool {
    const v = std.posix.getenv("ZINC_BATCHED_TC") orelse return true;
    return !(std.mem.eql(u8, v, "0") or std.ascii.eqlIgnoreCase(v, "off") or
        std.ascii.eqlIgnoreCase(v, "false") or std.ascii.eqlIgnoreCase(v, "no"));
}

// Effort 26 cycle 9: the cuBLAS dense Q4_K prefill GEMM is default-ON (opt out
// ZINC_BATCHED_CUBLAS=0/off/false/no). It is +76% on gemma-31b dense prefill at
// T=512 (the effort's #1 gap row) and neutral on gemma-26b (whose FLOPs are in
// the experts, not the small dense attn-proj GEMMs cuBLAS touches). The win is
// T-dependent (the full-weight dequant→fp16 round-trip is a fixed cost amortized
// over T tokens): +76% @T=512, +15% @T=128, break-even @T=64 — so the dispatch
// gates cuBLAS on T >= cublas_min_t (128) and falls back to the proven gemm_q4k_tc
// path for short prompts. qwen (no batched gemma path) and all decode are
// untouched (prefill-only path).
fn cublasDefaultOn() bool {
    const v = std.posix.getenv("ZINC_BATCHED_CUBLAS") orelse return true;
    return !(std.mem.eql(u8, v, "0") or std.ascii.eqlIgnoreCase(v, "off") or
        std.ascii.eqlIgnoreCase(v, "false") or std.ascii.eqlIgnoreCase(v, "no"));
}

/// Bytes for one expert's [rows × cols] slice in a stacked/fused MoE weight
/// tensor. cols is the quantized (contiguous) dim; rows is the number of output
/// rows. Matches the layout the dmmv kernels expect (a_offset in bytes).
fn expertSliceBytes(q: gguf.GGMLType, rows: u32, cols: u32) u32 {
    return rows * (cols / q.blockSize()) * q.bytesPerBlock();
}

fn readArchU32(gf: *const gguf.GGUFFile, arch: []const u8, suffix: []const u8) ?u32 {
    var buf: [128]u8 = undefined;
    const key = std.fmt.bufPrint(&buf, "{s}.{s}", .{ arch, suffix }) catch return null;
    return gf.getU32(key);
}

/// Read a per-layer u32 metadata array (e.g. head_count_kv). Falls back to a
/// scalar key (broadcast) if the value is not stored as an array.
fn readU32Array(allocator: std.mem.Allocator, gf: *const gguf.GGUFFile, arch: []const u8, suffix: []const u8, n: u32) ![]u32 {
    const out = try allocator.alloc(u32, n);
    errdefer allocator.free(out);
    var buf: [128]u8 = undefined;
    const key = std.fmt.bufPrint(&buf, "{s}.{s}", .{ arch, suffix }) catch return error.KeyFormat;
    if (gf.metadata.get(key)) |val| {
        switch (val) {
            .array => |arr| {
                for (0..n) |i| out[i] = if (i < arr.len) (arr[i].asU32() orelse 0) else 0;
                return out;
            },
            else => {
                const scalar = val.asU32() orelse 0;
                for (out) |*v| v.* = scalar;
                return out;
            },
        }
    }
    return error.MissingArray;
}

/// Read a per-layer bool metadata array (e.g. sliding_window_pattern).
fn readBoolArray(allocator: std.mem.Allocator, gf: *const gguf.GGUFFile, arch: []const u8, suffix: []const u8, n: u32) ![]bool {
    const out = try allocator.alloc(bool, n);
    errdefer allocator.free(out);
    var buf: [128]u8 = undefined;
    const key = std.fmt.bufPrint(&buf, "{s}.{s}", .{ arch, suffix }) catch return error.KeyFormat;
    if (gf.metadata.get(key)) |val| {
        switch (val) {
            .array => |arr| {
                for (0..n) |i| out[i] = if (i < arr.len) (arr[i].asBool() orelse false) else false;
                return out;
            },
            else => return error.MissingArray,
        }
    }
    return error.MissingArray;
}
