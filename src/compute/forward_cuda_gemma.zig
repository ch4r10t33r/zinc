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
const KvWritePush = extern struct { kv_dim: u32, dst_offset: u32 };
const GemmaAttnPush = extern struct {
    head_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    seq_len: u32,
    scale_bits: u32,
    window: u32,
};
const RmsRopePush = extern struct { head_dim: u32, eps: f32, rope_dim: u32, position: u32 };
const SwigluPush = extern struct { N: u32 };
const ScaleAccPush = extern struct { N: u32, scale: f32 };
const ScalarMulPush = extern struct { N: u32 };
const ArgmaxPush = extern struct { N: u32 };
// MoE router/combine kernels (byte-match kernels.cu).
const TopkPush = extern struct { n_experts: u32, k: u32 };
const MoeAccPush = extern struct { N: u32, n_used: u32, src_stride: u32 };
const MulVecPush = extern struct { N: u32, scale: f32 };
const ZeroPush = extern struct { N: u32 };
// Batched MoE expert matvec (one launch over all experts; ids read GPU-side).
const ExpertsPush = extern struct { M: u32, K: u32, slice: u32, x_stride: u32, n_used: u32, base: u32 = 0 };

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
    rms_norm_rope: CudaPipeline,
    dmmv: [6]CudaPipeline,
    dmmv_fast: [4]CudaPipeline,
    rope: CudaPipeline,
    kv_cache_write: CudaPipeline,
    gemma_attention: CudaPipeline,
    geglu: CudaPipeline,
    scale_accumulate: CudaPipeline,
    scalar_mul: CudaPipeline,
    argmax: CudaPipeline,
    // MoE (compiled unconditionally; dispatched only when n_experts>0)
    softmax_topk: CudaPipeline,
    moe_weighted_acc: CudaPipeline,
    moe_weighted_acc_scaled: CudaPipeline, // batched MoE: folds down scale GPU-side
    mul_vec_scaled: CudaPipeline,
    zero_vec: CudaPipeline,
    dmmv_q4k_experts: CudaPipeline, // batched fused gate/up over all experts
    dmmv_q5_1_experts: CudaPipeline, // batched down over all experts
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
        pipes.rms_norm_rope = try pipeline.createPipeline(ctx, src.ptr, "rms_norm_rope");
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
        pipes.rope = try pipeline.createPipeline(ctx, src.ptr, "rope");
        pipes.kv_cache_write = try pipeline.createPipeline(ctx, src.ptr, "kv_cache_write");
        pipes.gemma_attention = try pipeline.createPipeline(ctx, src.ptr, "gemma_attention");
        pipes.geglu = try pipeline.createPipeline(ctx, src.ptr, "geglu");
        pipes.scale_accumulate = try pipeline.createPipeline(ctx, src.ptr, "scale_accumulate");
        pipes.scalar_mul = try pipeline.createPipeline(ctx, src.ptr, "scalar_mul");
        pipes.argmax = try pipeline.createPipeline(ctx, src.ptr, "argmax");
        pipes.softmax_topk = try pipeline.createPipeline(ctx, src.ptr, "softmax_topk");
        pipes.moe_weighted_acc = try pipeline.createPipeline(ctx, src.ptr, "moe_weighted_acc");
        pipes.moe_weighted_acc_scaled = try pipeline.createPipeline(ctx, src.ptr, "moe_weighted_acc_scaled");
        pipes.mul_vec_scaled = try pipeline.createPipeline(ctx, src.ptr, "mul_vec_scaled");
        pipes.zero_vec = try pipeline.createPipeline(ctx, src.ptr, "zero_vec");
        pipes.dmmv_q4k_experts = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q4k_experts");
        pipes.dmmv_q5_1_experts = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q5_1_experts");
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
        inline for (std.meta.fields(Pipelines)) |f| {
            if (comptime std.mem.eql(u8, f.name, "dmmv")) {
                for (&self.pipes.dmmv) |*p| pipeline.freePipeline(p);
            } else if (comptime std.mem.eql(u8, f.name, "dmmv_fast")) {
                for (&self.pipes.dmmv_fast) |*p| pipeline.freePipeline(p);
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

        var cmd = try command.beginCommand(ctx);
        // pre-attention norm (gemma rms, +1 baked in)
        const rms = RmsPush{ .N = d.n_embd, .eps = d.rms_eps };
        cmd.dispatch(&self.pipes.rms_norm, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.hidden, &wan.gpu_buffer, &self.norm_buf }, &rms, @sizeOf(RmsPush), 0);
        // Q, K projections; V from Wv if present, else the raw K projection.
        self.dmmvDispatch(&cmd, wq, &self.norm_buf, &self.q_buf, g.q_dim, d.n_embd, 0, 0);
        self.dmmvDispatch(&cmd, wk, &self.norm_buf, &self.k_buf, g.kv_dim, d.n_embd, 0, 0);
        const v_src: *const CudaBuffer = if (wv_opt) |wv| blk: {
            self.dmmvDispatch(&cmd, wv, &self.norm_buf, &self.v_buf, g.kv_dim, d.n_embd, 0, 0);
            break :blk &self.v_buf;
        } else &self.k_buf;
        // per-head norms. V plain-normalize (no weight) is issued BEFORE the Q/K
        // norm+rope because v_src may alias k_buf (the un-normed K projection). V
        // is never roped, so it stays a standalone noweight rms_norm.
        const rms_h = RmsPush{ .N = g.head_dim, .eps = d.rms_eps };
        cmd.dispatch(&self.pipes.rms_norm_noweight, .{ g.n_kv_head, 1, 1 }, .{ 256, 1, 1 }, &.{ v_src, &self.v_buf }, &rms_h, @sizeOf(RmsPush), 0);
        // Q/K: per-head rms_norm fused with NEOX RoPE (this layer's inv_freq
        // table, attn_scale 1.0) — one launch each, no normalized round-trip.
        const inv_freq = if (g.is_swa) &self.inv_freq_swa else &self.inv_freq_full;
        const nr = RmsRopePush{ .head_dim = g.head_dim, .eps = d.rms_eps, .rope_dim = g.rope_dim, .position = pos };
        const nr_sh = g.head_dim * @sizeOf(f32);
        cmd.dispatch(&self.pipes.rms_norm_rope, .{ d.n_head, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.q_buf, &wqn.gpu_buffer, inv_freq, &self.q_buf }, &nr, @sizeOf(RmsRopePush), nr_sh);
        cmd.dispatch(&self.pipes.rms_norm_rope, .{ g.n_kv_head, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.k_buf, &wkn.gpu_buffer, inv_freq, &self.k_buf }, &nr, @sizeOf(RmsRopePush), nr_sh);
        // KV cache write at this position
        const kvw = KvWritePush{ .kv_dim = g.kv_dim, .dst_offset = pos * g.kv_dim };
        cmd.dispatch(&self.pipes.kv_cache_write, .{ ceilDiv(g.kv_dim, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.k_buf, &self.kv_k[L], &self.v_buf, &self.kv_v[L] }, &kvw, @sizeOf(KvWritePush), 0);
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
        cmd.dispatch(&self.pipes.rms_norm_residual, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.o_buf, &wpan.gpu_buffer, &self.hidden }, &rms, @sizeOf(RmsPush), 0);
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

        var cmd = try command.beginCommand(ctx);
        const rms = RmsPush{ .N = d.n_embd, .eps = d.rms_eps };
        // pre-ffn norm
        cmd.dispatch(&self.pipes.rms_norm, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.hidden, &wfn.gpu_buffer, &self.ffn_norm_buf }, &rms, @sizeOf(RmsPush), 0);
        // GeGLU FFN: gelu(gate) * up → down
        self.dmmvDispatch(&cmd, wgate, &self.ffn_norm_buf, &self.gate_buf, d.n_ff, d.n_embd, 0, 0);
        self.dmmvDispatch(&cmd, wup, &self.ffn_norm_buf, &self.up_buf, d.n_ff, d.n_embd, 0, 0);
        const sg = SwigluPush{ .N = d.n_ff };
        cmd.dispatch(&self.pipes.geglu, .{ ceilDiv(d.n_ff, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.gate_buf, &self.up_buf, &self.geglu_buf }, &sg, @sizeOf(SwigluPush), 0);
        self.dmmvDispatch(&cmd, wdown, &self.geglu_buf, &self.down_buf, d.n_embd, d.n_ff, 0, 0);
        // post-ffn norm (gemma rms) on the FFN output, fused with the residual add
        // into `hidden` (scale 1.0) — one launch, no down_buf round-trip.
        cmd.dispatch(&self.pipes.rms_norm_residual, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.down_buf, &wpfn.gpu_buffer, &self.hidden }, &rms, @sizeOf(RmsPush), 0);
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
            cmd.dispatch(&self.pipes.moe_weighted_acc, .{ ceilDiv(d.n_embd, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.moe_out_buf, &self.down_buf, &self.router_out_buf }, &ma, @sizeOf(MoeAccPush), 0);
            cmd.dispatch(&self.pipes.rms_norm, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.moe_out_buf, &wpn2.gpu_buffer, &self.moe_out_buf }, &rms, @sizeOf(RmsPush), 0);
            cmd.commitAndWait();
        }

        // --- combine: cur = post_ffw_norm(shared + moe); hidden += cur. ------
        {
            var cmd = try command.beginCommand(ctx);
            const acc = ScaleAccPush{ .N = d.n_embd, .scale = 1.0 };
            cmd.dispatch(&self.pipes.scale_accumulate, .{ ceilDiv(d.n_embd, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.shared_buf, &self.moe_out_buf }, &acc, @sizeOf(ScaleAccPush), 0);
            cmd.dispatch(&self.pipes.rms_norm, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.shared_buf, &wpost.gpu_buffer, &self.shared_buf }, &rms, @sizeOf(RmsPush), 0);
            cmd.dispatch(&self.pipes.scale_accumulate, .{ ceilDiv(d.n_embd, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.hidden, &self.shared_buf }, &acc, @sizeOf(ScaleAccPush), 0);
            cmd.commitAndWait();
        }
    }

    /// Multiply the residual stream by the learned per-layer output scale.
    fn layerOutScale(self: *ForwardGemma, L: u32) !void {
        const d = self.d;
        const ctx = self.ctx;
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
