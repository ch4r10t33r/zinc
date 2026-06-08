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
const SwigluPush = extern struct { N: u32 };
const ScaleAccPush = extern struct { N: u32, scale: f32 };
const ScalarMulPush = extern struct { N: u32 };
const ArgmaxPush = extern struct { N: u32 };

fn dmmvIdx(t: gguf.GGMLType) usize {
    return switch (t) {
        .q4_k => 0,
        .q5_k => 1,
        .q6_k => 2,
        .q8_0 => 3,
        .f32 => 4,
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
};

const Pipelines = struct {
    rms_norm: CudaPipeline,
    rms_norm_noweight: CudaPipeline,
    dmmv: [5]CudaPipeline,
    dmmv_fast: [4]CudaPipeline,
    rope: CudaPipeline,
    kv_cache_write: CudaPipeline,
    gemma_attention: CudaPipeline,
    geglu: CudaPipeline,
    scale_accumulate: CudaPipeline,
    scalar_mul: CudaPipeline,
    argmax: CudaPipeline,
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
    gate_buf: CudaBuffer, // [n_ff]
    up_buf: CudaBuffer, // [n_ff]
    geglu_buf: CudaBuffer, // [n_ff]
    down_buf: CudaBuffer, // [n_embd]
    logits_buf: CudaBuffer,
    argmax_buf: CudaBuffer,
    host_embed: []f32,

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
        };

        // ---- compile kernels -----------------------------------------------
        const src = try allocator.dupeZ(u8, KERNELS_CU);
        defer allocator.free(src);
        var pipes: Pipelines = undefined;
        pipes.rms_norm = try pipeline.createPipeline(ctx, src.ptr, "rms_norm");
        pipes.rms_norm_noweight = try pipeline.createPipeline(ctx, src.ptr, "rms_norm_noweight");
        pipes.dmmv[0] = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q4k");
        pipes.dmmv[1] = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q5k");
        pipes.dmmv[2] = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q6k");
        pipes.dmmv[3] = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q8_0");
        pipes.dmmv[4] = try pipeline.createPipeline(ctx, src.ptr, "dmmv_f32");
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
            .gate_buf = try buffer.createBuffer(ctx, d.n_ff * f4),
            .up_buf = try buffer.createBuffer(ctx, d.n_ff * f4),
            .geglu_buf = try buffer.createBuffer(ctx, d.n_ff * f4),
            .down_buf = try buffer.createBuffer(ctx, d.n_embd * f4),
            .logits_buf = try buffer.createBuffer(ctx, d.vocab * f4),
            .argmax_buf = try buffer.createBuffer(ctx, @sizeOf(u32)),
            .host_embed = try allocator.alloc(f32, d.n_embd),
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
            // rope_freqs.weight is a global F32 [head_dim_full/2] proportional-rope
            // factor table; download it from the GPU copy.
            const rf = try allocator.alloc(f32, half);
            defer allocator.free(rf);
            @memset(rf, 1.0);
            if (model.get("rope_freqs.weight")) |t| {
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

        return self;
    }

    pub fn deinit(self: *ForwardGemma) void {
        const a = self.allocator;
        inline for (.{ &self.hidden, &self.norm_buf, &self.q_buf, &self.k_buf, &self.v_buf, &self.attn_out_buf, &self.o_buf, &self.ffn_norm_buf, &self.gate_buf, &self.up_buf, &self.geglu_buf, &self.down_buf, &self.logits_buf, &self.argmax_buf, &self.inv_freq_swa, &self.inv_freq_full }) |b| {
            buffer.freeBuffer(b);
        }
        for (self.kv_k) |*b| buffer.freeBuffer(b);
        for (self.kv_v) |*b| buffer.freeBuffer(b);
        a.free(self.kv_k);
        a.free(self.kv_v);
        a.free(self.geom);
        a.free(self.host_embed);
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
                try self.ffnBlock(L);
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
        cmd.commitAndWait();

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
        // per-head norms. V plain-normalize (no weight) is issued BEFORE the K
        // norm because v_src may alias k_buf (the un-normed K projection). V is
        // never roped.
        const rms_h = RmsPush{ .N = g.head_dim, .eps = d.rms_eps };
        cmd.dispatch(&self.pipes.rms_norm, .{ d.n_head, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.q_buf, &wqn.gpu_buffer, &self.q_buf }, &rms_h, @sizeOf(RmsPush), 0);
        cmd.dispatch(&self.pipes.rms_norm_noweight, .{ g.n_kv_head, 1, 1 }, .{ 256, 1, 1 }, &.{ v_src, &self.v_buf }, &rms_h, @sizeOf(RmsPush), 0);
        cmd.dispatch(&self.pipes.rms_norm, .{ g.n_kv_head, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.k_buf, &wkn.gpu_buffer, &self.k_buf }, &rms_h, @sizeOf(RmsPush), 0);
        // RoPE q/k (NEOX) using this layer's effective inv_freq table.
        const inv_freq = if (g.is_swa) &self.inv_freq_swa else &self.inv_freq_full;
        const rope_q = RopePush{ .stride = g.head_dim, .rope_dim = g.rope_dim, .n_heads = d.n_head, .position = pos, .freq_base_bits = 0, .attn_scale_bits = 0 };
        cmd.dispatch(&self.pipes.rope, .{ d.n_head, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.q_buf, &self.q_buf, inv_freq }, &rope_q, @sizeOf(RopePush), 0);
        const rope_k = RopePush{ .stride = g.head_dim, .rope_dim = g.rope_dim, .n_heads = g.n_kv_head, .position = pos, .freq_base_bits = 0, .attn_scale_bits = 0 };
        cmd.dispatch(&self.pipes.rope, .{ g.n_kv_head, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.k_buf, &self.k_buf, inv_freq }, &rope_k, @sizeOf(RopePush), 0);
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
        // post-attention norm (gemma rms) on the attention output, then residual add.
        cmd.dispatch(&self.pipes.rms_norm, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.o_buf, &wpan.gpu_buffer, &self.o_buf }, &rms, @sizeOf(RmsPush), 0);
        const acc = ScaleAccPush{ .N = d.n_embd, .scale = 1.0 };
        cmd.dispatch(&self.pipes.scale_accumulate, .{ ceilDiv(d.n_embd, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.hidden, &self.o_buf }, &acc, @sizeOf(ScaleAccPush), 0);
        cmd.commitAndWait();
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
        // post-ffn norm (gemma rms) on the FFN output, then residual add.
        cmd.dispatch(&self.pipes.rms_norm, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.down_buf, &wpfn.gpu_buffer, &self.down_buf }, &rms, @sizeOf(RmsPush), 0);
        const acc = ScaleAccPush{ .N = d.n_embd, .scale = 1.0 };
        cmd.dispatch(&self.pipes.scale_accumulate, .{ ceilDiv(d.n_embd, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.hidden, &self.down_buf }, &acc, @sizeOf(ScaleAccPush), 0);
        cmd.commitAndWait();
    }

    /// Multiply the residual stream by the learned per-layer output scale.
    fn layerOutScale(self: *ForwardGemma, L: u32) !void {
        const d = self.d;
        const ctx = self.ctx;
        const ws = self.model.getLayer(L, "layer_output_scale.weight") orelse return; // optional
        var cmd = try command.beginCommand(ctx);
        const sm = ScalarMulPush{ .N = d.n_embd };
        cmd.dispatch(&self.pipes.scalar_mul, .{ ceilDiv(d.n_embd, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.hidden, &ws.gpu_buffer }, &sm, @sizeOf(ScalarMulPush), 0);
        cmd.commitAndWait();
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
};

fn ceilDiv(a: u32, b: u32) u32 {
    return (a + b - 1) / b;
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
