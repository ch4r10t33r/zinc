//! CUDA forward pass for the dense `qwen35` hybrid-SSM model (Qwen 3.5 9B).
//!
//! M2 bring-up — incremental greedy decode of a single token on NVIDIA. The CUDA
//! backend modules (device/buffer/pipeline/command) and the kernel library
//! (`src/shaders/cuda/kernels.cu`, NVRTC-compiled) are orchestrated here into a
//! real forward pass. Weights upload VERBATIM-quantized (Q4_K/Q5_K/Q6_K/Q8_0
//! blocks) via loader_cuda and are dequantized inside the DMMV kernels.
//!
//! Layer schedule (qwen35, full_attention_interval=4): layer L is full attention
//! when ((L+1) % 4 == 0) → L in {3,7,11,15,19,23,27,31}; the other 24 layers are
//! gated-delta-net SSM. Every layer is followed by a SwiGLU FFN block.
//!
//! @section Inference Runtime
const std = @import("std");
const buffer = @import("../cuda/buffer.zig");
const pipeline = @import("../cuda/pipeline.zig");
const command = @import("../cuda/command.zig");
const shim = @import("../cuda/c.zig").shim;
const gguf = @import("../model/gguf.zig");
const loader = @import("../model/loader_cuda.zig");

const log = std.log.scoped(.cuda_fwd);
const CudaBuffer = buffer.CudaBuffer;
const CudaPipeline = pipeline.CudaPipeline;
const LoadedTensor = loader.LoadedTensor;

/// The CUDA kernel library, bundled into the binary and NVRTC-compiled on load.
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
const AttnPush = extern struct {
    head_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    seq_len: u32,
    attn_scale_bits: u32,
    sink_offset: u32,
};
const KvWritePush = extern struct { kv_dim: u32, dst_offset: u32 };
const ConvPush = extern struct {
    conv_channels: u32,
    d_conv: u32,
    kernel_is_f16: u32,
    state_offset: u32,
};
const DeltaNetPush = extern struct {
    d_inner: u32,
    dt_rank: u32,
    head_v_dim: u32,
    d_state: u32,
    n_group: u32,
    ssm_a_is_f16: u32,
    dt_bias_is_f16: u32,
    has_dt_bias: u32,
    has_ssm_a: u32,
    n_tok: u32,
    conv_stride_tok: u32,
    ab_stride_tok: u32,
    y_stride_tok: u32,
};
const GatedNormPush = extern struct {
    d_inner: u32,
    dt_rank: u32,
    head_v_dim: u32,
    d_state: u32,
    norm_per_head: u32,
};
const SwigluPush = extern struct { N: u32 };
const SigmoidMulPush = extern struct { N: u32 };
const DeintPush = extern struct { head_dim: u32, n_head: u32 };
const ArgmaxPush = extern struct { N: u32 };

/// Map a GGUF quant type to its DMMV kernel pipeline (indexes ForwardCuda.dmmv).
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

/// Derived dims (one decode token, qwen35).
const Derived = struct {
    n_embd: u32,
    n_ff: u32,
    n_head: u32,
    n_kv_head: u32,
    head_dim: u32,
    q_dim: u32, // n_head * head_dim = 4096
    kv_dim: u32, // n_kv_head * head_dim = 1024
    vocab: u32,
    rms_eps: f32,
    n_layers: u32,
    full_attn_interval: u32,
    rope_dim: u32,
    rope_freq_base: f32,
    // SSM
    d_conv: u32,
    d_inner: u32,
    d_state: u32,
    dt_rank: u32,
    n_group: u32,
    head_v_dim: u32, // d_inner / dt_rank = 128
    conv_channels: u32, // d_inner + 2*n_group*d_state = 8192
    conv_state_len: u32, // (d_conv-1) * conv_channels per ssm layer
    ssm_state_len: u32, // dt_rank * head_v_dim * head_v_dim per ssm layer

    fn from(c: anytype) Derived {
        const head_dim = c.head_dim;
        const q_dim = c.n_heads * head_dim;
        const kv_dim = c.n_kv_heads * head_dim;
        const head_v_dim = c.ssm_d_inner / c.ssm_dt_rank;
        const conv_channels = c.ssm_d_inner + 2 * c.ssm_n_group * c.ssm_d_state;
        return .{
            .n_embd = c.hidden_dim,
            .n_ff = c.intermediate_dim,
            .n_head = c.n_heads,
            .n_kv_head = c.n_kv_heads,
            .head_dim = head_dim,
            .q_dim = q_dim,
            .kv_dim = kv_dim,
            .vocab = c.vocab_size,
            .rms_eps = c.rms_norm_eps,
            .n_layers = c.n_layers,
            .full_attn_interval = c.full_attn_interval,
            .rope_dim = c.rope_dim,
            .rope_freq_base = c.rope_freq_base,
            .d_conv = c.ssm_d_conv,
            .d_inner = c.ssm_d_inner,
            .d_state = c.ssm_d_state,
            .dt_rank = c.ssm_dt_rank,
            .n_group = c.ssm_n_group,
            .head_v_dim = head_v_dim,
            .conv_channels = conv_channels,
            .conv_state_len = (c.ssm_d_conv - 1) * conv_channels,
            .ssm_state_len = c.ssm_dt_rank * head_v_dim * head_v_dim,
        };
    }
};

/// All NVRTC-compiled kernels used by the decode path.
const Pipelines = struct {
    rms_norm: CudaPipeline,
    dmmv: [5]CudaPipeline, // q4k, q5k, q6k, q8_0, f32
    rope: CudaPipeline,
    kv_cache_write: CudaPipeline,
    naive_attention: CudaPipeline,
    sigmoid_mul: CudaPipeline,
    deinterleave: CudaPipeline,
    ssm_conv1d: CudaPipeline,
    ssm_delta_net: CudaPipeline,
    ssm_gated_norm: CudaPipeline,
    swiglu: CudaPipeline,
    argmax: CudaPipeline,
};

/// Per-token GPU forward state for qwen35 greedy decode.
pub const ForwardCuda = struct {
    allocator: std.mem.Allocator,
    ctx: ?*shim.CudaCtx,
    model: *loader.Model,
    d: Derived,
    max_ctx: u32,

    pipes: Pipelines,

    // activation scratch (device, f32)
    hidden: CudaBuffer,
    norm_buf: CudaBuffer,
    qfull_buf: CudaBuffer, // [2*q_dim] packed [Q|gate] interleaved per head (attn)
    q_buf: CudaBuffer,
    k_buf: CudaBuffer,
    v_buf: CudaBuffer,
    gate_buf: CudaBuffer, // max(n_embd, n_ff) for FFN gate / attn-z gate
    attn_out_buf: CudaBuffer, // q_dim for attn; conv_channels for SSM qkv
    ffn_norm_buf: CudaBuffer,
    up_buf: CudaBuffer,
    swiglu_buf: CudaBuffer,
    router_buf: CudaBuffer, // ssm alpha [dt_rank]
    down_buf: CudaBuffer, // ssm beta [dt_rank]
    logits_buf: CudaBuffer,
    argmax_buf: CudaBuffer, // u32 x1
    host_embed: []f32,

    // constant buffers
    inv_freq: CudaBuffer, // [rope_dim/2]
    sinks: CudaBuffer, // [n_layers*n_head] NaN

    // KV cache per attention layer (indexed by layer; only attn layers populated)
    kv_k: []CudaBuffer,
    kv_v: []CudaBuffer,
    // SSM state per layer (only ssm layers populated)
    ssm_conv_state: []CudaBuffer,
    ssm_state: []CudaBuffer,
    conv_off: []u32, // per-layer circular conv state offset

    /// Compile kernels, allocate every device buffer, upload inv_freq + sinks,
    /// zero the KV cache and SSM state.
    pub fn init(allocator: std.mem.Allocator, model: *loader.Model, max_ctx: u32) !ForwardCuda {
        const ctx = model.ctx;
        const d = Derived.from(model.config);

        const src = try allocator.dupeZ(u8, KERNELS_CU);
        defer allocator.free(src);

        var pipes: Pipelines = undefined;
        pipes.rms_norm = try pipeline.createPipeline(ctx, src.ptr, "rms_norm");
        pipes.dmmv[0] = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q4k");
        pipes.dmmv[1] = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q5k");
        pipes.dmmv[2] = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q6k");
        pipes.dmmv[3] = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q8_0");
        pipes.dmmv[4] = try pipeline.createPipeline(ctx, src.ptr, "dmmv_f32");
        pipes.rope = try pipeline.createPipeline(ctx, src.ptr, "rope");
        pipes.kv_cache_write = try pipeline.createPipeline(ctx, src.ptr, "kv_cache_write");
        pipes.naive_attention = try pipeline.createPipeline(ctx, src.ptr, "naive_attention");
        pipes.sigmoid_mul = try pipeline.createPipeline(ctx, src.ptr, "sigmoid_mul");
        pipes.deinterleave = try pipeline.createPipeline(ctx, src.ptr, "deinterleave_qgate");
        pipes.ssm_conv1d = try pipeline.createPipeline(ctx, src.ptr, "ssm_conv1d");
        pipes.ssm_delta_net = try pipeline.createPipeline(ctx, src.ptr, "ssm_delta_net");
        pipes.ssm_gated_norm = try pipeline.createPipeline(ctx, src.ptr, "ssm_gated_norm");
        pipes.swiglu = try pipeline.createPipeline(ctx, src.ptr, "swiglu");
        pipes.argmax = try pipeline.createPipeline(ctx, src.ptr, "argmax");
        log.info("nvrtc: compiled {d} kernel pipelines", .{16});

        const f4 = @sizeOf(f32);
        const max_act = @max(d.n_ff, d.conv_channels); // 12288 vs 8192 → 12288
        var self = ForwardCuda{
            .allocator = allocator,
            .ctx = ctx,
            .model = model,
            .d = d,
            .max_ctx = max_ctx,
            .pipes = pipes,
            .hidden = try buffer.createBuffer(ctx, d.n_embd * f4),
            .norm_buf = try buffer.createBuffer(ctx, d.n_embd * f4),
            .qfull_buf = try buffer.createBuffer(ctx, 2 * d.q_dim * f4),
            .q_buf = try buffer.createBuffer(ctx, d.q_dim * f4),
            .k_buf = try buffer.createBuffer(ctx, d.kv_dim * f4),
            .v_buf = try buffer.createBuffer(ctx, d.kv_dim * f4),
            .gate_buf = try buffer.createBuffer(ctx, max_act * f4),
            .attn_out_buf = try buffer.createBuffer(ctx, d.conv_channels * f4),
            .ffn_norm_buf = try buffer.createBuffer(ctx, d.n_embd * f4),
            .up_buf = try buffer.createBuffer(ctx, d.n_ff * f4),
            .swiglu_buf = try buffer.createBuffer(ctx, max_act * f4),
            .router_buf = try buffer.createBuffer(ctx, 64 * f4),
            .down_buf = try buffer.createBuffer(ctx, 64 * f4),
            .logits_buf = try buffer.createBuffer(ctx, d.vocab * f4),
            .argmax_buf = try buffer.createBuffer(ctx, @sizeOf(u32)),
            .host_embed = try allocator.alloc(f32, d.n_embd),
            .inv_freq = try buffer.createBuffer(ctx, (d.rope_dim / 2) * f4),
            .sinks = try buffer.createBuffer(ctx, d.n_layers * d.n_head * f4),
            .kv_k = try allocator.alloc(CudaBuffer, d.n_layers),
            .kv_v = try allocator.alloc(CudaBuffer, d.n_layers),
            .ssm_conv_state = try allocator.alloc(CudaBuffer, d.n_layers),
            .ssm_state = try allocator.alloc(CudaBuffer, d.n_layers),
            .conv_off = try allocator.alloc(u32, d.n_layers),
        };

        // Per-layer KV / SSM state, sized only for the layer type. We still
        // allocate a tiny stub for the "wrong" type so deinit is uniform.
        for (0..d.n_layers) |li| {
            const L: u32 = @intCast(li);
            self.conv_off[li] = 0;
            if (isFullAttn(L, d.full_attn_interval)) {
                self.kv_k[li] = try buffer.createBuffer(ctx, max_ctx * d.kv_dim * f4);
                self.kv_v[li] = try buffer.createBuffer(ctx, max_ctx * d.kv_dim * f4);
                self.ssm_conv_state[li] = try buffer.createBuffer(ctx, f4);
                self.ssm_state[li] = try buffer.createBuffer(ctx, f4);
            } else {
                self.kv_k[li] = try buffer.createBuffer(ctx, f4);
                self.kv_v[li] = try buffer.createBuffer(ctx, f4);
                self.ssm_conv_state[li] = try buffer.createBuffer(ctx, d.conv_state_len * f4);
                self.ssm_state[li] = try buffer.createBuffer(ctx, d.ssm_state_len * f4);
                // zero-init the SSM state
                try zeroBuffer(allocator, ctx, &self.ssm_conv_state[li], d.conv_state_len);
                try zeroBuffer(allocator, ctx, &self.ssm_state[li], d.ssm_state_len);
            }
        }

        // inv_freq[k] = 1 / (freq_base ^ (2k / rope_dim)), k = 0 .. rope_dim/2-1
        {
            const half = d.rope_dim / 2;
            const hf = try allocator.alloc(f32, half);
            defer allocator.free(hf);
            for (0..half) |k| {
                const exp = @as(f32, @floatFromInt(2 * k)) / @as(f32, @floatFromInt(d.rope_dim));
                hf[k] = 1.0 / std.math.pow(f32, d.rope_freq_base, exp);
            }
            buffer.upload(ctx, &self.inv_freq, std.mem.sliceAsBytes(hf));
        }

        // sinks: all NaN → attention sink disabled.
        {
            const n = d.n_layers * d.n_head;
            const sk = try allocator.alloc(f32, n);
            defer allocator.free(sk);
            @memset(sk, std.math.nan(f32));
            buffer.upload(ctx, &self.sinks, std.mem.sliceAsBytes(sk));
        }

        return self;
    }

    pub fn deinit(self: *ForwardCuda) void {
        const a = self.allocator;
        inline for (.{ &self.hidden, &self.norm_buf, &self.qfull_buf, &self.q_buf, &self.k_buf, &self.v_buf, &self.gate_buf, &self.attn_out_buf, &self.ffn_norm_buf, &self.up_buf, &self.swiglu_buf, &self.router_buf, &self.down_buf, &self.logits_buf, &self.argmax_buf, &self.inv_freq, &self.sinks }) |b| {
            buffer.freeBuffer(b);
        }
        for (self.kv_k) |*b| buffer.freeBuffer(b);
        for (self.kv_v) |*b| buffer.freeBuffer(b);
        for (self.ssm_conv_state) |*b| buffer.freeBuffer(b);
        for (self.ssm_state) |*b| buffer.freeBuffer(b);
        a.free(self.kv_k);
        a.free(self.kv_v);
        a.free(self.ssm_conv_state);
        a.free(self.ssm_state);
        a.free(self.conv_off);
        a.free(self.host_embed);
        inline for (std.meta.fields(Pipelines)) |f| {
            if (comptime std.mem.eql(u8, f.name, "dmmv")) {
                for (&self.pipes.dmmv) |*p| pipeline.freePipeline(p);
            } else {
                pipeline.freePipeline(&@field(self.pipes, f.name));
            }
        }
        self.* = undefined;
    }

    /// Run one greedy decode step for `token` at sequence position `pos`,
    /// returning the argmax token id. v0: embed → final rms_norm → LM head →
    /// argmax (layers are gated by `run_layers`).
    pub fn decodeStep(self: *ForwardCuda, token: u32, pos: u32, run_layers: bool) !u32 {
        const d = self.d;
        const ctx = self.ctx;

        // EMBED: CPU-dequant the token_embd row, upload to hidden.
        self.model.dequantEmbeddingRow(token, self.host_embed);
        buffer.upload(ctx, &self.hidden, std.mem.sliceAsBytes(self.host_embed));

        if (run_layers) {
            var L: u32 = 0;
            while (L < d.n_layers) : (L += 1) {
                if (isFullAttn(L, d.full_attn_interval)) {
                    try self.attentionLayer(L, pos);
                } else {
                    try self.ssmLayer(L);
                }
                try self.ffnBlock(L);
            }
        }

        // TAIL: final rms_norm → LM head → argmax.
        const out_norm = self.model.get("output_norm.weight") orelse return error.MissingTensor;
        const lm_head = self.model.get("output.weight") orelse return error.MissingTensor;

        var cmd = try command.beginCommand(ctx);
        const rms = RmsPush{ .N = d.n_embd, .eps = d.rms_eps };
        cmd.dispatch(&self.pipes.rms_norm, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.hidden, &out_norm.gpu_buffer, &self.norm_buf }, &rms, @sizeOf(RmsPush), 0);
        const lm = DmmvPush{ .M = d.vocab, .K = d.n_embd };
        cmd.dispatch(&self.pipes.dmmv[dmmvIdx(lm_head.info.type_)], .{ d.vocab, 1, 1 }, .{ 256, 1, 1 }, &.{ &lm_head.gpu_buffer, &self.norm_buf, &self.logits_buf }, &lm, @sizeOf(DmmvPush), 0);
        const am = ArgmaxPush{ .N = d.vocab };
        cmd.dispatch(&self.pipes.argmax, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.logits_buf, &self.argmax_buf }, &am, @sizeOf(ArgmaxPush), 0);
        cmd.commitAndWait();

        var tok: u32 = 0;
        buffer.download(ctx, &self.argmax_buf, std.mem.asBytes(&tok));
        return tok;
    }

    // ---- per-block builders -------------------------------------------------

    fn attentionLayer(self: *ForwardCuda, L: u32, pos: u32) !void {
        const d = self.d;
        const ctx = self.ctx;
        const wq = self.layer(L, "attn_q.weight"); // packed [Q | gate], M = 2*q_dim
        const wk = self.layer(L, "attn_k.weight");
        const wv = self.layer(L, "attn_v.weight");
        const wqn = self.layer(L, "attn_q_norm.weight");
        const wkn = self.layer(L, "attn_k_norm.weight");
        const wo = self.layer(L, "attn_output.weight");
        const wan = self.layer(L, "attn_norm.weight");

        var cmd = try command.beginCommand(ctx);
        const rms = RmsPush{ .N = d.n_embd, .eps = d.rms_eps };
        cmd.dispatch(&self.pipes.rms_norm, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.hidden, &wan.gpu_buffer, &self.norm_buf }, &rms, @sizeOf(RmsPush), 0);
        // Q+gate, K, V projections from norm_buf.
        // qwen35 packs the attention gate INTO attn_q: wq outputs 2*q_dim, laid out
        // per head as [Q(head_dim) | gate(head_dim)] interleaved across heads
        // ([Q0,g0,Q1,g1,...]). Project the full 2*q_dim, then deinterleave into
        // contiguous q_buf (the Q halves) and gate_buf (the gate halves).
        self.dmmvDispatch(&cmd, wq, &self.norm_buf, &self.qfull_buf, 2 * d.q_dim, d.n_embd, 0);
        const deint = DeintPush{ .head_dim = d.head_dim, .n_head = d.n_head };
        cmd.dispatch(&self.pipes.deinterleave, .{ ceilDiv(d.q_dim, 256), 1, 1 }, .{ 256, 1, 1 }, &.{ &self.qfull_buf, &self.q_buf, &self.gate_buf }, &deint, @sizeOf(DeintPush), 0);
        self.dmmvDispatch(&cmd, wk, &self.norm_buf, &self.k_buf, d.kv_dim, d.n_embd, 0);
        self.dmmvDispatch(&cmd, wv, &self.norm_buf, &self.v_buf, d.kv_dim, d.n_embd, 0);
        // per-head q/k rms norm
        const rms_h = RmsPush{ .N = d.head_dim, .eps = d.rms_eps };
        cmd.dispatch(&self.pipes.rms_norm, .{ d.n_head, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.q_buf, &wqn.gpu_buffer, &self.q_buf }, &rms_h, @sizeOf(RmsPush), 0);
        cmd.dispatch(&self.pipes.rms_norm, .{ d.n_kv_head, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.k_buf, &wkn.gpu_buffer, &self.k_buf }, &rms_h, @sizeOf(RmsPush), 0);
        // RoPE q/k (inv_freq buffer path: freq_base_bits=0)
        const rope_q = RopePush{ .stride = d.head_dim, .rope_dim = d.rope_dim, .n_heads = d.n_head, .position = pos, .freq_base_bits = 0, .attn_scale_bits = 0 };
        cmd.dispatch(&self.pipes.rope, .{ d.n_head, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.q_buf, &self.q_buf, &self.inv_freq }, &rope_q, @sizeOf(RopePush), 0);
        const rope_k = RopePush{ .stride = d.head_dim, .rope_dim = d.rope_dim, .n_heads = d.n_kv_head, .position = pos, .freq_base_bits = 0, .attn_scale_bits = 0 };
        cmd.dispatch(&self.pipes.rope, .{ d.n_kv_head, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.k_buf, &self.k_buf, &self.inv_freq }, &rope_k, @sizeOf(RopePush), 0);
        // KV cache write at this position
        const kvw = KvWritePush{ .kv_dim = d.kv_dim, .dst_offset = pos * d.kv_dim };
        const kv_grid = ceilDiv(d.kv_dim, 64);
        cmd.dispatch(&self.pipes.kv_cache_write, .{ kv_grid, 1, 1 }, .{ 64, 1, 1 }, &.{ &self.k_buf, &self.kv_k[L], &self.v_buf, &self.kv_v[L] }, &kvw, @sizeOf(KvWritePush), 0);
        // attention: out → attn_out_buf
        const seq_len = pos + 1;
        const attn = AttnPush{ .head_dim = d.head_dim, .n_heads = d.n_head, .n_kv_heads = d.n_kv_head, .seq_len = seq_len, .attn_scale_bits = 0, .sink_offset = L * d.n_head };
        cmd.dispatch(&self.pipes.naive_attention, .{ d.n_head, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.q_buf, &self.kv_k[L], &self.kv_v[L], &self.sinks, &self.attn_out_buf }, &attn, @sizeOf(AttnPush), seq_len * 4);
        // gate: attn_out *= sigmoid(gate)
        const sm = SigmoidMulPush{ .N = d.q_dim };
        cmd.dispatch(&self.pipes.sigmoid_mul, .{ ceilDiv(d.q_dim, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.attn_out_buf, &self.gate_buf, &self.attn_out_buf }, &sm, @sizeOf(SigmoidMulPush), 0);
        // O projection, accumulate into hidden
        self.dmmvDispatch(&cmd, wo, &self.attn_out_buf, &self.hidden, d.n_embd, d.q_dim, 1);
        cmd.commitAndWait();
    }

    fn ssmLayer(self: *ForwardCuda, L: u32) !void {
        const d = self.d;
        const ctx = self.ctx;
        const wan = self.layer(L, "attn_norm.weight");
        const wqkv = self.layer(L, "attn_qkv.weight");
        const wz = self.layer(L, "attn_gate.weight");
        const walpha = self.layer(L, "ssm_alpha.weight");
        const wbeta = self.layer(L, "ssm_beta.weight");
        const wconv = self.layer(L, "ssm_conv1d.weight");
        const wdt = self.layer(L, "ssm_dt.bias");
        const wa = self.layer(L, "ssm_a");
        const wnorm = self.layer(L, "ssm_norm.weight");
        const wout = self.layer(L, "ssm_out.weight");

        var cmd = try command.beginCommand(ctx);
        const rms = RmsPush{ .N = d.n_embd, .eps = d.rms_eps };
        cmd.dispatch(&self.pipes.rms_norm, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.hidden, &wan.gpu_buffer, &self.norm_buf }, &rms, @sizeOf(RmsPush), 0);
        // qkv (conv_channels) and z-gate (d_inner)
        self.dmmvDispatch(&cmd, wqkv, &self.norm_buf, &self.attn_out_buf, d.conv_channels, d.n_embd, 0);
        self.dmmvDispatch(&cmd, wz, &self.norm_buf, &self.gate_buf, d.d_inner, d.n_embd, 0);
        // alpha, beta (dt_rank)
        self.dmmvDispatch(&cmd, walpha, &self.norm_buf, &self.router_buf, d.dt_rank, d.n_embd, 0);
        self.dmmvDispatch(&cmd, wbeta, &self.norm_buf, &self.down_buf, d.dt_rank, d.n_embd, 0);
        // conv1d (+ SiLU) over the qkv stream → swiglu_buf (conv_out)
        const conv = ConvPush{ .conv_channels = d.conv_channels, .d_conv = d.d_conv, .kernel_is_f16 = boolU32(wconv.info.type_ == .f16), .state_offset = self.conv_off[L] };
        cmd.dispatch(&self.pipes.ssm_conv1d, .{ ceilDiv(d.conv_channels, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.attn_out_buf, &wconv.gpu_buffer, &self.ssm_conv_state[L], &self.swiglu_buf }, &conv, @sizeOf(ConvPush), 0);
        // delta-net scan: conv_out + dt_bias + alpha + beta + ssm_a + state → attn_out_buf (delta_out)
        const dn = DeltaNetPush{
            .d_inner = d.d_inner,
            .dt_rank = d.dt_rank,
            .head_v_dim = d.head_v_dim,
            .d_state = d.d_state,
            .n_group = d.n_group,
            .ssm_a_is_f16 = boolU32(wa.info.type_ == .f16),
            .dt_bias_is_f16 = boolU32(wdt.info.type_ == .f16),
            .has_dt_bias = 1,
            .has_ssm_a = 1,
            .n_tok = 1,
            .conv_stride_tok = d.conv_channels,
            .ab_stride_tok = d.dt_rank,
            .y_stride_tok = d.d_inner,
        };
        cmd.dispatch(&self.pipes.ssm_delta_net, .{ d.dt_rank, d.head_v_dim, 1 }, .{ d.head_v_dim, 1, 1 }, &.{ &self.swiglu_buf, &wdt.gpu_buffer, &self.router_buf, &self.down_buf, &wa.gpu_buffer, &self.ssm_state[L], &self.attn_out_buf }, &dn, @sizeOf(DeltaNetPush), 0);
        // gated norm: (delta_out, z) → swiglu_buf
        const norm_per_head: u32 = if (wnorm.info.numElements() == d.d_inner) 1 else 0;
        const gn = GatedNormPush{ .d_inner = d.d_inner, .dt_rank = d.dt_rank, .head_v_dim = d.head_v_dim, .d_state = d.d_state, .norm_per_head = norm_per_head };
        cmd.dispatch(&self.pipes.ssm_gated_norm, .{ d.dt_rank, 1, 1 }, .{ d.head_v_dim, 1, 1 }, &.{ &self.attn_out_buf, &self.gate_buf, &wnorm.gpu_buffer, &self.swiglu_buf }, &gn, @sizeOf(GatedNormPush), 0);
        // out projection, accumulate into hidden
        self.dmmvDispatch(&cmd, wout, &self.swiglu_buf, &self.hidden, d.n_embd, d.d_inner, 1);
        cmd.commitAndWait();

        // advance circular conv offset (host), AFTER this token's conv.
        self.conv_off[L] = (self.conv_off[L] + 1) % (d.d_conv - 1);
    }

    fn ffnBlock(self: *ForwardCuda, L: u32) !void {
        const d = self.d;
        const ctx = self.ctx;
        // qwen35 names the FFN norm `post_attention_norm.weight` (not ffn_norm).
        const wfn = self.layer(L, "post_attention_norm.weight");
        const wgate = self.layer(L, "ffn_gate.weight");
        const wup = self.layer(L, "ffn_up.weight");
        const wdown = self.layer(L, "ffn_down.weight");

        var cmd = try command.beginCommand(ctx);
        const rms = RmsPush{ .N = d.n_embd, .eps = d.rms_eps };
        cmd.dispatch(&self.pipes.rms_norm, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.hidden, &wfn.gpu_buffer, &self.ffn_norm_buf }, &rms, @sizeOf(RmsPush), 0);
        self.dmmvDispatch(&cmd, wgate, &self.ffn_norm_buf, &self.gate_buf, d.n_ff, d.n_embd, 0);
        self.dmmvDispatch(&cmd, wup, &self.ffn_norm_buf, &self.up_buf, d.n_ff, d.n_embd, 0);
        const sg = SwigluPush{ .N = d.n_ff };
        cmd.dispatch(&self.pipes.swiglu, .{ ceilDiv(d.n_ff, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.gate_buf, &self.up_buf, &self.swiglu_buf }, &sg, @sizeOf(SwigluPush), 0);
        self.dmmvDispatch(&cmd, wdown, &self.swiglu_buf, &self.hidden, d.n_embd, d.n_ff, 1);
        cmd.commitAndWait();
    }

    // Public wrappers for the per-block builders (used by the v1 driver to run
    // and inspect one block type at a time).
    pub fn attentionLayerPub(self: *ForwardCuda, L: u32, pos: u32) !void {
        return self.attentionLayer(L, pos);
    }
    pub fn ssmLayerPub(self: *ForwardCuda, L: u32) !void {
        return self.ssmLayer(L);
    }
    pub fn ffnBlockPub(self: *ForwardCuda, L: u32) !void {
        return self.ffnBlock(L);
    }

    /// Download the current hidden state (for sanity checks).
    pub fn readHidden(self: *ForwardCuda, out: []f32) void {
        buffer.download(self.ctx, &self.hidden, std.mem.sliceAsBytes(out[0..self.d.n_embd]));
    }

    /// Download the top of the logits buffer (for top-k reporting).
    pub fn readLogits(self: *ForwardCuda, out: []f32) void {
        buffer.download(self.ctx, &self.logits_buf, std.mem.sliceAsBytes(out[0..@min(out.len, self.d.vocab)]));
    }

    // ---- helpers ------------------------------------------------------------

    fn layer(self: *ForwardCuda, L: u32, suffix: []const u8) *const LoadedTensor {
        return self.model.getLayer(L, suffix) orelse {
            log.err("missing tensor blk.{d}.{s}", .{ L, suffix });
            @panic("missing tensor");
        };
    }

    /// Dispatch the right DMMV kernel for `w`'s quant type: y[M] = W[M,K] · x.
    fn dmmvDispatch(self: *ForwardCuda, cmd: *command.CudaCommand, w: *const LoadedTensor, x: *const CudaBuffer, y: *const CudaBuffer, M: u32, K: u32, acc_mode: u32) void {
        const push = DmmvPush{ .M = M, .K = K, .acc_mode = acc_mode };
        cmd.dispatch(&self.pipes.dmmv[dmmvIdx(w.info.type_)], .{ M, 1, 1 }, .{ 256, 1, 1 }, &.{ &w.gpu_buffer, x, y }, &push, @sizeOf(DmmvPush), 0);
    }
};

fn isFullAttn(L: u32, interval: u32) bool {
    return (L + 1) % interval == 0;
}

fn ceilDiv(a: u32, b: u32) u32 {
    return (a + b - 1) / b;
}

fn boolU32(b: bool) u32 {
    return if (b) 1 else 0;
}

/// Zero a device buffer by uploading a host-side zero array of `n` f32.
fn zeroBuffer(allocator: std.mem.Allocator, ctx: ?*shim.CudaCtx, buf: *CudaBuffer, n: u32) !void {
    const z = try allocator.alloc(f32, n);
    defer allocator.free(z);
    @memset(z, 0);
    buffer.upload(ctx, buf, std.mem.sliceAsBytes(z));
}
