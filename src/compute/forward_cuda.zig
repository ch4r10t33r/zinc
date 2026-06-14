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
const EmbedPush = extern struct { K: u32, vocab: u32 };
// MoE router/combine kernels (byte-match kernels.cu).
const TopkPush = extern struct { n_experts: u32, k: u32 };
const MoeAccPush = extern struct { N: u32, n_used: u32, src_stride: u32 };
const SigmoidAccPush = extern struct { N: u32 };
// Batched MoE expert matvec: one launch over all n_used experts, GPU-side ids.
const ExpertsPush = extern struct { M: u32, K: u32, slice: u32, x_stride: u32, n_used: u32, base: u32 = 0 };

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
    // MoE (0 for the dense qwen35; >0 for the qwen2_moe qwen36)
    n_experts: u32, // total routed experts (256)
    n_experts_used: u32, // top-k active experts per token (8)
    shexp_ff: u32, // shared-expert intermediate dim (512)

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
            .n_experts = c.n_experts,
            .n_experts_used = c.n_experts_used,
            .shexp_ff = c.shared_expert_intermediate_dim,
        };
    }
};

/// All NVRTC-compiled kernels used by the decode path.
const Pipelines = struct {
    rms_norm: CudaPipeline,
    dmmv: [5]CudaPipeline, // q4k, q5k, q6k, q8_0, f32
    dmmv_fast: [4]CudaPipeline, // q4k_fast, q5k_fast, q6k_fast, q8_0_fast (block=64)
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
    embed_q4k: CudaPipeline, // GPU-side Q4_K embedding-row dequant (Effort 25 c5)
    // MoE (qwen2_moe). Compiled unconditionally; only dispatched when n_experts>0.
    softmax_topk: CudaPipeline,
    moe_weighted_acc: CudaPipeline,
    sigmoid_scale_acc: CudaPipeline,
    dmmv_q4k_experts: CudaPipeline, // batched gate/up over all experts
    dmmv_q5k_experts: CudaPipeline, // batched down over all experts
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
    down_buf: CudaBuffer, // ssm beta [dt_rank]; MoE: slot-major down outputs [n_used*n_embd]
    logits_buf: CudaBuffer,
    argmax_buf: CudaBuffer, // u32 x1
    tok_in_buf: CudaBuffer, // u32 x1: device token id, source for the GPU embed lookup
    // async decode command ring (dense path): commitAsync'd layer commands are
    // stashed here and freed after the tail commitAndWait drains the shared
    // CUstream. Defaults so the init literal need not list them.
    pending: [1024]command.CudaCommand = undefined,
    n_pending: u32 = 0,
    // CUDA-graph decode replay (Effort 25, behind ZINC_CUDA_GRAPH). When set, the
    // dense per-step kernel chain is stream-captured into a cached CUgraphExec and
    // replayed as ONE graph launch — collapsing the ~480 launch-bound kernel
    // dispatches/token into a single submission. `capturing` is true only while
    // recording, switching `submit`/attention to a no-sync capture path.
    graph: ?*shim.CudaGraph = null,
    capturing: bool = false,
    // GPU-side embedding (Effort 25 cycle 5): when the token_embd tensor is Q4_K,
    // dequant the token's row on-GPU (reading the id from `tok_in_buf`) instead of
    // a per-token CPU dequant + full-row H2D. `embed_weight` points at the resident
    // token_embd.weight device buffer; null/false → fall back to the CPU path.
    embed_weight: ?*const CudaBuffer = null,
    embed_gpu: bool = false,
    // MoE scratch (only used when n_experts > 0)
    router_logits_buf: CudaBuffer, // [n_experts] f32 router logits
    router_out_buf: CudaBuffer, // [2*n_experts_used] u32: ids then weight-bits
    gate_scalar_buf: CudaBuffer, // [1] f32 shared-expert sigmoid gate logit
    host_embed: []f32, // PINNED: async embed H2D source (graph-capturable)
    host_tok: []u32, // PINNED: async argmax D2H dest (graph-capturable), len 1
    host_tok_in: []u32, // PINNED: token-id H2D source for the GPU embed lookup, len 1
    host_router_ids: []u32, // [n_experts_used] downloaded expert ids

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
        // The CUDA forward implements the qwen35/qwen36 hybrid-SSM family (dense +
        // qwen2_moe). Reject other architectures (e.g. gemma4) cleanly here rather
        // than dividing by zero on the absent SSM dims downstream.
        switch (model.config.architecture) {
            .qwen35, .qwen2_moe => {},
            else => |a| {
                log.err("CUDA backend: architecture '{s}' is not implemented (only qwen35/qwen36 family)", .{@tagName(a)});
                return error.UnsupportedArchitecture;
            },
        }
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
        // Fast single-row matvec variants (84-90% bandwidth peak vs ~12-15% base);
        // same DmmvPush ABI + [weight, x, y] buffer order, dispatched at block=64.
        pipes.dmmv_fast[0] = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q4k_fast");
        pipes.dmmv_fast[1] = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q5k_fast");
        pipes.dmmv_fast[2] = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q6k_fast");
        pipes.dmmv_fast[3] = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q8_0_fast");
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
        pipes.embed_q4k = try pipeline.createPipeline(ctx, src.ptr, "embed_lookup_q4k");
        pipes.softmax_topk = try pipeline.createPipeline(ctx, src.ptr, "softmax_topk");
        pipes.moe_weighted_acc = try pipeline.createPipeline(ctx, src.ptr, "moe_weighted_acc");
        pipes.sigmoid_scale_acc = try pipeline.createPipeline(ctx, src.ptr, "sigmoid_scale_acc");
        pipes.dmmv_q4k_experts = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q4k_experts");
        pipes.dmmv_q5k_experts = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q5k_experts");
        log.info("nvrtc: compiled {d} kernel pipelines", .{26});

        const f4 = @sizeOf(f32);
        const max_act = @max(d.n_ff, d.conv_channels); // 12288 vs 8192 → 12288
        // Batched MoE buffers hold all n_used experts' gate/up/swiglu (slot-major).
        const moe_act = @max(max_act, d.n_experts_used * d.n_ff);
        // MoE: down_buf is slot-major [n_used * n_embd]; keep ≥64 for the SSM beta reuse.
        const down_elems = @max(@as(u32, 64), d.n_experts_used * d.n_embd);
        // Tiny-but-nonzero stubs so the dense path (n_experts==0) still allocates/free uniformly.
        const router_logits_elems = @max(@as(u32, 1), d.n_experts);
        const router_out_elems = @max(@as(u32, 1), 2 * d.n_experts_used);
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
            .gate_buf = try buffer.createBuffer(ctx, moe_act * f4),
            .attn_out_buf = try buffer.createBuffer(ctx, d.conv_channels * f4),
            .ffn_norm_buf = try buffer.createBuffer(ctx, d.n_embd * f4),
            .up_buf = try buffer.createBuffer(ctx, moe_act * f4),
            .swiglu_buf = try buffer.createBuffer(ctx, moe_act * f4),
            .router_buf = try buffer.createBuffer(ctx, 64 * f4),
            .down_buf = try buffer.createBuffer(ctx, down_elems * f4),
            .logits_buf = try buffer.createBuffer(ctx, d.vocab * f4),
            .argmax_buf = try buffer.createBuffer(ctx, @sizeOf(u32)),
            .tok_in_buf = try buffer.createBuffer(ctx, @sizeOf(u32)),
            .router_logits_buf = try buffer.createBuffer(ctx, router_logits_elems * f4),
            .router_out_buf = try buffer.createBuffer(ctx, router_out_elems * @sizeOf(u32)),
            .gate_scalar_buf = try buffer.createBuffer(ctx, f4),
            .host_embed = try buffer.allocHost(f32, d.n_embd),
            .host_tok = try buffer.allocHost(u32, 1),
            .host_tok_in = try buffer.allocHost(u32, 1),
            .host_router_ids = try allocator.alloc(u32, @max(@as(u32, 1), d.n_experts_used)),
            .inv_freq = try buffer.createBuffer(ctx, (d.rope_dim / 2) * f4),
            .sinks = try buffer.createBuffer(ctx, d.n_layers * d.n_head * f4),
            .kv_k = try allocator.alloc(CudaBuffer, d.n_layers),
            .kv_v = try allocator.alloc(CudaBuffer, d.n_layers),
            .ssm_conv_state = try allocator.alloc(CudaBuffer, d.n_layers),
            .ssm_state = try allocator.alloc(CudaBuffer, d.n_layers),
            .conv_off = try allocator.alloc(u32, d.n_layers),
        };

        // GPU-side embed: enable only when token_embd.weight is Q4_K and the row
        // length is a whole number of 256-superblocks (always true for these
        // dense qwen models). Other quants (e.g. the 35B MoE's Q8_0) keep the CPU
        // dequant path. The device buffer pointer is stable for the model's life.
        if (model.get("token_embd.weight")) |t| {
            self.embed_weight = &t.gpu_buffer;
            self.embed_gpu = (t.info.type_ == .q4_k) and (d.n_embd % 256 == 0);
        }

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

        // CUDA-graph decode replay (opt-in via ZINC_CUDA_GRAPH). Only the dense
        // path (n_experts==0) is graph-capturable — the MoE router reads expert
        // ids back to the host mid-block, which is illegal during stream capture.
        if (d.n_experts == 0 and std.posix.getenv("ZINC_CUDA_GRAPH") != null) {
            self.graph = shim.cuda_graph_create();
            log.info("ZINC_CUDA_GRAPH: dense decode will replay via a captured CUDA graph", .{});
        }

        return self;
    }

    pub fn deinit(self: *ForwardCuda) void {
        const a = self.allocator;
        if (self.graph) |g| shim.cuda_graph_free(g);
        inline for (.{ &self.hidden, &self.norm_buf, &self.qfull_buf, &self.q_buf, &self.k_buf, &self.v_buf, &self.gate_buf, &self.attn_out_buf, &self.ffn_norm_buf, &self.up_buf, &self.swiglu_buf, &self.router_buf, &self.down_buf, &self.logits_buf, &self.argmax_buf, &self.tok_in_buf, &self.router_logits_buf, &self.router_out_buf, &self.gate_scalar_buf, &self.inv_freq, &self.sinks }) |b| {
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
        buffer.freeHost(self.host_embed);
        buffer.freeHost(self.host_tok);
        buffer.freeHost(self.host_tok_in);
        a.free(self.host_router_ids);
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

    /// Run one greedy decode step for `token` at sequence position `pos`,
    /// returning the argmax token id. v0: embed → final rms_norm → LM head →
    /// argmax (layers are gated by `run_layers`).
    pub fn decodeStep(self: *ForwardCuda, token: u32, pos: u32, run_layers: bool) !u32 {
        const d = self.d;
        const ctx = self.ctx;

        // EMBED. GPU path (Q4_K token_embd): stage the token id into the tiny pinned
        // host_tok_in and dequant its row on-GPU into `hidden` (no per-token CPU
        // dequant; H2D shrinks from a full n_embd row to 4 bytes). CPU fallback:
        // dequant the row into the pinned host_embed and upload it.
        const use_graph = run_layers and self.graph != null;
        if (self.embed_gpu) {
            self.host_tok_in[0] = token;
        } else {
            self.model.dequantEmbeddingRow(token, self.host_embed);
            // Non-graph path uploads (sync) here. The graph path defers the H2D into
            // decodeStepGraph so it is CAPTURED as the graph's first node — pinned.
            if (!use_graph) buffer.upload(ctx, &self.hidden, std.mem.sliceAsBytes(self.host_embed));
        }

        // TAIL tensors (resolved up-front so the graph and async paths share them).
        const out_norm = self.model.get("output_norm.weight") orelse return error.MissingTensor;
        const lm_head = self.model.get("output.weight") orelse return error.MissingTensor;

        // CUDA-graph replay path: capture the full dense per-step chain (embed +
        // layers + tail + argmax D2H) and launch it as one graph. Bit-identical
        // to the async chain (same kernels, same order, same single stream).
        if (use_graph) {
            return try self.decodeStepGraph(pos, &out_norm.gpu_buffer, &lm_head.gpu_buffer, lm_head.info.type_);
        }

        // Non-graph GPU embed: tiny token H2D, then the embed dispatch leads the
        // stream (ordered before the layer ops; drained by the tail commitAndWait).
        if (self.embed_gpu) {
            buffer.upload(ctx, &self.tok_in_buf, std.mem.sliceAsBytes(self.host_tok_in));
            try self.recordEmbed();
        }

        if (run_layers) {
            var L: u32 = 0;
            while (L < d.n_layers) : (L += 1) {
                if (isFullAttn(L, d.full_attn_interval)) {
                    try self.attentionLayer(L, pos);
                } else {
                    try self.ssmLayer(L);
                }
                if (d.n_experts > 0) try self.moeFfnBlock(L) else try self.ffnBlock(L);
            }
        }

        // TAIL: final rms_norm → LM head → argmax.
        var cmd = try command.beginCommand(ctx);
        self.tailDispatch(&cmd, &out_norm.gpu_buffer, &lm_head.gpu_buffer, lm_head.info.type_);
        cmd.commitAndWait(); // drains the whole stream incl. the async layer ops
        self.drainPending(); // free the stashed async commands (completion guaranteed)

        var tok: u32 = 0;
        buffer.download(ctx, &self.argmax_buf, std.mem.asBytes(&tok));
        return tok;
    }

    /// Dense decode step via CUDA-graph replay (Effort 25). `hidden` already holds
    /// the embedded token. Stream-captures the layer chain + tail into the cached
    /// CUgraphExec and launches it as one submission, eliminating the per-kernel
    /// launch + inter-kernel-bubble latency of the ~480-kernel launch-bound chain.
    /// While `capturing`, `submit` and the attention kernel take a no-sync,
    /// shape-invariant path so the captured topology is identical every step
    /// (only per-token push-constant scalars change → cheap in-place exec update).
    fn decodeStepGraph(self: *ForwardCuda, pos: u32, out_norm: *const CudaBuffer, lm_head: *const CudaBuffer, lm_type: gguf.GGMLType) !u32 {
        const d = self.d;
        const ctx = self.ctx;

        self.capturing = true;
        _ = shim.cuda_graph_begin(ctx);
        // EMBED as the graph's first node(s). GPU path: a 4-byte token-id H2D then
        // the embed_lookup dispatch dequants the row into `hidden` on-device (the
        // per-token CPU dequant is gone entirely). CPU fallback: H2D the pre-
        // dequantized pinned host_embed row. Either way it rides the one launch.
        if (self.embed_gpu) {
            buffer.uploadAsync(ctx, &self.tok_in_buf, std.mem.sliceAsBytes(self.host_tok_in));
            try self.recordEmbed();
        } else {
            buffer.uploadAsync(ctx, &self.hidden, std.mem.sliceAsBytes(self.host_embed));
        }
        var L: u32 = 0;
        while (L < d.n_layers) : (L += 1) {
            if (isFullAttn(L, d.full_attn_interval)) {
                try self.attentionLayer(L, pos);
            } else {
                try self.ssmLayer(L);
            }
            try self.ffnBlock(L); // dense only — graph path is gated on n_experts==0
        }
        var cmd = try command.beginCommand(ctx);
        self.tailDispatch(&cmd, out_norm, lm_head, lm_type);
        cmd.releaseCompleted(); // captured onto the stream; no event record / no sync
        // ARGMAX D2H as the graph's last node (pinned host_tok), so the predicted
        // token lands in host memory after the single graph-launch sync — no extra
        // per-token download sync.
        buffer.downloadAsync(ctx, &self.argmax_buf, std.mem.sliceAsBytes(self.host_tok));
        self.capturing = false;

        // End capture, instantiate-or-update the cached exec, launch + ONE sync
        // (drains the embed H2D, layer chain, tail, and argmax D2H together).
        _ = shim.cuda_graph_end_launch(ctx, self.graph.?);

        return self.host_tok[0];
    }

    /// Dispatch the GPU-side embedding lookup (Q4_K): dequant the token's row
    /// (id in `tok_in_buf`) into `hidden`. Routed through `submit` so it captures
    /// cleanly as the graph's first compute node and otherwise rides the async
    /// ring. One block per 256-superblock, 256 threads (one output each).
    fn recordEmbed(self: *ForwardCuda) !void {
        const d = self.d;
        var cmd = try command.beginCommand(self.ctx);
        const push = EmbedPush{ .K = d.n_embd, .vocab = d.vocab };
        const nsb = d.n_embd / 256;
        cmd.dispatch(&self.pipes.embed_q4k, .{ nsb, 1, 1 }, .{ 256, 1, 1 }, &.{ self.embed_weight.?, &self.tok_in_buf, &self.hidden }, &push, @sizeOf(EmbedPush), 0);
        self.submit(cmd);
    }

    /// Record the decode tail (final rms_norm → LM head matvec → argmax) onto
    /// `cmd`. Shared by the async and graph-capture paths.
    fn tailDispatch(self: *ForwardCuda, cmd: *command.CudaCommand, out_norm: *const CudaBuffer, lm_head: *const CudaBuffer, lm_type: gguf.GGMLType) void {
        const d = self.d;
        const rms = RmsPush{ .N = d.n_embd, .eps = d.rms_eps };
        cmd.dispatch(&self.pipes.rms_norm, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.hidden, out_norm, &self.norm_buf }, &rms, @sizeOf(RmsPush), 0);
        const lm = DmmvPush{ .M = d.vocab, .K = d.n_embd };
        // LM head is the single biggest matvec (vocab x n_embd, Q6_K). Use the fast
        // variant at block=64 like every other quant matvec; f32 falls back to base.
        const lm_idx = dmmvIdx(lm_type);
        if (lm_idx < 4) {
            cmd.dispatch(&self.pipes.dmmv_fast[lm_idx], .{ d.vocab, 1, 1 }, .{ 64, 1, 1 }, &.{ lm_head, &self.norm_buf, &self.logits_buf }, &lm, @sizeOf(DmmvPush), 0);
        } else {
            cmd.dispatch(&self.pipes.dmmv[lm_idx], .{ d.vocab, 1, 1 }, .{ 256, 1, 1 }, &.{ lm_head, &self.norm_buf, &self.logits_buf }, &lm, @sizeOf(DmmvPush), 0);
        }
        const am = ArgmaxPush{ .N = d.vocab };
        cmd.dispatch(&self.pipes.argmax, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.logits_buf, &self.argmax_buf }, &am, @sizeOf(ArgmaxPush), 0);
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
        self.dmmvDispatch(&cmd, wq, &self.norm_buf, &self.qfull_buf, 2 * d.q_dim, d.n_embd, 0, 0);
        const deint = DeintPush{ .head_dim = d.head_dim, .n_head = d.n_head };
        cmd.dispatch(&self.pipes.deinterleave, .{ ceilDiv(d.q_dim, 256), 1, 1 }, .{ 256, 1, 1 }, &.{ &self.qfull_buf, &self.q_buf, &self.gate_buf }, &deint, @sizeOf(DeintPush), 0);
        self.dmmvDispatch(&cmd, wk, &self.norm_buf, &self.k_buf, d.kv_dim, d.n_embd, 0, 0);
        self.dmmvDispatch(&cmd, wv, &self.norm_buf, &self.v_buf, d.kv_dim, d.n_embd, 0, 0);
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
        // naive_attention's dynamic shared mem holds `seq_len` f32 scores. Under
        // graph capture, request the MAX (max_ctx) so the kernel node's shared-mem
        // size is constant across steps — keeping the captured topology invariant
        // (only push scalars change) so the exec updates in place instead of
        // re-instantiating each token. The kernel still uses only seq_len entries.
        const attn_smem: u32 = if (self.capturing) self.max_ctx * 4 else seq_len * 4;
        cmd.dispatch(&self.pipes.naive_attention, .{ d.n_head, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.q_buf, &self.kv_k[L], &self.kv_v[L], &self.sinks, &self.attn_out_buf }, &attn, @sizeOf(AttnPush), attn_smem);
        // gate: attn_out *= sigmoid(gate)
        const sm = SigmoidMulPush{ .N = d.q_dim };
        cmd.dispatch(&self.pipes.sigmoid_mul, .{ ceilDiv(d.q_dim, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.attn_out_buf, &self.gate_buf, &self.attn_out_buf }, &sm, @sizeOf(SigmoidMulPush), 0);
        // O projection, accumulate into hidden
        self.dmmvDispatch(&cmd, wo, &self.attn_out_buf, &self.hidden, d.n_embd, d.q_dim, 1, 0);
        self.submit(cmd);
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
        self.dmmvDispatch(&cmd, wqkv, &self.norm_buf, &self.attn_out_buf, d.conv_channels, d.n_embd, 0, 0);
        self.dmmvDispatch(&cmd, wz, &self.norm_buf, &self.gate_buf, d.d_inner, d.n_embd, 0, 0);
        // alpha, beta (dt_rank)
        self.dmmvDispatch(&cmd, walpha, &self.norm_buf, &self.router_buf, d.dt_rank, d.n_embd, 0, 0);
        self.dmmvDispatch(&cmd, wbeta, &self.norm_buf, &self.down_buf, d.dt_rank, d.n_embd, 0, 0);
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
        self.dmmvDispatch(&cmd, wout, &self.swiglu_buf, &self.hidden, d.n_embd, d.d_inner, 1, 0);
        self.submit(cmd);

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
        self.dmmvDispatch(&cmd, wgate, &self.ffn_norm_buf, &self.gate_buf, d.n_ff, d.n_embd, 0, 0);
        self.dmmvDispatch(&cmd, wup, &self.ffn_norm_buf, &self.up_buf, d.n_ff, d.n_embd, 0, 0);
        const sg = SwigluPush{ .N = d.n_ff };
        cmd.dispatch(&self.pipes.swiglu, .{ ceilDiv(d.n_ff, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.gate_buf, &self.up_buf, &self.swiglu_buf }, &sg, @sizeOf(SwigluPush), 0);
        self.dmmvDispatch(&cmd, wdown, &self.swiglu_buf, &self.hidden, d.n_embd, d.n_ff, 1, 0);
        self.submit(cmd);
    }

    /// MoE FFN block (qwen2_moe / qwen36). Replaces the dense ffnBlock when
    /// n_experts>0. Steps: rms_norm → router logits → top-k softmax → for each of
    /// the k routed experts run its gate/up/SwiGLU/down (addressed by a byte slice
    /// into the stacked expert weight) into a slot-major down_buf → weighted
    /// accumulate into hidden → shared expert (sigmoid-gated) accumulate into
    /// hidden. The routed combine and shared expert both += into hidden, so the
    /// residual is already present (no separate add).
    fn moeFfnBlock(self: *ForwardCuda, L: u32) !void {
        const d = self.d;
        const ctx = self.ctx;
        const n_used = d.n_experts_used;
        const ef = d.n_ff; // expert_ff = intermediate_dim = 512
        const wfn = self.layer(L, "post_attention_norm.weight");
        const wrouter = self.layer(L, "ffn_gate_inp.weight"); // [hidden, n_experts] F32
        const wge = self.layer(L, "ffn_gate_exps.weight"); // stacked [hidden, ef, n_experts]
        const wue = self.layer(L, "ffn_up_exps.weight");
        const wde = self.layer(L, "ffn_down_exps.weight"); // stacked [ef, hidden, n_experts]

        // Per-expert byte strides into the stacked tensors (quant may vary per layer).
        const gate_slice = expertSliceBytes(wge.info.type_, ef, d.n_embd);
        const up_slice = expertSliceBytes(wue.info.type_, ef, d.n_embd);
        const down_slice = expertSliceBytes(wde.info.type_, d.n_embd, ef);
        // Fast batched path when experts are q4k (gate/up) + q5k (down): one
        // launch over all experts, ids read GPU-side, no host readback. Else the
        // per-slot host path (correct for any quant).
        const batched_experts = dmmvIdx(wge.info.type_) == 0 and dmmvIdx(wue.info.type_) == 0 and dmmvIdx(wde.info.type_) == 1;

        // --- Router: rms_norm → logits → top-k softmax. -----------------------
        {
            var cmd = try command.beginCommand(ctx);
            const rms = RmsPush{ .N = d.n_embd, .eps = d.rms_eps };
            cmd.dispatch(&self.pipes.rms_norm, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.hidden, &wfn.gpu_buffer, &self.ffn_norm_buf }, &rms, @sizeOf(RmsPush), 0);
            self.dmmvDispatch(&cmd, wrouter, &self.ffn_norm_buf, &self.router_logits_buf, d.n_experts, d.n_embd, 0, 0);
            const tk = TopkPush{ .n_experts = d.n_experts, .k = n_used };
            cmd.dispatch(&self.pipes.softmax_topk, .{ 1, 1, 1 }, .{ 64, 1, 1 }, &.{ &self.router_logits_buf, &self.router_out_buf }, &tk, @sizeOf(TopkPush), 0);
            if (batched_experts) {
                self.submit(cmd); // async: experts read the ids GPU-side, no readback
            } else {
                cmd.commitAndWait(); // sync: the fallback host-gathers the ids next
                self.drainPending();
            }
        }

        // Fallback only: download the chosen expert ids for the per-slot path.
        if (!batched_experts) {
            buffer.download(ctx, &self.router_out_buf, std.mem.sliceAsBytes(self.host_router_ids[0..n_used]));
        }

        // --- Routed experts → SwiGLU → down, slot-major into down_buf.
        {
            var cmd = try command.beginCommand(ctx);
            if (batched_experts) {
                const nrows_gu = n_used * ef;
                const pg = ExpertsPush{ .M = ef, .K = d.n_embd, .slice = gate_slice, .x_stride = 0, .n_used = n_used };
                cmd.dispatch(&self.pipes.dmmv_q4k_experts, .{ nrows_gu, 1, 1 }, .{ 64, 1, 1 }, &.{ &wge.gpu_buffer, &self.ffn_norm_buf, &self.gate_buf, &self.router_out_buf }, &pg, @sizeOf(ExpertsPush), 0);
                const pu = ExpertsPush{ .M = ef, .K = d.n_embd, .slice = up_slice, .x_stride = 0, .n_used = n_used };
                cmd.dispatch(&self.pipes.dmmv_q4k_experts, .{ nrows_gu, 1, 1 }, .{ 64, 1, 1 }, &.{ &wue.gpu_buffer, &self.ffn_norm_buf, &self.up_buf, &self.router_out_buf }, &pu, @sizeOf(ExpertsPush), 0);
                const sg = SwigluPush{ .N = nrows_gu };
                cmd.dispatch(&self.pipes.swiglu, .{ ceilDiv(nrows_gu, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.gate_buf, &self.up_buf, &self.swiglu_buf }, &sg, @sizeOf(SwigluPush), 0);
                const pd = ExpertsPush{ .M = d.n_embd, .K = ef, .slice = down_slice, .x_stride = ef, .n_used = n_used };
                cmd.dispatch(&self.pipes.dmmv_q5k_experts, .{ n_used * d.n_embd, 1, 1 }, .{ 64, 1, 1 }, &.{ &wde.gpu_buffer, &self.swiglu_buf, &self.down_buf, &self.router_out_buf }, &pd, @sizeOf(ExpertsPush), 0);
            } else {
                const sg = SwigluPush{ .N = ef };
                var j: u32 = 0;
                while (j < n_used) : (j += 1) {
                    const id = self.host_router_ids[j];
                    self.dmmvDispatch(&cmd, wge, &self.ffn_norm_buf, &self.gate_buf, ef, d.n_embd, 0, id * gate_slice);
                    self.dmmvDispatch(&cmd, wue, &self.ffn_norm_buf, &self.up_buf, ef, d.n_embd, 0, id * up_slice);
                    cmd.dispatch(&self.pipes.swiglu, .{ ceilDiv(ef, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.gate_buf, &self.up_buf, &self.swiglu_buf }, &sg, @sizeOf(SwigluPush), 0);
                    const down_push = DmmvPush{ .M = d.n_embd, .K = ef, .acc_mode = 0, .a_offset = id * down_slice, .y_offset = j * d.n_embd * @sizeOf(f32) };
                    const didx = dmmvIdx(wde.info.type_);
                    if (didx < 4) {
                        cmd.dispatch(&self.pipes.dmmv_fast[didx], .{ d.n_embd, 1, 1 }, .{ 64, 1, 1 }, &.{ &wde.gpu_buffer, &self.swiglu_buf, &self.down_buf }, &down_push, @sizeOf(DmmvPush), 0);
                    } else {
                        cmd.dispatch(&self.pipes.dmmv[didx], .{ d.n_embd, 1, 1 }, .{ 256, 1, 1 }, &.{ &wde.gpu_buffer, &self.swiglu_buf, &self.down_buf }, &down_push, @sizeOf(DmmvPush), 0);
                    }
                }
            }
            // weighted combine of the k slot outputs into hidden.
            const ma = MoeAccPush{ .N = d.n_embd, .n_used = n_used, .src_stride = d.n_embd };
            cmd.dispatch(&self.pipes.moe_weighted_acc, .{ ceilDiv(d.n_embd, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.hidden, &self.down_buf, &self.router_out_buf }, &ma, @sizeOf(MoeAccPush), 0);
            self.submit(cmd);
        }

        // --- Shared expert: gate/up/SwiGLU/down, scaled by sigmoid(gate logit).
        {
            const wgs = self.layer(L, "ffn_gate_shexp.weight");
            const wus = self.layer(L, "ffn_up_shexp.weight");
            const wds = self.layer(L, "ffn_down_shexp.weight");
            const wgi = self.layer(L, "ffn_gate_inp_shexp.weight"); // [hidden, 1] F32
            const sf = d.shexp_ff;
            var cmd = try command.beginCommand(ctx);
            self.dmmvDispatch(&cmd, wgs, &self.ffn_norm_buf, &self.gate_buf, sf, d.n_embd, 0, 0);
            self.dmmvDispatch(&cmd, wus, &self.ffn_norm_buf, &self.up_buf, sf, d.n_embd, 0, 0);
            const sg = SwigluPush{ .N = sf };
            cmd.dispatch(&self.pipes.swiglu, .{ ceilDiv(sf, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.gate_buf, &self.up_buf, &self.swiglu_buf }, &sg, @sizeOf(SwigluPush), 0);
            // shared-expert gate scalar = W_gi · norm (1×hidden).
            self.dmmvDispatch(&cmd, wgi, &self.ffn_norm_buf, &self.gate_scalar_buf, 1, d.n_embd, 0, 0);
            // down → down_buf[0..hidden], then hidden += sigmoid(gate)*down.
            self.dmmvDispatch(&cmd, wds, &self.swiglu_buf, &self.down_buf, d.n_embd, sf, 0, 0);
            const ss = SigmoidAccPush{ .N = d.n_embd };
            cmd.dispatch(&self.pipes.sigmoid_scale_acc, .{ ceilDiv(d.n_embd, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.hidden, &self.down_buf, &self.gate_scalar_buf }, &ss, @sizeOf(SigmoidAccPush), 0);
            self.submit(cmd); // async: deferred like the dense path
        }
    }

    // Public wrappers for the per-block builders (used by the v1 driver to run
    // and inspect one block type at a time).
    pub fn attentionLayerPub(self: *ForwardCuda, L: u32, pos: u32) !void {
        try self.attentionLayer(L, pos);
        self.waitPending(); // dump callers read hidden right after — make it sync
    }
    pub fn ssmLayerPub(self: *ForwardCuda, L: u32) !void {
        try self.ssmLayer(L);
        self.waitPending();
    }
    pub fn ffnBlockPub(self: *ForwardCuda, L: u32) !void {
        try self.ffnBlock(L);
        self.waitPending();
    }

    /// Async decode command ring. For the dense path (n_experts==0) each layer-op
    /// commits asynchronously and is stashed; the ops pipeline on the shared
    /// CUstream (the CPU never blocks per-op), then the tail's single
    /// commitAndWait drains the stream and `drainPending` frees the events. This
    /// removes the ~0.4 ms/op WSL2 CPU↔GPU sync round-trip (and the boost-
    /// starvation those idle gaps cause — see Effort 21 Cycles 7–8). MoE layers
    /// also pipeline now: only the per-layer router stays synchronous (it reads
    /// the chosen expert ids back to the host mid-block); the routed + shared
    /// expert commands defer like the dense path, so a MoE token drops from
    /// ~4 syncs/layer to 1 (the router) — recovering the starved GPU boost.
    fn submit(self: *ForwardCuda, cmd: command.CudaCommand) void {
        var c = cmd;
        if (self.capturing) {
            // Graph capture: the dispatches are already recorded onto the captured
            // stream. Don't record a completion event or sync — just free the
            // (unused) command handle; the single graph launch + its sync drains
            // all captured work at the end of the step.
            c.releaseCompleted();
            return;
        }
        if (self.n_pending < self.pending.len) {
            c.commitAsync();
            self.pending[self.n_pending] = c;
            self.n_pending += 1;
        } else {
            c.commitAndWait();
        }
    }

    /// Free the stashed async commands. Safe once a later same-stream
    /// commitAndWait (the tail) has drained the stream — completion is guaranteed.
    fn drainPending(self: *ForwardCuda) void {
        var i: u32 = 0;
        while (i < self.n_pending) : (i += 1) self.pending[i].releaseCompleted();
        self.n_pending = 0;
    }

    /// Wait on + free the stashed async commands, for callers that read GPU
    /// results before any tail sync (the per-block Pub wrappers).
    fn waitPending(self: *ForwardCuda) void {
        var i: u32 = 0;
        while (i < self.n_pending) : (i += 1) self.pending[i].wait();
        self.n_pending = 0;
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
    /// Quant types (idx < 4: q4k/q5k/q6k/q8_0) use the fast single-row variant at
    /// block=64; f32 (idx == 4) has no fast variant and keeps the base at block=256.
    fn dmmvDispatch(self: *ForwardCuda, cmd: *command.CudaCommand, w: *const LoadedTensor, x: *const CudaBuffer, y: *const CudaBuffer, M: u32, K: u32, acc_mode: u32, a_offset: u32) void {
        const push = DmmvPush{ .M = M, .K = K, .acc_mode = acc_mode, .a_offset = a_offset };
        const idx = dmmvIdx(w.info.type_);
        if (idx < 4) {
            cmd.dispatch(&self.pipes.dmmv_fast[idx], .{ M, 1, 1 }, .{ 64, 1, 1 }, &.{ &w.gpu_buffer, x, y }, &push, @sizeOf(DmmvPush), 0);
        } else {
            cmd.dispatch(&self.pipes.dmmv[idx], .{ M, 1, 1 }, .{ 256, 1, 1 }, &.{ &w.gpu_buffer, x, y }, &push, @sizeOf(DmmvPush), 0);
        }
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

/// Bytes for one expert's [rows × cols] slice in a stacked MoE weight tensor.
/// rows*cols must be the per-expert element count; `q.blockSize()` divides cols
/// (the contiguous/quantized dim) and `q.bytesPerBlock()` is the block width.
fn expertSliceBytes(q: gguf.GGMLType, rows: u32, cols: u32) u32 {
    return rows * (cols / q.blockSize()) * q.bytesPerBlock();
}

/// Zero a device buffer by uploading a host-side zero array of `n` f32.
fn zeroBuffer(allocator: std.mem.Allocator, ctx: ?*shim.CudaCtx, buf: *CudaBuffer, n: u32) !void {
    const z = try allocator.alloc(f32, n);
    defer allocator.free(z);
    @memset(z, 0);
    buffer.upload(ctx, buf, std.mem.sliceAsBytes(z));
}
