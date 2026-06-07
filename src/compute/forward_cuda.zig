//! CUDA forward pass for the dense `qwen35` hybrid (Qwen 3.5 9B) — SCAFFOLD.
//!
//! Status: M1 bring-up scaffold. Establishes the structure + integration points
//! for ZINC's CUDA decode/prefill, but is **not yet wired** into the compute
//! dispatch (`gpu/interface.zig` + `main.zig` three-way) — that lands with the
//! M1 kernel set. See `docs/cuda-backend.md` §5 and Efforts 20 (prefill) / 21
//! (decode) under `loops/efforts/`.
//!
//! Mirrors `src/compute/forward_metal.zig` (raw-pointer binds; an async
//! stream/event command ring) using the CUDA backend modules surfaced by
//! `gpu/interface.zig` when `is_cuda`:
//!   - device   : `../cuda/device.zig`   — CudaDevice: caps (cc/SMs/vram), ctx
//!   - buffer   : `../cuda/buffer.zig`   — device buffers + pinned H2D/D2H staging
//!   - pipeline : `../cuda/pipeline.zig` — NVRTC compile `.cu` → CUfunction
//!   - command  : `../cuda/command.zig`  — CUstream dispatch; commitAsync/wait ring
//! Kernels live in `src/shaders/cuda/kernels.cu`, NVRTC-compiled on load for the
//! running arch (sm_89 on the 4090, sm_120 on the 5090).
//!
//! Target is the **dense** 9B — no MoE — so the expert path of the 35B plan
//! (`softmax_topk`, routed/shared experts) is intentionally absent.
//!
//! @section Inference Runtime
const std = @import("std");
const gpu = @import("../gpu/interface.zig");

/// Exact `qwen35-9b-q4k-m` config — from the GGUF metadata (general.file_type=15
/// = Q4_K_M; 427 tensors). Confirmed on box 2026-06-06.
pub const Cfg = struct {
    pub const arch = "qwen35";
    pub const n_layers: u32 = 32;
    pub const n_embd: u32 = 4096; // hidden
    pub const n_ff: u32 = 12288; // dense SwiGLU FFN (no MoE)
    pub const n_head: u32 = 16;
    pub const n_head_kv: u32 = 4; // GQA 4:1
    pub const head_dim: u32 = 256; // attention.key_length = value_length
    pub const rms_eps: f32 = 1e-6;
    pub const vocab: u32 = 248320;
    pub const context_train: u32 = 262144; // runtime context is far smaller
    // Hybrid layer pattern: is_full_attn = ((L+1) % full_attention_interval == 0)
    //   → layers 3,7,…,31 = full attention (8); the other 24 = delta-net SSM.
    pub const full_attention_interval: u32 = 4;
    // RoPE: partial / mRoPE — applied to rope_dim of head_dim; sections×2 = rope_dim.
    pub const rope_dim: u32 = 64;
    pub const rope_freq_base: f32 = 1.0e7;
    pub const rope_sections = [4]u32{ 11, 11, 10, 0 };
    // Delta-net SSM (the 24 non-attention layers):
    pub const ssm_d_conv: u32 = 4;
    pub const ssm_d_inner: u32 = 4096;
    pub const ssm_d_state: u32 = 128;
    pub const ssm_dt_rank: u32 = 32;
    pub const ssm_group_count: u32 = 16;
};

// Per-tensor quant (the Q4_K_M mix) — drives which DMMV variants are needed:
//   token_embd            Q4_K  [4096, 248320]   output/LM head  Q6_K [4096, 248320]
//   full-attn layer : attn_q Q4_K, attn_k Q4_K, attn_v Q6_K, attn_output Q4_K,
//                     q_norm/k_norm/attn_norm F32
//   ssm layer       : attn_qkv Q5_K, attn_gate Q4_K, ssm_out Q5_K,
//                     ssm_alpha/beta Q8_0, ssm_a/conv1d/dt.bias/norm F32
//   ffn (every layer): gate Q4_K, up Q4_K, down Q6_K;  all *_norm F32
// Histogram (427): F32 177, Q4_K 132, Q5_K 48, Q8_0 48, Q6_K 22.
// => DMMV weight types required: Q4_K✓ Q8_0✓ F32✓  +  Q5_K(todo) Q6_K(todo).

/// CUDA forward pass for the dense qwen35 9B.
/// Milestones: M1 one correct token → M2 full prefill+decode → M3 fused/async perf.
pub const CudaForward = struct {
    allocator: std.mem.Allocator,
    // TODO M1 state (mirrors the Metal/Vulkan engines):
    //   device     : gpu.backend.CudaDevice
    //   pipelines  : compiled CUfunctions for the kernel set in the checklist below
    //   weights    : device buffers (mmap-staged H2D; quant-typed)
    //   kv_pool    : paged KV cache (attention layers)
    //   ssm_state  : conv ring `(d_conv-1)*inner` f32/layer + recurrent `dt_rank·…` f32/layer
    //   ring       : [N]command.CudaCommand pending ring over CUstream+CUevent (M3)

    /// Allocate the CUDA forward engine (M1 scaffold — GPU wiring is still TODO).
    /// @param allocator Backing allocator for engine-owned device and host buffers.
    /// @returns An initialized engine; device/kernel/buffer setup lands with the M1 kernels.
    pub fn init(allocator: std.mem.Allocator) !CudaForward {
        // TODO M1: select device (by UUID/cc), NVRTC-compile kernels.cu for the
        // running arch, allocate weight/scratch/KV/SSM buffers, stage weights H2D.
        return .{ .allocator = allocator };
    }

    /// Release all engine resources (buffers, streams/events, modules, context).
    /// @note No-op in the current scaffold; becomes meaningful once `init` allocates GPU state.
    pub fn deinit(self: *CudaForward) void {
        _ = self;
        // TODO: free buffers; destroy streams/events; unload modules; pop ctx.
    }

    /// M1 — one correct decode token, validated token-for-token vs the
    /// Metal/Vulkan reference. Forward order for the **dense** qwen35 9B
    /// (docs/cuda-backend.md §5 M1, minus the MoE path):
    ///   1. embedding gather (host) → hidden
    ///   per layer L:
    ///     2. rms_norm (input)                                    [done]
    ///     3. is_full_attn = ((L+1) % full_attention_interval == 0)
    ///          attention: DMMV Q/K/V → qk_norm + RoPE → kv_cache_write →
    ///                     softmax(QKᵀ)V (single query) → DMMV O
    ///          SSM      : DMMV in → ssm_conv1d → ssm_delta_net (recurrent) →
    ///                     ssm_gated_norm → DMMV out
    ///     4. scale_accumulate (residual)                         [done]
    ///     5. rms_norm (post-mixer)                               [done]
    ///     6. dense FFN: DMMV gate + DMMV up → swiglu → DMMV down [swiglu done]
    ///     7. scale_accumulate (residual)                         [done]
    ///   8. final rms_norm → DMMV lm_head (Q6_K) → argmax
    pub fn decodeStep(self: *CudaForward) !u32 {
        _ = self;
        return error.NotImplemented; // M1 TODO
    }

    /// M2 — batched prefill over the prompt (layer-major DMMV→GEMM; batched SSM
    /// selective scan). See Effort 20 for the prefill-specific kernel plan.
    pub fn prefill(self: *CudaForward, n_tokens: usize) !void {
        _ = self;
        _ = n_tokens;
        return error.NotImplemented; // M2 TODO
    }
};

// Reference the backend surface so the scaffold documents (and the compiler
// checks, once wired) the dependency on the CUDA modules.
comptime {
    if (gpu.is_cuda) {
        _ = gpu.backend; // ../cuda/device.zig
        _ = gpu.buffer_mod; // ../cuda/buffer.zig
        _ = gpu.pipeline_mod; // ../cuda/pipeline.zig
        _ = gpu.command_mod; // ../cuda/command.zig
    }
}

// Kernel checklist — `src/shaders/cuda/kernels.cu`, dense-9B M1 set:
//   done : rms_norm, dmmv_q4k, dmmv_f32, dmmv_q8_0, swiglu, scale_accumulate,
//          sigmoid_scale_acc          (all validated on 4090 sm_89 + 5090 sm_120)
//   todo : dmmv_q5k (attn_qkv, ssm_out), dmmv_q6k (LM head, attn_v, ffn_down),
//          qk_norm, RoPE (partial/mRoPE, sections [11,11,10,0], dim 64),
//          kv_cache_write, attention (naive softmax(QKᵀ)V → flash), ssm_conv1d,
//          ssm_delta_net (recurrent step + batched prefill), ssm_gated_norm,
//          argmax
