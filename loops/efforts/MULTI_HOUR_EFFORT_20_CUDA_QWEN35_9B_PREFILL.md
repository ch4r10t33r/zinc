# Effort 20 — CUDA Qwen 3.5 9B prefill bring-up

> **Status:** planning · M0.5 primitive layer done (5090 only) · backend not yet wired into the engine · prefill = **0 tok/s** (no `forward_cuda.zig`). Goal: correctness parity, then prefill perf vs llama.cpp CUDA on the 4090 + 5090.

Date: 2026-06-06

Pairs with **Effort 21 — CUDA Qwen 3.5 9B decode** (`MULTI_HOUR_EFFORT_21_CUDA_QWEN35_9B_DECODE.md`).
Shared bring-up plan & backend contract: **`docs/cuda-backend.md`**.

## Target model

- Catalog id / site artifact: `qwen35-9b-q4k-m` — the **smallest model ZINC catalogs** (status `.supported`; the 2b is a non-catalog dev artifact).
- GGUF: `unsloth/Qwen3.5-9B-Q4_K_M.gguf` — 5.68 GB on disk, **5.28 GiB** tensors, **8.95 B** params, sha256 `03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8`. On the box: `~/workspace/Qwen3.5-9B-Q4_K_M.gguf`.
- Architecture: `qwen35` — **dense** SSM+attention hybrid (**not** MoE). `full_attention_interval`-th layers are full attention; the rest are delta-net SSM. Dense SwiGLU FFN every layer.
  - This is the key divergence from `docs/cuda-backend.md`, whose target is the 35B-A3B **MoE**. The dense 9B **drops the entire MoE kernel set** (softmax_topk, expert DMMV, route-pack, shared-expert) → a much smaller kernel surface for first light.
- Config (**confirmed on box**, GGUF parse): 32 layers, hidden 4096, FFN 12288 (dense SwiGLU), 16 heads / 4 KV (GQA 4:1), head_dim 256, rms_eps 1e-6, vocab 248320, train ctx 262144. `full_attention_interval=4` → **8 full-attention + 24 delta-net SSM** layers. RoPE partial/mRoPE: dim 64, freq_base 1e7, sections [11,11,10,0]. SSM: d_conv 4, d_inner 4096, d_state 128, dt_rank 32, groups 16.
- Quant (**confirmed**, Q4_K_M; 427 tensors = F32 177 / Q4_K 132 / Q5_K 48 / Q8_0 48 / Q6_K 22): token_embd Q4_K; LM head + attn_v + ffn_down **Q6_K**; attn_qkv + ssm_out **Q5_K**; ssm_alpha/beta Q8_0; attn_q/k/o + attn_gate + ffn_gate/up Q4_K; all norms + ssm_a/conv1d/dt F32. **DMMV weight types needed: Q4_K✓ Q8_0✓ F32✓ + Q5_K, Q6_K (todo).**
- Primary metric: **prefill tok/s** on the public Long-Coding-Plan scenarios, token-for-token correct vs the Metal/Vulkan reference.

## Why this effort exists

ZINC has no NVIDIA path today — Vulkan in WSL2 is CPU-only (`docs/cuda-backend.md` §1), so the kernels only touch these GPUs via a native CUDA backend. Prefill is the harder hot path: in the RDNA campaign (Effort 17) ZINC core prefill was only **18%** of llama.cpp while decode already *led*. This effort takes CUDA prefill from **nonexistent → correct → competitive** on the 4090/5090.

### Baselines (the bar to clear)

| ref | hw | scenario | prefill tok/s |
|---|---|---|---:|
| **llama.cpp CUDA** | **4090** | **pp2048** | **8487 ± 164** (±1.9%, clean) |
| **llama.cpp CUDA** | **5090** | **pp2048** | **10049 ± 41** (±0.4%, clean) |
| llama.cpp CUDA | 4090/5090 | pp512 | noisy ±16–35% (too short → use pp2048) |
| llama.cpp | RDNA4 (Effort 17) | core (36 tok) | 548.94 |
| ZINC vulkan | RDNA4 (Effort 17) | core | 100.79 (18.4% of llama) |
| **ZINC cuda** | 4090 / 5090 | any | **0 (not wired)** |

Cycle-1 task: re-measure llama.cpp CUDA prefill on **both** GPUs with the real scenario prompts and enough reps for a stable median — the synthetic `pp512 ±27%` cannot be a target.

## Bring-up path (correctness first, then prefill perf)

Per `docs/cuda-backend.md` milestones, adapted to the dense 9B:

1. **Wire the backend** (blocks everything): `build.zig` `cuda` enum + `configureCudaModule` + `zig build cuda-smoke`; `src/gpu/interface.zig` `is_cuda` (from `build_options.backend`) routing `backend/buffer/pipeline/command` → `src/cuda/*`; three-way `main.zig` + `model_manager_runtime.zig`. Then **re-validate primitives + the 5 done kernels on BOTH GPUs** (M0.5 was 5090-only).
2. **Dense prefill kernel set** (subset of doc §5, no MoE): batched/layer-major DMMV→GEMM for Q4_K/Q6_K/Q8_0/F32 (gate/up/down, Q/K/V/O, SSM in/out, LM head); `rms_norm`✓, `swiglu`✓, `scale_accumulate`✓; RoPE + qk_norm; `kv_cache_write`; attention prefill (naive `softmax(QKᵀ)V` → flash); and the **batched SSM trio** (`ssm_conv1d`, `ssm_delta_net`, `ssm_gated_norm`) — the batched selective scan is the hardest kernel. Mirror the validated RDNA layer-major batched-SSM machinery; honor the conv-state / delta-recurrence / residual-ordering hazards Effort 17 flagged (do **not** just drop shape guards).
3. **Prefill perf (M3):** DP4a tiled GEMM (register blocking + padded shared mem), fused gate/up + fused down-acc, layer-major SSM prefix/segment paths, optional tensor-core `mma.sync` GEMM as an sm_120 (5090) win. Bench vs llama.cpp CUDA on both GPUs.

## Measurement contract

- Prompts: public Long-Coding-Plan scenarios — core (≈36 prompt toks), context-medium (≈174), context-long (≈322); raw mode, deterministic.
- Metric: prefill tok/s = prompt_tokens / prefill_time, **median of N warm reps**. Characterize cold/warm clock-ramp on each GPU first (CUDA analog of the Metal cold/warm regime).
- Correctness gate: **token-for-token** match vs Metal/Vulkan reference on the same prompt *before* any tok/s is recorded.
- Devices: report **4090 (sm_89)** and **5090 (sm_120)** separately; **pin by UUID** — 4090 `GPU-e59a6fce-1961-bafe-927c-06c0149f2370`, 5090 `GPU-5126d018-ec86-be8b-1bf5-b5ac323d3350` (device *index* is unreliable on this box; `nvidia-smi` ignores `CUDA_VISIBLE_DEVICES`).

## Cycle log

- **Cycle 0 (2026-06-06):** effort opened. Backend at M0.5 (primitive layer validated on 5090; `src/cuda/*` + `src/shaders/cuda/kernels.cu` = WIP, untracked). 5 kernels validated (`rms_norm`, `dmmv_q4k`, `swiglu`, `scale_accumulate`, `sigmoid_scale_acc`, ≤1e-5). llama.cpp CUDA reference on 4090: decode 98 t/s; prefill `pp512 ~8040` (noisy, must re-measure). **Next:** wire `build.zig`/`gpu/interface.zig`, validate primitives + 5 kernels on the **4090** (extend the 5090-only M0.5), confirm 9B config/quant on box, re-measure llama.cpp prefill scenarios on both GPUs.
- **Cycle 1 (2026-06-06):** foundation validated on **both GPUs** (was 5090-only). `kernels_test` (gcc + NVRTC, staged to `~/cuda_proto`) → **ALL PASS on 4090 (sm_89) and 5090 (sm_120)**: `rms_norm` 2.2e-7, `dmmv_q4k` 1.6e-7, `dmmv_f32` 3.3e-6, `dmmv_q8_0` 1.5e-7, `swiglu` 2.2e-7, `scale_accumulate` 7.7e-6, `sigmoid_scale_acc` 1.7e-5 (all ≤1.7e-5). 7 kernels now (doc listed 5 — `dmmv_f32`/`q8_0` added). UUID device-pin confirmed. **Prefill-blocking ports remaining:** batched/layer-major GEMM, `dmmv_q6k` (LM head), RoPE+qk_norm, attention prefill, batched SSM trio; plus build wiring + `forward_cuda`.
- **Cycle 2 (2026-06-06):** backend **wired into the build system**. `build.zig`: `cuda` enum + `configureCudaModule` (shim + NVRTC; links `cuda`/`nvrtc`, `/usr/lib/wsl/lib` for `libcuda`) + `cuda` exe branch + Linux-gated **`zig build cuda-smoke`**. `gpu/interface.zig`: `is_cuda` (build_options-driven) 3-way routing → `src/cuda/*` (default Metal build stays green, `zig build` exit 0). `src/compute/forward_cuda.zig` scaffold added (dense-9B M1 forward order + kernel checklist; not yet dispatched). **`zig build cuda-smoke` → PASS on both GPUs** (5090 sm_120, 4090 sm_89): NVRTC compile + vadd + dp4a async all green. **Next (prefill):** re-measure llama.cpp prefill scenarios cleanly, then port batched/layer-major GEMM + batched SSM prefill into the M1 path.
- **Cycle 3 (2026-06-06):** **9B config + quant confirmed on box** (dependency-free GGUF parse) and folded into `src/compute/forward_cuda.zig` (`Cfg` constants + a per-tensor quant table) and this file's Target section. Headlines: 32 layers (8 full-attn / 24 SSM via interval=4), hidden 4096, FFN 12288, 16h/4kv, head_dim 256, vocab 248320; SSM d_state 128 / dt_rank 32 / d_conv 4 / d_inner 4096. Quant histogram F32 177 / Q4_K 132 / Q5_K 48 / Q8_0 48 / Q6_K 22 → corrected kernel checklist: **DMMV needs both Q5_K and Q6_K** added (q4k/q8_0/f32 already green).
- **Cycle 4 (2026-06-06):** clean llama.cpp prefill bar measured (llama-bench `-r 10`, UUID-pinned): **pp2048 = 8487 ± 164 t/s (4090) / 10049 ± 41 t/s (5090)** — both tight. `pp512` is too short to time (±16–35% clock-ramp noise), so the bar is **pp2048**; replaces the noisy ~8040 placeholder. ZINC-cuda prefill target: reach parity at pp2048 once the M1 prefill path exists.
