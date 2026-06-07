# Effort 21 — CUDA Qwen 3.5 9B decode bring-up

> **Status:** planning · M0.5 primitive layer done (5090 only) · backend not yet wired · decode = **0 tok/s**. Goal: beat llama.cpp CUDA decode (**98 t/s on the 4090**) on the 4090 + 5090.

Date: 2026-06-06

Pairs with **Effort 20 — CUDA Qwen 3.5 9B prefill** (`MULTI_HOUR_EFFORT_20_CUDA_QWEN35_9B_PREFILL.md`).
Shared bring-up plan & backend contract: **`docs/cuda-backend.md`**.

## Target model

Same as Effort 20: `qwen35-9b-q4k-m` — `unsloth/Qwen3.5-9B-Q4_K_M.gguf` (5.28 GiB, 8.95 B), `qwen35` **dense** SSM+attention hybrid, Q4_K_M. On the box: `~/workspace/Qwen3.5-9B-Q4_K_M.gguf`. Dense → **no MoE kernels**. Confirmed config: **32 layers (8 full-attn + 24 delta-net SSM, interval=4)**, hidden 4096, FFN 12288, 16h/4kv, head_dim 256, vocab 248320; SSM d_state 128 / dt_rank 32 / d_conv 4 / d_inner 4096; RoPE partial dim 64, freq 1e7. Decode DMMV quant set: Q4_K✓ Q8_0✓ F32✓ + Q5_K, Q6_K (todo).

## Why this effort exists

Decode is the per-token autoregressive path and the one ZINC is historically *good* at: on RDNA (Effort 17) ZINC decode **beat** llama.cpp on every scenario (95–97 vs 85 tok/s). So once the CUDA kernels are numerically correct, decode parity-then-win is the **more reachable** of the two hot paths. The decode perf unlock — an async stream/event command ring that overlaps GPU exec with CPU command-building — already paid off on Metal/RDNA (+22% on the 2b SSM decode) and ports directly to `CUstream` + `CUevent`.

### Baselines (the bar to clear)

| ref | hw | decode tok/s |
|---|---|---:|
| **llama.cpp CUDA** | **4090** | **97 ± 1** (tg128 clean; 120 ± 26 at tg256 when boosting) |
| **llama.cpp CUDA** | **5090** | **~156–226** (tg256 156±25 / tg128 226±293) — bench-unstable, re-measure |
| llama.cpp | RDNA4 (Effort 17) | 84.96–85.51 |
| ZINC vulkan | RDNA4 (Effort 17) | 95.39–96.91 |
| **ZINC cuda** | 4090 / 5090 | **0 (not wired)** |

Bar on the 4090 = **98 t/s** (and ZINC beat llama.cpp by ~12% on RDNA, so parity+ is realistic once correct). 5090 should exceed it; measure both.

## Bring-up path (correctness first, then decode perf)

1. **Backend wiring** — shared with Effort 20 (`build.zig` `cuda` enum + `configureCudaModule`; `gpu/interface.zig` `is_cuda`; three-way `main.zig`/`model_manager_runtime.zig`). Validate primitives + the 5 done kernels on **both** GPUs.
2. **Dense decode kernel set** (single-token, doc §5 minus MoE): templated **DMMV matvec** Q4_K✓/Q6_K/Q8_0/F32 (+accumulate mode) for Q/K/V/O, SSM in/out, LM head; `rms_norm`✓, `swiglu`✓, `scale_accumulate`✓, `sigmoid_scale_acc`✓; RoPE + qk_norm; **paged `kv_cache_write`** + naive single-query attention (`softmax(QKᵀ)V`, one query); the **`ssm_delta_net` recurrent step** (per-row register-resident state — *the* decode-critical SSM kernel) + `ssm_conv1d` (ring state) + `ssm_gated_norm`; `argmax`.
   - State buffers: paged KV pool; SSM conv ring `(d_conv−1)*inner` f32/layer + recurrent `dt_rank·…` f32/layer (sizes from config — confirm on box).
   - **M1 gate:** one token, token-for-token vs Metal/Vulkan reference.
3. **Decode perf (M3):** the async `commitAsync`/`wait`/`releaseCompleted` pending-command ring over `CUstream`+`CUevent` (the proven SSM-decode win); fused `rms_norm`+DMMV and fused gate/up; warp32 reductions tuned for sm_89 vs sm_120 occupancy.

## Measurement contract

- Metric: decode tok/s = generated_tokens / gen_time, **steady-state** (exclude first token), median of N warm reps.
- Correctness gate: token-for-token vs Metal/Vulkan reference greedy decode on a fixed prompt *before* recording tok/s.
- Devices: **4090 + 5090 separately**, UUID-pinned (4090 `GPU-e59a6fce-…`, 5090 `GPU-5126d018-…`) — index unreliable; see Effort 20.

## Cycle log

- **Cycle 0 (2026-06-06):** opened. llama.cpp CUDA decode baseline = **98.0 ± 1.0 t/s on the 4090** (clean, tg128). Backend at M0.5; `ssm_delta_net` recurrent step + attention not yet ported. **Next:** wire backend, reach M1 single-token decode correctness for the dense 9B, then port the async stream/event ring.
- **Cycle 1 (2026-06-06):** foundation validated on **both GPUs** (was 5090-only). `kernels_test` → **ALL PASS on 4090 (sm_89) and 5090 (sm_120)**: `rms_norm`, `dmmv_q4k`/`f32`/`q8_0`, `swiglu`, `scale_accumulate`, `sigmoid_scale_acc` (all ≤1.7e-5) — 7 of the dense-decode kernels green on both. **Decode-blocking ports remaining:** `dmmv_q6k`, RoPE+qk_norm, paged `kv_cache_write` + single-query attention, the SSM trio (`ssm_delta_net` recurrent step + `ssm_conv1d` + `ssm_gated_norm`), `argmax`; then `forward_cuda` + the async `CUstream`/`CUevent` ring.
- **Cycle 2 (2026-06-06):** backend **wired into the build system** (shared with Effort 20): `build.zig` `cuda` enum + `configureCudaModule` + Linux-gated `zig build cuda-smoke`; `gpu/interface.zig` `is_cuda` 3-way routing → `src/cuda/*`; `src/compute/forward_cuda.zig` scaffold. Default Metal build green (`zig build` exit 0). **`zig build cuda-smoke` → PASS on both GPUs** (5090 sm_120 / 4090 sm_89): NVRTC + vadd + dp4a async green. **Next (decode):** port the M1 decode kernels (`dmmv_q6k`, RoPE+qk_norm, paged `kv_cache_write` + single-query attention, the `ssm_delta_net` recurrent step, `argmax`) and wire `forward_cuda.decodeStep` into dispatch for the first correct token.
- **Cycle 3 (2026-06-06):** **9B config + quant confirmed on box** and folded into `forward_cuda.zig` (`Cfg`) + the Target section above. For decode this fixes the per-layer dispatch (8 full-attn vs 24 SSM layers, interval=4) and the DMMV set: the LM-head `argmax` reads **Q6_K** [4096×248320]; SSM-layer `ssm_out` + `attn_qkv` are **Q5_K** — so the decode hot path needs **dmmv_q5k + dmmv_q6k** beyond the three already green. SSM recurrent state per layer: conv ring `(4-1)×8192` + recurrent `dt_rank 32 · d_state 128 · …`.
- **Cycle 4 (2026-06-06):** decode baseline measured, but **bench is boost-sensitive on this shared Windows box**: 4090 tg128 = 97±1 (tight) yet tg256 = 120±26; 5090 swung 156–226 (it's the display GPU → contention). Clean **prefill** bar did land (pp2048 8487/10049, see Effort 20). **Decode bar = 97 t/s on the 4090** (defensible); the **5090 needs clock-pinning (sudo `nvidia-smi -lgc`) or a warmup protocol** for a tight number — a measurement-contract TODO. ZINC decode beat llama.cpp by ~12% on RDNA, so 97+ is the realistic 4090 target once the kernels are correct.
