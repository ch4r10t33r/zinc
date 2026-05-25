# Effort 15 — Apple M1 Max Gemma 4 12B decode: close gap to llama.cpp

## Objective

Move local Gemma 4 12B (`gemma4-12b-q4k-m`, the 26B-A4B MoE) decode tok/s on
this M1 Max as close to local llama.cpp on the same machine and model as
possible, without regressing correctness or any other platform target.

This is the M1 Max sibling of Effort 11 (same model, Apple9 / M4). Effort 11's
correctness and GPU-routed MoE work has landed on `main`; Gemma 4 chat is
already coherent. This effort is **decode speed only** — Effort 14 style, not
a correctness pass.

Stop conditions:

- ZINC decode tok/s matches or exceeds local llama.cpp on this machine, or
- three consecutive Phase 3 attempts fail the microbench gate, or
- six total accepted Phase 4 cycles complete (autonomous-spend cap).

## Fixed inputs (do not drift)

- Machine: local Apple M1 Max, 32 GPU cores, 32 GB unified memory,
  `MTLGPUFamilyApple7`.
- Model id: `gemma4-12b-q4k-m` (Gemma 4 26B-A4B MoE, ~4B active, Q4_K_M).
- Model path: `/Users/stepan/Library/Caches/zinc/models/models/gemma4-12b-q4k-m/model.gguf`
  (16.9 GB on disk; fits comfortably in 32 GB UMA).
- Backend: Metal only. Do not touch Vulkan, RDNA shaders, the 35B-A3B path,
  the dense Gemma path, or any non-Metal builder.
- Comparator: `/Users/stepan/Workspace/llama.cpp/build-metal/bin/llama-cli`
  against the exact same GGUF.
- Prompt mode: **chat** (`--chat`). Raw completion is not a valid Gemma
  coherence gate — it emits repetition such as `is is is is`.
- Canonical prompt: `"What is the capital of France?"`
- Generation length: `-n 128`
- Sampling: greedy, `--temp 0`
- Build: `zig build -Doptimize=ReleaseFast` only. `--profile` numbers from
  any other build are not accepted as evidence.

## Run the loop with

```bash
ZINC_MODEL_ID=gemma4-12b-q4k-m \
ZINC_PROMPT_MODE=chat \
ZINC_TEST_PROMPT="What is the capital of France?" \
ZINC_MAX_TOKENS=128 \
ZINC_TARGET_TOK_PER_SEC=40 \
ZINC_STOP_ON_TARGET=0 \
ZINC_BENCHMARK_RUNS=5 \
ZINC_PROFILE_EVERY=1 \
ZINC_BUILD_OPTIMIZE=ReleaseFast \
ZINC_TEST_TIMEOUT_MS=300000 \
ZINC_RUN_TIMEOUT_MS=900000 \
bun loops/implement_metal.ts --effort 15 --agent claude --cycles 80
```

Omit `--resume` on the **first** run of this effort: a fresh start
initializes effort-15 state at cycle 1. Note a fresh start also runs
`cleanupOldRuns()`, which wipes the entire `.metal_optimize/` directory —
archive any other effort's run dir out of `.metal_optimize/` first if you
need to keep it. Add `--resume` for every run after the first to continue
the same run.

`--agent codex --model gpt-5.5 --reasoning xhigh` also works; the doc is
written for either agent. `ZINC_TARGET_TOK_PER_SEC=40` is a placeholder —
`ZINC_STOP_ON_TARGET=0` so the loop never stops on it; the real exit is the
llama.cpp comparator below. Re-tune the target after Phase 0 measures the
true baseline.

Important harness detail:

- `implement_metal.ts` must build the verifier binary with
  `zig build -Doptimize=ReleaseFast`.
- If the loop says `Building (zig build)` (no optimize flag) or the verifier
  measures a plain-`zig build` binary, stop and fix the harness before
  optimizing.
- Agent-side `--profile` numbers are not accepted unless the official loop
  verifier was built with the same optimize mode.

## Measurement protocol

Run this exactly before every accept/reject decision in any phase.

1. Clean stragglers:
   ```bash
   pkill -f 'zig-out/bin/zinc' || true
   pkill -f 'llama-cli' || true
   ```
2. 1 warmup run, then 3 timed runs. Report the **median** decode tok/s.
   Single-run numbers are not accepted as evidence. Each fresh ZINC process
   starts cold (cold/warm spread is real on this machine) — the warmup run
   is mandatory.
3. ZINC command:
   ```bash
   ./zig-out/bin/zinc \
     --model-id gemma4-12b-q4k-m \
     --prompt "What is the capital of France?" \
     --chat \
     -n 128 \
     --profile
   ```
4. llama.cpp command:
   ```bash
   ~/Workspace/llama.cpp/build-metal/bin/llama-cli \
     -m /Users/stepan/Library/Caches/zinc/models/models/gemma4-12b-q4k-m/model.gguf \
     -p "What is the capital of France?" \
     -n 128 -ngl 99 -fa on --temp 0 --no-warmup
   ```
   Decode tok/s is bandwidth-bound and prompt-content-independent, so the
   comparator does not need ZINC's chat template — only the *speed* number
   is compared. If `llama-cli` rejects the GGUF with an unknown-architecture
   error, the `build-metal` binary predates Gemma 4 support; rebuild
   llama.cpp from its current source (the source tree has `LLM_ARCH_GEMMA4`)
   before trusting the comparator.
5. Log into `loops/efforts/EFFORT_15_NOTES.md`:
   - decode tok/s (3 runs, median)
   - prefill tok/s (3 runs, median)
   - ZINC profile dispatch-ms / dispatch-byte breakdown
   - any anomalies (process collisions, thermal warnings, residency
     pressure warnings, cold/warm divergence, etc.)

## Known-blocked levers (do not propose these)

Gemma 4 is not Qwen 3. Two levers that were live in Effort 14 are closed here:

- **KV-cache q8_0 quantization is unavailable.** Gemma 4 has sliding-window
  attention; `defaultKvCacheQ8Enabled` returns `false` whenever
  `architecture == .gemma and rope_freq_base_swa > 0`
  (`src/compute/forward_metal.zig:418`). The decode KV path is f16 by design
  because Metal does not implement llama.cpp's ISWA quantized-cache K/V
  rotation. Do not propose Q8 KV cache as a decode bandwidth win unless that
  ISWA path is implemented first — and that is out of scope for this effort.
- **`.gemma`-gated kernel unblocking is not the lever.** Effort 14's biggest
  win came from unblocking Metal kernels artificially gated to `.gemma`.
  Gemma 4 *is* `.gemma`, so those gates are already open for this model.
  The inverse is worth a look (see Phase 1).

## Phase order (strictly sequential)

### Phase 0 — Lock baselines

- Create `loops/efforts/EFFORT_15_NOTES.md` (a skeleton is committed
  alongside this doc).
- Re-measure ZINC and llama.cpp on this exact M1 Max with the protocol above.
- Confirm chat output is coherent: it must contain `Paris`
  (Effort 11's accepted answer is `The capital of France is **Paris**.`).
- Capture ZINC `--profile` and identify which dispatch buckets dominate
  decode by ms. Note that `--profile` aggregates the ~9 s chat prefill and
  the 128-token decode — separate the decode-only slice; do not let prefill
  `commitAndWait` time mask the decode kernel ranking.
- Exit: both numbers, the coherence check, and the profile are written into
  the notes file. The gap versus llama.cpp is quantified as both absolute
  and percentage.

### Phase 1 — Identify hottest decode kernel

- From the Phase 0 profile, name the **single** dispatch bucket with the
  largest decode ms slice. For this MoE model the candidates are:
  - attention Q/K/V/O Q4_K matvec
  - flash attention (sliding-window decode)
  - router logits + top-k
  - routed expert gate/up (fused), GeGLU, expert down (Q5_1)
  - shared-expert gate/up/down
  - weighted expert-output accumulate
  - LM head (vocab projection)
- Inverse-gate check: grep `n_experts == 0`, `architecture == .qwen2`, and
  `architecture != .gemma` in `src/compute/forward_metal.zig`. The
  `gemma4-12b` MoE config (`n_experts == 8`) cannot reach dense-only fast
  paths such as the `architecture == .qwen2 and n_experts == 0` and
  `architecture == .gemma and n_experts == 0` batched/decode branches. If a
  dense fast path has a routed-MoE equivalent that is missing, that is a
  structural lever — record it in the notes file.
- Write a hypothesis in the notes file:
  "If kernel X drops by Δms per step, end-to-end decode rises by Δ%."

### Phase 2 — Compare against llama.cpp's kernel for the same shape

llama.cpp is locally cloned at `/Users/stepan/Workspace/llama.cpp`. Use it as
read-only reference. Do not copy code verbatim (different license posture,
different graph builder). Extract the **technique**, then reimplement in the
ZINC kernel.

#### Where llama.cpp Metal code lives

- All Metal shaders are inlined as a string in one big `.metal` file:
  `~/Workspace/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal`
- Host-side dispatch and pipeline construction:
  - `ggml-metal-ops.cpp` — graph op → kernel selection (best place to learn
    *which* kernel llama.cpp picks for which shape).
  - `ggml-metal-device.m` — pipeline state construction, function constants.
  - `ggml-metal-context.m` — command buffer plumbing.
- Useful one-liners:
  ```bash
  # MoE expert matvec (decode, N=1) and grouped matmul (prefill)
  grep -nE "kernel_mul_mv_id|kernel_mul_mm_id" \
    ~/Workspace/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal

  # Which dispatch shape llama.cpp picks for MoE / expert ops
  grep -nE "mul_mat_id|MUL_MAT_ID|n_expert" \
    ~/Workspace/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp

  # Flash attention (sliding-window vector decode kernel)
  grep -nE "kernel_flash_attn_ext_vec" \
    ~/Workspace/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal
  ```

#### ZINC ↔ llama.cpp kernel map for Gemma 4 12B MoE decode

| Op (decode, N=1) | ZINC shader (`src/shaders/metal/`) | llama.cpp kernel (`ggml-metal.metal`) |
|---|---|---|
| Q4_K matvec — attn Q/K/V/O | `dmmv_q4k.metal`, `dmmv_q4k_qk_dual.metal` | `kernel_mul_mv_q4_K_f32`, `kernel_mul_mv_ext_q4_f32_disp` |
| Flash attention (SWA decode) | `flash_attn.metal` | `kernel_flash_attn_ext_vec`, `kernel_flash_attn_ext_vec_reduce` |
| Router logits + top-k | `softmax_topk.metal` | `ggml_soft_max` + top-k argsort |
| Routed expert gate/up (fused) | `dmmv_q4k_moe_gate_up.metal`, `dmmv_q4k_moe_gate_up_geglu.metal` | `kernel_mul_mv_id_*` (q4_K id matvec) |
| Expert GeGLU activation | `geglu.metal` | fused into `mul_mat_id` consumer |
| Routed expert down (Q5_1) | `dmmv_q5_1_moe.metal`, `dmmv_q5_1_moe_cols.metal` | `kernel_mul_mv_id_*` (q5_1 id matvec) |
| Weighted expert accumulate | `moe_accumulate.metal`, `moe_weighted_acc_scaled.metal`, `moe_weighted_acc_shared.metal` | expert-weighted sum in `build_moe_ffn` |
| Shared expert gate/up/down | `dmmv_q4k_moe_gate_up*.metal`, `dmmv_q5_1_moe.metal` | dense `kernel_mul_mv_*` |
| LM head (vocab projection) | `dmmv_q4k_lmhead.metal`, `dmmv_q4k_lmhead_norm.metal`, `dmmv_q4k_lmhead_1024.metal` | `kernel_mul_mv_q4_K_f32` |
| RMS norm + residual / fused mul | `rms_norm_mul.metal`, `residual_rms_norm.metal`, `add_rms_norm.metal` | `kernel_rms_norm_fuse_impl` |
| RoPE | `rope_native.metal`, `rope_fused.metal`, `rope_kv_cache_write.metal` | `kernel_rope_norm`, `kernel_rope_neox` |

#### What to record in the divergence note

For the single hottest kernel only:

- Threadgroup memory usage (bytes per threadgroup).
- Use of simdgroup primitives: `simd_sum`, `simd_shuffle_*`, `simdgroup_matrix`.
- Tile shape: output rows per threadgroup, how K is split, expert-per-grid
  mapping for the `_moe` kernels.
- Vector type and width: `half4`, `float4`, packed `block_q4_K` / `block_q5_1`
  access.
- Prefetch / `threadgroup_barrier` vs `simdgroup_barrier` placement.
- Function constants llama.cpp uses to specialize variants.

#### Output

One-page note in `EFFORT_15_NOTES.md`:

```
## Cycle N — Phase 2 divergence note: <kernel name>

ZINC current approach:
  - <bullets>

llama.cpp approach (file:line):
  - <bullets>

Biggest divergence (the lever for Phase 3):
  - <one sentence>

Why this should help on M1 Max (Apple7) specifically:
  - <one sentence>
```

### Phase 3 — Implement one change, microbench first

- Implement exactly **one** technique from the divergence note. Do not
  bundle changes.
- Land it as a new kernel variant or a shape-routed branch. Do **not**
  replace the working kernel until the new one is proven.
- Microbench the new path on the exact hot shape using `bench-metal-shapes`.
  Extend that target with the Gemma 4 12B MoE hot shapes if they are not
  present:
  ```bash
  zig build bench-metal-shapes -- \
    --model ~/Library/Caches/zinc/models/models/gemma4-12b-q4k-m/model.gguf \
    --case <case> --iterations 100 --warmup 10
  ```
- Gate: microbench must beat the existing kernel by **≥5%** with
  byte-identical greedy chat output on the canonical prompt.
- If the gate fails: revert, write what was learned, return to Phase 2 with
  the next divergence on the list.

### Phase 4 — Whole-model validation

- Run the full measurement protocol with the new kernel enabled.
- Gate: 3-run median decode tok/s beats the prior baseline by **≥2×** the
  observed run-to-run noise (typically ≥2% absolute).
- Gate: greedy chat output is byte-identical against the prior baseline on
  the canonical prompt, and still contains `Paris`.
- If both gates pass: commit. Local commit only. Do not push, do not open a
  PR.
- If either gate fails: revert, log the result, return to Phase 2.

### Phase 5 — Loop to Phase 1

- Re-run `--profile`. The hottest kernel will have shifted.
- Apply Phases 1 through 4 on the new top kernel.
- Exit conditions are the termination block below.

## Hard rules

1. No regression land. Never merge a change that loses decode tok/s.
2. No correctness drift. Greedy chat output must be byte-identical against
   the pre-change baseline on the canonical prompt.
3. Microbench before whole-model. No exceptions.
4. One change per cycle. Bundles invalidate measurement.
5. Metal-only scope. Edits restricted to:
   - `src/shaders/metal/`
   - `src/metal/`
   - Metal branches inside `src/gpu/` and `src/compute/forward_metal.zig`
   - the `bench-metal-shapes` benchmark target if shapes need adding
6. No new external dependencies. Zig and Metal only.
7. One commit per accepted change. Measurement numbers (3-run median before,
   3-run median after) belong in the commit message.
8. 3-run median, not a single lucky run. Always warm up first.

## Scope boundary checklist (must answer yes to all before editing)

- [ ] Is the file under one of the Metal scopes listed in rule 5?
- [ ] Will this change affect only the Gemma 4 MoE path, or is it
      shape/arch-routed so the dense Gemma, Qwen, and 35B-A3B paths
      keep their current kernels?
- [ ] Is there a kill-switch (flag or shape gate) to revert at runtime if a
      regression surfaces later?

## Reference — Effort 11 (same model, M4)

Effort 11 (`MULTI_HOUR_EFFORT_11_METAL_GEMMA_M4.md`) optimized this exact
model on M4. Its kernels are shared code, so M1 inherits all of them. Read
its **"Known dead ends"** and **"Already landed foundations"** sections
before proposing a change.

Caveat: Effort 11's dead ends were measured on Apple9 / M4 (~546 GB/s, ~40
GPU cores). This is Apple7 / M1 Max (~400 GB/s, 32 GPU cores). A few M4 dead
ends may behave differently here — retrying one is allowed, but only with a
fresh M1 `bench-metal-shapes` number that justifies it. Do not retry an
Effort 11 dead end on a hunch.

The GPU-routed MoE, batched top-k routing, route packing, and the
`ZINC_GEMMA_MOE_VALIDATE=1` parity mode have all landed — do not
reimplement them.

## Termination

Exit on any one of:

- ZINC decode tok/s on this M1 Max matches or exceeds llama.cpp on the same
  machine and model.
- Three consecutive Phase 3 attempts on three different kernels all fail the
  microbench gate.
- Six total accepted Phase 4 cycles complete.

On exit, the notes file's "Outcome" section must contain:

- starting decode tok/s
- ending decode tok/s
- decode tok/s at each accepted cycle
- list of attempted-and-reverted changes with reasons
- the single most useful learning from the effort

## Why this plan is safe to run unattended

- Every change pays for itself with measured evidence before it lands.
- Scope is fenced to Metal-only and Gemma-4-MoE-only, so failures cannot
  damage other targets.
- Hard cycle cap prevents runaway spend if nothing works.
- Correctness gate is byte-identical greedy chat output, which is not
  game-able by a kernel that is only "close enough."
