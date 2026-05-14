# Effort 14 — Apple M1 Max Qwen3-8B decode: close gap to llama.cpp

## Objective

Move local Qwen3-8B-Q4_K_M decode tok/s on this M1 Max as close to local
llama.cpp on the same machine and model as possible, without regressing
correctness or any other platform target.

Stop conditions:

- ZINC decode tok/s matches or exceeds local llama.cpp on this machine, or
- three consecutive Phase 3 attempts fail the microbench gate, or
- six total accepted cycles complete (autonomous-spend cap).

## Fixed inputs (do not drift)

- Machine: local Apple M1 Max, 32 GPU cores, 32 GB unified memory,
  `MTLGPUFamilyApple7`.
- Model id: `qwen3-8b-q4k-m`
- Model path: `/Users/stepan/Library/Caches/zinc/models/models/qwen3-8b-q4k-m/model.gguf`
- Backend: Metal only. Do not touch Vulkan, RDNA shaders, the 35B-A3B path,
  or any non-Metal builder.
- Comparator: `/Users/stepan/Workspace/llama.cpp/build-metal/bin/llama-cli`
  against the exact same GGUF.
- Canonical prompt: `"The capital of France is"`
- Generation length: `-n 128`
- Sampling: greedy, `--temp 0`
- Build: `zig build -Doptimize=ReleaseFast` only. `--profile` numbers from
  any other build are not accepted as evidence.

## Measurement protocol

Run this exactly before every accept/reject decision in any phase.

1. Clean stragglers:
   ```bash
   pkill -f 'zig-out/bin/zinc' || true
   pkill -f 'llama-cli' || true
   ```
2. 1 warmup run, then 3 timed runs. Report the **median** decode tok/s.
   Single-run numbers are not accepted as evidence.
3. ZINC command:
   ```bash
   ./zig-out/bin/zinc \
     --model-id qwen3-8b-q4k-m \
     --prompt "The capital of France is" \
     -n 128 \
     --profile
   ```
4. llama.cpp command:
   ```bash
   ~/Workspace/llama.cpp/build-metal/bin/llama-cli \
     -m /Users/stepan/Library/Caches/zinc/models/models/qwen3-8b-q4k-m/model.gguf \
     -p "The capital of France is" \
     -n 128 -ngl 99 -fa on --temp 0 --no-warmup
   ```
5. Log into `loops/efforts/EFFORT_14_NOTES.md`:
   - decode tok/s (3 runs, median)
   - prefill tok/s (3 runs, median)
   - ZINC profile dispatch-byte breakdown
   - any anomalies (process collisions, thermal warnings, residency
     pressure warnings, etc.)

## Phase order (strictly sequential)

### Phase 0 — Lock baselines

- Re-measure ZINC and llama.cpp on this exact M1 Max with the protocol above.
- Capture ZINC `--profile` and identify which kernels dominate by ms.
- Exit: both numbers and the profile are written into the notes file. Gap
  versus llama.cpp is quantified as both absolute and percentage.

### Phase 1 — Identify hottest decoder kernel

- From the Phase 0 profile, name the **single** dispatch bucket with the
  largest decode ms slice.
- Cross-reference with the 8B GGUF tensor layout: lm_head, attn_qkv,
  attn_o, ffn_gate, ffn_up, ffn_down.
- Write a hypothesis in the notes file:
  "If kernel X drops by Δms per step, end-to-end decode rises by Δ%."

### Phase 2 — Compare against llama.cpp's kernel for the same shape

llama.cpp is locally cloned at `/Users/stepan/Workspace/llama.cpp`. Use it as
read-only reference. Do not copy code verbatim (different license posture,
different graph builder, different launch infrastructure) — extract the
**technique**, then reimplement in the ZINC kernel.

#### Where llama.cpp Metal code lives

- All Metal shaders are inlined as a string in one big `.metal` file:
  `~/Workspace/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal`
- Host-side dispatch and pipeline construction:
  - `ggml-metal-ops.cpp` — graph op → kernel selection (best place to learn
    *which* kernel llama.cpp picks for which shape).
  - `ggml-metal-device.m` — pipeline state construction, function constants.
  - `ggml-metal-context.m` — command buffer plumbing.
- Useful one-liners while doing Phase 2:
  ```bash
  # Find a kernel by name
  grep -n "kernel void kernel_mul_mv_q4_K_f32" \
    ~/Workspace/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal

  # Find which dispatch shape llama.cpp picks for q4_K matvec
  grep -nE "q4_K|MUL_MV_Q4_K" \
    ~/Workspace/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp

  # See all matvec kernels at once
  grep -nE "^kernel .* kernel_mul_mv_" \
    ~/Workspace/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal
  ```

#### ZINC ↔ llama.cpp kernel map for Qwen3-8B dense

| Op (decode, N=1) | ZINC shader (`src/shaders/metal/`) | llama.cpp kernel (`ggml-metal.metal`) |
|---|---|---|
| Q4_K matvec — FFN gate / up / down | `dmmv_q4k.metal`, `dmmv_q4k_dense_gate_up_geglu.metal` | `kernel_mul_mv_q4_K_f32`, `kernel_mul_mv_ext_q4_f32_disp`, `kernel_mul_mv_ext_q4x4_f32_disp` |
| Q4_K matvec — attn_qkv / attn_o | `dmmv_q4k.metal`, `dmmv_q4k_qk_dual.metal` | same as above |
| Q4_K matvec — lm_head (vocab proj) | `dmmv_q4k_lmhead.metal`, `dmmv_q4k_lmhead_norm.metal`, `dmmv_q4k_lmhead_1024.metal` | same as above |
| Flash attention (decode) | `flash_attn.metal`, `flash_attn_q8.metal` | `kernel_flash_attn_ext_vec`, `kernel_flash_attn_ext_vec_reduce` |
| RoPE (Qwen3 uses NORM/NeoX rope per arch) | `rope_native.metal`, `rope_fused.metal`, `rope_kv_cache_write.metal` | `kernel_rope_norm`, `kernel_rope_neox` |
| RMS norm + residual / fused mul | `rms_norm_mul.metal`, `residual_rms_norm.metal`, `add_rms_norm.metal` | `kernel_rms_norm_fuse_impl` |
| Embedding lookup (q4_K → f32) | `embed_dequant_q4k.metal` | `kernel_get_rows_q4_K` |

#### What to record in the divergence note

For the single hottest kernel only:

- Threadgroup memory usage (bytes per threadgroup).
- Use of simdgroup primitives: `simd_sum`, `simd_shuffle_*`, `simdgroup_matrix`.
- Tile shape: how many output rows per threadgroup, how K is split.
- Vector type and width: `half4`, `float4`, packed `block_q4_K` access.
- Prefetch / `metal::threadgroup_barrier` placement.
- Function constants (compile-time specialization) llama.cpp uses to pick
  between variants (`mul_mv_ext_q4_f32_disp` is a dispatcher into several
  specialized inner kernels — read its body before picking the lever).

#### Output

One-page note in `EFFORT_14_NOTES.md` formatted as:

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
- Land it as a new kernel variant or as a shape-routed branch. Do **not**
  replace the working kernel until the new one is proven.
- Microbench the new path on the exact hot shape using the existing
  `bench-metal-shapes` target (extend it with the 8B hot shapes if those
  aren't present yet).
- Gate: microbench must beat the existing kernel by **≥5%** with
  byte-identical greedy output on the canonical prompt.
- If gate fails: revert the change, write what was learned, return to
  Phase 2 with the next divergence on the list.

### Phase 4 — Whole-model validation

- Run the full measurement protocol with the new kernel enabled.
- Gate: 3-run median decode tok/s beats the prior baseline by **≥2×** the
  observed run-to-run noise (typically ≥2% absolute).
- Gate: greedy output is byte-identical against the prior baseline on the
  canonical prompt.
- If both gates pass: commit. Local commit only. Do not push, do not open
  a PR.
- If either gate fails: revert the change, log the result, return to
  Phase 2.

### Phase 5 — Loop to Phase 1

- Re-run `--profile`. The hottest kernel will have shifted.
- Apply Phases 1 through 4 on the new top kernel.
- Exit conditions are the termination block above.

## Hard rules

1. No regression land. Never merge a change that loses decode tok/s, even
   with explanation.
2. No correctness drift. Greedy output must be byte-identical against the
   pre-change baseline on the canonical prompt.
3. Microbench before whole-model. No exceptions.
4. One change per cycle. Bundles invalidate measurement.
5. Metal-only scope. Edits restricted to:
   - `src/shaders/metal/`
   - `src/metal/`
   - Metal branches inside `src/gpu/`
   - the `bench-metal-shapes` benchmark target if shapes need adding
6. No new external dependencies. Zig and Metal only.
7. One commit per accepted change. Measurement numbers (3-run median
   before, 3-run median after) belong in the commit message.
8. 3-run median, not a single lucky run.

## Scope boundary checklist (must answer yes to all before editing)

- [ ] Is the file under one of the four Metal scopes listed in rule 5?
- [ ] Will this change affect only the 8B path, or is it shape-routed so
      the 35B-A3B path keeps its current kernel?
- [ ] Is there a kill-switch (flag or shape gate) to revert at runtime if
      a regression surfaces later?

## Termination

Exit on any one of:

- ZINC decode tok/s on this M1 Max matches or exceeds llama.cpp on the
  same machine and model.
- Three consecutive Phase-3 attempts on three different kernels all fail
  to clear the microbench gate.
- Six total accepted Phase-4 cycles complete.

On exit, the notes file's "Outcome" section must contain:

- starting decode tok/s
- ending decode tok/s
- decode tok/s at each accepted cycle
- list of attempted-and-reverted changes with reasons
- the single most useful learning from the effort

## Status as of 2026-05-14 (after 21 loop cycles)

**Current state**: ~45.0 tok/s decode on Qwen3-8B Q4_K_M, M1 Max.
**Starting state**: 8.6 tok/s (cold) / 10.6 tok/s (warm).
**Improvement**: ~5× cold / ~4× warm.

**See `EFFORT_14_NOTES.md` for the full cycle history.** Before any new
cycle, read its "Consolidated cycle history" section. Key categorical
findings:

- **Dispatch-count reduction is NOT the lever on M1 Max** — falsified
  three independent ways (Cycles 1, 3, 4 of the loop, plus the pre-loop
  dispatch-consolidation and SwiGLU fusion changes). Do not propose
  another fusion that's primarily about saving dispatches.
- **precise → fast Metal math is essentially exhausted.** Grep
  `precise::` in `src/shaders/metal/` before proposing — most hot paths
  already use `fast::*`.
- **simdgroup-redundant-reduction pattern** has been applied to all
  four shaders that use broadcast barriers heavily (rms_norm_mul,
  residual_rms_norm, rope_kv_cache_write Q/K-norm, flash_attn). Look
  elsewhere.
- The biggest single win (Cycle 2, 5×) came from unblocking an
  optimized kernel that was artificially gated to `.gemma`. Keep
  hunting for those — grep `cfg.architecture == .gemma` and
  `architecture == .gemma` in `src/compute/forward_metal.zig`.

## Pivot priorities (post-Cycle-21)

The loop's stall-warning fired at Cycle 20+ because cycles 12-20
followed the same micro-template (precise→fast, eliminate broadcast
barriers). For categorical pivots, see
`EFFORT_14_NOTES.md` → "Still-open structural levers" — ordered by
expected impact. Top three for a fresh cycle:

1. **KV-cache quantization to q8_0** — llama.cpp uses this by default
   on Metal; ZINC's decode path still uses f16. Real bandwidth saving
   at any context length.
2. **Fused rms_norm + Q projection** — the Vulkan equivalent
   (`rms_norm_dmmv_q4k_alpha_beta`) shipped +0.6 tok/s on RDNA4. No
   Metal equivalent yet for the attention prologue.
3. **Single-kernel Q+K+V projection** — three outputs from one input
   read, no branch (unlike Cycle 1's failed Q+K dual which only fused
   two).

## Why this plan is safe to run unattended

- Every change pays for itself with measured evidence before it lands.
- Scope is fenced to Metal-only and 8B-only, so failures cannot damage
  other targets.
- Hard cycle cap prevents runaway spend if nothing works.
- Correctness gate is byte-identical greedy output, which is not
  game-able by a smart kernel that's "close enough."
