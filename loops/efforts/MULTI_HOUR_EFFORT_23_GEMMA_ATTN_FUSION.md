# Effort 23 — gemma attention-path kernel fusion (continue the dense decode lever)

> **Status:** OPEN. Scoped continuation of the two fusions already on main
> (`d134c5a1`): `rms_norm_residual` (post-attn/ffn norm+residual) and
> `rms_norm_rope` (per-head Q/K norm+RoPE) → gemma-31b dense ~28→~30 tok/s
> (~+7%). This effort fuses the **remaining small per-layer dispatches** on the
> gemma attention path, same lever.

Date opened: 2026-06-10. Pairs with `forward_cuda_gemma.zig` + `kernels.cu`.

## Why (the lever, already proven)
gemma decode is **boost-saturated but launch-latency/occupancy bound** (NOT
sync-bound — async is a proven NO-OP/regression, do not touch the command ring).
The only lever is **fewer/fatter kernels**: collapse tiny per-layer dispatches so
the GPU is fed. Two fusions landed (~+4% and ~+6%, stacking to ~+7%). A handful
of small dispatches remain.

## Targets (one per cycle, smallest-risk first)
1. **Per-head V rms_norm (`rms_norm_noweight`) + KV-cache write.** V is normed
   (no weight, never roped) then written to the KV cache — two tiny launches.
   Fuse into one `rms_norm_kvwrite` kernel (norm per head_dim, write to kv_v at
   `pos*kv_dim`). Bit-equivalent.
2. **Pre-attention `rms_norm` → fold the residual-stream read** if a write+read
   round-trip can be dropped (the norm output feeds Q/K/V matvecs; check whether
   a fused norm-into-first-matvec is worth it, else skip — matvecs dominate).
3. **FFN path**: any remaining `rms_norm`/elementwise pairs not already covered
   by `rms_norm_residual` (gemma FFN: pre-ffn norm → GeGLU → post-ffn fused).

## Hard rules (from memory — violating wastes the cycle)
- **Build:** isolated caches ALWAYS — `ZIG_LOCAL_CACHE_DIR=/tmp/lc-$RANDOM
  ZIG_GLOBAL_CACHE_DIR=/tmp/gc-$RANDOM ~/zig-0.15.2/zig build cuda-dbg
  -Dbackend=cuda -Dshaders=false`; verify the binary hash CHANGED.
- **Do NOT async gemma** (boost-saturated; regression). Fusion only.
- **4090-pinned** (`GPU-e59a6fce-…`); isolated box dir (`~/zinc-e23`), never
  `~/workspace/zinc`; do not disturb the parallel 5090 work.
- **Boost noise:** interleaved back-to-back A/B (NGEN≥160), same thermal state.
  Never claim a win from one boosted run.

## Validation contract (per cycle)
- `scripts/validate_catalog.sh` → **5/5 token-correct** (the fused kernel must be
  bit-equivalent: token-identical to the pre-fusion path). Break correctness →
  REVERT, document, do not commit.
- Interleaved A/B vs the pre-cycle binary; require a measurable, repeatable gain.
- Commit ONLY this effort's change to `perf/e23-<target>`; push (NOT main).
  Update the cycle log below. Negative result → revert + log it.

## Cycle log
- **2026-06-11 — Target 1 DONE (V rms_norm+KV-write fuse + K KV-write fold). VALIDATED WIN, branch `perf/e23-vk-norm-fuse`, pushed (NOT main).**
  New kernel `rms_norm_kvwrite` (kernels.cu): per-head V plain-normalize (no
  weight) writing STRAIGHT into the V cache at `pos*kv_dim` — replaces the
  `rms_norm_noweight(→v_buf)` + the V half of `kv_cache_write`. Also extended
  `rms_norm_rope` with a `dst_offset` push field so the K head writes its
  norm+roped result straight into `kv_k` at `pos*kv_dim` (Q passes 0 → own
  buffer), folding the K half of `kv_cache_write`. Net: the standalone
  `kv_cache_write` launch + the V round-trip are gone → **2 fewer tiny
  dispatches/layer** on the gemma attention path (both dense + MoE). Bit-equivalent
  (identical normalization, identical cache layout). `kv_cache_write` pipeline
  removed.
  **Build:** isolated caches; variant md5 `8e098573…` ≠ baseline `4278b4c6…`
  (real recompile, not stale). **Correctness:** `validate_catalog` → **5/5
  token-correct** (qwen 3×12/12; gemma4-26b 12/12; gemma4-31b teacher-forced
  11/12 = the documented near-tie, unchanged). **Perf (interleaved back-to-back
  A/B, NGEN=160, 4090, warmup ignored):** gemma-31b DENSE variant wins **4/4
  rounds** — 29.90/29.43, 29.81/29.39, 29.84/29.42, 29.89/29.39 → mean 29.86 vs
  29.41 = **+1.54%** (rock-steady, no boost noise). gemma-26b MoE (same attn-path
  fusion, FFN untouched) 3 rounds 52.18/52.84, 50.65/48.86, 54.08/48.48 → mean
  52.30 vs 50.06 = **+4.48%, net-positive, NO regression** (round 1 near-tie,
  boost-variable). Complements the two landed fusions (norm+residual, qk-norm+rope)
  — different part of the layer, stacks.
  **NEXT (Target 2/3):** pre-attention `rms_norm` fold into the first Q/K/V matvec
  (likely skip — matvecs dominate), or any remaining FFN norm/elementwise pairs.
