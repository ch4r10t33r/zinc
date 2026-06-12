# Effort 24 — CUDA batched prefill (wire the 5.9× gemm_*_tiled_v2 GEMMs)

> **Status:** OPEN. The top untapped CUDA prefill win. The validated
> `gemm_q4k/q5k/q6k_tiled_v2` GEMMs (Y[T,M]=A[T,K]·W[M,K]^T, ~9300 GFLOP/s,
> **5.9× over per-token matvec**) are in `kernels.cu` but UNWIRED. Production
> prefill (`main.zig`) runs `decodeStep`/`prefillStep` once per prompt token →
> re-reads every weight T times. Batch it: read each weight ONCE for all T
> tokens. Full design in memory `project_batched_prefill_design`.

Opened 2026-06-12. Stacks on the head-skip (`c78e64ed`, already on main).

## SCOPE — Phase 1: gemma-31b DENSE only (pure transformer → clean GEMM win)
Qwen is hybrid-SSM (3/4 layers = sequential state scan) and is OUT of scope
until a chunked/parallel SSM scan exists. gemma-26b MoE = Phase 2 (later). Do
gemma-31b dense first.

## APPROACH — ADDITIVE (forward_cuda_gemma.zig is the parallel gemma work's HOT
## file: add NEW methods + a toggle; do NOT rewrite existing decodeStep/blocks)
Add `ForwardGemma.prefillBatched(tokens) !u32` behind env `ZINC_BATCHED_PREFILL`:
process all T prompt tokens at once, return the last token's argmax (= first gen
token). Per layer:
- batched norms: loop T or a T-batched rms variant (cheap; not the bottleneck)
- Q/K/V + O projections, FFN gate/up/down: `gemm_*_tiled_v2`
  (`GemmPush{M,K,T,a_offset,x_offset,y_offset,acc_mode}`) over the T tokens
- attention: **Target 1 LOOPS** the existing single-query attention per token
  (each query masked to [0..its pos]) — defer the batched-attn kernel
- KV write: all T positions (batched, or loop)
Then the last token's existing tail (the head). Wire `main.zig` prefill to call
`prefillBatched` when `ZINC_BATCHED_PREFILL` is set, else the current per-token
loop. (The duck-typed prefill fn already special-cases the last token.)

## TARGETS (one per cycle; each must pass the gate before any commit)
1. **Batched skeleton** — `prefillBatched` with LOOPED attention + GEMM
   projections/FFN. The big first step; may span >1 cycle (use the cycle log).
2. **Batched causal-attention kernel** — replace the per-token attention loop
   (grid=(n_head,T), each query masked to its pos; or flash-style online softmax).
3. **Batched KV write** — all T positions in one launch (if still looped).
4. (later) gemma-26b MoE prefill; tensor cores via NVRTC `-I` (+2.2×).

## GATE (NON-NEGOTIABLE — the batched path must be OUTPUT-IDENTICAL)
- `ZINC_BATCHED_PREFILL=1 dbg_cuda gen <prompt> N <model>` GEN_IDS must be
  **byte-identical** to the per-token path. Extend `scripts/prefill_catalog.sh`
  to A/B batched-vs-per-token (it already A/Bs head-skip the same way). Mismatch
  → REVERT.
- `scripts/validate_catalog.sh` → 5/5 token-correct vs llama.cpp.
- Measure prefill tok/s via `scripts/prefill_catalog.sh` (ABBA-counterbalanced).

## HARD RULES (from memory — violating wastes the cycle)
- **Build:** isolated caches; verify the binary hash CHANGED.
- **Box:** 4090-pinned (`GPU-e59a6fce-1961-bafe-927c-06c0149f2370`); isolated
  dir `~/zinc-e24`, NEVER `~/workspace/zinc` (parallel 5090 research).
- **Coordination:** main moves FAST (active parallel 5090/gemma work). Branch
  off the LATEST origin/main each cycle, rebase often, keep changes ADDITIVE
  (new methods + toggle) to minimize conflicts. NEVER roll back parallel work.
- Commit only a validated, output-identical, FASTER increment to `perf/e24-*`;
  push (NOT main). Incomplete/negative → log it in the cycle log + continue.

## EXPECTED
gemma-31b prefill ~15-35 → ~90-200 tok/s (5.9× on the GEMM-able majority;
attention is a smaller FLOP share). Stacks on the head-skip's +4%.

## CYCLE LOG
- (none yet)
