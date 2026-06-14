# Effort 26 — Beat llama.cpp on every catalog row where ZINC trails (prefill graphs + MoE decode)

> **Status:** 🔬 OPEN (spawned 2026-06-13 from the published RTX 5090 catalog). Goal: ZINC > llama.cpp on EVERY row. Runs on the **5090**; the parallel Effort-25 decode loop owns the **4090** — do not disturb it.

Forward paths: `src/compute/forward_cuda_gemma.zig` (gemma4 dense+MoE), `src/compute/forward_cuda.zig` (qwen35/36 hybrid-SSM). Graph/stream layer: `src/cuda/command.zig`, `src/cuda/cuda_shim.c` (Effort 25 already wired `cuStreamBeginCapture`→`cuGraphExec`→`cuGraphLaunch` for DECODE — reuse it).

## The gap (RTX 5090, 2026-06-13, zinc vs llama tok/s — the rows to beat)

**PREFILL (the big gaps):**
- `gemma-4-31B` dense — **57.3 vs 381.5 (0.15×)** ⚠️ biggest gap
- `gemma-4-26B-A4B` MoE — **106.3 vs 413.9 (0.26×)** ⚠️
- `qwen36-35B-A3B` MoE — 45.5 vs 48.3 (0.94×) — close
- (qwen35-9B 1.14× ✓ and qwen36-27B 1.66× ✓ already BEAT llama — leave them.)

**DECODE:**
- `gemma-4-26B-A4B` MoE — **31% of llama** ⚠️ ; `qwen36-35B-A3B` MoE — **42%** ⚠️ (the big decode gaps)
- gemma-31B 82%, qwen35-9B 75%, qwen36-27B 91% (close; lower priority)

The two structural gaps: **gemma PREFILL** (pure-transformer GEMM prefill, where llama runs cuBLAS-class kernels) and **MoE DECODE**. (Qwen prefill already wins only because qwen36 is hybrid-SSM — slow prefill in BOTH engines; that's not a real lead, just a slow llama baseline.)

## Why we're behind, and the lever (the finding that directs this effort)

**Gemma prefill is LAUNCH-BOUND, not compute-bound.** Effort 25 profiled a batched gemma-31B prefill (T=413) at **7–12% GPU util / ~76 W of 575 W — the GPU sits ~90% idle.** So faster GEMM kernels are the WRONG move: Effort 24 proved the batched/tensor-core GEMM is 5.9× isolated but **end-to-end NEUTRAL** (the GPU is idle between kernels, not crunching). The fp16 tensor-core GEMM (`ZINC_BATCHED_TC`, default OFF) compiles and is ~2.2× isolated but is end-to-end neutral for the same reason.

**The lever is the SAME as Effort 25's decode win: CUDA GRAPHS.** Capture the per-layer prefill kernel chain once, `cuGraphLaunch` to replay it, collapsing ~16 launches/layer × 60 layers of per-kernel launch + inter-kernel-bubble latency. The 90% idle is the 4–7× headroom needed to beat llama. Effort 25 already built the capture machinery for decode — extend it to prefill.

## Targets (priority order; the BAR is BEAT llama on the 5090, interleaved-A/B-confirmed)

- **T1 — Gemma prefill via CUDA graphs (PRIMARY).** Extend Effort 25's `cuStreamBeginCapture`/`cuGraphExec` to `prefillBatched` in `forward_cuda_gemma.zig`. **KEY DESIGN:** prefill replays the chain ~ONCE per prompt, so a single whole-prompt graph isn't amortized (capture cost ≈ one replay). **CHUNK** prefill into fixed-size token chunks (e.g. C=128/256), capture the per-C-chunk per-layer chain ONCE, replay it `ceil(T/C)` times — amortizes the instantiate AND cuts the launch chain C-fold. Per-chunk topology is invariant (C fixed); KV cache + causal attention already handle cross-chunk dependencies (each chunk attends prior chunks via the cache). Per-chunk-varying scalars (chunk base position, KV offset) ride device-buffer reads or `cuGraphExecKernelNodeSetParams` so topology stays invariant (same trick decode used). Beat llama: gemma-31B > 381, gemma-26B > 414.
- **T2 — MoE decode gap (gemma-26B 31%, qwen36-35B-A3B 42%).** The router keeps a per-layer host sync and the small-M routed-expert matvecs are occupancy-starved. Levers: (a) **GPU-side expert gather** — drop the last router host round-trip (the e25 doc's MoE-graph blocker), so the MoE step is sync-free; (b) **wider/multi-row expert kernels** for the small-M matvecs (more output rows/block → occupancy); (c) **uniform-quant MoE** — the catalog 35B-A3B has MIXED expert quants (q4k/q5k/q6k) which blocks graph capture; a per-quant-group batched expert kernel (or a requant) unblocks the MoE decode graph. Token-correct gate.
- **T3 — qwen36-35B-A3B MoE prefill (0.94× → beat).** Just +6%. Same chunked-prefill-graph lever as T1 over the MoE prefill chain (batched experts already default-on).
- **T4 (stretch) — dense decode last mile (gemma-31B 82%, qwen35-9B 75%, qwen36-27B 91%).** e25 graphs are size-gated (help 9B, fade at 27B); the lever here is deeper kernel fusion (fewer/fatter launches — the Effort 23 playbook). Lower priority (already close).

## Plan (incremental, validate-before-commit; ONE target per cycle)

1. **Cycle 1 — CONFIRM THE REGIME (cheap, do first):** re-profile gemma-31B prefill util on the 5090 (`nvidia-smi --query-gpu=utilization.gpu,power.draw,clocks.sm` DURING a batched prefill) to re-confirm launch-bound (~10% util). Then an interleaved end-to-end A/B of the TC path (`ZINC_BATCHED_TC` on/off) — **if TC is NOT neutral (a real end-to-end GEMM win), take it first, it's free**; if neutral (expected), graphs are confirmed as the only lever. Log util + the TC A/B in the cycle log.
2. **Cycle 2+ — T1 chunked-prefill graph capture.** Validate 5/5 token-correct + prefill A/B (util before→after, zinc vs llama). Commit `perf/e26-prefill-graph-<step>`.
3. Then T2, T3, T4.

## Validation contract

- `scripts/validate_catalog.sh` 5/5 token-correct (`ZINC_GPU` = the 5090 UUID). Graph replay / new kernels must be bit-equivalent (or within the documented gemma near-tie tolerance). A token divergence = bug → fix or revert.
- **The BAR is BEAT LLAMA:** a target is "done" only when zinc tok/s > llama.cpp tok/s on that row, on the SAME 5090 + same gguf, interleaved-A/B-confirmed (beat boost noise). Use `~/workspace/llama.cpp/build/bin/llama-bench` or the perf suite for the llama baseline.
- Profile util before→after — the whole point is moving util UP; a "win" that doesn't raise util is suspect.
- Isolated-cache builds (`ZIG_LOCAL_CACHE_DIR`+`ZIG_GLOBAL_CACHE_DIR`, verify the binary md5 changed or you are measuring stale code).

## HARD RULES

- **5090-pinned** (UUID `GPU-5126d018-ec86-be8b-1bf5-b5ac323d3350`). The parallel **Effort-25 decode loop owns the 4090** (`GPU-e59a6fce-…`) + `~/zinc-e25` — DO NOT touch it. Util-gate A/B rounds (`--query-gpu=utilization.gpu`, NOT `--query-compute-apps` — hidepid hides foreign procs); skip a round if the 5090 is contended.
- Isolated box dir `~/zinc-e26`, **never** `~/workspace/zinc`. Box gotchas: `DECODE/PREFILL tok/s` print is on STDERR (`2>&1`); `nohup … >FILE 2>&1 &` + poll the FILE (a backgrounded `ssh '… bash'` orphans its remote script); `pkill -f <pat>` self-matches the ssh argv (kill by PID); gemma-31B `gen` reloads 18 GB/call (~45 s).
- **DO NOT async gemma decode** (boost-saturated, proven regression — Effort 23/25). Prefill graphs are fine (prefill is launch-bound).
- Branches not main; validate before commit; commit ONLY the scoped change; never commit host/IP/port; a neutral cycle is a logged negative (valuable) → revert the code.

## Cycle log

(append dated entries per cycle: cycle | target | change | built+md5-changed? | catalog 5/5? | A/B vs llama (util before→after, zinc tok/s vs llama tok/s) | branch/sha or revert+why | next)
