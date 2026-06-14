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

### Cycle 1 — 2026-06-13 — CONFIRM THE REGIME (+ took the free TC win). **PREMISE REFUTED; VALIDATED +25% TC win committed.**

**THE BIG FINDING — the effort's core premise is WRONG.** Re-profiled gemma-31b batched prefill on the **5090** (fine 30–50 ms `nvidia-smi -i 0` sampling across pp256/512/1024, isolated-cache build md5 `8c24738b`): during the compute window the GPU runs at **~100% util, full 2850 MHz boost, ~380–418 W** — it is **COMPUTE-bound, NOT launch-bound (~10% util)**. The earlier "7–12% util / 76 W" reading (Effort 25) was the model-load + clock-ramp phase, not steady-state compute. zinc prefill: 185/200/199 tok/s at pp256/512/1024 (scales flat = compute-saturated). **Therefore T1 (CUDA graphs over the prefill chain) is a DEAD END** — there are no launch bubbles to reclaim; same end-to-end-neutral fate as Effort 24's GEMM, for the opposite reason than documented (not "GPU idle" but "GPU saturated").

**llama baseline corrected.** llama-bench on the **same 5090 + same gguf**: gemma-31b prefill **pp256 2890 / pp512 3542 / pp1024 3499 t/s**. The effort file's cited "381.5" was a `performance_suite.mjs` artifact, not a fair warm prefill-throughput number. **The real gap is ~17×** (zinc ~200 vs llama ~3500), worse than the documented 0.15×. The gap is **kernel efficiency** — zinc runs a hand-written Q4_K-dequant tiled GEMM (`gemm[idx]`, 64×64, no tensor cores by default), llama runs cuBLAS/MMQ-class tensor-core GEMM.

**The free TC win (cycle-1 plan: "if TC is a real GEMM win, take it").** Effort 24's `ZINC_BATCHED_TC` (fp16 tensor-core dense GEMM, default-OFF, found neutral on the 4090) is a **real win on the 5090's Blackwell fp16 tensor cores.** A/B (ABBA x3, pp512, 5090): gemma-31b prefill **194.5 → 241.3 = +24%** (opt-in measure), and with the committed default-on binary **202.2 → 251.9 = +25%**. gemma-26b (MoE) neutral — TC only touches dense Q4_K GEMMs; its FLOPs are in experts. **Change:** flip `use_tc` to default-ON via `tcDefaultOn()` (opt out `ZINC_BATCHED_TC=0/off/false/no`); scoped to gemma prefill, qwen + all decode untouched. **Build** default-cache md5 `562a6f3b` (≠ TC-off base). **Catalog 5/5 token-correct** with TC as the default (5090; gemma4-31b 12/12, gemma4-26b 12/12). **Does NOT beat llama** (+25% → 252 vs 3542) but it's free and correct.

**Branch `perf/e26-tc-default-prefill` (commit `09895d36`, pushed, NOT main).** Disposition: validated win taken; T1 (prefill graphs) abandoned as a dead end (premise refuted).

**NEXT (redirect the effort):** the lever to actually beat/close-the-gap on gemma prefill is a **more efficient batched GEMM** — i.e. understand why zinc's TC GEMM (252 t/s) is still 14× behind llama's MMQ (3542). Candidates: (a) the default Q4_K TC kernel is occupancy/traffic-bound (the cycle-15 lowsmem kernel is hand-tuned but no async-copy/wgmma); (b) attention (`gemma_attention_batched`) may dominate, not the GEMM — **profile per-kernel time (nsys/per-kernel timers) BEFORE more GEMM work** to find the true prefill bottleneck (Effort 24 assumed GEMM and got neutral on the 4090 partly because it never confirmed GEMM was the critical path). T2 (MoE decode) is the other open structural gap and is independent of this finding.

### Cycle 2 — 2026-06-14 — PER-KERNEL PREFILL PROFILE (did cycle-1's NEXT) + 2 logged GEMM-knob negatives. **GEMM confirmed = 89% of prefill; attention only 11%. No code committed (profiling harness-only; both ready knobs in-noise).**

**TOOLING NOTE:** `nsys` is UNUSABLE on this WSL2 box — it prints "Connection timed out" (update/license check blocked by the `127.99/16` net-fence) and never produces a `.nsys-rep`; tried twice with `--trace=cuda --sample=none --backtrace=none`, both hung. **Workaround: HARNESS-ONLY host-timed sync points** (`ZINC_PREFILL_PROF` env; `waitPending()` + `std.time.nanoTimestamp()` around `attentionLayerBatched`/`ffnBlockBatched` and a per-layer command-split isolating the `gemma_attention_batched` dispatch). The forced syncs cost ~3% (prof prefill 244 vs 252 base) — fine for a relative split. Profiling code was reverted on the Mac worktree (git checkout) and the box source (restored from backup); box rebuilt back to the clean production binary `562a6f3b`. **DO NOT commit profiling instrumentation.**

**THE PROFILE (gemma-31b, T=512, 60 layers, 5090, production binary `562a6f3b` = TC default-on):**
- **FFN block (gate/up/down dense TC GEMMs + GeGLU + norms): 1119 ms = 58%**
- **Attn projections (Q/K/V/O dense TC GEMMs) + per-head norm/RoPE/KV-write: 607 ms = 31%**
- **Attention softmax kernel (`gemma_attention_batched`): 210 ms = 11%**
⇒ **~89% of prefill is the dense GEMM** (FFN 58% + attn-proj ~31%); the attention softmax is only 11%. **This RESOLVES the cycle-1 open question** (was the GEMM or attention the critical path?): the GEMM is. Effort 24's GEMM focus was on the right path; **attention is NOT worth optimizing** (even halving the 210 ms = only +5.5% end-to-end).

**NEGATIVE 1 — `ZINC_BATCHED_TC_M128_LOWSMEM` (the cycle-17 synthesis kernel: wider 128×64 M-tile + 12 KB shared, halves the dominant f16-A re-read; byte-identical) is NEUTRAL on the 5090.** A/B ABBA×3, pp512: default(m64-lowsmem) median 249.2 vs m128-lowsmem median 252.7 = **+1.4% (in-noise)**; GEN_IDS identical (494). **The fact that halving the activation (A) re-read traffic gave nothing means the GEMM is NOT A-traffic-bound — it's compute- and/or weight-traffic-bound** (the in-kernel Q4_K→fp16 dequant + fp16 wmma, and the 4-bit weight read). Stays opt-in.

**NEGATIVE 2 (borderline) — `ZINC_BATCHED_TC_NORMF16` (norm/GeGLU producers emit fp16 directly so every dense TC GEMM skips its redundant f32→fp16 recast pass; byte-identical) does NOT robustly clear noise.** pp512 ABBA×3: default median 244.3 vs NORMF16 254.2 (+3.5% but ranges overlap). pp1024 ABBA×4 (cleaner/longer): default median 237.4 vs NORMF16 240.3 = **only +1.0%, at the floor**; GEN_IDS identical (1024 token). Consistent with the cycle-23 lesson (single-launch fusions need locked clocks to clear this box's ±1% boost floor — and no `nvidia-smi -lgc` without sudo) and with Effort 24's 4090-neutral. **NOT flipped to default** (would be an unvalidated win). Recast overhead is a small slice of the 89% GEMM, so eliding it can't move the needle while the GEMM math dominates.

**CONCLUSION / REDIRECT:** the 14× gap is the GEMM **math/weight path**, not launch overhead (cycle 1), not activation traffic (neg 1), not recast overhead (neg 2), not attention (11%). The only lever that can close it is an **MMQ-style INT8 tensor-core GEMM that matmuls the quantized Q4_K weights directly** (`mma.sync` int8, no fp16 dequant — what llama's MMQ does), or async-copy/wgmma pipelining of the existing fp16 kernel. That is a multi-cycle kernel build, the correct next target for gemma prefill. T2 (MoE decode gap) remains the independent alternative. **All micro-opt knobs (m128 traffic, recast-elision, attention) are now confirmed dead/marginal — do not re-litigate them.**

**Disposition:** no perf branch (no validated code win); this is a logged diagnostic + 2 negatives. Catalog correctness untouched (production binary unchanged; all A/B GEN_IDS identical).
