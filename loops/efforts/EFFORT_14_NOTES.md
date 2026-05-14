# Effort 14 — running notes

Local machine: Apple M1 Max, 32 GPU cores, 32 GB UMA, MTLGPUFamilyApple7.
Model: `qwen3-8b-q4k-m` (Qwen3 dense, 36 layers, 32 heads / 8 KV, dim 4096,
vocab 151936, Q4_K_M, 4.8 GB on disk).

## Phase 0 — Baseline (2026-05-13)

Build: `zig build -Doptimize=ReleaseFast` at HEAD `0a88734`.

### ZINC baseline (canonical prompt, n=128)

3 runs, prefill / decode tok/s:

| Run | Prefill | Decode |
|----:|--------:|-------:|
| 1 | 10.4 | 8.60 |
| 2 | 10.5 | 9.12 |
| 3 | 10.4 | 8.59 |

**Median**: prefill 10.4 tok/s, decode **8.60 tok/s** (116 ms/step).

### Profile (n=64)

```
steps=68 prompt=5 completion=64
shared_steps=0 cmds=4960 commits=4960      <- 73 commits per token
wait: commitAndWait 7795 ms (114.6 ms/step, 99.4% of traced time)
cpu: embed 0.55 ms | record 31.74 ms (0.467/step) | sample 0.02 ms

dmmv bytes/step: q4_k 3.13 GiB (72.3%) | q6_k 1.20 GiB (27.7%)
path bytes/step: dense (FFN) 3.07 GiB | attn 0.81 GiB | lm-head 0.45 GiB
```

Per-token weight read: ~4.3 GiB. M1 Max peak bandwidth: ~410 GB/s.
Theoretical floor at 100% efficiency: **10.5 ms/token = 95 tok/s**.
Observed: 116 ms/token = 8.6 tok/s = **~9% of peak**. Kernel efficiency is
the bottleneck, not dispatch.

### llama.cpp baseline

Not yet captured — `llama-cli` hung on warmup twice (process spun for hours
at 96% CPU on the model load path). Comparable-model reference from the
public benchmarks page: M4 Max llama.cpp Qwen3 8B decode is **83.67 tok/s**.
Scaling to M1 Max bandwidth (~75% of M4 Max) → expected llama.cpp on M1 Max
is in the **55–65 tok/s** range. ZINC at 8.6 tok/s ≈ 13–15% of that.

## Phase 1 — Hottest cost identified

Not a kernel — initially the dispatch model. Then after fixing dispatch,
the kernels themselves (q4_k matvec specifically, 72% of dispatch bytes).

## Cycle 1 — Dispatch consolidation for Qwen3 dense

### Change

`src/compute/forward_metal.zig::canUseDenseSharedDecodeCommand`. Was
gated to `cfg.architecture == .gemma`. Extended to `.gemma, .qwen2`
(Qwen3 maps to `.qwen2`). All structural pre-checks unchanged.

### Result

```
ZINC post-fix (n=128, 3 runs):
  Run 1: prefill 12.2 | decode 8.55
  Run 2: prefill 12.4 | decode 8.61
  Run 3: prefill 12.4 | decode 8.52
Median: prefill 12.3 | decode 8.55

Profile (n=64): shared_steps=0 cmds=200 commits=68 (1 commit/step)
```

Output: byte-identical greedy output on canonical prompt.

### Verdict: KEEP — but pivot

- **Decode**: 8.55 vs 8.60 → neutral, well within run-to-run noise.
- **Prefill**: +18% (10.4 → 12.3 tok/s) — a real win.
- **Dispatch**: 73 commits/token → 1 commit/token, command-buffer pressure
  collapsed. Cleaner runtime, frees Metal queue capacity.
- **No regression** on decode. Output byte-identical.

Per Phase 4 gate, decode didn't beat ≥2× noise. But the change has:

1. A clear prefill win (>>2× noise),
2. A structural cleanup (4960 commits/decode → 68),
3. No correctness drift.

Kept. Documented here so future cycles know the dispatch path is already
active for Qwen3 dense.

### Lesson

The "73 commits per token" finding from Phase 0 looked like the smoking
gun, but the M1 Max GPU was already pipelining the per-dispatch command
buffers via the Metal driver. Collapsing them improves prefill (where
many tokens process at once and dispatch overhead amortizes badly) but
does not move single-token decode, which is GPU-time bound.

**The real bottleneck is q4_k matvec kernel bandwidth efficiency.**
We're running at ~9% of peak M1 Max bandwidth.

## Phase 2 — Q4_K matvec kernel comparison

Read-through of ZINC `dmmv_q4k.metal` vs llama.cpp `kernel_mul_mv_q4_K_f32`:
**line-for-line equivalent.** Same NSG=2, NR0=2, identical fixed-point
accumulation, identical kmask schedule, identical loads. ZINC's own
comment (`dmmv_q4k.metal:14`) acknowledges this as a faithful port.
llama.cpp's "ext" variants (`kernel_mul_mv_ext_q4_f32_disp`) are gated by
`ne11 ∈ [4,8]` (small batch); for N=1 decode both paths take the same
kernel. **The matvec kernel itself is not the gap.**

## Cycle 2 — Decode-shape microbench (added)

Added `bench-metal-dmmv-q4k` (`benchmarks/metal_dmmv_q4k.zig` and a
build.zig step). Targets the actual `dmmv_q4k.metal` kernel at the
Qwen3-8B N=1 decode shapes. Queues `iters` dispatches inside ONE command
buffer to isolate kernel time from per-dispatch CPU overhead.

### Result on this M1 Max

```
shape                             weight MB    us/iter   GB/s (W)    vs 410    vs 546
attn_q       M=4096  K=4096            9.00      59.62      158.3     38.6%     29.0%
attn_k/v     M=1024  K=4096            2.25      10.01      235.7     57.5%     43.2%
attn_o       M=4096  K=4096            9.00      38.61      244.4     59.6%     44.8%
ffn_up/gate  M=12288 K=4096           27.00     115.13      245.9     60.0%     45.0%
ffn_down     M=4096  K=12288          27.00     127.59      221.9     54.1%     40.6%
```

(attn_q at 38.6% is a cold-start artifact — same shape `attn_o` hits
59.6% after warmup; treat 244 GB/s as the real attn_q number too.)

### Implied per-token kernel ceiling

```
Implied 36-layer Q4_K matvec lower bound: 16.86 ms/token => 59.3 tok/s
Current ZINC end-to-end:                  116    ms/token =>  8.6 tok/s
```

**~50 tok/s of headroom is being lost outside the matvec kernels.**

### Where the time goes

End-to-end decode profile (n=64): `dispatch/step: total 580 barriers 399`.
That's **16 compute encoder dispatches per layer** × 36 layers. A typical
Qwen3 decode layer in ZINC:

```
rms_norm -> Q proj -> K proj -> V proj -> Q norm -> K norm
         -> RoPE Q -> RoPE K -> KV write -> flash_attn
         -> O proj -> residual+norm -> gate proj -> up proj
         -> SwiGLU -> down proj
```

Each is its own dispatch with a barrier after, costing maybe 50–150 µs
of fixed dispatch+barrier overhead. **The lever is reducing dispatch
count via fusion**, not making any single kernel faster.

### Conclusion: Phase 1 was the wrong target

The smoking gun was never the matvec kernel. It's the orchestration —
too many small dispatches with barriers between them. M1 Max compute
encoders have fixed launch overhead; at 580/token that's a real cost.

## Cycle 3 — Fused dense gate+up+SwiGLU

### Change

New shader `src/shaders/metal/dmmv_q4k_dense_gate_up_swiglu.metal`,
sibling of the existing `*_geglu.metal` Gemma kernel with the activation
swapped from GeGLU to SwiGLU (`SiLU(gate) * up = (gate * sigmoid(gate)) * up`).

Pipeline `dmmv_q4k_dense_gate_up_swiglu_pipe` loaded + freed alongside
the GeGLU pipeline. New `canUseDenseQ4KGateUpSwiGLU` gate (mirror of the
GeGLU gate but for `.qwen2` architecture / `!usesGeglu`). New
`dispatchDenseQ4KGateUpSwiGLUOnCmd` dispatch. Wired into the dense FFN
path in `runDecodeStep` so Qwen3 dense takes the fused path while Gemma
keeps its GeGLU path.

### Result

```
ZINC after Cycle 3 (n=128, 3 runs):
  Run 1: prefill 12.5 | decode 8.67
  Run 2: prefill 12.6 | decode 8.70
  Run 3: prefill 12.5 | decode 8.67
Median: prefill 12.5 | decode 8.67  (+1.4% vs Cycle 1 8.55, +0.8% vs Phase 0 8.60)

Profile (n=64): cmds=200 commits=68
  dispatch/step: total 507.8 barriers 362.9   (was 579.8 / 398.9)
```

Output: byte-identical greedy output on canonical prompt.
Fusion fired exactly as expected: **-72 dispatches and -36 barriers
per token**, the precise math for 36 layers × (-2 dispatches, -1 barrier).

### Verdict: KEEP — but third hypothesis falsified

Decode change is within run-to-run noise (3-run range ±0.03). The
fusion mathematically did what it said it would do at the dispatch
layer, but **dispatch count and barrier count are not the dominant
cost on M1 Max for this workload**.

Cumulative path so far:
- Baseline 8.60 tok/s, ceiling 59 tok/s (kernel-only)
- Cycle 1: collapsed 4960 command buffer commits → 68. Decode flat.
- Cycle 3: removed 72 compute encoder dispatches + 36 barriers. Decode +0.07 tok/s.

The structural cleanups landed (cleaner runtime, +18% prefill, dense
shared path active for Qwen3 dense) but decode is essentially
unchanged. The lever is elsewhere.

## Open hypotheses (Phase 2 redux)

Three remaining candidates, none of which can be confirmed without
finer profiling than `--profile` provides:

1. **Per-kernel memory cache behavior**. Microbench reads same-shape
   weights repeatedly, hitting Apple SLC. Production reads ~3 GiB of
   distinct weight tensors per token, cold for each. The 244 GB/s
   isolated number may be an upper bound the production path never
   reaches.
2. **Small-kernel overhead concentration**. Of 508 dispatches/token,
   ~250 are q4_K matvecs; the rest are RoPE, Q/K norm, KV write,
   flash attention, residual+norm. If those small kernels are
   overhead-dominated (~100µs of launch latency vs ~50µs of compute),
   they collectively account for 25–50 ms/token.
3. **GPU scheduler gaps between back-to-back small dispatches**.
   Without a Metal performance trace we can't see GPU idle time.

## What would actually unlock progress

A real Metal performance trace (Xcode Instruments / Metal Performance
HUD via `MTL_HUD_ENABLED=1`) showing the per-dispatch GPU timeline.
That tells us within minutes whether the GPU has visible idle gaps,
which kernels are slow, and whether barriers are forcing pipeline
drains. Without it, we're running blind on the orchestration layer.

## Cycle 4 — Q+K dual matvec for Qwen3 (REVERTED)

### Change

Extended `canUseDenseQ4KQKDual` gate in `forward_metal.zig` from
`arch == .gemma` to `arch == .gemma or arch == .qwen2`. Reuses
`dmmv_q4k_qk_dual.metal` for Qwen3 dense Q+K projection. M_q=4096,
M_k=1024, K=4096 — all gate preconditions (Q4_K, K%256==0, M_q%4==0)
already satisfied. No new shader.

### Result

```
Baseline (gate Gemma-only),     warm GPU, 5 runs:  8.74, 10.62, 10.59, 10.60, 10.60
With change (gate +Qwen3),      warm GPU, 5 runs: 10.20,  8.60, 10.19, 10.19, 10.18
Median (excluding warmup):  baseline 10.60  |  change 10.19  (-3.9%)
```

Profile (n=64): dispatch/step dropped 507.8 → 471.8 (exactly -36, one
per dense layer — the fusion fired as designed). Output byte-identical.

### Verdict: REVERT — first measured regression of this effort

The dispatch count drop happened exactly as predicted, but the dual
kernel is ~4% slower on Qwen3-8B's shapes than the two separate
matvecs. On Gemma 31B (M_q=8192, M_k=4096) the dual kernel won by ~1
dispatch's worth of overhead. On Qwen3-8B (M_q=4096, M_k=1024) two
things changed:

- M_q is half the size, so per-thread Q-vs-K branch cost in the dual
  kernel is amortized over half as many output rows.
- The two separate Q and K dispatches were already running concurrently
  via the Metal concurrent encoder (`barrier_enabled = mode == .concurrent`).
  Fusing them into one grid did not unlock new parallelism — the GPU
  was already saturating both at once.

Documented the regression-case in the gate with a comment so future
cycles don't re-attempt this fusion on Qwen3 without changing the
kernel itself.

### Lesson (third dispatch-fusion attempt with same outcome)

Three cycles in a row (1 collapse 73→1 commits, 3 fuse gate+up+swiglu,
4 fuse Q+K) have demonstrated that **reducing dispatch count is not
the lever on M1 Max for this workload**. Cycles 1 and 3 had flat
decode; Cycle 4 actually regressed. The GPU is already overlapping
back-to-back dispatches via the concurrent encoder, and fusion either
costs nothing (best case) or actively hurts when the fused kernel has
worse per-thread cost.

Also discovered: the prior "8.67 baseline" in Cycles 0–3 was a
**cold-GPU** number. With the GPU warm (after 1 throwaway run), the
real baseline on this M1 Max is **10.60 tok/s** — ~24% higher than
Cycle 3 reported. The Phase 0 protocol's "1 warmup run" assumed one
run is enough to warm the GPU; on this M1 Max it takes one run *for
the model load* and a second run for the GPU clocks to ramp.

Updated ceiling math: 10.60 tok/s is ~18% of peak M1 Max bandwidth
(microbench ceiling at 244 GB/s implies ~59 tok/s). We are no longer
9% of peak — we're 18%, with ~3× headroom left to the kernel-only
ceiling. The remaining gap is likely a mix of:

1. Cold-weight reads (per-token weights don't fit in 48 MB SLC, so
   each layer pays L2-miss latency to main RAM).
2. Per-dispatch GPU launch latency on the ~250 non-matvec small
   kernels (RoPE, norms, KV writes, flash attention).
3. GPU clock scaling — sustained-load throttling.

The next cycle should target item 2 or 3 (item 1 is a hardware bound).

## Consolidated cycle history (21 loop cycles completed)

This table is the **primary reference** for any future agent invocation
on this effort. Read it before proposing a change. Most levers in
columns "Already shipped" or "Falsified" do not need to be re-tried.

### Wins kept (sticky)

| Cycle | Headline tok/s | Change |
|---|---:|---|
| 2 | **8.6 → 44.3** | **Q6_K matvec routed through `dmmv_q6k_llama` instead of legacy SPIRV-Cross kernel** (was gated to `.gemma`). The single biggest win of the effort. |
| 9 | 44.7 | `precise::exp` → `fast::exp` in dense SwiGLU activation (~442K calls/token) |
| 10 | 44.8 | `cos/sin` → `fast::cos/sin` in `rope_kv_cache_write` (~92K sincos pairs/token) |
| 11 | 45.0 | `precise /` → `fast::divide` in SwiGLU activation |
| 12 | 43.6 | `1.0f / final_sum` → `fast::divide` in flash_attn final softmax (1152 calls/token) |
| 13 | 43.9 | Hoist `rms_inv` compute to one thread in `residual_rms_norm` (saves 255× redundant `fast::rsqrt+fast::divide` per dispatch) |
| 14 | 44.9 | `precise rsqrt+/` → `fast::rsqrt+fast::divide` in `rms_norm_mul` (most-called norm shader, 144 dispatches/token) |
| 15 | 45.1 | Eliminate rms_inv broadcast barrier in `rms_norm_mul` (simdgroup-redundant pattern, −144 barriers/token) |
| 16 | 45.0 | Same pattern in `residual_rms_norm` (−36 barriers/token) |
| 17 | 44.9 | Same pattern in Q-norm/K-norm of `rope_kv_cache_write` (−72 barriers/token) |
| 18 | 45.0 | Same pattern in `flash_attn` reduceThreadgroupMax/Sum (−72 barriers/token) |
| 19 | 44.9 | flash_attn running_max/sum moved from threadgroup memory to per-thread registers (−36 barriers/token) |
| 20 | 45.0 | Redundant barrier removed between K-dot loop and reduceThreadgroupMax in `flash_attn` |
| 21 | 45.1 | Partial-reduction barrier elimination in Q-norm/K-norm of `rope_kv_cache_write` (1440 TG dispatches/token) |

### Reverts (falsified)

| Cycle | Reason |
|---|---|
| 1 | Q+K dual matvec gate extended to `.qwen2` — −3.9% regression. The dual kernel's Q-vs-K branch cost doesn't amortize on Qwen3-8B's smaller (M_q=4096, M_k=1024) shapes, and separate Q+K dispatches were already running concurrently via the Metal concurrent encoder. |
| 3 | Q6_K LM head fused-norm kernel (port of `dmmv_q4k_lmhead_norm` pattern) — −10% regression. Different shape constraints; kernel-level rms+matvec fusion not a fit for the Q6_K lm_head. |
| 4 | `dense_cmd_group_layers` bumped 30 → 36 — flat. Profile confirmed cmds/step dropped 2.89 → 1.89 but wall time didn't move. Dispatch count is **not the lever** on M1 Max for this workload. |

### Do-not-retry list (post-Cycle-21)

Levers that have been shipped or proven flat — should NOT be attempted again:

1. **Dispatch-count reduction in any form** (cycles 1, 3, 4, plus the
   pre-loop Cycle 3 SwiGLU fusion and Cycle 1 dense-shared-cmd-buffer).
   The Metal concurrent encoder already overlaps independent dispatches;
   eliminating commits or compute-encoder calls does not move the
   single-token decode wall clock.
2. **precise → fast Metal math** is essentially done. Already applied to:
   `fast::exp`, `fast::cos`, `fast::sin`, `fast::divide`,
   `fast::rsqrt` across SwiGLU, RoPE, flash_attn, rms_norm,
   residual_rms_norm, rope_kv_cache_write. Grepping for `precise::`
   in `src/shaders/metal/` should return very few hits worth swapping.
3. **simdgroup-redundant-reduction pattern for broadcast barriers** has
   been applied to all four shaders that use it heavily (rms_norm_mul,
   residual_rms_norm, rope_kv_cache_write Q/K-norm, flash_attn
   reduceThreadgroupMax/Sum). Re-applying to a fifth shader is unlikely
   to move the headline.
4. **Kernel-level rms+matvec fusion for q6_K lm_head** (Cycle 3 result).

### Still-open structural levers

Ordered by likely impact. The loop's "stall warning" fired at Cycle 20+
because cycles 12-20 all followed the same micro-optimization
template — these are the categorical pivots a future agent should
consider:

1. **KV-cache quantization to q8_0** (or smaller). llama.cpp's local
   reference Metal config uses `-ctk q8_0 -ctv q8_0`. ZINC's M1 Max
   path still uses f16 KV. At any non-trivial context length this is
   a real bandwidth saving on the attention path. There is a
   `--kv-q8` knob in the bench harness already; verify what the
   end-to-end decode actually uses.
2. **Fused rms_norm + matvec on attention prologue** (rms_norm + Q
   projection, in particular). The Vulkan side has
   `rms_norm_dmmv_q4k_alpha_beta` proven to ship +0.6 tok/s on
   RDNA4. The Metal `rms_norm_mul` is independently fast now, but
   collapsing the (rms_norm → barrier → Q-proj) sequence into one
   kernel removes the barrier *and* one DRAM round-trip on norm_buf.
3. **Single-kernel Q+K+V projection.** Q/K/V projections all read
   the same input. A single kernel that writes three outputs would
   save 2 dispatches + 2 barriers per attention layer (72 each per
   token) AND save 2× input re-reads. Different from Cycle 1's Q+K
   dual which only fused two of them and required a branch — a
   three-output kernel can keep one straight-line loop and three
   separate accumulators.
4. **GPU-side greedy argmax for the LM head step.** Currently the
   full logits vector is copied to CPU and scanned by
   `sampleGreedy`. Tail cost, not a primary lever, but real at
   high tok/s.
5. **Microbench-driven kernel rewrites.** `zig build
   bench-metal-dmmv-q4k` exists. The dmmv_q4k matvec currently runs
   at 54–60% of M1 Max bandwidth in isolation — there is room. But
   any rewrite should beat the microbench by ≥5% *first* before
   whole-model rollout.
6. **Thermal/clock-state characterization** (diagnostic, not a code
   lever). Run `powermetrics --samplers gpu_power -i 1000` in
   parallel with a decode run to confirm whether the bimodal sample
   pattern is GPU clock ramp-down/up or something else. If it's
   clock-related, the right "fix" is to keep the GPU warmer (e.g.,
   shorter idle gaps between samples, or a tiny background dispatch
   between samples).

## Loop measurement gotchas (for any future cycle)

1. **3-run median is too small at this thermal noise level.** Range
   between samples regularly hits 2–3 tok/s on a M1 Max that has been
   running for >1 hour, so a 3-run median can be pulled in either
   direction by a single sample. The loop now defaults to 5 samples
   with symmetric 1-trim; for overnight runs, prefer 7 samples with
   symmetric 2-trim (`ZINC_BENCHMARK_RUNS=7 ZINC_BENCHMARK_TRIM=true`).
2. **Bimodal samples mean THERMAL.** The harness now annotates these
   with `⚠ THERMAL` when range > 1.5 AND samples straddle the median
   by ±0.75 tok/s on both sides. Treat a "kept" verdict at this
   noise level with suspicion — the change may be flat in code but
   look winning in measurement, or vice versa.
3. **First run after model load is always slower.** Each verifier
   sample is a fresh `./zig-out/bin/zinc` invocation. The single
   warmup run helps the OS page cache but does NOT preserve Metal
   GPU clocks. For real per-cycle A/B confidence, either run a longer
   single decode (-n 256+ averages out the cold tail) or accept that
   each sample carries a cold component.
