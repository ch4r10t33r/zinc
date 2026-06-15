# Effort 29 - Metal M4 Gemma 4 26B public-suite gap

Created: 2026-06-14

## Objective

Switch away from the Qwen36-27B plateau and continue M4 optimization on a
different model: `gemma4-26b-a4b-q4k-m`.

The target is the current public Metal suite gap against llama.cpp on the same
M4 machine. Gemma 26B-A4B is already correct and already uses the structural
route-packed Gemma MoE prefill path; this effort should not rediscover the old
"enter the routed path" work from Effort 11. The next useful work is to close
the remaining public-suite steady-state gap, starting with decode and guarding
prefill on the same public prompt.

Primary model:

- Model id: `gemma4-26b-a4b-q4k-m`
- Backend: Metal
- Machine: local M4 Max / Apple9, 64 GB unified memory
- Prompt mode: chat
- Primary metric: decode throughput
- Guard metric: prefill throughput

## Current public-suite facts

Source: `site/src/data/zinc-performance.json`, Metal target generated
`2026-06-13T23:22:53.827Z`.

Prompt:

```text
A teammate sent two local LLM benchmark screenshots with different tok/s numbers for the same model. Explain the likely causes in two short paragraphs, then give one concrete measurement rule.
```

Published comparison:

| metric | ZINC | llama.cpp | ZINC / llama |
| --- | ---: | ---: | ---: |
| prefill | 334.8 tok/s | 411.78 tok/s | 81.3% |
| decode | 69.43 tok/s | 83.49 tok/s | 83.2% |
| overall steady throughput | 94.83 tok/s | 114.28 tok/s | 83.0% |
| end-to-end throughput | 19.36 tok/s | 116.14 tok/s | 16.7% |
| total latency | 7489 ms | 1274 ms | 5.9x slower |

Interpretation:

- The steady prefill/decode gaps are both about 17-19%, so this is not a
  correctness or fallback emergency.
- The end-to-end latency gap is much larger than the steady prefill/decode gap.
  Do not ignore it if profile output shows cold pipeline compilation, load,
  prompt preparation, or finalization overhead. The Metal optimization harness
  can only promote `prefill` or `decode`, so this loop starts with decode and
  should add evidence for latency if the profile exposes it.
- Gemma 31B dense is already essentially tied with llama.cpp on steady decode in
  the same suite. Gemma 26B MoE is the better model switch because it has a real
  remaining gap and exercises a different MoE path from Qwen36-27B.

## Run the loop

Start a fresh run. Do not `--resume` Effort 11; its state mixes historical
route-pack bring-up, old baselines, and public-prompt prefill cycles.

```bash
PROMPT="A teammate sent two local LLM benchmark screenshots with different tok/s numbers for the same model. Explain the likely causes in two short paragraphs, then give one concrete measurement rule."

ZINC_MODEL_ID=gemma4-26b-a4b-q4k-m \
ZINC_METRIC_MODE=decode \
ZINC_PROMPT_MODE=chat \
ZINC_TEST_PROMPT="$PROMPT" \
ZINC_REFERENCE_TEXT="tokens per second" \
ZINC_MAX_TOKENS=96 \
ZINC_MIN_DECODE_TOKENS=64 \
ZINC_TARGET_TOK_PER_SEC=84 \
ZINC_STOP_ON_TARGET=0 \
ZINC_BENCHMARK_RUNS=3 \
ZINC_BENCHMARK_WARMUPS=1 \
ZINC_BENCHMARK_CONFIRM_RUNS=4 \
ZINC_PROFILE_EVERY=1 \
ZINC_BUILD_OPTIMIZE=ReleaseFast \
ZINC_TEST_TIMEOUT_MS=300000 \
ZINC_RUN_TIMEOUT_MS=900000 \
ZINC_CROSS_EFFORT_PROMPT="$PROMPT" \
ZINC_CROSS_EFFORT_METRIC=prefill \
ZINC_CROSS_EFFORT_PROMPT_MODE=chat \
ZINC_CROSS_EFFORT_MAX_TOKENS=32 \
ZINC_CROSS_EFFORT_EVERY=3 \
ZINC_GEMMA_PLATEAU_STALL_CYCLES=6 \
ZINC_AUTO_STOP_NO_BEST_CYCLES=24 \
ZINC_METAL_SHAPES_EVERY=0 \
ZINC_HARD_FAMILY_COOLDOWN=1 \
ZINC_WORKLOAD_RESET_ON_CHANGE=1 \
ZINC_CODEX_REASONING_EFFORT=xhigh \
bun loops/implement_metal.ts \
  --effort-file loops/efforts/MULTI_HOUR_EFFORT_29_METAL_M4_GEMMA26_PUBLIC_GAP.md \
  --agent codex --model gpt-5.5 --cycles 100
```

If decode hits the target or plateaus with a named decode bottleneck exhausted,
flip the main metric to prefill on the same prompt:

```bash
ZINC_METRIC_MODE=prefill
ZINC_TARGET_TOK_PER_SEC=412
ZINC_CROSS_EFFORT_METRIC=decode
ZINC_CROSS_EFFORT_MAX_TOKENS=96
```

## First-cycle requirements

Before changing kernels:

1. Confirm the verifier is building with `zig build -Doptimize=ReleaseFast`.
2. Capture the baseline with profile output.
3. Confirm Gemma chat output contains `tokens per second`.
4. Identify the decode bottleneck by phase, not by model family:
   - routed MoE expert gate/up/down
   - shared expert
   - attention
   - final RMS/LM head/sampling
   - command-buffer wait/barrier cadence
   - cold-start or total-latency overhead
5. Compare the first baseline against the public-suite number. If fresh decode
   is already far above 69.43 tok/s, update this file with the new floor before
   chasing stale site data.

## Live foundations from Effort 11

These are already part of the current codebase and should not be rebuilt:

- Gemma 26B supports GPU routed MoE and structural route-packed prefill.
- The public-prompt route-pack path has shown:

```text
prefill actual path: batched-route-pack default_batched=yes structural_batched=yes route_layers=30 queued_chunks=0
route pack occupancy: full 1356 tail 2044 singleton_tail 656 padding_slots 10400 util 61.8%
```

- The previous public-prompt prefill best was about `370 tok/s`, with decode
  guard near `70 tok/s`.
- Known-bad prefill work includes Q5_1 exact-6 tails, Q4_K exact-4 tails,
  broad route-pack alt block-width changes, and routine guard audits after
  `structural_batched=yes`.

## Likely decode directions

Use profile evidence before choosing one:

1. If decode is dominated by routed MoE, inspect whether decode still launches
   experts serially while prefill uses grouped route packing. A decode-side
   microbatch or grouped expert dispatch may be needed, but only if correctness
   can be preserved for one-token generation.
2. If shared expert Q8 dominates, consume exact-shape evidence before retuning.
   Effort 11 regressed several broad Q8 threadgroup and paired variants.
3. If LM head or sampling dominates, compare CPU and GPU paths. Prior Gemma GPU
   LM-head attempts regressed total time, so require current profile evidence.
4. If command/barrier cadence dominates, remove a named barrier/dispatch family
   through fusion rather than swapping barrier scopes cosmetically.
5. If end-to-end latency remains far worse while steady decode improves, add
   measurement coverage for pipeline compilation, load, and first-request setup.

## Keep/reject rules

Keep a change only if:

- Chat output remains coherent and contains `tokens per second`.
- `zig build test` passes.
- The official harness improves decode, or lands evidence that directly names
  the next decode bottleneck.
- Prefill cross-effort does not regress by more than 3%.
- The self-analysis includes before/after numbers from the same optimize mode.

Reject or revert a change if:

- It weakens correctness checks or changes the prompt/reference text to pass.
- It repeats Effort 11 dead ends without new exact-shape evidence.
- It optimizes raw completion while regressing chat mode.
- It improves prefill while silently lowering decode in this decode-focused run.
- It only changes measurement flags and produces no durable evidence.

## Validation after a meaningful keep

Run the public suite without writing site data:

```bash
bun tools/performance_suite.mjs \
  --target metal \
  --phase all \
  --models gemma4-26b-a4b-q4k-m \
  --runs 3 \
  --warmup 1 \
  --llama-cli /Users/zolotukhin/Workplace/llama.cpp/build/bin/llama-cli \
  --llama-server /Users/zolotukhin/Workplace/llama.cpp/build/bin/llama-server \
  --output /tmp/zinc-m4-gemma26-metal-$(date +%Y%m%d-%H%M%S).json \
  --no-site-write
```

Remove `--no-site-write` only when the suite result should update the website
data.

## 2026-06-15 fresh-run result

Run directory:

```text
.metal_optimize/2026-06-15T04-45-33
```

The run was stopped manually during cycle 12 because it had entered Q8/finalizer
churn with a persistent prefill guard regression. Do not resume this state as a
blind decode loop.

Outcome:

- Fresh baseline: `67.23 decode tok/s`.
- Best kept-correct decode: `68.51 tok/s` at cycle 3.
- Current HEAD after stop: `e4304807` (`pre-cycle-12`), measured around
  `68.4 decode tok/s`.
- Completed cycles: 11 kept/reverted decisions.
- Cross-effort prefill guard baseline: `355.60 tok/s` at cycle 1.
- Last cross-effort prefill guard: `332.80 tok/s` at cycle 9 (`-6.4%`).
- Public-suite site prefill row is `334.8 tok/s`, so the guard baseline was
  warmer than the site row, but the run still violated its own `-3%` guard.

Accepted changes in the stopped tree:

```text
10f26ff2 cycle 1  Gemma weighted post-norm/residual finalizer in decode
cfb2ddf1 cycle 2  dmmv_q8_0_pair_k2816 wiring for Gemma26 shared gate/up
01525bb8 cycle 3  Q8_0 LM-head partial argmax for greedy decode
674727f3 cycle 4  default-off gate for batched prefill weighted post-norm
7478c86b cycle 5  gate single-token weighted post-norm out of prefill
031f6413 cycle 6  Gemma26 Q8 LM-head argmax 1024-thread shape gate
784d4a4d cycle 7  restore Gemma26 prefill weighted post-norm by exact shape
6049306f cycle 8  decode-only shared-down Q8 128-thread selector
b87e24f1 cycle 9  decode-only shared gate/up Q8 K=2816 128-thread dispatch
45e639ee cycle 11 make batched prefill weighted post-norm opt-in again
```

Rejected change:

```text
cycle 10 Packed dmmv_q8_0_pair_k2816 reduction/writeback tail
```

Interpretation:

- Q8 LM-head partial argmax was the only material decode signal, and even that
  was small: about `+0.4 tok/s` accepted best movement.
- Shared expert Q8 launch-shape work did not produce a stable decode lift.
- The prefill guard oscillated between about `320-334 tok/s` after the warm
  cycle-1 guard, and neither finalizer gating direction recovered it.
- The profile at cycle 11 still shows structural batched route-pack active:

```text
prefill actual path: batched-route-pack default_batched=yes structural_batched=yes route_layers=30 queued_chunks=0
prefill 49 tokens in 146.7 ms (334.1 tok/s)
decode q8 hot #1: lm-head M=262144 K=2816 bytes=23.38 GiB calls=32
decode q8 hot #2: shared/shared-gate-up M=2112 K=2816 bytes=11.30 GiB calls=1920
decode q8 hot #3/#4: attention Q8 projections
decode dispatch/step: total 537.0 barriers 361.0
```

Next run guidance:

1. Do not continue the same decode Q8 threadgroup basin unless an exact-shape
   microbench or profile counter first proves the candidate's expected win.
2. Fix the harness policy before another long Gemma decode run: a cross-effort
   regression beyond `-3%` should either revert the offending candidate or stop
   the run. This run kept too many neutral changes while the guard was red.
3. Re-measure a clean no-agent baseline after a short cool-down. If prefill is
   around `333-335 tok/s`, treat cycle 1's `355.6 tok/s` guard as a warm/noisy
   outlier and update the guard baseline. If it returns to `355+`, restore a
   cleaner pre-cycle tree before continuing.
4. If continuing decode, the next credible target is not another Q8 launch
   retune. Use profile evidence to remove a named barrier/dispatch family in
   attention or Gemma MoE, or add measurement for the public-suite total-latency
   gap.
5. If switching to prefill, run a prefill-primary loop on this same public prompt
   with decode guarded, and focus on route-pack occupancy/tail waste rather than
   weighted post-norm toggles.
