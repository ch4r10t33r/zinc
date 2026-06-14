# Effort 25 - Metal Qwen 3.5 9B prefill on M4

Created: 2026-06-13

## Objective

Close the largest current Apple Silicon M4 public-suite gap after Effort 24:
Qwen 3.5 9B Q4_K_M prefill on the Metal backend.

Primary target:

- Model id: `qwen35-9b-q4k-m`
- File: `Qwen3.5-9B-Q4_K_M.gguf`
- Backend: Metal
- Machine: local M4 Max / Apple9, 64 GB unified memory
- Architecture: Qwen 3.5 dense SSM+attention hybrid
- Public-suite prompt mode: raw completion
- Primary metric: prefill throughput
- Guard metric: decode throughput

Use the managed model cache:

```bash
./zig-out/bin/zinc model pull qwen35-9b-q4k-m
```

The model is expected at:

```text
~/Library/Caches/zinc/models/models/qwen35-9b-q4k-m/model.gguf
```

## Why this is now the biggest M4 gap

Latest published M4 suite data, generated `2026-06-13T23:22:53.831Z`:

| scenario | ZINC prefill | llama.cpp prefill | ZINC decode | llama.cpp decode | overall |
|---|---:|---:|---:|---:|---:|
| core | 36.5 tok/s | 333.3 tok/s | 29.4 tok/s | 57.0 tok/s | 42.2% |
| context-medium | 36.6 tok/s | 79.5 tok/s | 28.9 tok/s | 57.4 tok/s | 48.3% |
| context-long | 36.4 tok/s | 87.5 tok/s | 28.5 tok/s | 57.9 tok/s | 44.2% |
| decode-extended | 36.8 tok/s | 444.8 tok/s | 28.9 tok/s | 58.0 tok/s | 43.0% |

The important smell is that ZINC prefill is flat around `36 tok/s` across
prompt sizes. That usually means the model is still traveling a decode-shaped
or under-batched prefill path instead of a true prompt-batched path. Decode is
also behind, but prefill is the larger gap and should be attacked first.

## Run the loop

Start with the public-suite raw prompt. Keep the guard on decode so prefill
work does not silently regress generation.

```bash
PROMPT='Developer question: two local LLM benchmark screenshots show different tok/s values for the same model. A useful answer explains likely causes and gives one fair measurement rule.

Answer:'

ZINC_MODEL_ID=qwen35-9b-q4k-m \
ZINC_METRIC_MODE=prefill \
ZINC_PROMPT_MODE=raw \
ZINC_TEST_PROMPT="$PROMPT" \
ZINC_REFERENCE_TEXT="token throughput" \
ZINC_MAX_TOKENS=96 \
ZINC_MIN_DECODE_TOKENS=32 \
ZINC_TARGET_TOK_PER_SEC=333 \
ZINC_STOP_ON_TARGET=0 \
ZINC_BENCHMARK_RUNS=3 \
ZINC_BENCHMARK_WARMUPS=1 \
ZINC_BENCHMARK_CONFIRM_RUNS=4 \
ZINC_PROFILE_EVERY=1 \
ZINC_BUILD_OPTIMIZE=ReleaseFast \
ZINC_TEST_TIMEOUT_MS=300000 \
ZINC_RUN_TIMEOUT_MS=900000 \
ZINC_CROSS_EFFORT_PROMPT="$PROMPT" \
ZINC_CROSS_EFFORT_METRIC=decode \
ZINC_CROSS_EFFORT_PROMPT_MODE=raw \
ZINC_CROSS_EFFORT_MAX_TOKENS=96 \
ZINC_CROSS_EFFORT_EVERY=3 \
ZINC_HARD_FAMILY_COOLDOWN=1 \
ZINC_WORKLOAD_RESET_ON_CHANGE=1 \
ZINC_CODEX_REASONING_EFFORT=xhigh \
bun loops/implement_metal.ts \
  --effort-file loops/efforts/MULTI_HOUR_EFFORT_25_METAL_QWEN35_9B_M4.md \
  --agent codex --model gpt-5.5 --cycles 100
```

For a baseline-only check:

```bash
ZINC_MODEL_ID=qwen35-9b-q4k-m \
ZINC_METRIC_MODE=prefill \
ZINC_PROMPT_MODE=raw \
ZINC_TEST_PROMPT="$PROMPT" \
ZINC_REFERENCE_TEXT="token throughput" \
ZINC_MAX_TOKENS=96 \
ZINC_MIN_DECODE_TOKENS=32 \
ZINC_BENCHMARK_RUNS=3 \
ZINC_BENCHMARK_WARMUPS=1 \
ZINC_RUN_TIMEOUT_MS=900000 \
bun loops/implement_metal.ts \
  --effort-file loops/efforts/MULTI_HOUR_EFFORT_25_METAL_QWEN35_9B_M4.md \
  --dry-run
```

## Baseline interpretation

Current public numbers:

- prefill: about `36.5 tok/s`
- decode: about `29 tok/s`
- llama.cpp prefill: `79-445 tok/s` depending on prompt scenario
- llama.cpp decode: about `57-58 tok/s`

Harness correctness sentinel:

- Use `ZINC_REFERENCE_TEXT="token throughput"`.
- The harness checks its short output preview, not the full generated text.
  The baseline first line contains `token throughput`; later lines contain
  `tokens per second`. Avoid `tok/s` or later-line strings, because they route a
  valid baseline into rollback/correctness-repair mode.

Milestones:

1. `50 tok/s` prefill: proves this is not hard-stuck on model correctness or
   startup noise.
2. `80 tok/s` prefill: reaches the long-context llama.cpp band.
3. `150 tok/s` prefill: proves a real batched prefill path is feeding the M4.
4. Keep decode above `28 tok/s` unless the prefill win is large enough to
   justify a follow-up decode recovery cycle.

## First-cycle requirements

Before changing production code:

1. Capture the baseline with `ZINC_PROFILE_EVERY=1`.
2. Identify whether `prefillBatched` is actually used for `qwen35-9b-q4k-m`.
3. Name the top prefill bucket: dense FFN, SSM projection/state update,
   attention, LM head/tail, command encode, or tokenizer/setup.
4. Compare Qwen 9B exact shapes against the 27B routes from Effort 24. Do not
   reuse Qwen 27B fixed shapes (`K=5120`, `K=17408`, 64 layers) on Qwen 9B.

Expected Qwen 9B shape facts to verify from GGUF/profile:

- 32 layers, with full attention every 4th layer and SSM otherwise
- hidden dimension `4096`
- dense FFN inner dimension around `12288`
- no MoE routing
- Qwen attention gate is interleaved with Q by head; do not assume a contiguous
  `[Q_all | gate_all]` split.

## Likely useful directions

1. If prefill is decode-shaped, wire the existing Metal batched prefill path for
   Qwen 3.5 9B before tuning individual matvec kernels.
2. If batched prefill is already active, add exact-shape `bench-metal-shapes`
   evidence for Qwen 9B prefill before changing default-on shaders.
3. Check whether Qwen 9B dense Q4_K gate/up/down paths are missing the
   llama-style row-pair/dual dispatches that Effort 24 added for 27B.
4. Check SSM prefill separately from dense FFN. Qwen 9B has fewer layers and
   smaller dimensions than 27B, so command overhead and state-update kernels may
   be a larger share.
5. Avoid MoE route-pack work entirely. This model is dense.

## Stop / pivot rules

- Do not keep profile-only churn unless it unlocks a named next speed path.
- Do not add another shape selector unless the profile names that shape as hot.
- Do not change prompt preparation or reference text to improve metrics.
- If three cycles show prefill flat with `prefillBatched` inactive, pivot to the
  batching/orchestration gate instead of shader retuning.
- If five cycles show prefill flat with `prefillBatched` active, add exact-shape
  microbench evidence before the next production runtime edit.
- For this effort, speed-neutral `optimization` cycles are now treated as churn by
  the Metal harness. Label evidence-only work as `@@@STEP_KIND: analysis` or
  `@@@STEP_KIND: enablement`; production optimization work must move the accepted
  prefill median.

## 2026-06-14 overnight analysis

Overnight run:

- Results dir: `.metal_optimize/2026-06-14T05-19-50`
- Log: `/tmp/zinc_m4_qwen35_effort25_20260613_221950.log`
- Cycles: 21
- Initial measured baseline: `38.4 prefill tok/s`
- Harness-selected best: `38.1 prefill tok/s`
- Decode guard at tail: `35.12 tok/s` vs `35.18 tok/s` baseline (`-0.2%`)

Conclusion: no production speedup landed. The run explored queued one-command
prefill, deeper SSM/full-attention prefix materialization, packed Q/gate batched
deinterleave, fused Q4_K gate/up+SwiGLU, KV/RoPE fusion, full-prefix materialized
tails, and small-batch Q4_K/Q6_K tail kernels. All stayed in the `37.7-38.1
tok/s` band while the pre-agent baseline was already around `38.4 tok/s`.

Harness finding: fresh runs were locking the workload without seeding
`currentBest` / `bestTokPerSec` from the pre-agent baseline, so the first neutral
cycle could become the promoted tree. `implement_metal.ts` was updated to seed
the accepted baseline immediately and reject neutral `optimization` keeps on this
structural Qwen3.5 9B M4 prefill effort.

Next useful work should be analysis/enablement, not another default-on kernel
variant:

1. Capture a profile that explicitly compares ZINC's remaining token-major replay
   against llama.cpp's layer-major graph for layers 1-31.
2. Add counters that report how many Qwen3.5 9B layers/tokens are truly
   layer-major materialized, plus the first fallback reason.
3. Only after those counters name the blocker, implement a dense-hybrid
   layer-major prefill path. Do not repeat 32-token tiling, tail kernels, or
   prefix-depth variants without new evidence.
