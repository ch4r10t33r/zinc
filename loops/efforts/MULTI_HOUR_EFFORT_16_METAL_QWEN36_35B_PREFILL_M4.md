# Effort 16 - Metal Qwen 3.6 35B-A3B prefill on M4

Created: 2026-05-19

## Objective

Make prompt prefill for `Qwen3.6-35B-A3B-UD-Q4_K_XL` fast and measurable on
the local Apple Silicon M4 Metal backend.

This is not Effort 5. Effort 5 targets local Metal decode throughput for the
same model. This effort is scored on prefill throughput and first-token
latency. Decode must stay coherent, but decode-only wins are not success here.

Primary target:

- Model id: `qwen36-35b-a3b-q4k-xl`
- File name: `Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`
- Backend: Metal
- Machine: local M4 Max / Apple9, 64 GB unified memory, about 546 GB/s peak
- Architecture: Qwen 3.6 hybrid SSM + attention + routed MoE

Use the managed model cache unless there is a specific reason not to:

```bash
./zig-out/bin/zinc model pull qwen36-35b-a3b-q4k-xl
```

## Run the loop

The Metal loop must score prefill, not decode. This effort depends on
`ZINC_METRIC_MODE=prefill`.

Use a long enough prompt that prefill is not dominated by process startup,
pipeline warmup, or a 5-token raw smoke test:

```bash
PROMPT='Read the following engineering context and answer the final fragment with the next word. ZINC is a local inference engine with Metal and Vulkan backends. The Metal backend uses GGUF tensors, command encoders, quantized matvec kernels, batched prefill helpers, rotary embeddings, flash attention, SSM recurrent blocks, routed MoE experts, and a managed model cache. Performance work must separate prompt prefill from autoregressive decode because prompt tokens can often be processed in a layer-major batch while generated tokens remain serial. Correctness still matters: the final fragment is intentionally simple. The capital of France is'

ZINC_MODEL_ID=qwen36-35b-a3b-q4k-xl \
ZINC_METRIC_MODE=prefill \
ZINC_PROMPT_MODE=chat \
ZINC_TEST_PROMPT="$PROMPT" \
ZINC_MAX_TOKENS=16 \
ZINC_TARGET_TOK_PER_SEC=90 \
ZINC_BENCHMARK_RUNS=5 \
ZINC_BENCHMARK_WARMUPS=1 \
ZINC_PROFILE_EVERY=1 \
ZINC_BUILD_OPTIMIZE=ReleaseFast \
ZINC_TEST_TIMEOUT_MS=300000 \
ZINC_RUN_TIMEOUT_MS=1800000 \
ZINC_CODEX_REASONING_EFFORT=xhigh \
bun loops/implement_metal.ts --resume --effort 16 --agent codex --model gpt-5.5 --cycles 100
```

Use `--agent claude` if desired. Keep the same env contract either way.

For one-shot baseline only:

```bash
ZINC_MODEL_ID=qwen36-35b-a3b-q4k-xl \
ZINC_METRIC_MODE=prefill \
ZINC_PROMPT_MODE=chat \
ZINC_TEST_PROMPT="$PROMPT" \
ZINC_MAX_TOKENS=16 \
ZINC_BENCHMARK_RUNS=5 \
ZINC_BENCHMARK_WARMUPS=1 \
ZINC_RUN_TIMEOUT_MS=1800000 \
bun loops/implement_metal.ts --effort 16 --dry-run
```

## Current standing

Updated after cycle 231 of the follow-up run:

- Current accepted tree is measuring `69.9 prefill tok/s` on the Effort 16
  chat prompt. Treat `69.9 prefill tok/s` as the comparison baseline for the
  next cycle unless the harness reports a colder fresh baseline.
- The original measured baseline was about `34.1 prefill tok/s`, so the loop
  has banked about `2.05x` accepted throughput on the same prompt.
- Cycle 231 is the new reference point. It routed Qwen3.6 prompt F32 routers
  through fused `router_f32_topk_batched` with input-offset support, leaving
  decode on the older path. The official verifier reported five samples:
  `69.90, 69.90, 69.90, 70.00, 69.90 prefill tok/s` and output `Paris.`
- The cycle-231 jump was `+18.3 prefill tok/s` over the prior best
  (`51.6 -> 69.9`). Do not optimize from pre-cycle-231 router assumptions.
- Cycles 199-230 plateaued around `51.0-51.6 prefill tok/s`. Most narrow
  Q8/repacked/fixed-K/TG128 retunes measured neutral or were reverted. Do not
  continue that family without exact-shape evidence and a same-cycle A/B.
- Earlier wins remain useful history: cycle 89 token-major F32 shared-gate MoE
  combine moved the tree from roughly `36.4` to `43.4`; cycle 100 early prompt
  graph commit and follow-up SSM/terminal cleanup moved the tree into the
  mid-40s; cycle 199 reduced SSM delta barriers and moved the tree to `51.6`.
- The short raw prompt `The capital of France is` regressed to repeated `is`
  when the Qwen SSM prefill projection chunk ran on very short prompts. The
  production gate now requires at least 32 prompt tokens for that path. Preserve
  that short-prompt coherence guard.
- Codex subprocesses in this harness cannot run local Metal model commands;
  they fail with `Metal device not available`. Do not run
  `./zig-out/bin/zinc --model-id qwen36-35b-a3b-q4k-xl` or
  `ZINC_QWEN36_* ./zig-out/bin/zinc` inside the agent. Use `zig build` and
  `zig build test`; the outer loop owns all Metal measurement and validation.
- The local llama.cpp checkout no longer has `ggml-metal.m`. Use
  `ggml-metal-context.m`, `ggml-metal-ops.cpp`, and `ggml-metal.metal` directly
  when the stalled prompt requests reference study.

Cycle-231 accepted profile snapshot:

```text
Prefill: 134 tokens in 1918.2 ms (69.9 tok/s)
Metal profile (request): steps=136 prompt=134 completion=2 shared_steps=136 cmds=5 commits=3
dispatch/step: total 582.2 barriers 388.0
barriers/step: embed 1.0 attn 75.1 ssm 116.1 router 77.1 gpu-moe 118.7 fallback-moe 0.0 dense 0.0 final 0.0
dmmv bytes: q8_0 182.46 GiB (67.4%) q4_k 45.45 GiB (16.8%) q5_1 0.00 GiB (0.0%) q5_k 27.76 GiB (10.3%) q6_k 2.63 GiB (1.0%)
path bytes: ssm 132.98 GiB attn 33.38 GiB dense 0.00 GiB moe-expert 75.84 GiB shared 16.52 GiB lm-head 1.51 GiB router 10.37 GiB
prefill buckets: ssm proj 98.69 GiB recurrent conv/delta/gated 3887/3887/3887 out 32.27 GiB | router 10.21 GiB topk 5227 cpu 0.00 ms
prefill buckets: moe gate/up 46.20 GiB down 28.49 GiB | shared gate/up 10.85 GiB down 5.42 GiB | waits 1 commits 1847.98 ms
q8 hot #1: ssm M=8192 K=2048 bytes=65.53 GiB calls=3947
q8 hot #2: ssm M=4096 K=2048 bytes=32.76 GiB calls=3947
q8 hot #3: ssm M=2048 K=4096 bytes=32.76 GiB calls=3947
q8 hot #4: attn M=8192 K=2048 bytes=20.37 GiB calls=1227
```

Best next directions for 90+:

1. Profile first. After cycle 231, `ssm` is the largest path-byte bucket
   (`132.98 GiB`) and `gpu-moe`/`ssm` are the largest barrier buckets. Do not
   make router/top-k the default next target unless a fresh profile moves it
   back on top.
2. Attack SSM projection/branch reuse that preserves the 32-token short-prompt
   guard and exact token-order recurrence. The profile says SSM projection is
   `98.69 GiB` and SSM out is `32.27 GiB`; split those before editing.
3. Attack MoE expert launch/buffer fusion, not broad route-pack validation.
   The expert path is still `75.84 GiB`, while router is now only `10.37 GiB`.
4. Remove real command/encoder/barrier work in the hot buckets. The current
   profile has `388.0` barriers/step, led by `gpu-moe 118.7`, `ssm 116.1`,
   and `router 77.1`.
5. If continuing cycle 231 directly, limit it to prefill-only
   `router_f32_topk_batched` threadgroup/input-offset variants with OFF/ON
   medians in the same cycle. Do not turn this into another route-pack or
   shared-gate validator pass.
6. Explore layer-major work beyond layer 0 only when the dependency is explicit:
   hidden for layer N depends on all prior layer MoE outputs. A safe candidate
   must prove the candidate input equals the token-major input before replacing
   production work.
7. After another default-on prefill jump, re-run the installed-model coherence
   sweep for Qwen 8B, Qwen 35B, Qwen 27B, and Gemma 12B/31B.
   The loop's Paris gate is necessary but not sufficient for all-model safety.

Known facts before the first Effort 16 baseline:

- Effort 5 measured local Metal decode on this model at about `38.11 tok/s`.
- The local llama.cpp reference from Effort 5 reported prompt throughput around
  `109.8-111.6 tok/s` on the same machine and model.
- ZINC does not yet have a dedicated M4/Qwen3.6-35B prefill effort file.
- `src/compute/forward_metal.zig::canUseBatchedPrefill` currently rejects
  models with `cfg.n_experts > 0` and models with `cfg.ssm_d_inner > 0` in
  the generic non-Gemma path. Qwen3.6 35B-A3B has both. Expect the current
  production path to behave like token-major decode during prompt ingestion.

The first cycle must record the real baseline:

```bash
zig build -Doptimize=ReleaseFast

./zig-out/bin/zinc \
  --model-id qwen36-35b-a3b-q4k-xl \
  --prompt "$PROMPT" \
  -n 16 \
  --profile
```

Capture these lines:

```text
Prefill: ...
Generated ... tok/s
Metal profile: ...
record breakdown: ...
dmmv bytes/request: ...
path bytes/request: ...
```

## Milestones

Phase 0:

- prefill metric is parsed by the loop via `ZINC_METRIC_MODE=prefill`
- 5-run baseline is recorded with prompt token count and sample range
- profile output identifies the largest prompt-side buckets

Phase 1:

- reach at least `90 tok/s` prefill on the Effort 16 prompt
- output remains a coherent Paris answer
- `zig build test` passes

Phase 2:

- reach at least `120 tok/s` prefill, roughly local llama.cpp prompt parity
- remaining gap is explained by named buckets, not "prefill is token-serial"

Stretch:

- reach at least `180 tok/s` prefill on a prompt of at least 96 tokens
- no decode regression greater than 3% on Effort 5's `-n 128` benchmark

## Benchmark contract

This effort is scored on prefill throughput:

- `ZINC_METRIC_MODE=prefill` is required.
- Prompt should be at least 96 tokens after the chat template is applied.
- `ZINC_MAX_TOKENS` should stay small, usually 8-16, because the measured
  work is prompt ingestion.
- Use 5 timed samples with one warmup once the baseline is stable.
- Do not accept a change based only on a single lucky sample.
- Correctness gate remains the France prompt output containing `Paris`. Use
  chat mode so Qwen's template emits a closed think scaffold instead of making
  the first 16 output tokens only a thinking marker.

Keep rules:

1. Keep only if `zig build test` passes.
2. Keep only if output is coherent and contains `Paris`.
3. Keep production-on performance changes only when median prefill improves by
   at least 2% over best or a large named bucket drops in profile.
4. Foundation keeps are allowed at 0% only when default-off and they add a
   validator, profiler, or exact-shape microbenchmark needed by this effort.
5. Any default-on prefill batching must have a validation mode that compares
   logits or intermediate tensors against the current per-token path.
6. Do not regress Effort 5 decode by more than 3% when a prefill milestone
   lands.

Useful supporting commands:

```bash
zig build bench-metal -- \
  --model-id qwen36-35b-a3b-q4k-xl \
  --warmup 1 --runs 3 -n 128

zig build bench-metal-shapes -- \
  --model-id qwen36-35b-a3b-q4k-xl
```

If `bench-metal` does not accept `--model-id` on the current tree, use the
managed cache path:

```bash
zig build bench-metal -- \
  -m "$HOME/Library/Caches/zinc/models/models/qwen36-35b-a3b-q4k-xl/model.gguf" \
  --warmup 1 --runs 3 -n 128
```

## Why this is hard

The model is not dense Qwen3-8B and not Gemma 31B. The existing dense batched
prefill path is useful reference code, but it cannot be enabled blindly.

The per-layer dependency is:

```text
hidden[token]
  -> attn_norm
  -> attention OR SSM branch
  -> hidden[token] += branch_output
  -> ffn_norm
  -> routed MoE + shared expert
  -> hidden[token] += ffn_output
  -> next layer
```

For SSM layers, token order matters inside the recurrent state update:

```text
hidden[token]
  -> SSM qkv/z/alpha/beta projections
  -> conv/state recurrence in token order
  -> gated norm + ssm_out
  -> residual
  -> MoE for the same token and layer
```

For MoE layers, routing depends on each token's `ffn_norm`. A real batched
prefill path must route all prompt tokens, group or pack selected experts,
run expert work over multiple tokens, then scatter weighted results back to
token order.

## Execution order

### Step 0 - Baseline and parser check

Run the dry-run command above and confirm:

- the loop prints `prefill tok/s`, not decode tok/s
- the generated text contains `Paris`
- profile output is captured on cycle 1
- sample range is not too wide for keep/reject decisions

If the loop cannot parse prefill, fix the harness before touching kernels.

### Step 1 - Prefill phase visibility on Metal

Before optimizing, make the prompt-side profile useful.

The profile should distinguish at least:

- embedding/dequant
- attention projections
- attention/flash/KV writes
- SSM projections
- SSM conv/delta/gated norm/out
- router/top-k
- MoE gate/up/down
- shared expert
- final norm + LM head
- command commit/wait count

Do not add profiling that is always on. Gate any extra detail behind an env
flag such as `ZINC_METAL_PREFILL_PROFILE=1` or reuse `--profile` if the output
volume stays reasonable.

Acceptance:

- default behavior unchanged
- `--profile` or the env flag emits prompt-side buckets on the Effort 16 prompt
- no measurable slowdown when profiling is off

### Step 2 - Validation harness for hybrid prefill

Add safety rails before replacing prompt work.

Start with a default-off validation mode, for example
`ZINC_QWEN36_35B_PREFILL_VALIDATE=1`, that compares a candidate batched result
against the current per-token reference.

Recommended first comparison:

1. Select one layer and a small prompt chunk, e.g. 4 or 8 tokens.
2. Run the normal per-token path and capture:
   - `ffn_norm`
   - router logits
   - selected expert ids and weights
   - shared expert gate output
   - MoE output before residual
   - post-layer hidden
3. Run the candidate batched or grouped path into scratch buffers.
4. Print max absolute diff, L2 diff, and worst index.

Do not start by validating the full model logits only. Intermediate diffs are
faster to debug and make reduction-order changes easier to reason about.

Acceptance:

- validator is default-off
- validator can run on the Effort 16 prompt without changing production output
- failures identify the layer, token, tensor, max diff, and worst index

### Step 3 - Dead-tail prefill cleanup

Remove output-only work from non-terminal prompt tokens before larger batching.

Candidates:

- final norm + LM head only for the final prompt token
- argmax only when needed for the first generated token
- any decode-only staging or readback that does not affect KV/SSM/MoE state

This is lower risk than layer-major MoE batching and gives a cleaner baseline.

Acceptance:

- logits for the final prompt token match the old path within tolerance
- no change to generated output
- profile shows the skipped bucket shrinking or disappearing for non-terminal
  prompt tokens

### Step 4 - Batch SSM projections without changing recurrence

The first structural target should keep the SSM recurrence exact.

Candidate shape:

```text
for one SSM layer and prompt chunk:
  collect attn_norm[token] for the chunk
  batched qkv/z/alpha/beta projections over [N, hidden_dim]
  feed projected per-token values into the existing conv/delta recurrence in
  token order
  continue with gated norm, out projection, residual, and MoE
```

The goal is to reduce repeated projection weight reads while leaving the
stateful part of the SSM path in token order.

Start with one layer/chunk behind validation. Then expand:

1. one SSM layer, chunk 4
2. all SSM layers, chunk 4
3. chunk 8 or 16
4. default-off production flag
5. default-on only after prefill median and correctness are stable

Known risk:

- batching projections before the hidden stream is correct for that layer will
  corrupt later work. Preserve the exact per-layer order.

### Step 5 - Batched/router-grouped MoE prefill

MoE is the largest likely prompt-side win, but it has the largest correctness
blast radius. Do it after the validator exists.

Target flow:

```text
ffn_norm[token] for N prompt tokens
  -> router logits [N, n_experts]
  -> top-k ids/weights [N, k]
  -> pack route slots by expert
  -> expert gate/up over grouped token columns
  -> SwiGLU
  -> expert down over grouped token columns
  -> scatter weighted outputs back to [N, hidden_dim]
  -> shared expert contribution
  -> residual add
```

Existing Metal reference pieces to inspect first:

- `softmax_topk_batched.metal`
- `moe_route_pack.metal`
- `moe_route_ids.metal`
- `moe_route_gather.metal`
- `moe_route_scatter_scaled.metal`
- Gemma batched MoE path in `recordGemmaBatchedPrefillMoeOnCmd`

Production references:

- llama.cpp `ggml_metal_op_mul_mat_id`
- llama.cpp `kernel_mul_mm_id_map0`
- llama.cpp `kernel_mul_mm_id`
- vLLM fused MoE route packing and aligned block-size code

Acceptance:

- route ids and weights match per-token top-k for the same inputs
- grouped expert output matches per-token expert output within tolerance
- full generated answer remains coherent
- prefill improves on at least the Effort 16 prompt and one longer prompt

### Step 6 - Only then tune kernels

Do not start with broad threadgroup sweeps.

Use exact-shape evidence first:

- router: `n_experts x hidden_dim`
- shared expert gate/up/down
- selected expert gate/up/down shapes
- SSM qkv/z/out projections
- LM head on final prompt token only

Add or extend `bench-metal-shapes` cases before changing production routing.
Keep shape-specific kernels narrow and gated.

## Known non-goals

- Do not weaken the Paris correctness gate.
- Do not optimize decode-only throughput under this effort.
- Do not blindly relax `canUseBatchedPrefill` for `n_experts > 0` or
  `ssm_d_inner > 0`.
- Do not make `ZINC_BATCHED_PREFILL=1` default for this model without
  validation.
- Do not copy large GGUFs into ad hoc local paths. Use managed cache.
- Do not start with a broad Q8/Q4 threadgroup sweep without exact-shape data.
- Do not treat a 5-token prompt as evidence for sustained prefill.

## Likely files

- `loops/implement_metal.ts` - harness metric mode and prompt plumbing
- `benchmarks/metal_inference.zig` - benchmark/profile output
- `benchmarks/metal_q8_shapes.zig` - exact-shape cases
- `src/compute/forward_metal.zig` - prefill orchestration, validators, MoE/SSM
- `src/shaders/metal/softmax_topk_batched.metal`
- `src/shaders/metal/moe_route_*.metal`
- `src/shaders/metal/dmmv_q4k*.metal`
- `src/shaders/metal/dmmv_q5k*.metal`
- `src/shaders/metal/dmmv_q6k*.metal`
- `src/shaders/metal/gemm_q4k.metal`
- `src/shaders/metal/gemm_q6k.metal`

## Done means

- Effort 16 can be run with `bun loops/implement_metal.ts --effort 16`.
- The loop scores `prefill tok/s`.
- The current M4 baseline is recorded.
- Accepted production changes improve prefill, preserve coherent output, and
  do not meaningfully regress Effort 5 decode.
