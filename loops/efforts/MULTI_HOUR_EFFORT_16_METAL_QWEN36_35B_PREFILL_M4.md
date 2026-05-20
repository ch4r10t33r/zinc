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
ZINC_TARGET_TOK_PER_SEC=120 \
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

Updated after cycle 120 of the follow-up run:

- Accepted best is `44.8 prefill tok/s` on the Effort 16 chat prompt.
- The original measured baseline was about `34.1 prefill tok/s`, so the loop
  banked roughly +31% accepted throughput.
- The largest accepted jump was cycle 89: token-major Qwen F32 shared-gate MoE
  combine, adapted from vLLM top-k weighted reduce and llama.cpp `mul_mat_id`
  discipline. That moved the accepted best from about `36.4` to `43.4`.
- Cycle 100 added the current best `43.8 prefill tok/s` via a 16-token early
  prompt graph commit. Cycles 108-120 moved the accepted tree to `44.8`
  with final-layer prompt tail skipping, layer-0 SSM branch precompute, and
  smaller barrier/materialization cleanups.
- The last 10-cycle self-review at cycle 120 showed no accepted movement above
  `44.8`. Treat this as a stall. Do not spend more cycles on small variants of
  early prompt split, terminal-only barriers, or layer-0 materialization unless
  a fresh profile proves the named bucket moved.
- Repeated fast-looking route-packed/F32 shared-gate promotions produced
  bang-only output (`!!!!!!!!!!!!!!!!`) despite measuring around `36.8-44.5`.
  Do not count these as progress and do not enable them by default without
  full active-prompt validation and a `Paris` output.
- Dual-Q8 SSM projection grouping also produced bang-only output around
  `43.5`; keep that family behind validators until the tensor diff is known.
- The token-major F32 shared-gate reducer rewrite was tried twice and measured
  slower (`44.1` in cycle 117 after the repaired neutral-keep guard). Do not
  retry one-threadgroup-per-token shared-gate reduction without new evidence.
- Codex subprocesses in this harness may fail direct Metal runs with
  `Metal device not available`; the outer loop's benchmark gate still has the
  real Metal measurement. Do not burn cycles retrying in-agent Metal smokes.
- The local llama.cpp checkout no longer has `ggml-metal.m`. Use
  `ggml-metal-context.m`, `ggml-metal-ops.cpp`, and `ggml-metal.metal` directly
  when the stalled prompt requests reference study.

Best next directions for 50+:

1. Use the route-pack validators to find the exact tensor/layer where F32
   shared-gate route packing diverges. A promotion needs layer, prompt-token
   count, max abs diff, flag-on command, and a `Paris` run.
2. Attack remaining SSM projection/conv/delta launch overhead with exact-shape
   validators or microbenchmarks before default-on changes.
3. Explore layer-major work beyond layer 0 only when the dependency is explicit:
   hidden for layer N depends on all prior layer MoE outputs. A safe candidate
   must prove the candidate input equals the token-major input before replacing
   production work.
4. Add profiling or validation that explains the remaining `ssm 132.98 GiB`,
   `moe-expert 75.84 GiB`, and `barriers/step 504.4` buckets. A neutral keep is
   only useful if it directly unlocks a default-on structural change.

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
