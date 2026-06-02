# Effort 11 - Metal Gemma 4 26B-A4B MoE on M4: prefill + decode

## Objective

Close the public-suite gap for `gemma4-26b-a4b-q4k-m` on the local Apple
Silicon M4 Metal backend. Correctness remains mandatory, but the current
problem is no longer "make Gemma coherent" or "escape the CPU fallback MoE
path." The current problem is that the accepted GPU-routed Gemma MoE path is
still far behind llama.cpp for both prefill and decode.

Primary target:

- Model: `gemma4-26b-a4b-q4k-m` from the managed cache.
- Machine: Mac Studio M4 Max, Apple GPU family Apple9, 64 GB unified memory.
- Prompt mode: chat template. Raw completion is not a valid Gemma quality gate.
- Metrics: improve both `prefill_tps` and `decode_tps`; do not trade one down
  silently to move the other.

## Current baseline - full Metal suite, 2026-06-01

Artifact source: `site/src/data/zinc-performance.json`.

Suite details:

- Target: `metal`
- ZINC commit: `09fc713586018212072e02e1f1c2c8323b967720`
- llama.cpp baseline commit: `af6528e6d`
- Runs: 1 warmup, then 3 measured runs per scenario
- Baseline method: same machine, same GGUF, llama.cpp Metal

| Scenario | ZINC prefill | ZINC decode | llama.cpp prefill | llama.cpp decode | Overall vs llama.cpp |
| --- | ---: | ---: | ---: | ---: | ---: |
| Quick Chat | 34.0 tok/s | 30.01 tok/s | 365.45 tok/s | 88.44 tok/s | 26.3% |
| Coding Review | 32.6 tok/s | 28.76 tok/s | 951.42 tok/s | 85.97 tok/s | 18.0% |
| Incident Context | 31.6 tok/s | 27.37 tok/s | 141.70 tok/s | 85.85 tok/s | 25.2% |
| Long Coding Draft | 33.9 tok/s | 30.16 tok/s | 517.55 tok/s | 85.64 tok/s | 29.6% |

Diagnosis:

- Decode is only about 34% of llama.cpp on the core scenario.
- Prefill is worse: about 9% of llama.cpp on Quick Chat and about 3% on
  Coding Review.
- Gemma 4 31B dense is near llama.cpp in the same suite, and Qwen 35B MoE is
  also near llama.cpp. This is not a generic Metal backend failure.
- The gap is specific to Gemma 4 26B-A4B MoE: fused q4_k gate/up experts,
  q5_1/q5_k/q6_k/q4_k expert down variants, GeGLU, expert scales, and the
  routed batched/prefill shape.
- Old effort notes saying `canUseGpuRoutedBatchedMoe()` rejects `.gemma` are
  obsolete. Gemma has a GPU-routed path now; the next work is coverage proof,
  grouped expert execution, and profile-grounded decode isolation.

## Current controller result - 2026-06-01 local M4 prefill loop

The first 100-cycle prefill controller run improved the Paris chat harness from
about 36 tok/s to a promoted best of **88.30 prefill tok/s**. The 100 tok/s
target was not reached. The best tree is cycle 98:

```text
cycle 98: Collapsed Gemma26 exact-20 queued prefill tail from [1,5,7,6,1] to [1,5,7,7].
```

Cross-effort decode was not the problem in that run. The latest decode guard
reported about **70.6 tok/s**, roughly +99% above the guard baseline.

What actually moved prefill:

- Cycle 11/12 class: dispatch collapse and routed-MoE expert-down cleanup.
- Cycle 25: queued prefill scheduling.
- Cycle 54: fused F32 router/top-k path.
- Cycle 59: fused gate-scale RMS + router path.
- Cycle 98: exact-20 queued-prefill tail schedule `[1,5,7,7]`.

Latest useful profile evidence near cycle 98:

```text
prefill wait: ~211.81 ms
dmmv prefill bytes: ~49.25 GiB
path bytes: attn 30.29 GiB, moe-expert 22.78 GiB, shared 14.50 GiB, lm-head 6.57 GiB, router 1.09 GiB
q8 hot #1: shared M=2112 K=2816 bytes=9.66 GiB calls=1642
queued prefill: prompt_tokens 20, chunks 4, first_chunks [1,5,7,7]
```

Interpretation:

- The post-80 run is no longer about entering the Gemma routed path; it is about
  extracting the remaining 11.7 tok/s from queued-prefill scheduling, shared
  expert Q8 shapes, and the guard blockers that keep the full batched-prefill
  path from being a public-suite default.
- Neutral same-family optimization keeps are now churn. After six cycles
  without a promoted best, a speed-neutral `optimization` should be reverted
  unless it adds or consumes exact-shape evidence.
- Do not optimize from pre-cycle-59 or cycle-49 numbers. Treat 88.30 tok/s as
  the current floor and cycle 98 as the schedule baseline.

## Loop recipes

The default correctness oracle is still the Paris chat prompt. For public-suite
prefill work, set `ZINC_REFERENCE_TEXT` to a stable word from the expected
answer and start a fresh loop instead of resuming the exact-20 Paris state.

### Decode-focused loop

```bash
ZINC_MODEL_ID=gemma4-26b-a4b-q4k-m \
ZINC_METRIC_MODE=decode \
ZINC_PROMPT_MODE=chat \
ZINC_TEST_PROMPT="What is the capital of France?" \
ZINC_MAX_TOKENS=96 \
ZINC_MIN_DECODE_TOKENS=16 \
ZINC_TARGET_TOK_PER_SEC=50 \
ZINC_STOP_ON_TARGET=0 \
ZINC_BENCHMARK_RUNS=5 \
ZINC_BENCHMARK_WARMUPS=1 \
ZINC_PROFILE_EVERY=1 \
ZINC_BUILD_OPTIMIZE=ReleaseFast \
ZINC_TEST_TIMEOUT_MS=300000 \
ZINC_RUN_TIMEOUT_MS=900000 \
ZINC_CROSS_EFFORT_PROMPT="What is the capital of France?" \
ZINC_CROSS_EFFORT_METRIC=prefill \
ZINC_CROSS_EFFORT_PROMPT_MODE=chat \
ZINC_CROSS_EFFORT_MAX_TOKENS=16 \
ZINC_CROSS_EFFORT_EVERY=3 \
ZINC_CODEX_REASONING_EFFORT=xhigh \
bun loops/implement_metal.ts --resume --effort 11 --agent codex --model gpt-5.5 --cycles 100
```

### Prefill-focused loop

```bash
ZINC_MODEL_ID=gemma4-26b-a4b-q4k-m \
ZINC_METRIC_MODE=prefill \
ZINC_PROMPT_MODE=chat \
ZINC_TEST_PROMPT="What is the capital of France?" \
ZINC_MAX_TOKENS=16 \
ZINC_TARGET_TOK_PER_SEC=100 \
ZINC_STOP_ON_TARGET=0 \
ZINC_BENCHMARK_RUNS=5 \
ZINC_BENCHMARK_WARMUPS=1 \
ZINC_PROFILE_EVERY=1 \
ZINC_BUILD_OPTIMIZE=ReleaseFast \
ZINC_TEST_TIMEOUT_MS=300000 \
ZINC_RUN_TIMEOUT_MS=900000 \
ZINC_CROSS_EFFORT_PROMPT="What is the capital of France?" \
ZINC_CROSS_EFFORT_METRIC=decode \
ZINC_CROSS_EFFORT_PROMPT_MODE=chat \
ZINC_CROSS_EFFORT_MAX_TOKENS=96 \
ZINC_CROSS_EFFORT_EVERY=3 \
ZINC_GEMMA_PLATEAU_STALL_CYCLES=6 \
ZINC_METAL_SHAPES_EVERY=3 \
ZINC_METAL_SHAPES_ARGS="--case gemma26_prefill_hot --pipeline production --route-tokens 20 --iterations 80 --warmup 10" \
ZINC_CODEX_REASONING_EFFORT=xhigh \
bun loops/implement_metal.ts --resume --effort 11 --agent codex --model gpt-5.5 --cycles 100
```

### Public-prompt prefill loop

Use this after cycle 118. Do not `--resume` from the Paris loop: its 88.30 tok/s
best is an exact-20-token schedule baseline and is not comparable to the public
site prompt.

```bash
PROMPT="Write an implementation plan for adding a stable benchmark preset to a local LLM CLI. Include the command shape, warmup policy, metrics to collect, failure handling, llama.cpp comparison, and how the site should display prefill, decode, latency, and overall prompt+decode throughput."

ZINC_MODEL_ID=gemma4-26b-a4b-q4k-m \
ZINC_METRIC_MODE=prefill \
ZINC_PROMPT_MODE=chat \
ZINC_TEST_PROMPT="$PROMPT" \
ZINC_REFERENCE_TEXT=benchmark \
ZINC_MAX_TOKENS=32 \
ZINC_TARGET_TOK_PER_SEC=200 \
ZINC_STOP_ON_TARGET=0 \
ZINC_BENCHMARK_RUNS=5 \
ZINC_BENCHMARK_WARMUPS=1 \
ZINC_PROFILE_EVERY=1 \
ZINC_BUILD_OPTIMIZE=ReleaseFast \
ZINC_TEST_TIMEOUT_MS=300000 \
ZINC_RUN_TIMEOUT_MS=900000 \
ZINC_CROSS_EFFORT_PROMPT="What is the capital of France?" \
ZINC_CROSS_EFFORT_METRIC=decode \
ZINC_CROSS_EFFORT_PROMPT_MODE=chat \
ZINC_CROSS_EFFORT_MAX_TOKENS=96 \
ZINC_CROSS_EFFORT_EVERY=3 \
ZINC_GEMMA_PLATEAU_STALL_CYCLES=6 \
ZINC_METAL_SHAPES_EVERY=3 \
ZINC_METAL_SHAPES_ARGS="--case gemma26_prefill_hot --pipeline production --route-tokens 70 --iterations 80 --warmup 10" \
ZINC_CODEX_REASONING_EFFORT=xhigh \
bun loops/implement_metal.ts --effort 11 --agent codex --model gpt-5.5 --cycles 60
```

### Public-suite validation

Run this after any meaningful keep and before claiming progress on the web
numbers:

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

Remove `--no-site-write` only when the suite result should update the site
data.

## Reference implementations

The useful pattern is not a one-token microkernel retune. Production MoE
engines batch or group routed tokens by expert.

Read these local references before grouped work:

- `/Users/zolotukhin/Workplace/llama.cpp/src/llama-graph.cpp`
  - `build_moe_ffn`: creates `selected_experts [n_expert_used, n_tokens]`,
    gathers router weights, calls `mul_mat_id` for gate/up/down, then weights
    and sums expert outputs.
- `/Users/zolotukhin/Workplace/llama.cpp/ggml/src/ggml-metal/ggml-metal-ops.cpp`
  - `ggml_metal_op_mul_mat_id`: switches to grouped matrix-matrix when
    `has_simdgroup_mm`, row count is large enough, and `n_tokens >= 32`.
- `/Users/zolotukhin/Workplace/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal`
  - `kernel_mul_mm_id_map0`: builds per-expert token counts and packed ids.
  - `kernel_mul_mm_id`: consumes that map with one grouped expert kernel.
- `/Users/zolotukhin/Workplace/vllm/vllm/model_executor/layers/fused_moe/`
  - `fused_moe.py`, `moe_align_block_size.py`, and router modules show the
    same topk -> sort/pack -> grouped expert matmul -> unpermute pattern.

## Landed foundations

These are already on `main`; do not reimplement them from scratch.

- `canUseGpuRoutedBatchedMoe()` has a Gemma branch for fused q4_k
  `ffn_gate_up_exps.weight`, scaled expert down, GeGLU, shared expert, and
  post-MoE normalization.
- Validation guard: `ZINC_GEMMA_MOE_VALIDATE=1`.
- Batched-prefill guard: `ZINC_GEMMA_BATCHED_PREFILL=1` or `validate`.
- Batched routing and packing shaders:
  - `src/shaders/metal/router_f32_topk_batched.metal`
  - `src/shaders/metal/router_f32_topk_batched_shared_gate.metal`
  - `src/shaders/metal/moe_route_pack.metal`
  - `src/shaders/metal/moe_route_pack_blocks.metal`
  - `src/shaders/metal/moe_route_ids.metal`
  - `src/shaders/metal/moe_route_gather.metal`
- Gemma MoE expert and scatter kernels:
  - `src/shaders/metal/dmmv_q4k_moe_gate_up_geglu.metal`
  - `src/shaders/metal/dmmv_q4k_moe_cols.metal`
  - `src/shaders/metal/dmmv_q5_1_moe_cols.metal`
  - `src/shaders/metal/dmmv_q5k_moe_cols.metal`
  - `src/shaders/metal/dmmv_q6k_moe_cols.metal`
  - `src/shaders/metal/moe_route_scatter_scaled.metal`
  - `src/shaders/metal/moe_route_scatter_direct_scaled.metal`
  - `src/shaders/metal/gemma_moe_post_norm_residual.metal`

## Definition of success

Minimum milestone:

- Paris chat coherence is preserved.
- `zig build test` passes.
- No public-suite scenario regresses.
- Decode improves from about 30 tok/s to at least 35 tok/s.
- Prefill improves from about 32-34 tok/s to at least 50 tok/s.

Target milestone:

- Decode reaches at least 50 tok/s.
- Prefill reaches at least 100 tok/s.
- Overall Quick Chat reaches at least 45% of llama.cpp.
- The profile names the remaining bottleneck instead of only showing generic
  aggregate wait time.

Stretch milestone:

- Decode reaches at least 70 tok/s.
- Prefill reaches at least 200 tok/s.
- Overall public-suite score reaches at least 70% of llama.cpp.

## Execution order

### Step 0 - Rebaseline and separate metrics

Do not start by changing kernels. First confirm the loop is measuring the
right thing:

1. Build line says `zig build -Doptimize=ReleaseFast`.
2. The verifier uses `ZINC_METRIC_MODE=decode` or `ZINC_METRIC_MODE=prefill`
   explicitly.
3. `--profile` is not using `ZINC_METAL_KERNEL_TIMING=1` unless the pass is
   explicitly out-of-band.
4. Decode and prefill are read from their own lines, not inferred from
   aggregate request latency.
5. The public suite is the final judge for published progress.

### Step 1 - Prove current Gemma routed-path coverage

The historical effort spent many cycles trying to enter the GPU-routed path.
That path exists now. Before optimizing, prove whether every layer and every
scenario actually uses it.

Add or use profile evidence that answers:

- Did `canUseGpuRoutedBatchedMoe()` return true for every Gemma MoE layer?
- If it returned false, which tensor, scale, bias, quant type, or pipeline was
  the exact blocker?
- In prefill, did `canUseGemmaBatchedPrefill()` return true under
  `ZINC_GEMMA_BATCHED_PREFILL=validate`? If not, name the exact guard.
- Are route-pack counts and packed ids actually used, or only compiled?

Acceptance:

- Profile output identifies `gpu-moe`/Gemma routed phases, not fallback MoE.
- Any remaining fallback is named with a guard reason.
- `ZINC_GEMMA_MOE_VALIDATE=1` passes for at least one short chat prompt.

### Step 2 - Decode isolation before retuning kernels

Decode is about 30 tok/s, far below llama.cpp's about 88 tok/s. Do not attack
random Q8 kernels without evidence.

Required work:

- Capture per-phase decode profile for the 96-token Paris loop recipe.
- Compare production phase time against exact-shape microbenchmarks for the
  same tensor shape before changing a kernel.
- If a phase is bandwidth-near in microbench but slow in production, suspect
  command ordering, barriers, temporary buffer churn, or CPU/GPU waits.
- If a phase is slow in both production and microbench, only then retune the
  kernel.

Do not repeat the old broad Q8 threadgroup/repack basin unless the exact shape
being changed has before/after `bench-metal-shapes` evidence.

### Step 3 - Prefill routed MoE audit

Prefill is the bigger public-suite gap. The expected winning shape is:

```text
tokens [N,H]
  -> router logits [N,E]
  -> topk [N,K]
  -> route pack: counts[E], ids[E,N*K]
  -> grouped gate/up by expert and token block
  -> batched GeGLU
  -> grouped down by expert and token block
  -> scatter weighted outputs back to [N,H]
  -> post norms/residuals on GPU
```

Required checks:

- Counts sum to `N * n_experts_used`.
- Packed ids decode as `token = id / K`, `slot = id % K`.
- Empty experts do not launch expensive dense blocks.
- Shared expert gate/up/down is batched across prompt tokens.
- `ffn_down_exps_scale` is folded into route weights exactly once.
- Post-FFN Gemma norms and residual order match the decode path.

Acceptance:

- `ZINC_GEMMA_BATCHED_PREFILL=validate` compares last-token logits against the
  per-token path and stays near float noise.
- The public-suite Quick Chat and Coding Review prefill numbers improve
  without decode regression.

### Step 4 - Grouped expert kernels

If Step 3 shows route packing is correct but prefill still rereads expert
weights too often, use the existing column-grouped kernels before attempting a
full llama.cpp `mul_mat_id` port.

First candidates:

- Fused q4_k gate/up GeGLU path for `ffn_gate_up_exps.weight`.
- q5_1/q5_k/q6_k/q4_k grouped down path as required by the model tensors.
- `NUM_COLS=4` first; only test 8 after 4 wins.

Acceptance:

- Unit or validation test compares grouped output against the per-token routed
  path for at least two tokens routed to the same expert.
- Max abs diff stays under the existing Gemma validation threshold.
- Public-suite prefill improves; decode does not regress beyond the
  cross-effort threshold.

### Step 5 - Command-buffer and barrier cleanup

Only after the routed/grouped shape is known correct:

- Keep router, topk, route pack, expert gate/up, activation, expert down,
  scatter, post-norm, and residual in as few command buffers as dependency
  rules allow.
- Remove barriers only when buffer hazards are proven absent.
- Watch `cmds`, `commits`, `barriers/step`, and GPU wait time in profile.

Acceptance:

- Same correctness.
- Same or better public-suite numbers.
- Profile shows a real reduction in commits, barriers, or wait time.

### Step 6 - Simdgroup-matrix grouped matmul

Only consider this after Steps 1-5 have named grouped expert matmul as the
remaining bottleneck.

Port the llama.cpp shape more directly:

- Expert-contiguous token counts.
- Packed route ids.
- Per-expert grouped matrix-matrix using `simdgroup_matrix`.
- Separate correctness gate before default enablement.

This is higher risk and should not be the first implementation step.

## Keep/reject rules

Keep a change only if:

- Paris chat remains coherent.
- `zig build test` passes.
- The official verifier, not an agent-only timing probe, improves the target
  metric or the public suite improves without cross-effort regression.
- The self-analysis names the phase it targeted and includes before/after
  numbers from the same optimize mode.

Reject or revert a change if:

- It loses Paris coherence.
- It weakens validation to pass.
- It moves work from profile-visible GPU time into unprofiled CPU work.
- It improves raw completion but regresses chat mode.
- It improves decode while prefill regresses by more than the harness
  cross-effort threshold, or vice versa.
- It cites obsolete cycle-49 numbers as the current Gemma baseline.
- It changes Q8 kernel/threadgroup/repack logic without exact-shape evidence.

## Known dead ends - do not repeat

These were already measured locally and should not be retried unless the
surrounding path has changed substantially:

- Q5_1 MoE down threadgroup width retunes: 128/256 thread variants regressed.
- Store-only MoE accumulate to remove zero fill regressed.
- No-logits zero-token prefill did not improve total time.
- GPU Q8 LM head for Gemma worsened total time because GPU wait rose.
- GPU post-MoE post-norm/residual tail regressed when tried as an isolated
  micro-change.
- Fused Q5_1 expert-down + weighted accumulate regressed.
- Broad Gemma GPU-routed decode MoE broke correctness in early cycles; the
  current routed path is newer, so validate coverage before changing semantics.
- Grouped Q4_K column input-addressing changes regressed cycle 15.
- Gemma K-as-V V-unit-norm handling in batched prefill regressed cycle 18.
- Q5_1 four-rows-per-workgroup variants regressed cycles 24 and 28.
- CPU Q8 LM-head scheduling, row pairing, unused-logit skip-store, and GPU
  argmax variants regressed or broke tests in cycles 25, 26, 29, and 30.
- vLLM-style projection-to-activation fused expert kernel regressed badly in
  cycle 34.
- Shared expert dual Q4_K gate/up fusion regressed in cycle 35.
- 64-thread Q4_K fused gate/up groups regressed in cycle 37.
- K-cap/cached Q4_K widening beyond the current `K <= 3072` should not be
  tried unless profile identifies the `K=2816` large-M Q4_K path as the
  remaining bottleneck and excludes the known-bad `K=4096` case.
- Raising shared Q8 gate/up threadgroup size to 512 regressed cycle 44.
- Large attention Q/O quad-row Q8 path regressed cycle 45.
- 256-thread paired Q8 K/V override regressed cycle 46.
- K=2816-specialized paired Q8 shader regressed cycle 48.
- Fused paired Q8 shared gate/up GeGLU regressed cycle 50.
- Shared-expert down Q8 through Apple9 512-thread path regressed cycle 51.
- CPU Q8 LM-head heap allocation, worker scheduling, row pairing, 16-lane dot,
  and CPU final-norm movement all regressed or failed to beat cycle 49.
- Gemma Q8 `attn_output.weight` K=4096 special shader and repacked layout both
  regressed cycles 54 and 70. The accepted path is the 256-thread runtime Q8
  shader.
- Default-on Gemma batched prefill without validation-on logits parity regressed
  cycle 69.
- Post-cycle-90 weighted-finalizer, sigmoid/cache, and narrow Q8 finalizer
  retunes did not beat the cycle-98 promoted best.
- Queued-prefill schedule `[1,6,7,6]` regressed versus `[1,5,7,7]` on the
  exact-20 Paris harness.
- Overlapping expert-down/shared GeGLU attempts regressed in the post-80 window.
- No-code study cycles are not useful unless they land measurement coverage or
  update this effort file with a concrete, current no-code conclusion.

## Next best targets after cycle 98

1. Consume `bench-metal-shapes --case gemma26_prefill_hot --pipeline production`
   evidence before another shared-Q8 or finalizer retune. Use `--route-tokens
   20` for Paris-only reproduction and `--route-tokens 70` for the public
   coding-plan prompt.
2. Audit the exact guard blockers preventing the full Gemma batched-prefill path
   from becoming default-on under public-suite prompt lengths.
3. Validate the cycle-98 tree on the public performance suite before publishing
   site numbers or optimizing only the 20-token Paris shape further.

## Measurement gates

Every kept source change must include:

```bash
zig build test
zig build -Doptimize=ReleaseFast
./zig-out/bin/zinc --model-id gemma4-26b-a4b-q4k-m \
  --prompt "What is the capital of France?" --chat -n 96 --profile
```

If a change touches generic Metal kernels, also run at least one Qwen smoke:

```bash
./zig-out/bin/zinc --model-id qwen35-9b-q4k-m \
  --prompt "What is the capital of France?" --chat -n 32
```

Before site publication, rerun the public-suite command in this file and
update `site/src/data/zinc-performance.json` only from that suite result.

## Files likely to change

- `src/compute/forward_metal.zig` - Gemma MoE routing, scratch buffers, prefill
  orchestration, profile guard reasons.
- `src/shaders/metal/moe_route_pack*.metal` - only if packed id/block layout
  needs extension.
- `src/shaders/metal/dmmv_*_moe_cols.metal` - grouped expert kernels.
- `src/shaders/metal/moe_route_scatter*.metal` - scatter/accumulate and scaled
  route handling.
- `src/shaders/metal/gemma_moe_post_norm_residual.metal` - post-MoE tail only
  if validation proves ordering or bandwidth issues.
- `loops/implement_metal.ts` - harness prompt or guard text only.
