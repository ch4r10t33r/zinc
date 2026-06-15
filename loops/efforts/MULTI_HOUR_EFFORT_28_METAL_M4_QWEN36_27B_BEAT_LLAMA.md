# Effort 28 - Metal M4 beat llama.cpp on Qwen 3.6 27B

Created: 2026-06-14

## Objective

Beat llama.cpp on the Apple Silicon M4 public suite, starting with the largest
known remaining disadvantage after the recent 9B dense prefill fix:
`qwen36-27b-q4k-m`.

Primary target:

- Model id: `qwen36-27b-q4k-m`
- File: `Qwen3.6-27B-Q4_K_M.gguf`
- Backend: Metal
- Machine: local M4 Max / Apple9, 64 GB unified memory
- Architecture: Qwen 3.6 dense hybrid with SSM layers
- Public-suite prompt mode: raw completion
- Primary metric: prefill throughput
- Guard metric: decode throughput

Use the managed model cache:

```bash
./zig-out/bin/zinc model pull qwen36-27b-q4k-m
```

The model is expected at:

```text
~/Library/Caches/zinc/models/models/qwen36-27b-q4k-m/model.gguf
```

## Why this is the largest current M4 disadvantage

Latest public M4 suite data, generated `2026-06-13T23:22:53.827Z`:

| scenario | ZINC prefill | llama.cpp prefill | ZINC decode | llama.cpp decode | overall |
|---|---:|---:|---:|---:|---:|
| core | 15.9 tok/s | 104.6 tok/s | 15.44 tok/s | 21.95 tok/s | 55.63% |
| context-medium | 15.8 tok/s | 27.8 tok/s | 15.19 tok/s | 21.83 tok/s | 63.07% |
| context-long | 15.7 tok/s | 25.3 tok/s | 14.94 tok/s | 21.39 tok/s | 64.33% |
| decode-extended | 15.8 tok/s | 141.3 tok/s | 15.16 tok/s | 21.36 tok/s | 59.41% |

The old 9B dense row in this same public JSON is stale: after commit
`dbf2eb5a`, the public core prompt locally measured about `331 prefill tok/s`
against a llama.cpp core prefill around `333 tok/s`. That row still has a
decode gap, but it is no longer the obvious public-suite prefill failure. The
27B dense-hybrid row remains below llama.cpp in every scenario and is still
only about `56%` overall on the core row.

This effort therefore targets 27B first. The relative prefill gap is the bigger
lever than decode:

- core prefill: `15.9 / 104.6 = 15%` of llama.cpp
- decode: `15.44 / 21.95 = 70%` of llama.cpp
- decode-only work cannot recover the full public-suite score; prefill must
  move substantially while decode is guarded

## What llama.cpp is doing better

The local llama.cpp Metal backend has separate quantized matrix-matrix prompt
kernels and matrix-vector decode kernels:

- `kernel_mul_mm_q4_K_f16`, `kernel_mul_mm_q6_K_f16`, and related `mul_mm_id`
  kernels in `ggml-metal.metal`
- `kernel_mul_mv_id_q4_K_f32`, `kernel_mul_mv_id_q6_K_f32`, and related decode
  kernels in the same file
- graph encoding routes `GGML_OP_MUL_MAT` / `GGML_OP_MUL_MAT_ID` through the
  Metal encoder, and the backend explicitly keys off the operation batch size
  before deciding whether large matrix multiplies should stay on Metal

The public ZINC shape is suspiciously flat: 27B prefill sits around
`15.7-15.9 tok/s` across short, medium, long, and extended scenarios. That
usually means one of two things:

1. The active public path is still effectively token-shaped for important
   layers or prompt segments.
2. The path is layer-major, but SSM recurrence, command scheduling, or the
   Q4_K/Q6_K prompt kernels are slow enough that batching is not feeding the M4
   like llama.cpp's graph does.

The first cycle must prove which one is true before editing production kernels.

## Evidence from the first run

Cycles 1 through 9 proved the first bottleneck, and the result is important:
raising the dense layer-major prefix cap by itself is not currently activating
the materialized path. The measured prefill stayed flat at `15.8-16.0 tok/s`
while the profile still reported:

```text
prefill path evidence: path=queued-token-major prompt_tokens=36 target_layer_tokens=2304 observed_layer_visits=2304 ffn_visits=2304 materialized_layer_tokens=0 replay_layer_tokens=2304
prefill largest byte bucket: dense_ffn_gate_up 215.16 GiB
Metal profile: dense SSM prefill projection disabled: layer0 dense guard alpha-type types qkv=q6_k gate=q4_k alpha=f32 beta=f32
```

Do not extend the existing dense prefix cap again until the profile line changes
to `materialized_layer_tokens > 0`. A cap-only patch is already known to be
noise on this workload.

The next useful cycle must do one of these:

1. Enable or explain the layer-0 dense SSM prefill projection blocker for
   `qkv=q6_k`, `gate=q4_k`, `alpha=f32`, and `beta=f32`.
2. Add a validator or profile counter that proves why the projection cannot be
   materialized safely yet, with layer and tensor names.
3. Attack the actual active largest bucket, `dense_ffn_gate_up`, by inspecting
   the Q4_K batched prompt kernel/body against llama.cpp's prompt matmul path.

Instrumentation-only work is acceptable if it produces a sharper blocker name.
Another prefix-cap extension without materialized layer tokens is not useful.

Updated run evidence after the F32 tail guard fix:

```text
Prefill: 36 tokens in 2124.7 ms (16.9 tok/s)
prefill path evidence: path=partial-layer-major+token-replay prompt_tokens=36 target_layer_tokens=2304 observed_layer_visits=2196 ffn_visits=2304 materialized_layer_tokens=108 replay_layer_tokens=2196
prefill largest byte bucket: dense_ffn_gate_up 205.07 GiB
layer-major prefill: materialized_tokens 36 layers 3/64 layer_tokens 108/2304 first_token_major_layer 3 reason prefix-full-attn-guard-failed
next replay blocker: layer 3 kind full-attn reason prefix-full-attn-guard-failed next_bytes projection 2.02 GiB dense_ffn 5.81 GiB remaining_bytes projection 148.31 GiB dense_ffn 329.95 GiB
```

The first non-cap fix worked: materialization is now active for layers 0-2 and
prefill improved from `16.0` to about `17.0 tok/s`. The next structural
blocker is the first full-attention boundary at layer 3. Do not widen a prefix
constant unless the change also implements or proves the full-attention boundary
handoff needed to get past `first_token_major_layer 3`.

Follow-up evidence: a later enablement patch attempted to add detailed
full-attention guard diagnostics, but the captured profile still only showed
the generic `prefix-full-attn-guard-failed` line. Before acting on the detailed
reason, make sure the diagnostic is wired into the actually executed replay
boundary path and appears in the profile artifact.

Latest named blocker after persistent guard fields:

```text
next replay blocker: layer 3 kind full-attn reason prefix-full-attn-guard-failed ... guard_layer 3 guard_kind full-attn full_attn_guard q-shape ssm_guard ok dense_guard ok
```

This narrows the next behavior change to the full-attention Q projection
shape/layout eligibility for layer 3. SSM and dense FFN guards are not the
reason this boundary stops today.

If `q-shape` remains, the next profile must include the actual Q tensor name,
type, element count, derived `q_rows`, resolved `q_dim`, and expected packed
Q+gate shape. Do not guess from the source helper alone; the live model may be
using a packed or split Q/gate layout that differs from the test helper.

Latest breakthrough:

```text
Prefill: 36 tokens in 1550.6 ms (23.2 tok/s)
prefill path evidence: path=partial-layer-major+token-replay prompt_tokens=36 target_layer_tokens=2304 observed_layer_visits=1656 ffn_visits=2304 materialized_layer_tokens=828 replay_layer_tokens=1476
prefill largest byte bucket: dense_ffn_gate_up 137.83 GiB
layer-major prefill: materialized_tokens 36 layers 23/64 layer_tokens 828/2304 first_token_major_layer 23 reason configured-prefix-limit ssm_layers 18 full_attn_layers 5 dense_ffn_layers 23
next replay blocker: layer 23 kind full-attn reason configured-prefix-limit ... full_attn_guard ok ssm_guard ok dense_guard ok
```

The 256-wide full-attention shape fix cleared the layer-3 `q-shape` blocker and
improved prefill from about `17.0` to `23.2 tok/s`. Since the next replay
boundary now reports `configured-prefix-limit` and all guards are `ok`, extending
the prefix limit is justified again. Extend in bounded chunks and keep profiling
the next blocker; do not skip correctness or decode guard checks.

A later 48-column fused Q4_K gate/up tile improved the accepted result slightly
to `23.4 tok/s`, but did not change the structural profile: materialized layers
remain `23/64`, and the next replay blocker is still `configured-prefix-limit`
with all guards `ok`. Treat further tile variants as secondary until another
prefix-extension attempt is measured.

Latest prefix-extension result:

```text
Prefill: 36 tokens in 1434.8 ms (25.1 tok/s)
prefill path evidence: materialized_layer_tokens=972 replay_layer_tokens=1332
layer-major prefill: layers 27/64 layer_tokens 972/2304 first_token_major_layer 27 reason configured-prefix-limit ssm_layers 21 full_attn_layers 6 dense_ffn_layers 27
next replay blocker: layer 27 kind full-attn reason configured-prefix-limit ... full_attn_guard ok ssm_guard ok dense_guard ok
```

The 23-to-27 layer cap extension improved prefill from `23.4` to `25.1 tok/s`.
If the next cycle changes the cap, extend only one more full-attention interval
to `31` and profile again.

Latest 31-layer result:

```text
Prefill: 36 tokens in 1320.5 ms (27.3 tok/s)
prefill path evidence: materialized_layer_tokens=1116 replay_layer_tokens=1188
layer-major prefill: layers 31/64 layer_tokens 1116/2304 first_token_major_layer 31 reason configured-prefix-limit ssm_layers 24 full_attn_layers 7 dense_ffn_layers 31
next replay blocker: layer 31 kind full-attn reason configured-prefix-limit ... full_attn_guard ok ssm_guard ok dense_guard ok
```

The 27-to-31 layer cap extension improved prefill to about `27.2-27.3 tok/s`.
If extending again, move only one more full-attention interval to `35`, then
profile the next blocker.

Latest 35-layer result:

```text
Prefill: 36 tokens in 1199.4 ms (30.0 tok/s)
prefill path evidence: materialized_layer_tokens=1260 replay_layer_tokens=1044
layer-major prefill: layers 35/64 layer_tokens 1260/2304 first_token_major_layer 35 reason configured-prefix-limit ssm_layers 27 full_attn_layers 8 dense_ffn_layers 35
next replay blocker: layer 35 kind full-attn reason configured-prefix-limit ... full_attn_guard ok ssm_guard ok dense_guard ok
```

The 31-to-35 layer cap extension improved prefill to about `29.9-30.0 tok/s`.
If extending again, move one interval to `39`, then profile. The largest
remaining byte bucket is still dense FFN gate/up, but replay bytes are dropping
materially with each interval.

Latest 39-layer result:

```text
Prefill: 36 tokens in 1091.3 ms (33.0 tok/s)
prefill path evidence: materialized_layer_tokens=1404 replay_layer_tokens=900
layer-major prefill: layers 39/64 layer_tokens 1404/2304 first_token_major_layer 39 reason configured-prefix-limit ssm_layers 30 full_attn_layers 9 dense_ffn_layers 39
next replay blocker: layer 39 kind full-attn reason configured-prefix-limit ... full_attn_guard ok ssm_guard ok dense_guard ok
```

The 35-to-39 layer cap extension improved prefill to about `32.9-33.0 tok/s`.
If extending again, move one interval to `43`, then profile.

Latest 43-layer result:

```text
Prefill: 36 tokens in 980.8 ms (36.7 tok/s)
prefill path evidence: materialized_layer_tokens=1548 replay_layer_tokens=756
layer-major prefill: layers 43/64 layer_tokens 1548/2304 first_token_major_layer 43 reason configured-prefix-limit ssm_layers 33 full_attn_layers 10 dense_ffn_layers 43
next replay blocker: layer 43 kind full-attn reason configured-prefix-limit ... full_attn_guard ok ssm_guard ok dense_guard ok
```

The 39-to-43 layer cap extension improved prefill to about `36.7-36.8 tok/s`.
If extending again, move one interval to `47`, then profile.

Latest 47-layer result:

```text
Prefill: 36 tokens in 862.7 ms (41.7 tok/s)
prefill path evidence: materialized_layer_tokens=1692 replay_layer_tokens=612
layer-major prefill: layers 47/64 layer_tokens 1692/2304 first_token_major_layer 47 reason configured-prefix-limit ssm_layers 36 full_attn_layers 11 dense_ffn_layers 47
next replay blocker: layer 47 kind full-attn reason configured-prefix-limit ... full_attn_guard ok ssm_guard ok dense_guard ok
```

The 43-to-47 layer cap extension improved prefill to about `41.7 tok/s`. If
extending again, move one interval to `51`, then profile. At this point the
public core llama.cpp prefill target is still far away, but the M4 disadvantage
has shifted from "no materialized prefix" to "how much of the model can safely
stay layer-major".

Latest 51-layer result:

```text
Prefill: 36 tokens in 753.4 ms (47.8 tok/s)
prefill path evidence: materialized_layer_tokens=1836 replay_layer_tokens=468
layer-major prefill: layers 51/64 layer_tokens 1836/2304 first_token_major_layer 51 reason configured-prefix-limit ssm_layers 39 full_attn_layers 12 dense_ffn_layers 51
next replay blocker: layer 51 kind full-attn reason configured-prefix-limit ... full_attn_guard ok ssm_guard ok dense_guard ok
```

The 47-to-51 layer cap extension improved accepted prefill to about `47.3 tok/s`
while the decode guard measured `15.32 tok/s` (`-1.7%` vs the guard baseline,
still above the `15 tok/s` floor). If extending again, move one interval to
`55`, then profile and keep watching decode.

Latest 55-layer result:

```text
Prefill: 36 tokens in 639.3 ms (56.3 tok/s)
prefill path evidence: materialized_layer_tokens=1980 replay_layer_tokens=324
layer-major prefill: layers 55/64 layer_tokens 1980/2304 first_token_major_layer 55 reason configured-prefix-limit ssm_layers 42 full_attn_layers 13 dense_ffn_layers 55
next replay blocker: layer 55 kind full-attn reason configured-prefix-limit ... full_attn_guard ok ssm_guard ok dense_guard ok
```

The 51-to-55 layer cap extension improved accepted prefill to `56.3 tok/s`.
If extending again, move one interval to `59`, then profile. Only `324` layer
tokens are still replayed on the public core prompt.

Latest 59-layer result:

```text
Prefill: 36 tokens in 519.5 ms (69.3 tok/s)
prefill path evidence: materialized_layer_tokens=2124 replay_layer_tokens=180
layer-major prefill: layers 59/64 layer_tokens 2124/2304 first_token_major_layer 59 reason configured-prefix-limit ssm_layers 45 full_attn_layers 14 dense_ffn_layers 59
next replay blocker: layer 59 kind full-attn reason configured-prefix-limit ... full_attn_guard ok ssm_guard ok dense_guard ok
```

The 55-to-59 layer cap extension improved accepted prefill to `69.0 tok/s`.
If extending again, move one final interval to `63`, then profile. This leaves
only the final layer/tail outside the layer-major prefix for the 36-token public
core prompt.

Latest 63-layer result:

```text
Prefill: 36 tokens in 393.2 ms (91.5 tok/s)
prefill path evidence: materialized_layer_tokens=2268 replay_layer_tokens=36
layer-major prefill: layers 63/64 layer_tokens 2268/2304 first_token_major_layer 63 reason configured-prefix-limit ssm_layers 48 full_attn_layers 15 dense_ffn_layers 63
token-major replay: replay_layers 1 ssm/full_attn/dense 0/1/1 replay_layer_tokens 36
measured replay GPU: total 27.65 ms
next replay blocker: layer 63 kind full-attn reason configured-prefix-limit ... full_attn_guard ok ssm_guard ok dense_guard ok
```

The 59-to-63 layer cap extension improved accepted prefill to `91.8 tok/s`.
The last remaining replay is the final full-attention layer plus dense FFN. If
the code can safely materialize the final layer and hand off to final norm/LM
head, that is the next runtime step. If not, profile why the final-layer handoff
cannot be layer-major.

Latest full-prefix result:

```text
Prefill: 36 tokens in 368.6 ms (97.7 tok/s profile, 95.3 tok/s accepted)
prefill path evidence: path=layer-major materialized_layer_tokens=2304 replay_layer_tokens=0
layer-major prefill: layers 64/64 layer_tokens 2304/2304 reason complete
token-major replay: steps 0 replay_layers 0
prefill largest byte bucket: lm_head 0.97 GiB
prefill queued prefill work: final dispatch 39 barriers 38 record 0.17 ms
```

The 63-to-64 layer cap extension removed token-major replay and improved
accepted prefill to `95.3 tok/s`. The remaining public-core gap to llama.cpp is
now roughly 10%. Do not extend prefix caps further. Focus on the complete
layer-major path: LM head/final tail, materialized dense gate/up kernels, SSM
out, and better profile attribution for layer-major bytes.

A later Q6_K 48-column GEMM variant was kept as neutral enablement around
`96.1 tok/s` and did not clearly promote the best result. Do not add more tile
variants without a profile or exact-shape benchmark proving the current tile is
the limiter.

The follow-up plain Q4_K 48-column GEMM route for 33-48 prompt tokens was also
kept as neutral enablement (`~96.4 tok/s`). It may remain useful coverage, but
it did not move the best accepted prefill score. Treat additional 48-wide tile
variants as low priority unless an exact-shape benchmark shows a bottleneck.

Important LM-head finding: the final prefill LM head already runs on the final
prompt token only via an input offset. The profiler's `lm_head 0.97 GiB` bucket
is therefore not evidence that the path computes an `N x vocab` logits matrix.
It still names the visible final-tail bucket, but next changes should target
the actual Q6_K K=5120 DMMV/argmax/barrier tail rather than removing nonexistent
all-token logits work.

A scoped Q6_K K=5120 LM-head DMMV grouping change from 4 to 16 simdgroups per
threadgroup was accepted as a small improvement (`97.1 -> 97.5 tok/s`). This
reduced vocab-row threadgroups by roughly 4x without changing per-simdgroup
math. The result was positive but below the promotion band and somewhat bimodal,
so do not assume dispatch count is now the only remaining bottleneck. Profile
LM-head time before trying an even larger grouping.

The follow-up 16-to-32 simdgroup LM-head trial was neutral (`97.1 tok/s`
confirmed, best still `97.5 tok/s`) with decode unchanged around `15.6 tok/s`.
Do not continue increasing that grouping unless a fresh exact-shape microbench
or command-timing profile shows the K=5120 Q6_K LM head is still launch-limited.

Corrected layer-major byte attribution then showed the actual largest complete
prefix bucket is dense FFN gate/up, not LM head:

```text
prefill largest byte bucket: dense_ffn_gate_up 215.16 GiB
path bytes: ssm 229.16 GiB attn 59.56 GiB dense 646.53 GiB lm-head 31.08 GiB
prefill buckets: dense ffn total 347.39 GiB gate 107.58 GiB up 107.58 GiB down 132.23 GiB
prefill buckets: ssm proj 86.90 GiB ... out 34.80 GiB
prefill buckets: attn proj 22.51 GiB out 9.49 GiB
```

Next optimization work should focus on the fused Q4_K dense gate/up path for the
exact 27B shape before revisiting LM head. A useful next step is either an
exact-shape microbench for the fused gate/up kernel or a measured kernel change
that reduces dense gate/up bytes, improves vector reuse, or reduces barriers
without splitting the fused route back into slower separate GEMMs.

The split hot-shape profile names the concrete materialized prefill shapes:

```text
prefill q4_k hot #1: dense/dense-gate M=17408 K=5120 bytes=107.58 GiB calls=64
prefill q4_k hot #2: dense/dense-up   M=17408 K=5120 bytes=107.58 GiB calls=64
prefill q4_k hot #3: dense/dense-down M=5120  K=17408 bytes=53.79 GiB calls=32
prefill q6_k hot #1: dense/dense-down M=5120  K=17408 bytes=78.44 GiB calls=32
prefill q6_k hot #4: lm-head/none     M=248320 K=5120 bytes=0.97 GiB calls=1
```

This makes the largest actionable prefill gap the fused Q4_K gate/up kernel for
`M=17408, K=5120, N=36` plus the dense-down Q6_K/Q4_K pair. Prefer exact-shape
benchmarking and changes to `gemm_q4k_gate_up_swiglu.metal` before more
LM-head work.

Route profiling for the fused dense gate/up path showed the locked 36-token
prompt uses the N48 tile for all 64 materialized calls:

```text
fused dense gate/up GEMM tiles:
  n32 calls 0 logical_cols 0 computed_cols 0 over 0.0%
  n48 calls 64 logical_cols 2304 computed_cols 3072 over 33.3%
  n64 calls 0 logical_cols 0 computed_cols 0 over 0.0%
```

This makes the next concrete experiment an exact N36 fused gate/up kernel or
another route that reduces the 33.3% overcompute while preserving correctness.
A dual-A shared-memory variant was attempted and failed the synthetic comparison
test, so do not reintroduce it without first making the small shader comparison
pass against the existing N48 route.

A later production fused gate/up shader change that kept separate gate/up A
tiles and reused one B tile compiled and passed tests, but measured neutral
(`96.9 tok/s`, best still `97.5 tok/s`). The profile still showed the same
N48 route and 33.3% overcompute. Treat barrier reduction inside the current N48
route as low confidence; focus on exact N36 work reduction, dense-down timing,
or a correctness-tested microbench before more shared-memory reshuffling.

An effective-N40 tail route for 33-40 token fused gate/up prefill passed the
synthetic N36 comparison and routed correctly:

```text
n40 calls 64 logical_cols 2304 computed_cols 2560 over 11.1%
n48 calls 0
```

However it measured neutral/slower (`96.3 tok/s`, best still `97.5 tok/s`).
Reducing token-column overcompute alone did not improve the public prompt. This
suggests the fused gate/up hotspot is more likely limited by quantized weight
load/decode, memory layout, or broader dense FFN scheduling than by the skipped
tail MMA. Next work should either measure dense-down Q6_K/Q4_K timing directly
or build an exact-shape microbench for the fused gate/up kernel before adding
more production routes.

The follow-up real-N40 physical tile (`NR1=40`, guarded B loads, 20 KiB
threadgroup memory) was correct and slightly better than the effective-N40
route (`96.3 -> 96.7 tok/s` current), with decode stable around `15.66 tok/s`,
but it still did not beat the best `97.5 tok/s`. The profile still reports
dense gate/up as the largest bucket and N40 calls for all 64 layers. Do not keep
iterating on N40 variants without a microbench or GPU timing; move to dense-down
Q6_K/Q4_K timing or gate/up memory-layout evidence.

A generic Q4_K/Q6_K N40 route for dense-down passed synthetic comparisons but
regressed badly (`94.1 tok/s` vs current `96.7`) and was reverted. Avoid generic
N40 dense-down routes unless a microbench explains the regression first. The
next useful direction is timing/attribution or a layout/packing change, not more
33-40 token tail variants.

GEMM kernel timing labels were added for fused dense gate/up and Q4_K/Q6_K
dense-down paths. This is enablement only; the regular optimization loop does
not set `ZINC_METAL_KERNEL_TIMING=1`, so the normal profile will not show those
per-kernel timings. Before another dense FFN route change, run one timing pass
with that env var to compare fused gate/up vs dense-down GPU time directly.

Dense-down route profiling then showed both dense-down formats are still on N48
with 33.3% token-column overcompute:

```text
q4 n48 calls 32 logical_cols 1152 computed_cols 1536 over 33.3%
q6 n48 calls 32 logical_cols 1152 computed_cols 1536 over 33.3%
```

Because the generic Q4_K/Q6_K N40 route already regressed, do not interpret this
as permission to retry generic N40 immediately. Use it to prioritize timing and
memory-layout evidence for the existing N48 dense-down route.

Measured layer-major prefix GPU duration is now visible in the normal profile:

```text
cmds 1 total 362.13 ms avg 362.13 ms eff 1383.8 GiB/s
path_byte_weighted_gpu_ms ssm/attn/dense 87.95/23.13/251.05
```

This confirms dense FFN dominates the measured prefill GPU command by a large
margin. The byte-weighted split is not per-kernel timing, but it is enough to
stop chasing attention or LM-head work for this prompt. The next dense work
needs real timing or layout evidence for fused gate/up versus dense-down.

The denser byte-weighted split then reported:

```text
dense GPU ms: gate/up/down 77.76/77.76/95.58 gate+up 155.52
dense-down q4/q6/other 38.88/56.70/0.00
```

This makes Q6_K dense-down the largest single dense sub-bucket, while fused
gate+up remains the largest combined bucket. Since generic dense-down N40
regressed, the next Q6_K work should focus on the existing N48 dense-down
kernel's memory layout or exact-shape microbenching rather than another
33-40 token route.

A default-on Q6_K/Q4_K dense-down direct-accumulate path reduced prefill dense
barrier bookkeeping (`other` dropped from roughly 302 to 238) but did not improve
throughput (`96.5 tok/s`, best still `97.5`) or measured prefix GPU time
(`~363.9 ms`). The separate dense-down residual tail is therefore not the
limiting cost for this prompt. Prefer actual dense GEMM kernel/memory-layout
evidence next.

Parallelizing the Q4_K/Q6_K accumulate store across the full threadgroup also
kept neutral (`96.9 tok/s`) and measured worse prefix GPU time (`~368.8 ms`).
Do not keep optimizing the direct-accumulate store path unless an exact-shape
microbench shows a local win; the old non-accumulate dense-down path may be
preferable despite the extra barrier.

Ping-ponging layer-major prefix hidden buffers removed more scheduling work
(`dense other` barriers down to roughly 175, layer-major dispatch/barriers
around 832/655) but still measured neutral (`96.7 tok/s`, GPU time `~362.8 ms`).
This reinforces that command scheduling/copyback cleanup is mostly exhausted;
the gap is inside the dense GEMM kernels or weight layout.

Disabling the dense-down direct-accumulate path by default also stayed neutral
after confirmation (`96.3 tok/s`, GPU time `~365.1 ms`, decode `15.55 tok/s`).
Both direct-accumulate and separate dense-down-tail schedules are now measured
as non-wins. Do not spend more cycles toggling that gate; use a microbench or
kernel-level timing to find the dense GEMM issue.

Exploratory cycles added a dedicated materialized dense FFN prefill case for
this effort (`qwen27b_prefill_dense_ffn`) and later a production-weighted route
summary, but the loop plateau-restored the source tree back to the best cycle
after 20 cycles without a promoted improvement. That benchmark enablement is
therefore recorded in git history and this effort file, but it is not present in
the final best tree. Before another production dense GEMM edit, reapply the
benchmark case from the exploratory cycle or recreate it, run it, and use its
result to pick one route or kernel target.

After comparing llama.cpp's Metal dense matmul path, vectorizing the final
SwiGLU writeback in the fused Q4_K gate/up kernel was kept as a small current
improvement (`96.4 -> 96.9 tok/s`) but still did not beat the best `97.5 tok/s`.
This supports the idea that final writeback is visible but not dominant; the
remaining dense gap is likely in quantized weight decode / main matrix multiply
or layout rather than scalar SwiGLU writeback alone.

The dense FFN shape benchmark production-weighted route summary was also an
exploratory enablement that did not survive the best-tree restore. Reapply it
only if the next run is explicitly evidence-gathering rather than preserving the
fastest source tree.

## Plateau result

The loop stopped after 43 cycles due to plateau auto-stop, then restored the
promoted-best source tree from cycle 23:

```text
Best kept-correct: 97.50 prefill tok/s
Target: 105 tok/s
Plateau stop: 20 cycles since promoted-best cycle 23
```

The restored best tree keeps the small Q6_K LM-head grouping improvement. Later
dense FFN profiling, benchmark, N40, direct-accumulate, ping-pong, and
writeback experiments are preserved in history/notes but are not active in the
final source tree unless manually reapplied.

## Run the loop

Start with the public-suite raw core prompt. It is short enough for fast cycles
and exposes the worst overall 27B public row. Keep decode as the cross-effort
guard so prefill work cannot silently damage generation.

```bash
PROMPT='Developer question: two local LLM benchmark screenshots show different tok/s values for the same model. A useful answer explains likely causes and gives one fair measurement rule.

Answer:'

ZINC_MODEL_ID=qwen36-27b-q4k-m \
ZINC_METRIC_MODE=prefill \
ZINC_PROMPT_MODE=raw \
ZINC_TEST_PROMPT="$PROMPT" \
ZINC_REFERENCE_TEXT="<think>" \
ZINC_MAX_TOKENS=96 \
ZINC_MIN_DECODE_TOKENS=32 \
ZINC_TARGET_TOK_PER_SEC=105 \
ZINC_STOP_ON_TARGET=0 \
ZINC_BENCHMARK_RUNS=3 \
ZINC_BENCHMARK_WARMUPS=1 \
ZINC_BENCHMARK_CONFIRM_RUNS=4 \
ZINC_PROFILE_EVERY=1 \
ZINC_BUILD_OPTIMIZE=ReleaseFast \
ZINC_TEST_TIMEOUT_MS=300000 \
ZINC_RUN_TIMEOUT_MS=1200000 \
ZINC_CROSS_EFFORT_PROMPT="$PROMPT" \
ZINC_CROSS_EFFORT_METRIC=decode \
ZINC_CROSS_EFFORT_PROMPT_MODE=raw \
ZINC_CROSS_EFFORT_MAX_TOKENS=96 \
ZINC_CROSS_EFFORT_EVERY=3 \
ZINC_HARD_FAMILY_COOLDOWN=1 \
ZINC_WORKLOAD_RESET_ON_CHANGE=1 \
ZINC_CODEX_REASONING_EFFORT=xhigh \
bun loops/implement_metal.ts \
  --effort-file loops/efforts/MULTI_HOUR_EFFORT_28_METAL_M4_QWEN36_27B_BEAT_LLAMA.md \
  --agent codex --model gpt-5.5 --cycles 100
```

For a baseline-only check:

```bash
ZINC_MODEL_ID=qwen36-27b-q4k-m \
ZINC_METRIC_MODE=prefill \
ZINC_PROMPT_MODE=raw \
ZINC_TEST_PROMPT="$PROMPT" \
ZINC_REFERENCE_TEXT="<think>" \
ZINC_MAX_TOKENS=96 \
ZINC_MIN_DECODE_TOKENS=32 \
ZINC_BENCHMARK_RUNS=3 \
ZINC_BENCHMARK_WARMUPS=1 \
ZINC_RUN_TIMEOUT_MS=1200000 \
bun loops/implement_metal.ts \
  --effort-file loops/efforts/MULTI_HOUR_EFFORT_28_METAL_M4_QWEN36_27B_BEAT_LLAMA.md \
  --dry-run
```

## First-cycle requirements

Before changing production code:

1. Capture a fresh 27B baseline with `ZINC_PROFILE_EVERY=1`.
2. Report the active prefill path: batched vs replay, prompt tokens, layer
   tokens, dispatch count, barrier count, command-buffer commits, and waits.
3. If the current profile does not expose those fields for 27B, add counters
   first and mark the cycle `@@@STEP_KIND: analysis` or
   `@@@STEP_KIND: enablement`.
4. Name the largest prefill bucket: dense FFN gate/up, dense FFN down, SSM
   projection, SSM recurrence, attention, LM head, command scheduling, CPU
   staging, or tokenizer/setup.
5. Compare against llama.cpp's Metal shape: prompt work should be matrix-matrix
   where the dependency graph allows it; decode work should be matrix-vector.

Do not take a shader rewrite before this evidence exists.

Harness sentinel note: this raw prompt starts with a thinking block and the
Metal loop's output extractor evaluates the first output line. Use `<think>` as
the correctness sentinel for this prefill-scored run instead of waiting for the
final answer text.

## Likely useful directions

1. If the public path is token-shaped for large 27B segments, make the next
   change structural: make the materialized layer-major path actually execute,
   preserving SSM recurrence order and comparing final logits against the
   existing path. Do not just raise the layer cap.
2. If the path is already layer-major, attack the named largest bucket. For
   dense FFN, prefer a measured Q4_K gate/up or Q6_K down kernel improvement
   tied to the exact 27B shape. For SSM, separate projection cost from recurrent
   state-update cost before editing.
3. Add a 27B-specific validator if a proposed layer-major change can diverge:
   report layer, token, tensor name, max abs diff, RMS diff, and the flag-on
   command. Keep risky paths default-off until validated.
4. Use llama.cpp's `mul_mm` vs `mul_mv` split as the north star. A public-prompt
   prefill improvement should move real matrix-matrix work, not just reduce a
   small dispatch or barrier counter.
5. Once prefill reaches the `50 tok/s` band, rerun the full 27B public matrix;
   earlier 27B attempts have helped one scenario while hurting another.

## Known traps

- Do not carry routed-expert route-pack assumptions into this dense 27B effort.
- Do not extend the dense prefix layer cap again while
  `materialized_layer_tokens=0`.
- Do not repeat the prior 27B decode fused gate/up attempts unless a fresh
  profile explains why the earlier versions reduced dispatches but lost speed.
- Do not sweep Q4_K/Q6_K tile constants without exact-shape evidence or a
  microbench that names the bottleneck.
- Do not chase LM head first unless the latest profile moves it above dense FFN
  or SSM buckets.
- Do not change the prompt, max tokens, or reference text to manufacture a win.
- Do not accept one-token or short-completion throughput as sustained decode.

## Milestones

1. `25 tok/s` prefill: clears the worst long-context llama.cpp prefill band.
2. `50 tok/s` prefill: proves the M4 is seeing real prompt batching.
3. `105 tok/s` prefill: reaches the public core llama.cpp band.
4. Keep decode at or above `15 tok/s` during prefill work.
5. Final bar: 27B public-suite overall above `100%` of llama.cpp in at least
   three of four scenarios, with no correctness regression.

## Completion criteria

This effort is complete only when:

- the largest current 27B bottleneck is named by profile output
- ZINC has a validated default-on improvement for that bottleneck
- `zig build test` passes
- the 27B public-suite matrix is re-run and compared against llama.cpp
- any remaining gap is explained by a named profile bucket, not a stale hunch
