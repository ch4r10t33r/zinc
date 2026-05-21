# Effort 15 - RDNA4 Qwen 3.6 27B dense-hybrid prefill and decode

Date: 2026-05-18

Target model:

- RDNA node GGUF: `/root/models/Qwen3.6-27B-Q4_K_M.gguf`
- Harness id: `qwen36-27b-q4k-m`
- Architecture shape: dense Qwen 3.6 hybrid with SSM layers, not MoE.
- Important tensor/log facts seen on the RDNA path:
  - `hidden_dim = 5120`
  - `intermediate_dim = 17408`
  - `output.weight = q6_k`, dims `(5120, 248320)`
  - SSM fastpath line: `ssm qkv=q6_k gate=q4_k alpha=f32 beta=f32 conv=f32 out=q5_k`
  - Dense FFN: `ffn_gate=q4_k`, `ffn_up=q4_k`, `ffn_down=q6_k`

## Executive summary

ZINC is not yet faster than llama.cpp on Qwen 3.6 27B. The gap is
different from the 35B-A3B work because this is not a routed MoE model.
The dominant decode bucket is dense FFN, then SSM. The dominant prefill
problem is structural: the RDNA batched prefill path explicitly rejects
SSM models, so 27B prompt processing still runs the full per-token
decode-style loop.

The next effort should be a structural prefill effort first, then a
decode cleanup effort. Direct single-shader micro-fusions already tested
on this exact model were mostly mixed or negative. The only prefill path
with enough headroom is layer-major batching that preserves SSM and
residual dependencies. Do not simply flip `canUseBatchedPrefillRdna` to
allow `cfg.ssm_d_inner > 0`; an earlier SSM batched attempt reached GPU
hard resets / `QueueSubmitFailed` and had a real dependency bug.

Targets needed to beat llama.cpp, using the latest measured matrix:

| scenario | ZINC prefill | llama prefill | prefill uplift needed | ZINC decode | llama decode | decode uplift needed |
|---|---:|---:|---:|---:|---:|---:|
| core | 17.06 | 61.12 | 3.58x | 20.79 | 34.43 | 1.66x |
| context-medium | 24.15 | 195.01 | 8.07x | 29.42 | 34.40 | 1.17x |
| context-long | 26.04 | 69.89 | 2.68x | 30.17 | 44.33 | 1.47x |
| decode-extended | 19.04 | 97.29 | 5.11x | 25.82 | 31.29 | 1.21x |

The realistic milestone order:

1. Establish a clean 27B-only baseline and profile artifact.
2. Add a validation-only layer-major prefill harness for this SSM+dense
   shape without replacing production output.
3. Batch dense FFN prefill per layer and token chunk, because it is the
   largest per-token weight stream and does not have the SSM recurrent
   dependency.
4. Batch SSM projections per layer and token chunk, but keep delta/state
   recurrence exact until a validated layer-major state path exists.
5. Only then chase decode improvements; keep those flag-gated and judged
   by the full four-scenario matrix.

## Current measured standing

Artifact: `/tmp/zinc-rdna-qwen36-27b-20260517-120205.json`

This run compared ZINC to llama.cpp on the same RDNA node using the
four public scenario shapes.

| scenario | ZINC prefill tok/s | ZINC decode tok/s | llama prefill tok/s | llama decode tok/s | ZINC prefill pct | ZINC decode pct |
|---|---:|---:|---:|---:|---:|---:|
| core | 17.06 | 20.79 | 61.12 | 34.43 | 27.9% | 60.4% |
| context-medium | 24.15 | 29.42 | 195.01 | 34.40 | 12.4% | 85.5% |
| context-long | 26.04 | 30.17 | 69.89 | 44.33 | 37.3% | 68.1% |
| decode-extended | 19.04 | 25.82 | 97.29 | 31.29 | 19.6% | 82.5% |

Artifact: `/tmp/zinc-rdna-qwen36-27b-after-20260517-121618.json`

Clean ZINC-only rerun:

| scenario | ZINC prefill tok/s | ZINC decode tok/s |
|---|---:|---:|
| core | 17.49 | 21.21 |
| context-medium | 23.97 | 29.43 |
| context-long | 26.99 | 31.80 |
| decode-extended | 19.14 | 26.95 |

Interpretation:

- Short `core` prefill is a 5-token prompt and has TTFT noise. Keep it
  in the matrix, but do not optimize from it alone.
- `context-medium`, `context-long`, and `decode-extended` show the real
  prefill problem: ZINC does not amortize prompt tokens.
- Decode is closer than prefill but still behind. The gap ranges from
  17% to 40% depending on scenario, with long-context decode the most
  painful.

2026-05-20 loop checkpoint:

- Best accepted effort-15 checkpoint on the current site-aligned Coding
  Review prefill benchmark is `49.91 tok/s` (`7cec485`), after moving the
  Qwen3.6-27B dense prefill segment range into the late layer-major path.
- The old synthetic Paris prompt can still report about `148 tok/s`, but
  it is not the controlling workload and should not drive keep/revert
  decisions.
- The current biggest profile bucket after the recent keeps remains
  `dense_ffn` (recent refreshes around `2.4 s` on the site-aligned
  prompt). Segment boundary additions near the lower layers are now
  repeatedly measured dead; use fresh subphase data before editing.

## Decode phase budget

A clean single-stream profile on a 48-token decode prompt produced:

```
Generated 48 tokens at about 27.23 tok/s
avg GPU decode token = 35.56 ms

Top-level phases:
  attention   2.27 ms
  ssm        12.46 ms
  dense_ffn  18.29 ms
  tail        1.90 ms

SSM subphases:
  proj        5.39 ms
  conv        0.52 ms
  delta       4.38 ms
  gnorm       0.36 ms
  out         2.20 ms

Dense FFN subphases:
  gateup     10.96 ms
  down        7.35 ms
```

Approximate decode share:

| bucket | ms/token | share |
|---|---:|---:|
| dense FFN | 18.29 | 51% |
| SSM | 12.46 | 35% |
| attention | 2.27 | 6% |
| final tail | 1.90 | 5% |

This model is not primarily an attention problem. Attention work matters
for long context, but the first decode improvements should target dense
FFN and SSM. A flash-attention-only plan will not close the 27B gap.

## Why prefill is slow today

The relevant entry point is `prefillBatched` in `src/compute/forward.zig`.
It calls `canUseBatchedPrefillRdna` before using the batched RDNA
orchestration. That gate currently rejects:

```zig
if (cfg.n_experts > 0) return false;
if (cfg.ssm_d_inner > 0) return false;
```

For Qwen 3.6 27B, `cfg.ssm_d_inner > 0`, so the model falls back to
`prefillBatch`. That fallback processes prompt tokens through the same
token-major path used by decode. Every prompt token rereads the dense
FFN and SSM projection weights.

The current batched prefill implementation is a good fit for dense
attention-only/Gemma-like paths: it keeps `scratch_hidden` as
`[n_tokens, hidden_dim]` and runs layer-major batched attention and dense
FFN. It is not yet a safe fit for Qwen 3.6 27B because SSM layers carry
state and because every layer's SSM branch feeds the hidden stream before
the layer's dense FFN and before later layers.

The dependency that must be preserved for each layer and token is:

```text
hidden[token]
  -> attn_norm
  -> attention OR SSM branch
  -> hidden[token] += branch_output
  -> ffn_norm
  -> dense FFN
  -> hidden[token] += ffn_output
  -> next layer
```

For SSM layers, this is stricter:

```text
hidden[token]
  -> SSM projections (qkv/z/alpha/beta)
  -> conv/state recurrence in token order
  -> gated norm + ssm_out
  -> hidden[token] += ssm_output
  -> dense FFN for the same layer/token
```

Skipping the SSM output for each token while trying to run a single
post-loop batched SSM update corrupts `hidden[token]`, which then
corrupts later projections and the FFN. This exact problem is already
documented in `runSsmLayerGpu`: the prior production skip was removed
because it produced incoherent hidden evolution.

## Why decode is slow today

Dense FFN weight traffic per layer is large:

```text
gate q4_k: 17408 x 5120 weights x 144/256 bytes ~= 50 MB
up   q4_k: 17408 x 5120 weights x 144/256 bytes ~= 50 MB
down q6_k:  5120 x 17408 weights x 210/256 bytes ~= 73 MB

dense FFN stream per layer ~= 173 MB, before activation reads/writes
```

Every layer has the dense FFN. Even if the exact layer count changes by
checkpoint, the decode profile confirms this traffic is the largest
bucket: `dense_ffn=18.29 ms`, with `gateup=10.96 ms` and `down=7.35 ms`.

SSM also has meaningful weight traffic plus recurrent state work:

```text
wqkv: q6_k, M=conv_channels, K=hidden_dim
z:    q4_k, M=d_inner,       K=hidden_dim
alpha/beta: f32, small M=dt_rank
out:  q5_k, M=hidden_dim,    K=d_inner
delta_net: recurrent state update, not just matvec traffic
```

For decode, there is no cross-token batch to amortize weight reads:
autoregressive dependency forces one generated token at a time. That is
why decode wins must come from fewer dispatches, better single-token
kernel shape, or cutting unnecessary memory traffic. Re-reading weights
once per token is fundamental unless speculative/multi-token decoding is
introduced, which is outside this effort.

## Failed or rejected attempts on this exact 27B target

Do not repeat these without a new measurement hypothesis and a clear
reason the prior failure mode no longer applies.

| attempt | result | decision |
|---|---|---|
| Widen dense fused gate+up+SwiGLU eligibility to include `inter_dim=17408` | mixed; core/context-long worse, only some decode rows looked better | rejected |
| Dense-delta gate variant | mixed; medium/long decode worse | rejected |
| `ZINC_SSM_DELTA_COLS8=0` default flip | quick profile sometimes faster, full matrix mixed/worse | rejected |
| Q6_K+Q4_K fused SSM qkv+z pair | engaged but regressed; SSM proj got slower | rejected |
| Q6_K K=17408 specialization for dense FFN down | regressed decode and dense_ffn | rejected |
| Wider Q4/Q6 variants for dense FFN/down | flat or negative across matrix | rejected |
| Widened fused attention o-proj merge to `hidden_dim=5120` | severe long-context regression, attention exploded | rejected |
| `ZINC_SSM_DELTA_NORMED_QK=1` | regressed | rejected |
| Full SSM batched prefill gate flip | GPU hard resets / `QueueSubmitFailed`; dependency bug in hidden evolution | hard ban |
| Prefix SSM projection replay (`ZINC_QWEN36_27B_SSM_PREFILL_PROJ=z/qkv/both`) | coherent after keeping fused RMS+alpha+beta, but context-medium regressed from ~29.24 tok/s to 29.09-29.17 tok/s at L1 and 27.91-28.97 tok/s at L2/L4/L8 | default-off diagnostic only |
| Forcing separate SSM alpha/beta path (`ZINC_FUSED_SSM_AB=0`) | immediate `QueueSubmitFailed` on Qwen 3.6 27B | do not use as a prerequisite for 27B prefill work |
| Layer-major dense segment lower-bound additions | layer-1 extension measured old 4-62 override at `50.16 tok/s` vs new layer-1 schedule at `50.07 tok/s`; earlier prefix/layer-depth sweeps were flat or worse | rejected unless a new profile proves lower layers are hot |
| Partial hidden scratch copy fused with first attention RMS norm at segment handoff (`ZINC_QWEN36_27B_PARTIAL_ATTN_NORM_STORE`) | OFF median `49.96 tok/s` `[51.32, 49.82, 49.96]`; ON median `49.53 tok/s` `[49.53, 49.42, 49.62]`; also moved attention RMS outside the normal phase timer | rejected |

Rejected matrix details:

| artifact | core | context-medium | context-long | decode-extended |
|---|---:|---:|---:|---:|
| clean ZINC | 17.49 / 21.21 | 23.97 / 29.43 | 26.99 / 31.80 | 19.14 / 26.95 |
| fused-wide rejected | 17.05 / 20.61 | 23.48 / 30.66 | 25.29 / 27.70 | 18.69 / 27.16 |
| dense-delta rejected | 18.21 / 21.10 | 23.98 / 27.66 | 25.81 / 30.87 | 19.12 / 27.00 |
| q6wide rejected | 16.32 / 20.10 | 23.58 / 28.94 | 25.72 / 28.03 | 18.27 / 26.76 |
| q4wide rejected | 17.41 / 20.85 | 24.04 / 28.80 | 25.88 / 31.88 | 18.68 / 25.86 |
| wide rejected | 16.49 / 22.70 | 23.53 / 30.05 | 25.74 / 28.89 | 18.16 / 26.40 |

Each cell is `prefill tok/s / decode tok/s`.

2026-05-18 follow-up:

- Default dense-FFN prefix depth remains one layer for context prompts.
  The real coding-review prompt measured ~29.23-29.24 tok/s prefill at
  L1; L2/L4/L8 with SSM projection replay were all slower.
- Very short prompts are the one safe exception. The 5-token raw smoke
  prompt improved from roughly 18-20 tok/s at L1 to ~23.9-24.1 tok/s at
  L8 with identical output tokens. Production now chooses L8 only when
  `prompt_len <= 8`; longer prompts keep L1.
- The SSM qkv/z replay path is left behind
  `ZINC_QWEN36_27B_SSM_PREFILL_PROJ={qkv,z,both}` for diagnostics. It is
  not a fix for the context prefill gap because the replay setup/copy
  overhead cancels the saved projection work.

2026-05-20 interrupted-cycle analysis:

- I stopped the loop after it had produced a candidate source edit, then
  measured the candidate directly against its own control instead of
  allowing more cycles to build on it.
- Candidate: fuse partial hidden scratch copy and first attention-layer
  RMS norm at a full-attention segment handoff.
- Result: control/OFF median `49.96 tok/s`; candidate/ON median
  `49.53 tok/s`. The candidate was reverted.
- Lesson: do not keep moving work out of the normal `attention`/`dense_ffn`
  profile sections for cosmetic copy removal. It can make phase budgets
  less trustworthy while not moving the real prefill wall time.
- Next loops should pivot from boundary-extension variants to
  measurement-backed dense subphase work: shaderstats for the current
  fused Q4_K gate/up/SwiGLU path and Q6_K batched down/kpar path, then
  one code change tied to the top subphase.

## Measurement contract

Always measure on a clean RDNA node. Stop stale GPU users first:

```bash
source .env

ssh -p "$ZINC_PORT" "$ZINC_USER@$ZINC_HOST" "\
  pkill -f 'zig-out/bin/zinc' || true; \
  pkill -f 'llama-server' || true; \
  pkill -f 'llama-cli' || true"
```

Full four-scenario comparison:

```bash
source .env

bun tools/performance_suite.mjs \
  --target rdna \
  --models qwen36-27b-q4k-m \
  --phase all \
  --rdna-sync \
  --rdna-build \
  --rdna-start-llama \
  --runs 3 \
  --warmup 1 \
  --timeout-ms 1800000 \
  --no-site-write \
  --output /tmp/zinc-rdna-qwen36-27b-$(date +%Y%m%d-%H%M%S).json
```

ZINC-only cycle check after a small change:

```bash
source .env

bun tools/performance_suite.mjs \
  --target rdna \
  --models qwen36-27b-q4k-m \
  --phase zinc \
  --rdna-sync \
  --rdna-build \
  --runs 3 \
  --warmup 1 \
  --timeout-ms 1800000 \
  --no-site-write \
  --output /tmp/zinc-rdna-qwen36-27b-zinc-only-$(date +%Y%m%d-%H%M%S).json
```

Extract medians from an artifact:

```bash
jq -r '.targets[0].models[0].scenarios[] |
  [.id,
   .zinc.prefill_tps.median,
   .zinc.decode_tps.median,
   (.baseline.prefill_tps.median // "na"),
   (.baseline.decode_tps.median // "na")] | @tsv' \
  /tmp/zinc-rdna-qwen36-27b-ARTIFACT.json
```

Profile a sustained decode prompt:

```bash
source .env

rsync -az --delete \
  --exclude '.zig-cache' --exclude 'zig-out' --exclude 'node_modules' \
  --exclude '.DS_Store' --exclude 'site' \
  -e "ssh -p $ZINC_PORT" . "$ZINC_USER@$ZINC_HOST:/root/zinc/"

ssh -p "$ZINC_PORT" "$ZINC_USER@$ZINC_HOST" "\
  cd /root/zinc && \
  zig build -Doptimize=ReleaseFast && \
  RADV_PERFTEST=coop_matrix ./zig-out/bin/zinc \
    -m /root/models/Qwen3.6-27B-Q4_K_M.gguf \
    --prompt 'LLM benchmark reports should separate prompt processing from token generation because' \
    -n 48 \
    --profile"
```

Collect these lines every time:

```text
Prefill: ...
Prefill profile: ...
Prefill GPU phases: ...
Prefill SSM subphases totals: ...
Generated ... tok/s
Modeled decode bandwidth: ...
PROFILE: avg GPU decode token=...
PROFILE: avg GPU phases ...
PROFILE: avg SSM subphases ...
PROFILE: avg dense_ffn subphases ...
PROFILE: avg tail subphases ...
```

## Acceptance rules

A performance keep must satisfy all of these:

1. `zig build -Doptimize=ReleaseFast` succeeds on the RDNA node.
2. Local `zig build test` succeeds before commit.
3. No `QueueSubmitFailed`, no GPU reset, no `amdgpu: ring ... timeout`
   in `dmesg`.
4. The output remains coherent on a short raw prompt and the
   decode-extended prompt.
5. For production-on changes, compare the full four-scenario matrix, not
   only one CLI prompt.
6. Keep if median prefill or decode improves materially and no primary
   scenario regresses more than 2%, unless the change is an explicitly
   default-off diagnostic/foundation.
7. For prefill changes, `ZINC_BATCHED_PREFILL=validate` or an equivalent
   new validator must compare final logits against the per-token path.
8. Any SSM prefill change that resets the GPU is rejected even if a
   shorter prompt once passes.

Recommended perf thresholds:

- Prefill production keep: at least +10% on one context scenario and no
  context scenario down more than 2%.
- Decode production keep: at least +3% on the `decode-extended` median
  or at least +2% across two of the four decode medians with no long
  context regression.
- Foundation keep: allowed at 0% only if default-off and it adds
  validation/profiling needed for the structural work.

## Work plan

### Track 0 - Baseline hygiene and profiling

Goal: make every later cycle attributable.

Steps:

1. Run the full matrix and save the artifact under `/tmp`.
2. Run the sustained `--profile -n 48` prompt.
3. Record Mesa/GECC/coopmat state:

```bash
ssh -p "$ZINC_PORT" "$ZINC_USER@$ZINC_HOST" "\
  dpkg -l mesa-vulkan-drivers | tail -1; \
  cat /sys/module/amdgpu/parameters/ras_enable; \
  vulkaninfo --summary | sed -n '1,80p'"
```

4. Confirm no stale processes:

```bash
ssh -p "$ZINC_PORT" "$ZINC_USER@$ZINC_HOST" "\
  pgrep -af 'zinc|llama-server|llama-cli' || true"
```

Deliverable: a short note in the next effort log with:

- artifact path
- ZINC/llama table
- profile phase table
- driver state
- commit SHA

### Track 1 - Prefill validation harness for Qwen 3.6 27B

Goal: build the safety rail before changing production prefill.

Do not start by relaxing `canUseBatchedPrefillRdna`. Instead add a
separate default-off path, for example `ZINC_QWEN36_27B_PREFILL_VALIDATE=1`,
that can run a candidate layer-major calculation and compare it with the
current per-token reference.

The first validator should not attempt to replace the full model output.
It should capture one layer/chunk and compare intermediate tensors:

1. Pick a small token chunk, e.g. 4 or 8 prompt tokens.
2. Run the current per-token path.
3. Capture reference tensors for one layer:
   - pre-FFN `ffn_norm`
   - dense gate output
   - dense up output
   - SwiGLU output
   - dense down output
   - post-FFN hidden
4. Re-run the candidate batched dense FFN for that same layer/chunk into
   scratch buffers.
5. Compare max absolute error and top-k worst indices.

Start with dense FFN because it is stateless across tokens once
`ffn_norm[token]` is known. Do not start with SSM delta recurrence.

Suggested validation tolerances:

- Dense FFN intermediate DMMV outputs: `1e-3` to `3e-3` absolute.
- Final hidden after residual: `3e-3` absolute.
- Final logits after a whole validate run: `1e-3` may be too strict for
  changed reduction order; use intermediate diffs to decide, then set a
  documented threshold.

Useful existing infrastructure:

- `ensureBatchedScratchCapacity`
- `dispatchProjectionBatched`
- `dispatchResidualRmsNorm`
- `dispatchFfnActivation`
- current validate mode in `prefillBatched`
- SSM capture comments/buffers around `runSsmLayerGpu`

Deliverable: default-off validator, no production behavior change.

### Track 2 - Batched dense FFN prefill per layer

Goal: stop rereading the dense FFN weights once per prompt token.

Current per-token dense FFN sequence:

```text
ffn_norm[token]
  -> gate DMMV q4_k
  -> up DMMV q4_k
  -> SwiGLU
  -> down DMMV q6_k
  -> hidden[token] += down
```

Candidate layer-major sequence:

```text
for layer L:
  produce ffn_norm[token] for all tokens in chunk, preserving prior branch output
  batched gate: q4_k [M=17408, K=5120] x [K, N]
  batched up:   q4_k [M=17408, K=5120] x [K, N]
  activation:   swiglu over [N, 17408]
  batched down: q6_k [M=5120, K=17408] x [K, N]
  residual add per token
```

This is the biggest prefill lever because the dense FFN streams about
173 MB of weights per layer per token today. With a chunk of N tokens,
the target is to stream those weights once per layer/chunk instead of N
times. Even after activation traffic and scratch writes, this is the
only route to multi-x prefill movement.

Implementation notes:

- Use existing batched projection helpers first. They may not be the
  final fastest GEMM shape, but they are already wired and safer.
- Keep chunk sizes conservative at first: 4, 8, 16. Do not jump to 128
  until validation passes and scratch memory is audited.
- Store scratch as token-major if that matches existing helpers. Avoid a
  huge layout refactor in the first production attempt.
- Run only one layer in validate mode first, then all layers default-off,
  then production-on after matrix proof.
- If the existing batched DMMV path dispatches one workgroup per
  row/token and underutilizes, pivot to the in-tree `mul_mm_q4k` style
  only after correctness is proven.

Risks:

- A batched dense FFN cannot be run before the layer's attention/SSM
  branch has updated `hidden[token]`.
- For SSM layers, all tokens' branch outputs must be exact before this
  layer's dense FFN can batch.
- Scratch memory can get large:
  - gate/up scratch for 16 tokens: `16 x 17408 x 4` each ~= 1.1 MB each.
  - down/hidden scratch is smaller.
  - This is fine, but keep all buffers sized explicitly.

Acceptance:

- Default-off validator passes for chunks 4, 8, and 16.
- Production-on chunk 8 or 16 improves context prefill by at least 10%.
- Decode does not regress more than noise; the code should not touch
  decode unless helpers are shared.

### Track 3 - Batched SSM projection prefill

Goal: amortize the SSM projection weights while preserving token-order
state recurrence.

Do this only after Track 1 validation exists. The SSM delta/state update
is the risky part; the four projections are less risky:

```text
attn_norm[token]
  -> wqkv q6_k
  -> z q4_k
  -> alpha f32
  -> beta f32
```

Candidate sequence for one SSM layer and token chunk:

1. Produce `attn_norm[token]` for all tokens in the chunk.
2. Batched project `wqkv`, `z`, `alpha`, `beta`.
3. Run conv/delta recurrence in token order or with a separately
   validated batched delta path.
4. Run gated norm and `ssm_out`.
5. Apply residual to each token before dense FFN.

The first production version may still run delta recurrence token by
token. That can still win if SSM projection weight reads are a large
share of prefill. The decode profile says SSM projection is about 5.39
ms of a 12.46 ms SSM bucket, so it is worth isolating.

Do not use the old destructive "skip runSsmLayerGpu and dispatch one
batched delta post-loop" shape. It corrupts later hidden states.

Acceptance:

- Compare `wqkv`, `z`, `alpha`, and `beta` outputs against per-token
  captures.
- Compare `delta_out`, `ssm_out`, and post-SSM hidden after the
  recurrence path.
- No GPU reset on a long-context prompt.
- Context prefill improves beyond Track 2, or the code stays default-off.

### Track 4 - Decode dense FFN, only after prefill foundation

Goal: improve single-token decode without repeating rejected wide/fused
variants.

The direct attempts already say "do not simply widen existing fused
gate/up/SwiGLU". Instead:

1. Add more granular profile counters if needed:
   - separate dense gate q4_k and dense up q4_k, not just `gateup`
   - separate dense down q6_k
   - record dispatch counts per token
2. If shaderstats shows register pressure or occupancy collapse, build a
   shape-specific variant behind an env flag. Candidate:
   - dense Q4_K gate/up/SwiGLU with `NUM_ROWS=1` for `M=17408`,
     designed to reduce register pressure versus the rejected wide fused
     path.
3. Test with full matrix. If it only helps `core`, reject it.

Known-safe decode insight:

- `dispatchDmmvAcc` for dense down already fuses down projection with
  residual add when no post-FFN norm exists. Do not undo that.

Rejected decode idea:

- A one-dispatch `ssm_out + ffn_norm` is not straightforward. `ssm_out`
  is a row-parallel DMMV that writes/accumulates pieces of `hidden`.
  `ffn_norm` requires the RMS of the completed hidden vector. There is
  no global synchronization inside one Vulkan dispatch across all rows.
  A correct version is at least two dispatches or a different work
  decomposition. Treat it as a research item, not a quick fusion.

### Track 5 - Decode SSM

Goal: reduce the 12.46 ms SSM bucket without repeating failed qkv+z and
delta variants.

Open candidates:

1. SSM `out` q5_k row/block tuning.
   - Current profile: `ssm_out ~= 2.20 ms`.
   - Prior wide tests focused on Q4/Q6 FFN and qkv/z, not this q5_k
     output projection shape.
   - Build only if shaderstats shows poor occupancy or memory pattern.
2. SSM projection diagnostics.
   - Split `ssm_proj` timing into wqkv, z, alpha, beta for one profiling
     cycle.
   - If alpha/beta are tiny but high-overhead, revisit fused RMS+alpha+beta
     only as a shape-specific cleanup.
3. Delta-net variants.
   - Current cols8 default is not clearly bad in the full matrix.
   - `ZINC_SSM_DELTA_NORMED_QK=1` regressed; do not retry.
   - Any new delta variant needs a clean state/output diff, not just text.

Acceptance:

- `--profile` shows the target SSM subphase moved.
- Full decode matrix improves, especially `context-long`.
- No prefill regression, because SSM helpers are shared.

### Track 6 - Long-context attention

Goal: address the `context-long` decode gap after dense FFN/SSM.

Attention is only about 2.27 ms on the short sustained profile, but the
long-context scenario has the largest decode deficit. After the dense
and SSM buckets are stabilized, run:

```bash
ZINC_FA_PROFILE_LAYER=1 RADV_PERFTEST=coop_matrix ./zig-out/bin/zinc \
  -m /root/models/Qwen3.6-27B-Q4_K_M.gguf \
  --prompt "$(cat /tmp/long-context-prompt.txt)" \
  -n 32 \
  --profile
```

Open candidates:

- seq-len-dependent flash attention split/merge parameters, borrowing
  the Effort 11 lesson that long-context and short-context optimal
  dispatch shapes differ.
- KV-cache write fused into K projection on attention layers. Lower ROI
  on 27B than dense FFN/SSM, but structurally safe if kept narrow.

Do not port a broad new attention architecture before proving attention
is a top bucket on the 27B long-context prompt.

## Immediate next-cycle guidance after the 49.91 checkpoint

1. Refresh `ZINC_PREFILL_PROFILE=1` on the exact site-aligned Coding
   Review prompt and record `dense_ffn` subphases.
2. If `gateup` is still dominant, collect `RADV_DEBUG=shaderstats` for
   the current fused Q4_K gate/up/SwiGLU kernel before editing it.
3. If `down` is dominant, collect shaderstats for the current Q6_K
   batched down/kpar path and target down+acc fusion or register pressure.
4. If benchmark samples are bimodal around `50` and `53 tok/s`, run an
   interleaved old-vs-new control. Do not accept a sub-1% change from an
   unpaired run.
5. Treat the layer-major segment boundary as provisionally settled.
   Do not add layers 1, 2, or 3 again without a fresh profile showing
   those layers are now the reason `dense_ffn` is hot.

## Suggested cycle schedule

Cycle 1:

- Run baseline matrix and sustained profile.
- Add a short running note with artifact paths and driver state.

Cycle 2:

- Add default-off dense FFN layer/chunk validator.
- No production behavior change.

Cycle 3:

- Validate one layer, chunk 4 and 8.
- Fix diffs until intermediate outputs match.

Cycle 4:

- Extend validator across all layers default-off.
- Confirm no GPU reset.

Cycle 5:

- Enable batched dense FFN prefill for a small chunk behind
  `ZINC_QWEN36_27B_DENSE_PREFILL=1`.
- Measure ZINC-only matrix.

Cycle 6:

- Tune chunk size 4/8/16.
- Keep only if context prefill moves.

Cycle 7:

- Add SSM projection capture/validator.
- No production behavior change.

Cycle 8:

- Batch SSM projections default-off, keep recurrence exact.
- Measure prefill SSM subphase.

Cycle 9:

- If prefill is moving, integrate dense FFN + SSM projection batching
  under one default-off flag and run full all-phase matrix.

Cycle 10:

- Decode-only diagnostic: split dense gate/up/down and SSM projection
  sub-buckets further if the prefill work did not already expose them.

Cycle 11+:

- Try exactly one decode candidate at a time. Prefer q5_k SSM out or
  shape-specific dense gate/up/SwiGLU with lower register pressure. Do
  not mix decode experiments with prefill restructuring in the same keep.

## Files to read before editing

Primary:

- `src/compute/forward.zig`
  - `canUseBatchedPrefillRdna`
  - `prefillBatched` / `prefillBatchedImpl`
  - `prefillBatch`
  - `runSsmLayerGpu`
  - dense FFN path around the `dense_ffn` profile phase
  - `dispatchProjectionBatched`
  - `dispatchDmmvAcc`
- `src/compute/dmmv.zig`
  - batched Q4_K/Q6_K dispatch helpers
  - k-parallel Q4/Q6 paths
- `src/compute/elementwise.zig`
  - residual RMS norm
  - SSM conv/delta/gated norm dispatches
- `src/shaders/dmmv_q4k_batch*.comp`
- `src/shaders/dmmv_q6k_batch*.comp`
- `src/shaders/dmmv_q4k.comp`
- `src/shaders/dmmv_q6k.comp`
- `src/shaders/dmmv_q5k.comp`
- `src/shaders/ssm_delta_net*.comp`
- `src/shaders/residual_rms_norm.comp`
- `src/shaders/rms_norm_add.comp`

Historical context:

- `loops/efforts/MULTI_HOUR_EFFORT_6_RDNA_QWEN36_PREFILL.md`
- `loops/efforts/MULTI_HOUR_EFFORT_8_RDNA_BATCHED_PREFILL.md`
- `loops/efforts/MULTI_HOUR_EFFORT_10_QWEN36_DECODE.md`
- `loops/efforts/MULTI_HOUR_EFFORT_11_RDNA_DECODE_LONG_CONTEXT.md`

## Final decision rule

This effort is successful when one of these is true:

1. ZINC beats llama.cpp on 27B decode in at least three of four
   scenarios and does not lose prefill by more than 20% in any context
   scenario.
2. ZINC improves 27B context prefill by at least 2x over the current
   clean baseline while preserving decode.
3. The effort produces a default-off, validated layer-major SSM+dense
   prefill foundation with no GPU resets and a measured partial prefill
   win, even if full llama.cpp parity needs another effort.

The most likely good outcome is #3 first, then #2. Expecting #1 from
single-token shader tweaks alone is not realistic on this model.
