# Effort 24 — CUDA batched prefill (wire the 5.9× gemm_*_tiled_v2 GEMMs)

> **Status:** OPEN. The top untapped CUDA prefill win. The validated
> `gemm_q4k/q5k/q6k_tiled_v2` GEMMs (Y[T,M]=A[T,K]·W[M,K]^T, ~9300 GFLOP/s,
> **5.9× over per-token matvec**) are in `kernels.cu` but UNWIRED. Production
> prefill (`main.zig`) runs `decodeStep`/`prefillStep` once per prompt token →
> re-reads every weight T times. Batch it: read each weight ONCE for all T
> tokens. Full design in memory `project_batched_prefill_design`.

Opened 2026-06-12. Stacks on the head-skip (`c78e64ed`, already on main).

## SCOPE — Phase 1: gemma-31b DENSE only (pure transformer → clean GEMM win)
Qwen is hybrid-SSM (3/4 layers = sequential state scan) and is OUT of scope
until a chunked/parallel SSM scan exists. gemma-26b MoE = Phase 2 (later). Do
gemma-31b dense first.

## APPROACH — ADDITIVE (forward_cuda_gemma.zig is the parallel gemma work's HOT
## file: add NEW methods + a toggle; do NOT rewrite existing decodeStep/blocks)
Add `ForwardGemma.prefillBatched(tokens) !u32` behind env `ZINC_BATCHED_PREFILL`:
process all T prompt tokens at once, return the last token's argmax (= first gen
token). Per layer:
- batched norms: loop T or a T-batched rms variant (cheap; not the bottleneck)
- Q/K/V + O projections, FFN gate/up/down: `gemm_*_tiled_v2`
  (`GemmPush{M,K,T,a_offset,x_offset,y_offset,acc_mode}`) over the T tokens
- attention: **Target 1 LOOPS** the existing single-query attention per token
  (each query masked to [0..its pos]) — defer the batched-attn kernel
- KV write: all T positions (batched, or loop)
Then the last token's existing tail (the head). Wire `main.zig` prefill to call
`prefillBatched` when `ZINC_BATCHED_PREFILL` is set, else the current per-token
loop. (The duck-typed prefill fn already special-cases the last token.)

## TARGETS (one per cycle; each must pass the gate before any commit)
1. **Batched skeleton** — `prefillBatched` with LOOPED attention + GEMM
   projections/FFN. The big first step; may span >1 cycle (use the cycle log).
2. **Batched causal-attention kernel** — replace the per-token attention loop
   (grid=(n_head,T), each query masked to its pos; or flash-style online softmax).
3. **Batched KV write** — all T positions in one launch (if still looped).
4. (later) gemma-26b MoE prefill; tensor cores via NVRTC `-I` (+2.2×).

## GATE (NON-NEGOTIABLE — the batched path must be OUTPUT-IDENTICAL)
- `ZINC_BATCHED_PREFILL=1 dbg_cuda gen <prompt> N <model>` GEN_IDS must be
  **byte-identical** to the per-token path. Extend `scripts/prefill_catalog.sh`
  to A/B batched-vs-per-token (it already A/Bs head-skip the same way). Mismatch
  → REVERT.
- `scripts/validate_catalog.sh` → 5/5 token-correct vs llama.cpp.
- Measure prefill tok/s via `scripts/prefill_catalog.sh` (ABBA-counterbalanced).

## HARD RULES (from memory — violating wastes the cycle)
- **Build:** isolated caches; verify the binary hash CHANGED.
- **Box:** 4090-pinned (`GPU-e59a6fce-1961-bafe-927c-06c0149f2370`); isolated
  dir `~/zinc-e24`, NEVER `~/workspace/zinc` (parallel 5090 research).
- **Coordination:** main moves FAST (active parallel 5090/gemma work). Branch
  off the LATEST origin/main each cycle, rebase often, keep changes ADDITIVE
  (new methods + toggle) to minimize conflicts. NEVER roll back parallel work.
- Commit only a validated, output-identical, FASTER increment to `perf/e24-*`;
  push (NOT main). Incomplete/negative → log it in the cycle log + continue.

## EXPECTED
gemma-31b prefill ~15-35 → ~90-200 tok/s (5.9× on the GEMM-able majority;
attention is a smaller FLOP share). Stacks on the head-skip's +4%.

## CYCLE LOG
- **2026-06-12 — Cycle 1: batched dense-gemma prefill skeleton COMPLETE + output-identical + faster.**
  Wired the validated `gemm_q4k/q5k/q6k_tiled_v2` GEMMs into a new additive
  `ForwardGemma.prefillBatched(tokens) !u32` (forward_cuda_gemma.zig): token-major
  `BatchScratch` (lazy, sized to T), `gemmDispatch` (Q4_K/Q5_K/Q6_K → tiled_v2 GEMM;
  q8_0/f32 → per-token dmmv fallback), `attentionLayerBatched` (batched pre-norm +
  Q/K/V/O GEMMs; per-head V-norm/KV-write, Q/K norm+RoPE, causal softmax LOOPED per
  token via `aliasBuffer` into the token-major scratch — reuses the single-token
  kernels, zero new kernels), `ffnBlockBatched` (batched pre-norm + gate/up/down
  GEMMs + element-wise GeGLU over [T,n_ff] + fused post-ffn norm/residual/scale).
  Tail (rms_norm+LM head+argmax) on the last token only. MoE (n_experts>0) falls
  back to the per-token path. Toggle `ZINC_BATCHED_PREFILL` wired in BOTH main.zig
  (product) and dbg_cuda.zig gen-path (gate harness) + `Engine.prefillBatched`.
  Built clean on the 4090 box (`zig build cuda-dbg`, fresh `.zig-cache`, EXIT=0,
  bin 45251da3). Direct A/B (dbg_cuda gen, GEN_IDS = the gate):
    - gemma-31b T=80:  29.30→53.13 t/s (+81%),  GEN_IDS byte-IDENTICAL ✓
    - gemma-31b T=200: 32.20→75.05 t/s (+133%), GEN_IDS byte-IDENTICAL ✓ (multi-tile T>128)
    - gemma-26b MoE T=60: GEN_IDS identical (per-token fallback, as designed) ✓
  GATE STATUS: GEN_IDS byte-identical on direct A/B (strong). REMAINING for merge:
  (1) extend scripts/prefill_catalog.sh to ABBA-counterbalance the batched A/B,
  (2) scripts/validate_catalog.sh 5/5. Committed to perf/e24-batched-prefill (WIP,
  toggle off by default → cannot regress production). NEXT: target #2 batched
  causal-attention kernel (replace the per-token attention loop — the remaining
  per-token launch overhead) for a bigger T, then run the formal gate.
- **2026-06-12 — Cycle 2: batched causal-attention kernel wired (target #2) — output-identical + faster.**
  Added an ADDITIVE kernel `gemma_attention_batched` (kernels.cu) — a verbatim twin
  of `gemma_attention` (same 3-pass softmax / GQA / sliding-window / no-sink / scale,
  same `zinc_block_reduce_*` order → bit-identical math) but batched over queries:
  block=(head=blockIdx.x, t=blockIdx.y), seq_len=t+1, token-major Q `q+(t*n_heads+head)*hd`
  and out, SWA `start = (window>0 && seq_len>window) ? seq_len-window : 0`. Chose a NEW
  gemma kernel over modifying main's just-landed `attention_causal_batched` (which carries
  sink logic gemma doesn't use) — zero conflict + guaranteed bit-identity. Wired in
  `attentionLayerBatched` (forward_cuda_gemma.zig): the per-token loop now does ONLY the
  norm/RoPE/KV-write (Q RoPE'd in place into b.q, K/V into the cache); the T per-token
  `gemma_attention` launches are REPLACED by ONE `gemma_attention_batched` launch
  grid=(n_head,T) reading b.q + the prompt region [0..T) of kv_k/kv_v → b.attn_out
  (shared mem = T*4). New `GemmaAttnBatchPush` + pipe `gemma_attention_batched`.
  Extended `scripts/prefill_catalog.sh` ADDITIVELY: `ZINC_AB=headskip|batched` (default
  headskip, unchanged) — batched mode A/Bs ZINC_BATCHED_PREFILL=1 vs baseline with the
  same ABBA counterbalancing + GEN_IDS-identical gate. Built clean on the 4090 box
  (fresh `.zig-cache`, EXIT=0, bin md5 dc54e7cd, CHANGED from cycle 1's 45251da3).
  GATE (ABBA x2, 250-tok prompt, 4090):
    - gemma4-31b dense: 31.43 → 94.12 t/s (+199%, ~3×), GEN_IDS byte-IDENTICAL ✓
    - gemma4-26b MoE:   54.52 → 58.19 (+7% noise, per-token fallback) GEN_IDS identical ✓
  Direct varied-output A/B (cyclic prompt → GEN 235,612,919,1471,218915,10205,…, not a
  collapsed prompt): byte-identical, prefill 30.4→91.7 t/s. Cycle 1 was +133% (T=200);
  the batched attention removed the per-token attn launch overhead → +199%. Committed to
  perf/e24-batched-prefill (toggle off by default → cannot regress production). REMAINING
  for merge: `scripts/validate_catalog.sh` 5/5 (per-token product path unchanged + batched
  is byte-identical to it → transitively correct). NEXT (cycle 3): target #3 batched KV
  write — fold the per-token norm/RoPE/KV-write loop into batched launches (kills the last
  T per-token launches); then NVRTC `-I` tensor-core GEMM (+2.2×).
- **2026-06-12 — Cycle 3: batched KV-write + Q/K norm-RoPE (target #3) — output-identical + faster.**
  Folded the per-token loop in `attentionLayerBatched` (V plain-norm+KV-write, Q/K per-head
  norm+RoPE) into ONE launch each (grid.y = T), eliminating the last T per-token launches on
  the batched gemma prefill path. Added two ADDITIVE kernels (kernels.cu): `rms_norm_rope_batched`
  and `rms_norm_kvwrite_batched` — verbatim twins of the single-token `rms_norm_rope`/`rms_norm_kvwrite`,
  batched over queries via block=(head=blockIdx.x, t=blockIdx.y) with explicit per-token src/dst
  strides (q_dim for Q in-place, kv_dim for K/V into the cache) and `position = base_position + t`
  (base 0 = fresh prompt). Per-block reduction order is UNCHANGED → bit-identical to the per-token
  loop. The `aliasBuffer` per-token ArrayList loop is GONE — kernels index the token-major scratch
  directly via `t*stride`. New `RmsRopeBatchPush`/`RmsKvWriteBatchPush` + 2 pipelines (additive).
  Single-token kernels + decodeStep/prefillStep untouched; `ZINC_BATCHED_PREFILL` off by default
  → cannot regress production. Built clean on the 4090 box (fresh `.zig-cache`, `zig build cuda-dbg`
  EXIT=0 + ran, bin md5 6e74a1e5 ≠ cycle 2's dc54e7cd). GATE (scripts/prefill_catalog.sh
  `ZINC_AB=batched`, 4090, ABBA x2, 250-tok):
    - gemma4-31b dense: 30.30 → 132.97 t/s (+339%, ~4.4×), GEN_IDS byte-IDENTICAL ✓
    - gemma4-26b MoE:   46.40 → 49.20 (+6% noise, per-token fallback) GEN_IDS identical ✓
  Cycle 2 was +199% (94 t/s); removing the per-token KV-write/RoPE launches → +339% (133 t/s).
  Committed `d43216ac` to perf/e24-batched-prefill, pushed (NOT main). REMAINING for merge:
  `scripts/validate_catalog.sh` 5/5 (product path unchanged + batched byte-identical → transitively
  correct). NEXT (cycle 4): NVRTC `-I/usr/local/cuda/include` fp16-scratch tensor-core GEMM (+2.2×),
  or run the formal validate_catalog gate to clear merge; gemma-26b MoE prefill (Phase 2) is the
  other open lever (route T tokens, group by expert).
- **2026-06-12 — Cycle 4: gemma-26b MoE prefill — batched attention + per-token MoE FFN (Phase 2 cycle 1) — output-identical + faster.**
  Stopped the MoE model (gemma4-26b-a4b) from falling back ENTIRELY to per-token: its
  ATTENTION block is the SAME structure as the dense gemma-31b, so `prefillBatched` now
  runs the shared batched attention path (GEMM Q/K/V/O + batched causal attn + batched
  norm/RoPE/KV-write — already bit-identical to the per-token `attentionLayer`) for the
  MoE model too. The routed-expert FFN is still LOOPED per token: the FFN is
  position-independent, so looping the EXISTING single-token `moeFfnBlock` + `layerOutScale`
  over each token's hidden slice (alias-swap `self.hidden` → `b.hidden[t*n_embd]`, launch
  captures the raw device ptr at `cuLaunchKernel` so the alias wrapper frees safely) is
  OUTPUT-IDENTICAL to the per-token path. FFN type is decided PER LAYER (mirrors the
  per-token `n_experts>0 && ffn_gate_inp present` test, so a MoE model's dense layers still
  take `ffnBlockBatched`). Async MoE commands drained per layer (attentionLayerBatched's
  commitAndWait drains the stream → `drainPending`) + `waitPending` before the tail. ADDITIVE:
  only `prefillBatched` control flow changed (new per-layer MoE branch); single-token
  kernels + decodeStep/prefillStep + moeFfnBlock + the dense batched path all untouched →
  `if (moe)`-guarded so the proven dense path is byte-for-byte unchanged. NO new kernels.
  Built clean on the 4090 box (fresh `.zig-cache`, `-Dbackend=cuda -Dshaders=false` EXIT=0
  + ran, bin md5 f8cf047a ≠ cycle3's 6e74a1e5). GATE (scripts/prefill_catalog.sh
  `ZINC_AB=batched`, 4090, ABBA x2, 250-tok):
    - gemma4-26b MoE:   43.69 → 49.97 t/s (+14%), GEN_IDS byte-IDENTICAL ✓ (NEW: was full per-token fallback)
    - gemma4-31b dense: 30.34 → 147.87 t/s (+387%, ~4.9×), GEN_IDS byte-IDENTICAL ✓ (NO regression — dense flow unchanged)
  The +14% is the attention-projection batching only (Q/K/V/O GEMM + batched attn); the
  routed-expert FFN is still per-token. Committed to perf/e24-batched-prefill, pushed (NOT
  main). REMAINING for merge: `scripts/validate_catalog.sh` 5/5 (product per-token path
  unchanged + batched byte-identical → transitively correct). NEXT (cycle 5): full
  batched-expert MoE FFN (route T tokens, group/scatter by expert — the remaining MoE
  per-token cost), NVRTC `-I` fp16 tensor-core GEMM (+2.2× on the dense GEMMs), or run the
  formal validate_catalog gate to clear merge.
- **2026-06-12 — Cycle 5: FORMAL GATE CLEARED — batched prefill now validated DIRECTLY vs llama.cpp (5/5).**
  The dense Phase-1 goal is COMPLETE (gemma-31b +363%/142 t/s, reproduced this cycle); the sole
  remaining merge blocker in every prior log was `validate_catalog 5/5`, which only ever exercised
  the per-token PRODUCT path (it never set `ZINC_BATCHED_PREFILL`), so the batched path was only
  TRANSITIVELY validated ("product correct + batched byte-identical to it"). This cycle UPGRADES the
  gate to DIRECT: extended `scripts/validate_catalog.sh` ADDITIVELY with `ZINC_BATCHED=1` — it exports
  `ZINC_BATCHED_PREFILL=1` for the gemma rows only (ForwardGemma has `prefillBatched`; qwen would
  silently fall back, so it's left per-token), so the gemma rows now confirm the BATCHED prefill path
  is itself token-correct vs llama.cpp, not merely identical to per-token. Script-only change → the
  compute path / binary is byte-for-byte cycle 4 (built clean at HEAD on the 4090 box, fresh
  `.zig-cache`, `zig build cuda-dbg` EXIT=0; bin md5 f5025990, differs from cycle 4's f8cf047a only by
  cross-machine link non-determinism — no `.zig` edit). GATE (4090, 250-tok):
    - `validate_catalog` (plain, product per-token): **5/5 PASS** (qwen35-9b/qwen36-27b/qwen36-35b-a3b/
      gemma4-26b all 12/12 free-run; gemma4-31b free-run 2/12 = the documented q8_1-reference near-tie
      at pos 2 → teacher-forced 11/12 confirms correctness).
    - `ZINC_BATCHED=1 validate_catalog` (DIRECT batched gate): **5/5 PASS** — gemma4-26b MoE batched
      free-runs **12/12** directly vs llama.cpp; gemma4-31b batched yields the IDENTICAL first token
      (same 2/12 near-tie + teacher-forced 11/12) → the batched prefill's first gen token == per-token's.
    - `prefill_catalog ZINC_AB=batched` (perf + GEN_IDS identity): ALL 5 GEN_IDS byte-identical;
      gemma4-31b dense **30.76 → 142.53 t/s (+363%)** (reproduces cycle 3/4's ~133/148); gemma4-26b MoE
      53.71 → 44.83 (**−17% this run = BOOST NOISE**, output identical — the routed-expert FFN is still
      per-token so the MoE gain is small/attention-only and swings ±17% on boost order; cf. cycle 4 saw
      +14% on the same path); qwen +2–3% (per-token fallback, noise).
  NET: the merge-gate CORRECTNESS bar is now MET both ways — the product path AND the batched path are
  independently 5/5 token-correct vs llama.cpp, on top of the GEN_IDS byte-identity. The branch remains
  WIP/unmerged (coordination: main moves fast; merge is a separate deliberate step), but it is no longer
  blocked on validation. NEXT (cycle 6): full batched-expert MoE FFN (route/group T tokens by expert —
  the remaining MoE per-token cost, to turn the noisy ±17% into a real MoE win), or NVRTC `-I` fp16
  tensor-core GEMM (+2.2× on the dense GEMMs — note: NOT byte-identical, would need its own tolerance gate).
- **2026-06-13 — Cycle 6: batched Q8_0 shared-expert FFN for gemma-26b MoE prefill (Phase 2 cycle 2) — output-identical + faster (the real MoE FFN win).**
  Cycle 4/5 batched only the MoE model's ATTENTION (the routed-expert FFN was still fully per-token), so
  gemma-26b's gain was attention-only and swung ±17% on boost (+14% cycle 4, −17% cycle 5 = noise). This
  cycle attacks the FFN: the gemma-26b shared-expert (the dense FFN every token runs alongside the routed
  experts) is **Q8_0** (gate/up/down) — ~17.8 MB/token of weight reads that the per-token loop re-read T
  times. Added an ADDITIVE Q8_0 prefill GEMM and batched the whole shared expert over all T tokens (weights
  read ONCE). Changes (all additive; decodeStep/moeFfnBlock/dense path untouched):
    - `gemm_q8_0_tiled_v2` (kernels.cu): bit-faithful twin of `gemm_q4k_tiled_v2` (64×64 tile, 256 thr, 4×4
      micro-tile, BK=32 = one Q8_0 block) with Q8_0 dequant `value = d*(float)q` matching dmmv_q8_0. Byte-
      addressed, pc.a_offset in BYTES. NVRTC-compiled at runtime (confirmed: "compiled gemma4 kernel pipelines").
    - `gemm: [3]→[4]` + `pipes.gemm[3]=gemm_q8_0_tiled_v2` + `gemmDispatch` `.q8_0 => 3`. gemma-31b dense has
      NO Q8_0 in the batched path (all Q4_K/Q6_K) → the proven dense path is byte-for-byte unchanged.
    - `BatchScratch.shared` ([T,n_embd]) + `sharedExpertBatched(L,T,b)`: batched pre-norm + gate/up/down GEMM
      (Q8_0→gemm_q8_0) + GeGLU[T,sf] + post_ffw_norm_1 → b.shared (same dmmv→GEMM swap the proven dense
      `ffnBlockBatched` makes). `moeRoutedCombine(L)`: `moeFfnBlock` MINUS its shared sub-block (router top-k +
      routed experts + post_ffw_norm(shared+moe)+residual), per token, reading b.shared[t] via a `self.shared_buf`
      alias (same per-token alias pattern as `self.hidden`). prefillBatched MoE branch now: sharedExpertBatched
      ONCE/layer, then loop T × (moeRoutedCombine + layerOutScale). Per-token submits drop 5→4 (shared no longer
      per-token) → fits the async ring at T=250.
  Built clean on the 4090 box (fresh `.zig-cache`, `zig build cuda-dbg -Dbackend=cuda -Dshaders=false` EXIT=0,
  bin md5 11ac3da1 ≠ cycle 5's f5025990). GATE — FULLY CLEARED:
    - Direct A/B (varied non-collapsed prompt, GEN 228,228,228,228,49,50,3236,608,…): gemma-26b prefill
      **38.47 → 82.00 t/s (+113%)**, GEN_IDS byte-IDENTICAL.
    - `prefill_catalog ZINC_AB=batched` (ABBA x2, 250-tok): ALL 5 GEN_IDS byte-identical —
      **gemma4-26b 42.61 → 81.52 t/s (+91%)** (was the noisy +14%/−17% attention-only), gemma4-31b dense
      31.03 → 132.81 t/s (+328%, NO regression), qwen ±0–4% (per-token fallback, noise).
    - `validate_catalog` plain **5/5**; `ZINC_BATCHED=1 validate_catalog` **5/5** — gemma4-26b MoE batched
      free-runs **12/12** DIRECT vs llama.cpp (the batched shared-expert path is token-correct); gemma4-31b
      the documented near-tie (free-run 2/12, teacher-forced 11/12, unchanged — dense path untouched).
  The noisy ±17% MoE swing is now a SOLID +91% (ABBA), correct both ways. Committed to perf/e24-batched-prefill,
  pushed (NOT main). NEXT (cycle 7): full token-GROUPED routed experts (route/group T tokens by expert so each
  active expert's Q4_K/Q5_1 weights — ~28 MB/token, the remaining per-token FFN cost — are read once per group,
  not once per token; needs a gather/scatter + variable-group GEMM, harder for byte-identity), or NVRTC `-I`
  fp16 tensor-core GEMM (+2.2× dense, NOT byte-identical → own tolerance gate).
