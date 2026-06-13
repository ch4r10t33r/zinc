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
- **2026-06-13 — Cycle 7: batched router for gemma-26b MoE prefill (Phase 2 cycle 3) — output-identical + faster.**
  Cycle 6 batched the Q8_0 shared-expert FFN; the routed-expert sub-block was still fully per-token, and EACH
  token re-ran the router: rms_norm_noweight + per-channel scale + an **F32 `ffn_gate_inp` matvec (~2 MB/token,
  re-read T times = ~500 MB at T=250)** + top-k softmax + its own command submit. This cycle BATCHES the router
  over all T tokens — the F32 router weight is read ONCE and one fewer command submits per token (4→3 on the
  prerouted path). Changes (all additive; decodeStep/moeFfnBlock/dense path untouched):
    - 3 ADDITIVE kernels (kernels.cu): `gemm_f32_tiled_v2` (bit-faithful twin of gemm_q4k_tiled_v2 — 64×64 tile,
      256 thr, 4×4 micro-tile, BK=32 — but NO dequant: stages the W tile straight from a row-major f32 weight;
      the batched twin of looping dmmv_f32, token-correct not bit-exact, same class as the proven quant GEMMs),
      `softmax_topk_batched` (verbatim softmax_topk over grid {T} — per-block routing math byte-for-byte the
      single-token kernel's), `mul_vec_scaled_batched` (a[t*row+i]=a[t*row+i]·b[i]·scale, broadcasting the
      per-channel ffn_gate_inp.scale across tokens).
    - `gemm_f32` pipeline + `softmax_topk_batched`/`mul_vec_scaled_batched` pipes; `BatchScratch.router_in`
      [T,n_embd] / `router_logits` [T,n_experts] / `router_table` [T,2·n_used] (sized 1 on the dense model).
    - `routerBatched(L,T,b)`: norm→scale→gemm_f32 logits→softmax_topk_batched → b.router_table, async on the
      shared stream (the per-token expert launches read the finished table by stream order — no host sync).
      Mirrors moeRoutedCombine's router sub-block op-for-op (same residual input: all tokens' PRE-FFN hidden,
      identical to the per-token path where each token routes before its own FFN residual add).
    - `moeRoutedCombine(L, prerouted)`: when prerouted, SKIP the router sub-block (caller aliases
      self.router_out_buf to b.router_table's row t); experts + combine unchanged. prefillBatched MoE branch
      calls routerBatched ONCE/layer (only on the Q4_K-gate_up + Q5_1-down async expert path), then loops T ×
      (alias router_table[t] → moeRoutedCombine(prerouted=true) + layerOutScale).
  Built clean on the 4090 box (fresh source rsync, warm `.zig-cache`, `zig build cuda-dbg -Dbackend=cuda
  -Dshaders=false` EXIT=0 + ran, bin md5 0a080dd5 ≠ cycle 6's 11ac3da1). GATE — FULLY CLEARED:
    - `prefill_catalog ZINC_AB=batched` (ABBA x2, 250-tok, 4090): GEN_IDS byte-identical —
      **gemma4-26b 44.70 → 103.92 t/s (+132%)** (up from cycle 6's 81.5 t/s — the batched router added ~+27%
      absolute on top of the batched shared expert), gemma4-31b dense 30.75 → 136.55 t/s (+344%, NO regression
      — dense flow untouched).
    - `ZINC_BATCHED=1 validate_catalog` (DIRECT batched gate, 4090): **5/5 PASS** — gemma4-26b MoE batched
      free-runs **12/12** DIRECT vs llama.cpp (the batched-router prefill path is token-correct); gemma4-31b the
      documented near-tie (free-run 2/12, teacher-forced 11/12, unchanged); qwen 12/12 (per-token fallback intact
      → the product decode path is unregressed). Plain validate transitively covered (default path byte-unchanged).
  Committed to perf/e24-batched-prefill, pushed (NOT main). NEXT (cycle 8): the routed gate_up/down expert GEMMs
  are the last per-token FFN cost (~36 MB/token Q4_K/Q5_1) — token-GROUPED routing (build per-expert token lists
  from b.router_table, expert-major grouped GEMM so each expert's weight is read once per group not once per
  token; needs gather/scatter, harder for byte-identity), or NVRTC `-I` fp16 tensor-core GEMM (+2.2× dense, own
  tolerance gate). b.router_table is now the device-side foundation a grouped-expert pass consumes.
- **2026-06-13 — Cycle 8: token-batched routed-expert matvecs for gemma-26b MoE prefill (Phase 2 cycle 4) — output-identical + faster (the last big per-token FFN cost).**
  Cycle 7 batched the router; the routed gate/up/down expert matvecs themselves were STILL looped per
  token (~36 MB/token: Q4_K gate_up + Q5_1 down). This cycle batches them over all T prompt tokens in
  single launches (grid.y = T) instead of T separate single-token launches. NOT token-grouped (no
  gather/scatter) — each (token t, expert-slot e, row) block reads token t's own router row, input slice,
  and output slice, so the per-block dequant + reduction is byte-for-byte the single-token kernel; only
  the launch dimension is batched. Changes (all additive; decodeStep/moeFfnBlock/dense path untouched):
    - 2 ADDITIVE kernels (kernels.cu): `dmmv_q4k_experts_batched` / `dmmv_q5_1_experts_batched` — bit-faithful
      twins of `dmmv_q4k_experts`/`dmmv_q5_1_experts` (same dequant + `zinc_block_reduce_sum` order) with
      per-token src/dst strides (`x_tok_stride`/`y_tok_stride`) + `routing_stride` so block=(g=blockIdx.x →
      slot·M+row, t=blockIdx.y) reads ids from `b.router_table[t*2·n_used + e]`. New `ExpertsBatchPush`
      (per-token strides appended to `ExpertsPush`) + 2 pipelines.
    - `BatchScratch.{moe_norm_e, gate_e, up_e, geglu_e, down_e}` ([T, …] slot-major, size 1 on dense).
    - `moeRoutedExpertsBatched(L,T,b)`: batched pre_ffw_norm_2 → b.moe_norm_e, gate (base 0) + up (base
      gu_half) Q4_K expert matvecs → b.gate_e/up_e, GeGLU over [T, n_used·ef] → b.geglu_e, Q5_1 down →
      b.down_e [T, n_used·n_embd] slot-major. Async on the shared stream; push params mirror the per-token
      `dmmv_*_experts` dispatches exactly. The unscaled down output (down scale folded at accumulate via
      moe_weighted_acc_scaled, unchanged) → bit-identical.
    - `moeRoutedCombine(L, prerouted, preexperts)`: when preexperts, SKIP pre_ffw_norm_2 + the expert
      matvecs + GeGLU (done batched) and read this token's b.down_e slice via the `self.down_buf` alias →
      only the zero+weighted-accumulate + post_ffw_norm_2 + scale_accumulate + post_ffw_norm + residual
      combine remain per token. `pre` (Q4_K gate_up + Q5_1 down) ⟹ batched ⟹ scaled accumulate, consistent.
    prefillBatched MoE branch now: sharedExpertBatched + routerBatched + moeRoutedExpertsBatched ONCE/layer,
    then loop T × (moeRoutedCombine(prerouted, preexperts) + layerOutScale) — the per-token loop is now just
    the lightweight accumulate/combine tail. Host-readback fallback keeps its full per-token path.
  Built clean on the 4090 box (fresh `.zig-cache`, `zig build cuda-dbg -Dbackend=cuda -Dshaders=false`
  EXIT=0, bin md5 d14f13bc ≠ cycle 7's 0a080dd5; NVRTC compiled the 2 new kernels at runtime). GATE — FULLY CLEARED:
    - `prefill_catalog ZINC_AB=batched` (ABBA x2, 250-tok, 4090): GEN_IDS byte-identical —
      **gemma4-26b MoE 39.72 → 135.16 t/s (+240%)** (UP from cycle 7's 103.92 — batching the routed-expert
      matvecs is the last big FFN lever), gemma4-31b dense 30.47 → 137.96 t/s (+353%, NO regression —
      dense flow untouched).
    - `ZINC_BATCHED=1 validate_catalog` (DIRECT batched gate, 4090): **5/5 PASS** — gemma4-26b MoE batched
      free-runs **12/12** DIRECT vs llama.cpp (the batched routed-expert prefill path is token-correct);
      gemma4-31b the documented near-tie (free-run 2/12, teacher-forced 11/12, unchanged); qwen 12/12
      (per-token fallback intact → product decode path unregressed).
  Committed `3b1afc78` to perf/e24-batched-prefill, pushed (NOT main). The gemma-26b MoE prefill FFN is now
  FULLY batched (shared expert + router + routed experts); the only remaining per-token work is the cheap
  accumulate/combine tail. NEXT (cycle 9): either token-GROUPED routed experts (gather tokens by expert so
  each expert's weight is read once per group not once per token — a memory-traffic win beyond launch
  batching, but needs gather/scatter and is harder for byte-identity), batch the per-token accumulate/combine
  tail too (kills the last T launches/layer), or NVRTC `-I` fp16 tensor-core GEMM (+2.2× dense, own tolerance gate).
- **2026-06-13 — Cycle 9: batched accumulate/combine tail for gemma-26b MoE prefill (Phase 2 cycle 5) — output-identical + faster (the LAST per-token FFN launches gone).**
  After cycle 8 the gemma-26b MoE FFN was batched through the routed-expert matvecs, but the FINAL stage —
  per token: zero accumulator → weighted-combine (down scale folded) → post_ffw_norm_2 → shared+=moe →
  post_ffw_norm → hidden+=cur → layer_output_scale — was still a T-iteration loop (T × ~7 single-token
  launches/layer + T alias-buffer create/free pairs). This cycle batches that tail, so on the GPU-side async
  expert path (Q4_K gate_up + Q5_1 down) the prefill MoE FFN has NO per-token loop at all. ADDITIVE
  (decodeStep/moeFfnBlock/moeRoutedCombine/dense path untouched):
    - 1 ADDITIVE kernel (kernels.cu): `moe_weighted_acc_scaled_batched` — bit-faithful twin of
      `moe_weighted_acc_scaled` (same j-loop FMA order, same GPU-side `escale[id]` down-scale fold) with
      grid.y = T + per-token strides (`a_tok_stride`/`b_tok_stride`/`routing_stride`) so block=(i=blockIdx.x,
      t=blockIdx.y) reads token t's own accumulator/down/routing slices → per-(t,i) math byte-for-byte the
      single-token kernel. New `MoeAccBatchPush` + 1 pipeline. The other 6 tail ops needed NO new kernel:
      `zero_vec`/`scale_accumulate`/`scalar_mul` are element-wise → one launch over the whole [T,n_embd] tile
      (N=T·n_embd, contiguous token-major → each element identical to the per-token launch); `rms_norm`
      (post_ffw_norm_2, post_ffw_norm) already indexes token=blockIdx.x → grid.x=T reproduces the per-token
      reduction block-for-block. The standalone per-token `layerOutScale` is folded in as the final batched
      `scalar_mul` (self-skips when the layer has no output scale).
    - `BatchScratch.moe_out_e` [T,n_embd] (the batched post_ffw_norm_2 accumulator; size T·n_embd on dense too,
      cheap) + `moeRoutedCombineBatched(L,T,b)` (the 7 batched launches above, async on the shared stream —
      stream order guarantees it reads the experts/router buffers after they're written). prefillBatched MoE
      branch: on the `pre` (batched) path it now calls sharedExpertBatched + routerBatched +
      moeRoutedExpertsBatched + **moeRoutedCombineBatched** ONCE/layer — the per-token loop is GONE. The non-`pre`
      host-readback fallback keeps its per-token `moeRoutedCombine(false,false)` + `layerOutScale` loop (aliasing
      b.hidden/b.shared per token), byte-for-byte unchanged.
  Built clean on the 4090 box (fresh `.zig-cache`, `zig build cuda-dbg -Dbackend=cuda -Dshaders=false` EXIT=0,
  bin md5 5a1b0597 ≠ cycle 8's d14f13bc; NVRTC compiled the new kernel at runtime). GATE — FULLY CLEARED:
    - `prefill_catalog ZINC_AB=batched` (ABBA x2, 250-tok, 4090): GEN_IDS byte-identical —
      **gemma4-26b MoE 46.72 → 160.49 t/s (+244%)** (UP from cycle 8's 135.16 — killing the per-token combine
      loop is the last FFN lever), gemma4-31b dense 30.11 → 130.21 (+332%, NO regression — dense flow untouched).
    - `ZINC_BATCHED=1 validate_catalog` (DIRECT batched gate, 4090): **5/5 PASS** — gemma4-26b MoE batched
      free-runs **12/12** DIRECT vs llama.cpp (the fully-batched MoE prefill path is token-correct); gemma4-31b
      the documented near-tie (free-run 2/12, teacher-forced 11/12, unchanged); qwen 12/12 (per-token fallback
      intact → product decode path unregressed).
  Committed to perf/e24-batched-prefill, pushed (NOT main). The gemma-26b MoE prefill FFN is now FULLY batched
  end-to-end (shared expert + router + routed experts + accumulate/combine + output scale) — zero per-token
  launches on the async expert path. NEXT (cycle 10): token-GROUPED routed experts (gather tokens by expert →
  each expert weight read once per group, a memory-traffic win beyond launch batching; needs gather/scatter,
  harder for byte-identity), batch the attention layer's remaining per-token bits if any, or NVRTC `-I` fp16
  tensor-core GEMM (+2.2× dense, own tolerance gate).
- **2026-06-13 — Cycle 10: async-pipeline the batched prefill layer loop (remove per-layer host syncs) — output-identical + faster.**
  After cycle 9 the gemma MoE/dense prefill FFN+attention were fully batched, but the layer loop still
  BLOCKED THE CPU twice (dense) / once (MoE) per layer: `attentionLayerBatched`, `ffnBlockBatched` and
  `sharedExpertBatched` each ended with `cmd.commitAndWait()` — a full `cuStreamSynchronize` round-trip. On
  WSL2 each is ~0.4 ms and, worse, the idle gaps starve the GPU boost clock (the exact pathology the merged
  DECODE async ring fixes). This cycle removes them: all three helpers now `self.submit(cmd)` (commitAsync +
  stash) on the SAME single shared CUstream the async MoE stages already use (confirmed `m->stream = c->stream`
  in cuda_shim.c → every launch is implicitly ordered, so cross-layer buffer reuse stays byte-identical — only
  CPU blocking is removed, not GPU execution order). The mid-loop `if (moe) drainPending()` (which relied on
  attention's commitAndWait draining the stream) is GONE; a single unconditional `waitPending()` before the
  tail now frees every layer's stashed command (ring depth = blocks/layer × n_layers ≈ 5×48 for gemma-26b,
  well under the 1024 ring; submit() self-falls-back to sync if it ever fills). The per-token MoE FALLBACK
  path (non-`pre`) is untouched — its inner `moeRoutedCombine` still commitAndWaits around its host id
  readback, which safely drains any stashed attention/shared command. ADDITIVE: only `prefillBatched` +
  its three batched helpers changed; decodeStep/prefillStep/per-token path/single-token kernels untouched;
  NO new kernels. Built clean on the 4090 box (fresh `.zig-cache`, `zig build cuda-dbg -Dbackend=cuda
  -Dshaders=false` EXIT=0 + ran, bin md5 2eca13ad ≠ cycle 9's 5a1b0597). GATE — FULLY CLEARED:
    - `prefill_catalog ZINC_AB=batched` (ABBA x2, 250-tok, 4090): GEN_IDS byte-identical —
      **gemma4-26b MoE 46.27 → 213.46 t/s (+361%)** (UP from cycle 9's 160.49 — removing the per-layer
      host syncs is a large MoE win: more layers × launches = more sync gaps removed + boost held),
      **gemma4-31b dense 30.02 → 140.56 t/s (+368%)** (UP from cycle 9's 130.21).
    - `ZINC_BATCHED=1 validate_catalog` (DIRECT batched gate, 4090): **5/5 PASS** — gemma4-26b MoE batched
      free-runs **12/12** DIRECT vs llama.cpp (the async-pipelined prefill path is token-correct); gemma4-31b
      the documented near-tie (free-run 2/12, teacher-forced 11/12, unchanged); qwen 12/12 (per-token
      fallback intact → product decode path unregressed).
  Committed to perf/e24-batched-prefill, pushed (NOT main). The batched prefill layer loop now runs with NO
  per-layer CPU↔GPU sync (one drain before the tail). NEXT (cycle 11): token-GROUPED routed experts (gather
  tokens by expert → each expert weight read once per group, a memory-traffic win beyond launch batching;
  needs gather/scatter, harder for byte-identity), or NVRTC `-I` fp16 tensor-core GEMM (+2.2× dense, own
  tolerance gate — NOT byte-identical).
- **2026-06-13 — Cycle 11: fp16 tensor-core (wmma) GEMM for the dense Q4_K projections/FFN — opt-in, token-correct, +6%.**
  Took the last NEXT-list lever: the NVRTC `-I/usr/local/cuda/include` path (landed in `a0d463af`) lets NVRTC
  resolve `<mma.h>`, so the dense batched Q4_K GEMMs (attn Q/K/V/O + FFN gate/up — the bulk of the dense FLOPs)
  can run on the fp16 tensor cores instead of the f32 register-tiled GEMM. Added an ADDITIVE kernel `gemm_q4k_tc`
  (kernels.cu): same `Y[T,M]=A[T,K]·W[M,K]^T` and SAME Q4_K dequant unpack as `gemm_q4k_tiled_v2`, but the inner
  product runs on wmma 16×16×16 fragments — the dequant'd W tile (m-major `Ws[r*BK+k]`) and f32 activations
  (k-major `As[k*BT+t]`) are cast to `__half` in shared mem, accumulated in fp32, stored col-major into a
  token-major `Cs` tile, then guard-copied to `Y[T,M]`. 64×64 tile / 256 thr = 8 warps, BK=32 (1 Q4_K sub-block
  = 2 wmma k-steps); each warp owns 2 accumulator fragments (M-blocks fm, fm+2 at T-block ft) → all 4×4=16
  (m,t)-blocks covered once; static shared = 24 KB. Wired OPT-IN behind `ZINC_BATCHED_TC` (read once per
  `prefillBatched` into `self.use_tc`): `gemmDispatch` routes Q4_K (idx 0) to `gemm_q4k_tc` only when `use_tc`,
  same `GemmPush`/grid/block — so with the toggle OFF the dispatch is byte-for-byte the proven f32 path.
  NOT byte-identical when ON (fp16 input rounding) → its gate is token-correctness vs llama.cpp, NOT the
  GEN_IDS byte-identity gate. Extended `scripts/validate_catalog.sh` + `scripts/prefill_catalog.sh` ADDITIVELY
  to pass `ZINC_BATCHED_TC=1` through (gemma rows / batched mode). ADDITIVE everywhere: 1 new kernel + 1 new
  pipeline + 1 bool + a 1-line dispatch branch; decodeStep/prefillStep/per-token path/the proven dense+MoE
  batched paths all unchanged when the toggle is off. Built clean on the 4090 box (fresh `.zig-cache`,
  `zig build cuda-dbg -Dbackend=cuda -Dshaders=false` EXIT=0 + ran, bin md5 602a6918 ≠ cycle 10's 2eca13ad;
  NVRTC compiled `gemm_q4k_tc` at runtime — "compiled gemma4 kernel pipelines"). GATE — its own tolerance gate
  CLEARED:
    - `ZINC_BATCHED=1 ZINC_BATCHED_TC=1 validate_catalog` (TC path direct vs llama.cpp, 4090): **5/5 PASS** —
      gemma4-26b MoE (TC on its dense-attn Q4_K GEMMs) free-runs **12/12**; gemma4-31b the SAME documented
      near-tie as the f32 path (free-run 2/12, teacher-forced 11/12) → fp16 TC rounding introduced **NO new
      divergence**; qwen 12/12 (per-token fallback). So the TC path is token-correct within tolerance.
    - Default batched path (toggle OFF) unchanged: 3-way direct A/B (gemma-31b, 220-tok) baseline /
      batched / batched+TC ALL GEN_IDS byte-identical (the off-path is provably the f32 path; on-path matched
      here too on this prompt).
    - Perf (ABBA B T T B, gemma-31b 250-tok, 4090): batched ~138.0 t/s (134.4, 141.6) → **batched+TC ~146.6
      t/s (146.7, 146.5) = +6.2%** over the already-batched dense path; baseline per-token was ~32 t/s
      (so batched+TC is ~4.6× the per-token prefill).
  The +6% is modest because the dense GEMM is MEMORY-bound here (Q4_K dequant + f32 activation reads dominate;
  the fp16 multiply was never the bottleneck) — tensor cores help the FLOP-heavy share only. Committed to
  perf/e24-batched-prefill, pushed (NOT main). Kept opt-in (off by default) → cannot regress production or the
  proven byte-identical batched gate. NEXT (cycle 12): make the TC path memory-efficient to unlock more than
  +6% — cache the dequant'd weights in fp16 once (kill per-GEMM dequant) and/or keep activations fp16 across
  the layer (kill the f32↔fp16 recast per GEMM); OR token-GROUPED routed experts (gather by expert =
  memory-traffic win, harder byte-identity).
- **2026-06-13 — Cycle 12: TC GEMM reads a PRE-CONVERTED fp16 activation (halve the dominant A read) — +18% over f32 batched / +8% over cycle-11 plain TC, byte-identical to plain TC.**
  Took cycle 11's own NEXT item: the +6% plain-TC gain was small because `gemm_q4k_tc` re-reads the f32
  activation A from global ONCE PER OUTPUT M-BLOCK (grid.x = M/64), so the f32 A traffic is ~(M/64)× the
  weight traffic and dominates the memory-bound dense GEMM. This cycle pre-converts A to fp16 ONCE per GEMM
  (`f32_to_f16`) and feeds a half-width A to the TC kernel — halving + de-duplicating the dominant read.
  ADDITIVE (decodeStep/per-token path/dense+MoE batched f32 paths/cycle-11 plain TC all untouched):
    - 2 ADDITIVE kernels (kernels.cu): `f32_to_f16` (element-wise `y[i]=__float2half(x[i])`) and
      `gemm_q4k_tc_f16a` — IDENTICAL to `gemm_q4k_tc` (same Q4_K dequant, same wmma 16×16×16 schedule, same
      Cs store / guarded copy) EXCEPT A arrives already fp16 (no per-load `__float2half`). The downcast uses
      the SAME `__float2half` the plain TC kernel applied in shared → the staged half bits are identical →
      `gemm_q4k_tc_f16a`'s output is BYTE-FOR-BYTE `gemm_q4k_tc`'s (confirmed empirically below), only the
      global A read is halved + read once not once-per-M-block.
    - `gemm_q4k_tc_f16a` + `f32_to_f16` pipelines + `BatchScratch.act_f16` ([T, max activation] halves) +
      `F32ToF16Push`. `gemmDispatch`: when `use_tc` and Q4_K (idx 0), pre-convert A → `act_f16` then dispatch
      `gemm_q4k_tc_f16a` (cycle-12 default). `x_offset == 0` on the batched GEMM path (contiguous [T,K]) so the
      whole [T·K] tile converts at offset 0; `act_f16` reuse across a layer's GEMMs is safe (stream-ordered).
    - A/B knob `ZINC_BATCHED_TC_PLAIN` (additive bool): forces cycle-11 `gemm_q4k_tc` (f32-A re-read) so the
      f16-A memory-traffic win is measurable in isolation. Off → the f16-A path. Both gated under `ZINC_BATCHED_TC`.
  Built clean on the 4090 box (fresh source rsync, `zig build cuda-dbg -Dbackend=cuda -Dshaders=false` EXIT=0,
  bin md5 33e6e1a1 ≠ cycle 11's 602a6918; NVRTC compiled the 2 new kernels at runtime). GATE — its own
  tolerance gate CLEARED (TC is fp16 → token-correctness, not GEN byte-identity vs f32):
    - `ZINC_BATCHED=1 ZINC_BATCHED_TC=1 validate_catalog` (f16-A TC path direct vs llama.cpp, 4090): **5/5 PASS**
      — qwen35-9b/qwen36-27b/qwen36-35b-a3b/gemma4-26b all **12/12**; gemma4-31b the SAME documented near-tie as
      the f32/plain-TC path (free-run 2/12, teacher-forced 11/12) → the f16-A pre-convert introduced **NO new
      divergence**.
    - f16-A == plain-TC byte-identity (the WIP's core claim): direct ABBA (gemma-31b, 250-tok) P=plain-TC
      F=f16a-TC → GEN_IDS IDENTICAL on every run → `gemm_q4k_tc_f16a` is byte-for-byte `gemm_q4k_tc`, as
      designed. Default path (`ZINC_BATCHED_TC` unset) is the unchanged f32 batched GEMM.
    - Perf (ABBA, gemma-31b 250-tok, 4090): plain-TC ~147.9 t/s → **f16-A TC ~159.6 t/s (+7.9%** over cycle-11
      plain TC); and batched-no-TC ~137.2 → **f16-A TC ~161.5 (+17.7%** over the f32 batched path — UP from
      cycle 11's plain-TC +6%). Per-token baseline ~32 t/s → f16-A TC is **~5×** the per-token prefill.
  The f16-A path roughly TRIPLES cycle 11's TC win (+6%→+18% over f32) by attacking the activation read
  traffic that made the GEMM memory-bound — exactly what the cycle-11 NEXT note predicted. Committed to
  perf/e24-batched-prefill, pushed (NOT main). Opt-in (off by default) → cannot regress production or the
  proven byte-identical batched gate. NEXT (cycle 13): cache the dequant'd Q4_K weights in fp16 ONCE per layer
  (kill the per-GEMM Q4_K dequant — the other half of the TC GEMM's traffic) and/or keep activations fp16
  across the whole layer (avoid the per-GEMM f32→fp16 recast); OR token-GROUPED routed experts (gather by
  expert = memory-traffic win beyond launch batching, harder byte-identity).
