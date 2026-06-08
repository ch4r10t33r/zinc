# Effort 22 — CUDA gemma4 forward (catalog completeness on the 4090)

> **Status:** dense COHERENT (Cycle 1) · **1/5 still blocked (gemma4-26b MoE)** · goal: `gemma4-31b` (dense) ✅ + `gemma4-26b-a4b` (MoE) generate **coherent text** via ZINC CUDA on the 4090, completing the 5/5 catalog. Catalog now **4/5** on the 4090.

Date: 2026-06-07. Pairs with the qwen35/qwen36 CUDA work (Efforts 20/21, `forward_cuda.zig`).

## Why this effort

Catalog coherence on the 4090 (ZINC CUDA) is **3/5**: `qwen35-9b`, `qwen36-27b`, `qwen36-35b-a3b` (MoE) all generate coherent text — `forward_cuda.zig` handles the qwen35/qwen36 hybrid-SSM family (dense + qwen2_moe). The **2 gemma4 models are unsupported** (`forward_cuda` is qwen-only; loading gemma4 now returns a clean `UnsupportedArchitecture`, commit `e430b9a`, instead of a div-by-zero panic). On the 4090, **CUDA is the only backend** (WSL2 exposes no NVIDIA Vulkan), so gemma-on-4090 requires a gemma4 CUDA forward. (Gemma already works on ZINC's `zinc_rt`/Vulkan backend on RDNA4.)

## Target models (on box at `~/workspace/models/`, all fit the 4090's 24 GB)

- **`gemma-4-31B-it-Q4_K_M.gguf`** (18 GB) — gemma4 **dense**. Confirmed config: 60 layers, n_embd 5376, n_ff 21504, n_head 32, **per-layer n_kv** (`[16×5, 4, …]` — SWA layers 16 KV, full layers 4 KV), **head_dim 512 (full) / 256 (SWA)**, rms_eps 1e-6, **rope freq_base 1e6 (full) / 1e4 (SWA)**, rope_dim 512, **sliding_window 1024**, **swa_pattern `[T,T,T,T,T,F,…]` (period 6)**, **final_logit_softcapping 30.0**, `n_embd_per_layer_input = 0` (no altup), vocab 262144.
- **`gemma-4-26B-A4B-it-UD-Q4_K_M.gguf`** (16 GB) — gemma4 **MoE** (`a4b`).

## gemma4 architecture (from llama.cpp `src/models/gemma4.cpp`) — what's NEW vs qwen35

A standard transformer (no SSM) but with several gemma-specific pieces:

1. **Scaled embeddings** — `inpL = tok_embd[token] * sqrt(n_embd)` (token input only).
2. **Per-layer input embeddings (altup)** — gated behind `n_embd_per_layer > 0`. **`gemma4-31b` has `embedding_length_per_layer_input = 0` → NONE** (the dense 31b is a plain transformer + SWA, no altup; that's a big simplification). Only handle this subsystem if the 26b/MoE config sets it >0 — check at that point.
3. **Sliding-window attention (SWA)** — per-layer `is_swa` pattern (period ~6: 5 SWA + 1 full). SWA layers use `n_swa` window + separate head dims (`n_embd_head_k_swa`); full layers use `rope_freqs` (proportional rope). `naive_attention` is full-context only → needs a windowed-mask variant (or pass the window to the existing kernel).
4. **Per-head q/k RMS norm** — `attn_q_norm`/`attn_k_norm` `[n_embd_head]` (like qwen3; reuse the per-head rms_norm path).
5. **4 norms/layer** — `attn_norm` (pre-attn) + `attn_post_norm` (post-attn) + `ffn_norm` (pre-ffn) + `ffn_post_norm` (post-ffn). The MoE layers add `ffn_pre_norm_2`/`ffn_post_norm_1`/`ffn_post_norm_2`.
6. **gemma RMSNorm = (1 + weight)·x̂** — the norm weight is offset by 1. Either a gemma-norm kernel variant or add 1.0 to the norm weights at load.
7. **GeGLU activation** — `gelu(gate) * up` (not SiLU). New `geglu` kernel (or a `gelu` + reuse the elementwise mul).
8. **Final logit soft-cap** (`f_final_logit_softcapping`, gemma2-style) + a **logits bias** (suppress-tokens → -inf). Confirm gemma4-31b uses these.
9. **MoE (26b)** — fused `ffn_gate_up_exps` `[n_embd, n_ff_exp*2, n_expert]` + `ffn_down_exps` + a **shared expert** (a normal FFN) + per-expert scale. Reuse the qwen36 MoE kernels (`softmax_topk`, `moe_weighted_acc`) with the fused gate_up layout.

## Plan (incremental, correctness-first — mirror the qwen35 bring-up)

1. **Loader config** — extend `loader_cuda.zig` + `ModelConfig` to extract the gemma4 keys (`*.embedding_length_per_layer`, `*.attention.sliding_window`, the per-layer `is_swa` array, `*.attention.{key,value}_length{,_swa}`, `*.final_logit_softcapping`, head dims). Don't break the qwen path.
2. **New kernels** (`kernels.cu`, additive): `gelu`/`geglu`; gemma-norm (or pre-+1 the weights); windowed/causal multi-context attention mask for SWA. Validate each in `kernels_test`.
3. **gemma4 dense forward** — a `forwardGemma` path (branch on `architecture == .gemma`), token-by-token decode: scaled embed → per-layer-embed project → for each layer {attn_norm → attn(SWA or full + rope) → attn_post_norm → +residual → ffn_norm → GeGLU → ffn_post_norm → +residual → per-layer-embed gate/proj/post-norm} → final norm → (softcap) → lm_head → argmax.
4. **Validate** — `dbg_cuda.zig` per-layer residual dump vs a llama.cpp eval-callback `l_out-N` reference (the exact method that found the qwen35 gate bug). Iterate to token-for-token match, then run `validate_cuda_decode.sh` (parameterized for gemma).
5. **MoE (26b)** — wire the gemma MoE FFN (fused gate_up_exps + shared expert) into the gemma forward; validate.

## Validation contract

- Per-layer residual diff vs llama.cpp (`l_out-N`) to pinpoint the first divergent layer (NOT just argmax — the qwen35 gate bug showed argmax can match at pos 0 while a mid-layer is wrong).
- Token-for-token greedy vs llama.cpp on a fixed prompt before declaring coherent; then the full `validate_cuda_decode.sh` suite.
- 4090-pinned (`GPU-e59a6fce-…`); the 26b/MoE is 16 GB (fits 24 GB).

## Cycle log

- **Cycle 1 (2026-06-07) — gemma4 DENSE coherent on the 4090. ✅**
  New `src/compute/forward_cuda_gemma.zig` (`ForwardGemma`), additive kernels in
  `kernels.cu` (`geglu`, `gemma_attention` windowed + scale, `rms_norm_noweight`,
  `scalar_mul`), and a `main.zig` branch (`runCuda` → shared `runCudaDecode`
  helper; qwen path unchanged). Result on `gemma-4-31B-it-Q4_K_M`, prompt "The
  capital of France is" → **" Paris.\n\nThe capital of France is Paris.\n\nThe"** —
  coherent, contains "Paris". Qwen35-9b re-checked through the shared decode path:
  still coherent (no regression).
  Key arch findings nailed this cycle (all confirmed from the GGUF + `gemma4.cpp`):
  - gemma RMSNorm `(1+w)` offset is **baked into the GGUF weights** (attn_q_norm
    ≈ 1.0234) → reuse the standard `rms_norm` kernel; V uses no-weight normalize.
  - **Per-layer geometry** read from GGUF arrays: SWA layers (pattern `[T×5,F]`,
    period 6) = head_dim 256, 16 KV heads, rope base 1e4, window 1024; full layers
    (i%6==5) = head_dim 512, 4 KV heads, rope base 1e6 + `rope_freqs` proportional
    factors (folded into a host-precomputed inv_freq table).
  - **Attention scale = 1.0** (gemma4 `f_attention_scale`, no 1/sqrt(d)).
  - **Alternative attention**: full layers omit `attn_v`; V = the raw K projection
    (pre-norm, pre-rope), then plain per-head rms-normalize, never roped.
  - **`layer_output_scale`** (per-layer scalar, e.g. 0.089/0.879/0.036) multiplies
    the whole residual stream at the end of each layer — applied via `scalar_mul`.
  - Embeddings scaled by `sqrt(n_embd)`; LM head tied to `token_embd.weight`
    (no `output.weight`); final-logit soft-cap skipped (monotonic → argmax-safe).
  - **NEXT:** validate per-layer-diff vs llama.cpp `l_out-N` (argmax is right but a
    mid-layer could still drift); then wire the **gemma4 MoE (26b)** FFN
    (fused `ffn_gate_up_exps` + shared expert + 3 extra norms) into `ForwardGemma`.

## Refs

llama.cpp `src/models/gemma4.cpp` (build graph), `src/llama-model.cpp` gemma4 hparams. The qwen35 bring-up (Effort 21 Cycle 5) for the per-layer-diff method; `scripts/validate_cuda_decode.sh` for the gate.
