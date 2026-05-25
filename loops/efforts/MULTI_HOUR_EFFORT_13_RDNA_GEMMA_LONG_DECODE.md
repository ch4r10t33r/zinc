# Effort 13 — RDNA Gemma 4 MoE long-decode parity with llama.cpp

## Status

Planned. This effort exists because the public RDNA benchmark matrix shows
two different Gemma stories:

| model | scenario | ZINC decode | llama.cpp decode | ZINC / llama |
|---|---:|---:|---:|---:|
| Gemma 4 26B A4B IT Q4K M | context-long | 125.60 tok/s | 105.76 tok/s | 119% |
| Gemma 4 26B A4B IT Q4K M | decode-extended | 53.69 tok/s | 102.72 tok/s | 52% |
| Gemma 4 31B IT Q4K M | context-long | 42.01 tok/s | 32.03 tok/s | 131% |
| Gemma 4 31B IT Q4K M | decode-extended | 24.96 tok/s | 29.17 tok/s | 86% |

The apparent context-long wins are not sustained-decode evidence. In that
scenario, ZINC stops after 2 generated tokens (`Paris<turn|>`) while the
llama.cpp baseline reports 8 generated tokens. The `decode-extended`
scenario forces 32 generated tokens and is the relevant signal.

Primary target: **Gemma 4 26B A4B IT Q4K M decode-extended on RDNA4**.
Current public result is 53.69 tok/s vs llama.cpp at 102.72 tok/s. A good
first milestone is 75 tok/s. The parity milestone is 95-105 tok/s.

## Latest llama.cpp reading

Upstream inspected at `ggml-org/llama.cpp@389ff61d7`
(`389ff61d77b5c71cec0cf92fe4e5d01ace80b797`, 2026-05-10).

The useful reference points are:

- `src/llama-graph.cpp::build_moe_ffn`
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp::ggml_vk_topk_moe`
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp::ggml_vk_mul_mat_id`
- `ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_base.glsl`
- `ggml/src/ggml-vulkan/vulkan-shaders/topk_moe.comp`

The "secret sauce" is structural, not a single arithmetic trick:

1. **Top-k MoE stays on GPU.** llama.cpp recognizes softmax/top-k/get_rows
   MoE patterns and replaces them with a fused `topk_moe` Vulkan dispatch.
   It supports early-softmax, normalized weights, sigmoid+bias, and late
   softmax variants.
2. **Selected experts are addressed by an ID tensor.** `ggml_mul_mat_id`
   dispatches the selected experts directly. The matvec-ID path is selected
   when `src2->ne[1] <= 8`, which is exactly decode-time top-k MoE.
3. **Per-expert bias/scale can fuse into the matvec-ID shader.** The Vulkan
   backend fuses `MUL_MAT_ID + ADD_ID`, `MUL_MAT_ID + MUL`, and
   `MUL_MAT_ID + ADD_ID + MUL` when the ID tensor matches.
4. **Q8_1 activation quant is conditional.** On AMD, llama.cpp only enables
   Q8_1 activation quant for large enough K and supported quant formats. It
   is not assumed to win every GEMV shape.
5. **Flash attention is adaptive.** llama.cpp chooses scalar vs coopmat,
   collapses GQA when shape-legal, and uses split-K to restore occupancy when
   a collapsed GQA dispatch would underfill the GPU.

For this Gemma 26B long-decode gap, item 1-3 are the main target. The
decode-extended prompt is only ~34 prompt tokens, so long-context attention
is not the dominant explanation for a 2x sustained decode gap.

## Current ZINC bottleneck hypothesis

ZINC's decode path already has a GPU-routed MoE implementation, but Gemma is
explicitly excluded:

```zig
const use_gpu_moe = config.architecture != .gemma and
    config.architecture != .gpt_oss and
    fused_gate_up == null and
    ...
```

The comment is directionally correct: Gemma has extra semantics that the
generic GPU MoE path did not handle when it was written:

- router input uses unit RMS norm plus optional `ffn_gate_inp.scale`
- expert input may use `pre_ffw_norm_2`
- Gemma 4 26B uses fused `ffn_gate_up_exps`
- routed down projection may need `ffn_down_exps.scale`
- shared and routed branches can require `post_ffw_norm_1/2` or
  `post_ffw_norm`

But the fallback cost is high. In the CPU-routed path, every MoE decode
layer does:

1. router DMMV on GPU
2. copy router logits to `router_staging`
3. `submitAndWait`
4. CPU `topKSoftmax`
5. reset/begin a new command buffer
6. serial loop over `n_experts_used`
7. per selected expert: gate DMMV, up DMMV, SwiGLU, down DMMV,
   scale-accumulate

That is exactly the pattern llama.cpp avoids with `topk_moe` +
`mul_mat_id`. The dense Gemma 31B result supports the diagnosis: without
MoE it is much closer to llama.cpp (86% on decode-extended), while Gemma
26B MoE is only 52%.

## Measurement contract

Use the same workload family as the published site matrix:

```bash
bun loops/optimize_perf.ts --effort 13 --model gemma412b --cycles 999
```

The `gemma412b` key is legacy naming in the harness; it points at
`/root/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf`.

Baseline must record:

- median decode tok/s on the 32-token decode-extended prompt
- `--profile` phase summary for one run
- `cpu_moe_fallbacks`
- `moe_router`, `moe_topk`, `moe_gate_up`, `moe_swiglu`, `moe_down`,
  `moe_weighted_acc`, `shared_*`, `final_lm_head`
- generated text for coherence

Do not trust `context-long` as the primary metric unless both engines
generate the same token count. It is useful as a regression guard only.

## Implementation plan

### Step 0: prove the fallback dominates

Run Gemma 26B decode-extended with `--profile -n 32 --chat` and capture the
phase budget. The minimum proof before changing code:

- `cpu_moe_fallbacks > 0`
- `moe_routed` or its sub-phases are a top decode bucket
- at least one `PROFILE_FALLBACK: cpu_moe` line appears

If the profile says final LM head or dense attention is larger than MoE,
stop and update this effort file before implementing.

### Step 1: GPU top-k validation for Gemma

Allow the existing GPU `dispatchSoftmaxTopk` to run on Gemma router logits
after the Gemma router scale. Do not consume its output yet.

Validation mode:

- still run the CPU fallback for the actual output
- copy GPU `router_output_buf` for the first token/layer
- compare IDs and weights against CPU `topKSoftmax`
- fail closed if IDs differ or weights exceed a small tolerance

This isolates the router/top-k part from expert math.

### Step 2: support fused `ffn_gate_up_exps`

Gemma 26B uses a single fused gate+up expert tensor. The existing GPU MoE
fast path assumes either separate gate/up tensors or two descriptor bindings
with offset zero. It also rejects `fused_gate_up != null`.

Two viable implementations:

1. Add offset-aware `pushDispatch5/6` helpers and bind the same buffer twice:
   binding 0 at expert slice base for gate, binding 1 at
   `up_base_offset` for up. Keep `expert_stride` as the full fused expert
   slice.
2. Add a Gemma-specific fused-gate-up MoE shader with an explicit
   `up_base_offset` push constant.

Prefer option 1 if descriptor offsets work with all push-descriptor and
fallback descriptor paths already used by ZINC. Prefer option 2 if adding
offset variants makes the call sites brittle.

Keep the CPU fallback path live behind a flag until validation is green.

### Step 3: consume GPU top-k for routed experts

Enable the GPU expert path for Gemma only after Step 1 and Step 2 validate.
Required semantics:

- router input: unit RMS norm, then `ffn_gate_inp.scale` when present
- expert input: `pre_ffw_norm_2` when present, otherwise `ffn_norm_buf`
- routed expert weights: selected-only normalized softmax, matching current
  CPU `topKSoftmax`
- down scale: fold `ffn_down_exps.scale[eid]` into the accumulation weight,
  or apply a tiny GPU scale-by-selected-expert kernel before accumulation
- post norms: preserve `post_ffw_norm_2`, `post_ffw_norm_1`, and
  `post_ffw_norm` ordering exactly

The first accepted version does not need to beat llama.cpp. It only needs to
remove the router readback and serial expert loop while preserving coherence.

### Step 4: remove per-expert serial dispatch where possible

After correctness, measure whether the remaining GPU path still dispatches
per expert or uses `Y = n_used` workgroup dimension. The desired shape is
one dispatch for all selected gate/up slots, one dispatch for activation, one
dispatch for down, and one weighted accumulation. That is the ZINC equivalent
of llama.cpp `mul_mat_vec_id`.

Do not port llama.cpp's full grouped-GEMM `count_experts` path for decode
first. With one token, top-k <= 8 matvec-ID is the right starting point.
Grouped GEMM matters for prefill or batched request execution.

### Step 5: only then try Q8_1 activation quant

Once Gemma GPU MoE is coherent and faster, test Q4_K x Q8_1 activation
quant on the Gemma expert shapes. The Qwen result is not transferable:
Q8_1 regressed some Qwen decode paths, but Gemma's fused gate/up and down
shapes differ. Treat it as a measured follow-up, not a default.

## Known traps

- Do not optimize the `context-long` Gemma "win" first. It is an early-stop
  artifact for ZINC and not sustained decode.
- Do not re-run generic DMMV micro-tunes before removing Gemma's CPU MoE
  fallback. A 2x MoE model gap is unlikely to come from a 1-2% matvec detail.
- Do not port llama.cpp grouped-GEMM `mul_mat_id` first. It is valuable, but
  decode top-k <= 8 should use the lighter matvec-ID shape.
- Do not silently skip Gemma post-norm or per-expert scale tensors. That will
  look fast and break coherence.
- Do not remove the CPU fallback until all catalog coherence prompts pass.

## Success criteria

Required before keeping a change:

- `zig build test`
- coherence sweep green across Qwen3.6 35B, Qwen3 8B, and Gemma 26B/31B
- Gemma 26B decode-extended output remains coherent for the benchmark prompt
- no crash on Gemma 31B dense path
- no regression >2% on Qwen3.6 35B decode

Performance milestones:

- Step 1 validation only: no speed requirement
- first GPU-routed Gemma MoE: >=65 tok/s on Gemma 26B decode-extended
- strong keep: >=75 tok/s
- parity target: 95-105 tok/s

## Secondary target: Gemma 31B dense

Gemma 31B dense decode-extended is 24.96 vs 29.17 tok/s, or 86% of
llama.cpp. Do not mix this with the 26B MoE effort until the MoE fallback is
removed. Once 26B is fixed, revisit 31B with:

- final LM-head profile
- Q4_K/Q8_1 activation quant on exact Gemma dense shapes
- dense FFN DMMV row-shape comparison against llama.cpp `mul_mat_vec_q4_k`
- flash-attention path only if context-heavy prompts show a real sustained
  gap with equal generated token counts
