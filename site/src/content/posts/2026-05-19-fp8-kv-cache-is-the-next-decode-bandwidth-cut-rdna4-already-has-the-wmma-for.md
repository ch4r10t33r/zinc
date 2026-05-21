---
title: "FP8 KV cache is the next decode bandwidth cut RDNA4 already has the wmma for"
date: "2026-05-19"
tags:
  - zinc
  - rdna4
  - amd
  - vulkan
  - kv-cache
  - fp8
  - quantization
  - qwen3
  - flash-attention
  - llama-cpp
  - local-llm-inference
keywords:
  - FP8 KV cache RDNA4
  - vLLM FP8 attention quantization Qwen3
  - Radeon AI PRO R9700 FP8 WMMA E4M3FN
  - llama.cpp Vulkan FP8 KV cache
  - gfx1201 aiter arch table FP8
  - Q8_0 vs FP8 KV cache symmetric fused FA
  - long-context decode bandwidth R9700
  - 54 percent ITL slope FP8 KV
  - Flash Attention FP8 fused kernel local
  - Qwen3 128k context KV cache budget
excerpt: "The Red Hat and AWS team published a state-of-FP8-KV-cache benchmark in April that put Qwen3's decode ITL slope at 54 percent of BF16 on an H100, with Flash Attention 3 doing the QK and PV matmuls in FP8 the whole way. RDNA4 ships the same E4M3FN WMMA instruction Triton already uses on MI350X, but no local Vulkan engine has a Qwen3 attention kernel that reads FP8 K and V. That is the next bandwidth cut on the R9700 decode loop, and the only thing standing between it and shipping is one cooperative-matrix shader."
---

The vLLM team at Red Hat and AWS [published a benchmark on April 22](https://vllm-project.github.io/2026/04/22/fp8-kvcache.html) that put Qwen3's decode inter-token latency at 54 percent of the BF16 baseline once the KV cache and the attention matmuls all ran in FP8. That is a halving of the slope, not the intercept, and it holds across needle-in-a-haystack at 128k and MRCR at 1M on Qwen3.5-27B. The Flash Attention 3 kernel they ship for Hopper does the QK and ScoreV matmuls in FP8 directly against an FP8 KV cache. Two prior accuracy issues are fixed in the same release. The break-even point for Llama 3.1 8B dropped from 25k tokens to 7k.

A single sentence in the vLLM blog frames the question for everyone running this loop on something that is not an H100: "for memory-bound decoding the per-token cost of the KV cache can be reduced to 54% of its BF16 counterpart in the best cases." On a Radeon AI PRO R9700 at the 117 tok/s Qwen3.6-35B-A3B decode rate zinc [published two weeks ago](/blog/2026-05-09-how-we-made-amd-qwen-inference-faster-than-llama-cpp-in-six-weeks-on-the-radeon-ai-pro-r9700), the KV cache reads land at roughly 1.65 ms per token at 16k context. Half of that is the slot the FP8 KV win is sitting in, and the silicon to take it is already on the card.

This post is about why FP8 KV is the next visible decode bandwidth cut on RDNA4, what the path through llama.cpp's Vulkan backend currently allows, and what zinc has to write to put the wmma instruction RDNA4 already ships under the attention kernel where the KV cache lives.

## What FP8 KV cache actually does on decode

The decode-side picture is the simplest of the three numerics decisions an inference engine has to make on the KV cache. The cache holds K and V tensors as they were produced by the projection layers at each step. On a memory-bound decode loop, the cost of attention is the bandwidth of reading those tensors back. FP16 storage gives 2 bytes per element. Q8_0 in llama.cpp's GGUF format gives 1.0625 bytes per element (a one-byte payload plus a shared FP16 scale across a 32-element block). FP8 E4M3FN gives 1 byte per element and runs the QK and ScoreV matmuls in the same numeric format, with online softmax rescaling in between, no dequantization step in the middle.

The difference between Q8_0 and FP8 on decode is not the storage size, which is within rounding. It is what the attention shader does with the read. The Q8_0 path on llama.cpp's HIP backend currently fuses cleanly only when both K and V are quantized to the same type; [a recent PSA on the llama.cpp discussions](https://github.com/ggml-org/llama.cpp/discussions/22411) made this concrete: `-ctk q4_0 -ctv q4_0` gets the fused FA kernel, while `-ctk q4_0 -ctv f16` silently falls back to a separate kernel that is materially slower. Symmetric quantization is what unlocks the fused path. The shader still dequantizes back to FP16 inside the inner loop, because the WMMA path it uses on RDNA3 and earlier RDNA4 builds does not have an FP8-by-FP8 fused-multiply-add.

FP8 KV is what closes that gap. The KV is stored as E4M3FN. The attention shader loads it as E4M3FN. The matmul instruction takes two FP8 inputs and accumulates into FP32. The dequantize-then-multiply hop disappears. On Hopper this is `wgmma.fp8`. On Blackwell with FlashInfer it is the default path. On RDNA4's gfx1201, the equivalent is `v_wmma_f32_16x16x16_fp8_fp8`, which AMD documents on [the RDNA4 matrix core developer page](https://gpuopen.com/learn/using_matrix_core_amd_rdna4/) and which a Triton kernel will compile down to automatically once the architecture detection picks the right device tier.

## Where RDNA4 actually sits today

The wmma instruction exists. The path to it is the part that is missing. The recent zinc piece on [the FP4 wave and FP8 wmma](/blog/2026-05-09-the-fp4-wave-breaks-at-rdna4-and-fp8-wmma-already-does-what-local-qwen3-needs) walked through where the RDNA4 matrix path lands at 383 TFLOPS for FP8 weights, the same throughput the data-center MI350X gets on the same format. The two cards use identical E4M3FN. The MI350X has had the kernel work in vLLM for months. The R9700 has not, because of a two-line arch-table miss.

That miss is documented in [ROCm TransformerEngine issue 520](https://github.com/ROCm/TransformerEngine/issues/520): gfx1201 is missing from `_ARCH_TO_DEVICE` in `aiter/ops/triton/utils/arch_info.py`, so any Triton kernel that consults the table to decide whether the device supports FP8 silently falls back to FP32 dequantization. ROCm 7.2.1 ships with that miss in place. The community fix is a one-line dictionary entry mapping `gfx1201` to `MI350X`. A community user at vLLM [demonstrated a 63 percent decode speedup on Qwen3-30B with FP8 weights](https://github.com/vllm-project/vllm/issues/28649) once that entry is patched in and a handful of RDNA4-specific kernel configs are tuned. That report is for weight quantization, not KV cache. The same arch-table patch is the precondition for either path.

For the KV cache specifically, the kernel work is still ahead of the patch. vLLM's path to FP8 KV is Flash Attention 3 on Hopper or FlashInfer on Blackwell. Neither has a RDNA4 build. llama.cpp's HIP path supports symmetric Q8_0 KV with a fused FA kernel, but there is no FP8 type in its KV cache enum; the `ggml_type` list in the public header does not yet include `GGML_TYPE_F8_E4M3`. The Vulkan backend, which is the one that runs the R9700 at 117 tok/s, inherits the same restriction. There is no FP8 KV path on any local engine today.

## The shape of the decode bandwidth win

The simplest way to see the win is to put the per-token bandwidth budget against the kernel path that has to read it. On a Qwen3.6-35B-A3B layer profile, the KV cache pulls roughly 5 GB per second of GDDR6 traffic at 117 tok/s on a 32 GB R9700, which the [AMD product page lists at 644.6 GB/s peak](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html). The bandwidth slice is small, but it is the linear-in-context slice, and the slope is the part the FP8 cut hits.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-05-19-fp8-kv-cache-rdna4-bandwidth-and-support.svg" alt="A two-panel data visualization on a soft slate-blue background. The left panel is a grouped vertical bar chart titled per-token KV cache bytes and decode ITL slope on Qwen3-class models. The chart has four KV formats on the x axis: FP16, Q8 zero, FP8 E4M3FN, and Q4 zero. Each format has two paired bars: a tall blue bar showing per-token KV cache bytes per element across both K and V, scaled on the left y axis from zero to two and a half bytes, and a shorter orange bar showing decode ITL slope as a percentage of BF16 baseline, scaled on the right y axis from zero to one hundred percent. The FP16 bars read 2.000 bytes and 100 percent. The Q8 zero bars read 1.0625 bytes and roughly 70 percent, with a small caption noting it requires symmetric K and V types to keep the fused FA path. The FP8 E4M3FN bars read 1.000 byte and 54 percent, with a bold orange outline on both bars and a callout that reads measured on H100 with Flash Attention 3 on Llama 3.1 8B per the vLLM April 22 2026 blog. The Q4 zero bars read 0.5625 bytes and a dashed orange estimate at roughly 55 percent, with a caption noting the slope is empirical for Q8 zero and the FP8 number is the validated measurement. The right panel is a five-by-four support matrix grid titled where the FP8 KV cache kernel actually runs. The columns are H100 Hopper, B200 Blackwell, MI350X CDNA, and R9700 RDNA4 gfx1201. The rows are vLLM Flash Attention 3, vLLM FlashInfer, SGLang, llama dot cpp HIP, and llama dot cpp Vulkan. Each cell is colored: a green filled square means production native FP8 KV plus FP8 attention, a yellow square means weight FP8 only or KV FP8 with software fallback, a gray square means not built, and a red X means hardware gap. The cells read: top row vLLM FA3 has green at H100, gray at B200, gray at MI350X, gray at R9700. Second row vLLM FlashInfer has gray, green, yellow, gray. Third row SGLang has green, yellow, yellow, gray. Fourth row llama dot cpp HIP has yellow with note Q8 zero symmetric, gray, yellow, yellow. Fifth row llama dot cpp Vulkan has yellow Q8 zero symmetric, yellow Q8 zero, yellow Q8 zero, yellow Q8 zero with a thick black outline. A footer reads the only cells with native FP8 KV cache plus FP8 attention today are the data center accelerator slots; RDNA4 has the v wmma f32 16x16x16 fp8 fp8 instruction documented in the RDNA4 ISA PDF but no local engine wires it under attention." loading="lazy" />
  <figcaption>The left panel pairs storage bytes against measured decode slope. The FP8 number is the validated vLLM Hopper result; Q8_0 and Q4_0 entries are bracketed by the symmetric-K-V fused FA discussion on llama.cpp HIP. The right matrix shows where the kernel runs in production today. The bottom-right cell is the R9700 row, and it sits at Q8_0 only on every engine.</figcaption>
</figure>

Two things to notice. The left panel is the headline argument: a one-byte KV format that also runs the matmul in the same numeric format collapses the slope to 54 percent of BF16 because the attention kernel reads half the cache bytes and does the math without a dequantize hop. The Q8_0 line is roughly 70 percent because the fused FA path still has to dequantize inside the loop, and the storage savings on a per-element basis are slightly less than FP8. The right panel is the situational map. Every cell with a green check mark today is a CUDA cell or a CDNA cell. Every cell with an R9700 row is yellow. That is the gap to take.

## What llama.cpp's Vulkan backend allows today

The current best path on a R9700 for KV bandwidth is symmetric Q8_0. `-ctk q8_0 -ctv q8_0` with `-fa 1` enables the fused Vulkan FA kernel that the [wave32 commit from May 11](/blog/2026-05-11-the-wave32-commit-that-closes-rdna4-long-context-flash-attention-gap) already specialized for RDNA4. The bandwidth saving on the KV read is real. The kernel still dequantizes inside the inner loop, but the read is from a one-byte cache, not a two-byte one. On the 35B-A3B decode at 16k context, the difference between FP16 KV and Q8_0 KV is most of the gap to the FP8 number; the FP8 win on top is the dequant elision plus the FP8 attention matmul.

The reason llama.cpp's Vulkan backend has not added an FP8 KV type is not a missing instruction. It is a missing kernel and a missing GGUF tensor type. The GGUF format on disk does not store the KV cache; that is allocated at runtime. The change is on the engine side: a new tensor type for `f8_e4m3`, a Vulkan compute shader for `flash_attn_batched.comp` that loads K and V as FP8 and runs the matmuls through `VK_KHR_cooperative_matrix` against an FP8 cooperative-matrix tile, and the per-head FP8 scales that vLLM tracks through `reshape_and_cache_flash`. The first two are kernel work. The third is bookkeeping the engine already does for Q8_0 K and Q8_0 V, in a slightly different shape.

## What the work looks like, concretely

The pieces break down cleanly. First, a Vulkan shader that loads K and V as packed E4M3FN bytes, runs the QK matmul through a cooperative-matrix tile in FP8, applies the online softmax rescale in FP32 in registers, and runs the PV matmul against an FP8 V tile. RADV exposes the FP8 cooperative-matrix path on gfx1201 through the same `VK_KHR_cooperative_matrix` extension the FP16 attention kernel already uses; the [chips and cheese RDNA4 LLVM piece](https://chipsandcheese.com/p/examining-amds-rdna-4-changes-in-llvm) walks through the wmma family the compiler ships against. The kernel scaffolding is what zinc already wrote for FP16 FA on RDNA4 in the wave32 path. The bookkeeping is per-head FP8 scales, which is the same data the vLLM `reshape_and_cache_flash` kernel writes through.

Second, a quantize-on-write path that the projection layer feeds into. The K and V projection outputs are FP16 today; they have to be downcast to E4M3FN with the per-head scale before they are written to the cache. That step is a small fused kernel that runs once per layer per decode token and costs almost nothing in absolute terms.

Third, a two-level accumulation policy on the QK matmul. The vLLM team's [Flash Attention 3 PR 104](https://github.com/vllm-project/flash-attention/pull/104) addressed an FP8 accuracy issue on Hopper that dropped needle-in-a-haystack accuracy from 91 percent down to 13 percent before they added a register-resident FP32 accumulator. The same precision pitfall applies to any FP8 attention path with a long contraction dimension, including RDNA4. The fix is to break the inner loop into chunks and accumulate into an actual FP32 register rather than into the WMMA's internal accumulator, which on contraction depths past 100k tokens loses precision. The fix is mechanical; it has to be in the first cut, not added after the fact.

Fourth, a calibration option for the per-head scales. Uncalibrated scales at `1.0` are the simplest case and what the vLLM benchmark uses; for models where uncalibrated scales drop accuracy below 95 percent, the engine has to expose a calibration step. For zinc on Qwen3.6-35B-A3B, the right first step is uncalibrated; the model is close enough to the Qwen3 family the vLLM team validated that no surprises are likely. The calibration hook is a flag.

## The tradeoff and what it does not buy

FP8 KV cache is not a free win. The two-level accumulation costs register pressure on the QK matmul. The FP8 ScoreV matmul has the same register pressure on long context. The break-even point on Hopper was 7k tokens for Llama 3.1 8B and roughly 8k for gpt-oss-20b with skip-sliding-window. The break-even on RDNA4 will be a function of the wmma dispatch granularity and the LDS-to-VGPR move pattern on gfx1201, which is different enough from Hopper that the curve will need to be measured, not predicted. On short contexts the constant overhead may dominate the slope savings, and the right interface is therefore a flag the user passes alongside the model, not a hard default.

The win is also model-dependent. Qwen3.5-27B at MRCR up to 1M recovers the full BF16 baseline AUC under FP8 KV plus FP8 attention. Qwen3-30B-A3B-Instruct-2507 recovers 94 to 98 percent of AUC depending on the model weights. Kimi-K2.5 on FlashMLA needs calibrated scales to stay close. The local engine has to expose the knob and document where it works.

The win does not change the LMHead profile from [Saturday's post](/blog/2026-05-16-what-qwen3-151k-lmhead-costs-on-rdna4-decode). The LMHead matmul is a separate tensor, not in the attention loop, and its cost is set by the vocabulary size and the hidden size. FP8 KV touches the attention slot, not the output projection slot. The two are stacked, not interchangeable.

## What comes next on zinc

The order of work is straightforward. Add `GGML_TYPE_F8_E4M3` to the cache type enum and the quantize-on-write kernel that the K and V projections feed into. Write a Vulkan attention shader that loads FP8 K and V and runs the matmuls through `VK_KHR_cooperative_matrix` with FP32 two-level accumulation. Validate the per-head scale path against the FP16 baseline at 8k, 32k, and 128k context on Qwen3.6-35B-A3B. Measure the ITL slope and compare against the BF16 baseline; the target is the 54 percent number the vLLM team got on H100.

The previous KV cache bandwidth cut, from FP16 to Q8_0, was the [April 26 argument](/blog/2026-04-26-why-fp16-kv-cache-is-the-wrong-default-for-128k-context-on-32gb-rdna4). That post made the case that FP16 was the wrong default for a 32 GB card at 128k context. The next cut is the one that runs the attention kernel in the same numeric format the storage uses, with no dequantize hop in the middle. The instruction is on the chip. The kernel is the work.
