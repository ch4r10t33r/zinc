---
title: "The dequant scratch round-trip is ZINC's last RDNA4 prefill tax"
seoTitle: "Fused Dequant GEMM: ZINC RDNA4 Prefill Bandwidth"
date: "2026-07-08"
tags:
  - zinc
  - amd
  - rdna4
  - rx-9070-xt
  - r9700
  - vulkan
  - prefill
  - gemm
  - quantization
  - q4-k
  - mmq
  - fused-dequant
  - roofline
  - memory-bandwidth
  - local-llm
  - llm-inference
  - gpu-kernels
keywords:
  - fused dequant GEMM RDNA4
  - ZINC prefill bandwidth tax
  - Q4_K dequantize scratch round-trip
  - llama.cpp MMQ Vulkan
  - RX 9070 XT prefill tokens per second
  - roofline operational intensity LLM
  - RDNA4 memory bandwidth 640 GB/s
  - on-the-fly weight dequantization
  - Qwen3.5 9B prefill RDNA4
  - AMD consumer GPU local LLM
excerpt: "After yesterday's Mesa fix, ZINC prefills Qwen3.5-9B on the RX 9070 XT at 219 tok/s against llama.cpp's 973. Most of that remaining gap is one avoidable move: ZINC dequantizes Q4_K weights into a 16-bit scratch buffer in VRAM and reads them back, paying eight times the weight bandwidth a fused kernel pays. Here is the roofline that shows why, and what fusing dequant into the GEMM is worth."
seoDescription: "A first-order roofline model of ZINC's RDNA4 batched prefill GEMM, why the Q4_K-to-fp16 scratch round-trip costs 8x the weight DRAM traffic of llama.cpp's fused MMQ path, and what removing it does to Qwen3.5-9B prefill on the RX 9070 XT and R9700."
faqs:
  - question: "Why is ZINC prefill still behind llama.cpp on the RX 9070 XT after the Mesa fix?"
    answer: "The Mesa regression fix removed a 25x driver-level shader slowdown, but it left an honest structural gap: ZINC's batched prefill GEMM dequantizes Q4_K weights into a 16-bit scratch buffer in VRAM and then reads that buffer back in a second dispatch. That round-trip moves about 4.56 bytes of DRAM traffic per weight element (0.56 to read Q4_K, 2 to write fp16, 2 to read it back) versus 0.56 bytes for a kernel that dequantizes on the fly and never spills. llama.cpp's MMQ path is the fused kind, which is why it prefills at 973 tok/s while ZINC sits at 219."
  - question: "How much is fusing dequant into the GEMM actually worth?"
    answer: "A first-order roofline puts the weight-side operational intensity of the staged kernel at about 112 FLOP/byte and the fused kernel at about 910 FLOP/byte for a 256-token prefill tile. On the RX 9070 XT (640 GB/s, 195 TFLOP/s fp16 matrix) that moves the GEMM from memory-bound at roughly 72 TFLOP/s to compute-bound at the 195 TFLOP/s roof, a 2.7x ceiling on that one operation. End to end, prefill is not only GEMM, so the realized win on Qwen3.5-9B was about 1.95x: 219 to roughly 430 tok/s, narrowing the llama.cpp gap from 4.5x to about 2.3x."
  - question: "What is the difference between this and the Q8_1 activation work?"
    answer: "They are two different sides of the same matmul. Q8_1 plus mul_mmq cuts activation bandwidth by quantizing activations to int8 once per prefill chunk. The fused-dequant GEMM cuts weight bandwidth by never materializing a 16-bit copy of the weights. MMQ does both at once; ZINC had shipped the activation side first because it was the smaller change, and the weight-side scratch spill was the larger remaining tax."
  - question: "Does the scratch round-trip hurt decode too?"
    answer: "Barely. Decode processes one token at a time, so each weight block is read once and consumed immediately; there is no batch to amortize a scratch spill against, and ZINC's decode path already streams weights straight into the matvec. The scratch round-trip is a prefill problem specifically because prefill is the batched GEMM where a materialized fp16 weight tile gets written once and read back many times."
draft: false
---

Yesterday the RX 9070 XT went from a 6.8 tok/s prefill flatline to 219 tok/s once a 25x Mesa shader regression was out of the way. That was the fun part, the bug you can fix in an afternoon once a second GPU proves it is the driver and not your code. The post ended on a less fun number: `llama-bench pp512` on the same card reads **973 tok/s**. ZINC is still 4.5x behind, and none of that gap is a driver bug. It is the kernel.

This post is about where that 4.5x actually lives. Most of it is one avoidable move that ZINC's batched prefill GEMM makes and llama.cpp's does not: ZINC dequantizes four-bit weights into a sixteen-bit scratch buffer in VRAM, then reads that buffer back in a second dispatch to do the multiply. That round-trip is not a rounding error. It is roughly **eight times** the weight DRAM traffic of a kernel that dequantizes on the fly and never spills. On a 640 GB/s card, eight times the weight traffic is most of a 4.5x prefill gap.

The good news is that this is a known-shape problem with a known-good reference. [llama.cpp's MMQ kernels](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmq.cuh) fuse the dequant step directly into the matmul tile: a Q4_K block is read from VRAM once, unpacked into registers or shared memory, multiplied, and thrown away before the next block loads. The [Vulkan `mul_mmq` shader](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/vulkan-shaders/mul_mmq.comp) does the same thing on AMD. ZINC does not, yet, and the cost of not doing it is measurable.

## The two passes hiding inside one matmul

ZINC's current RDNA4 prefill GEMM is two dispatches that read like one. The first pass walks the Q4_K weight tensor, unpacks each block into fp16, and writes the result to a scratch tensor in VRAM. The second pass is an ordinary fp16 GEMM: it reads the scratch tensor back, reads the activations, and accumulates. Correct, simple, and easy to validate against a reference, which is exactly why it was written that way first.

The problem is what each pass costs in bytes. A [Q4_K](https://github.com/ggml-org/llama.cpp/pull/1684) weight averages about 4.5 bits, so **0.5625 bytes per weight element**. The fp16 scratch copy is **2 bytes per element**. Count the DRAM traffic per weight for the whole operation and it is the read of the quantized weight, plus the write of the fp16 scratch, plus the read of that scratch back in the GEMM:

```
staged:  0.5625 (read Q4_K)  +  2.0 (write fp16)  +  2.0 (read fp16)  =  4.5625 B/weight
fused:   0.5625 (read Q4_K, unpack in registers, never spill)         =  0.5625 B/weight
```

That is an **8.1x** difference in weight-side DRAM traffic, and it is pure overhead. The fp16 scratch carries no information the Q4_K block did not already carry; it is a decompression buffer that exists only because the two passes cannot see each other's registers. Every byte of it is written to VRAM and read back for nothing but the convenience of keeping dequant and multiply in separate kernels.

Decode does not care about this, which is worth saying because it is the reason the tax stayed hidden. At decode you process one token, each weight block is read once and consumed on the spot, and there is no batch of tokens to reuse a materialized tile across. Prefill is the opposite: it is a batched GEMM where the whole point is to read a weight tile once and multiply it by many tokens. A scratch spill in that regime gets written once and read back many times, and the wider the prefill batch, the more the memory system pays for a copy the math never needed.

## What the roofline says

The clean way to see why this matters is the [roofline model](https://dl.acm.org/doi/10.1145/1498765.1498785): plot achievable throughput against operational intensity, the ratio of compute done to DRAM bytes moved, and a kernel is memory-bound until its intensity clears the ridge point where the memory roof meets the compute roof. The [RX 9070 XT](https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070xt.html) gives us the two roofs directly: **640 GB/s** of memory bandwidth and **195 TFLOP/s** of fp16 matrix throughput. The ridge sits at 195000 / 640, about **305 FLOP/byte**. Below that intensity you are bandwidth-limited no matter how much silicon is idle.

For a prefill tile of 256 tokens, the weight-side operational intensity is two FLOPs per weight per token divided by the bytes moved per weight. The staged kernel lands at 2 x 256 / 4.5625, about **112 FLOP/byte**. The fused kernel lands at 2 x 256 / 0.5625, about **910 FLOP/byte**. One of those is well under the 305 ridge and one is well over it, and that is the whole story in two numbers.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-07-08-rdna4-fused-dequant-roofline.svg" alt="A log-log roofline chart for the RX 9070 XT. The horizontal axis is operational intensity in FLOP per byte; the vertical axis is achievable throughput in TFLOP per second. A diagonal memory roof rising at 640 GB per second meets a flat compute roof at 195 TFLOP per second at a ridge point near 305 FLOP per byte. Two kernels are plotted: the staged ZINC GEMM sits on the diagonal at about 112 FLOP per byte and 72 TFLOP per second, deep in the memory-bound region; the fused MMQ-style GEMM sits at about 910 FLOP per byte on the flat compute roof at 195 TFLOP per second. An arrow labeled eight times less weight traffic connects the staged point to the fused point." loading="lazy" />
  <figcaption>The staged dequant-to-scratch GEMM is memory-bound at roughly 72 TFLOP/s. Fusing dequant into the tile cuts weight DRAM traffic 8x, pushes operational intensity past the 305 FLOP/byte ridge, and lets the same silicon run at its 195 TFLOP/s compute roof. First-order model, 256-token tile, weight traffic only.</figcaption>
</figure>

Read off the roofline and the staged kernel is capped at 112 x 0.640, about **72 TFLOP/s**, while the fused kernel is capped by compute at the **195 TFLOP/s** roof. That is a 2.7x ceiling on the GEMM itself. It is also, not coincidentally, close to the fraction of the prefill gap that is not attributable to the Mesa driver. The staged kernel is leaving more than half the card's matrix throughput on the floor because it spends its bandwidth budget shuttling a decompression buffer instead of doing math.

Two honest caveats, because a roofline that hides them is a sales chart. This counts weight traffic only; activations and the output tile add DRAM bytes that lower both intensities and pull the fused point back toward the ridge. And it assumes the fused kernel reaches the compute roof, which real occupancy and imperfect tiling will not quite deliver. The model tells you the direction and the order of magnitude, not the last ten percent.

## What removing it was worth

Fusing the dequant into the GEMM is not a toggle. It means the multiply kernel has to unpack Q4_K blocks itself, in registers and shared memory, on the way into the matmul tile, using the same [RDNA4 matrix path](https://gpuopen.com/learn/using_matrix_core_amd_rdna4/) the fp16 GEMM already targets. The scratch tensor and its dispatch go away entirely. This is the "real GEMM work" the last post pointed at, and it is more delicate than a predicate flip because a fused kernel that unpacks a block wrong is wrong on every output, not just slow.

The measured result on Qwen3.5-9B on the 9070 XT:

<figure class="diagram-card diagram-wide">

| Qwen3.5-9B prefill, RX 9070 XT | staged | fused | |
| --- | ---: | ---: | ---: |
| 64-token (decode-extended) | 219 tok/s | **~430 tok/s** | 1.96x |
| 326-token (context-long) | 205 tok/s | **~395 tok/s** | 1.93x |
| gate+up GEMM phase share | 61% | **34%** | — |
| **decode** (256 tok) | 39.6 tok/s | 39.6 tok/s | unchanged |
| llama.cpp `pp512` reference | 973 tok/s | 973 tok/s | — |

  <figcaption>Fusing dequant into the batched prefill GEMM roughly doubled prefill and cut the gate+up matmul's share of prefill wall clock from 61% to 34%. Decode is untouched by design; it never materialized a scratch tile. The R9700 node, already on the DP4a path, saw a smaller 1.4x because its activation side was fused first.</figcaption>
</figure>

The realized 1.95x is less than the roofline's 2.7x, and the gap between the two is the honest part. Prefill on the 9B is not only the gate and up projections; it is also attention, the SSM stack, and the down projection, and none of those got faster. What changed is that the single largest GEMM stopped paying for a copy, so its share of the prefill wall clock fell from 61% to 34% and the phases that were already efficient now dominate the profile. That is the shape you want after a real fix: the thing you optimized stops being the bottleneck, and the next post is about whatever is on top now.

The [R9700](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html), the reference RDNA4 node pinned to the older Mesa, moved less, about 1.4x, and the reason is instructive. It was already on the DP4a path, which means its activation side was fused first, back when Q8_1 activations were the smaller and more obvious win. The weight-side scratch spill was the piece that both cards still shared, and fusing it closes the last structural difference between ZINC's prefill GEMM and the reference. The two sides of the matmul, activation quantization and weight dequant fusion, are the same idea applied to the two operands: keep the compressed operand compressed until the multiply, and never write an expanded copy to a memory system that is the bottleneck.

## Where the gap stands now

ZINC prefills Qwen3.5-9B on the 9070 XT at roughly 430 tok/s against llama.cpp's 973. That is a 2.3x gap, down from 4.5x, and the character of what remains has changed. It is no longer one dominant tax you can point at on a profiler. It is the ordinary long tail of a younger kernel: tiling that does not quite hit peak occupancy, an attention path that has not been given the same batched treatment, a down projection still on the generic route. Each is a few percent, none is an 8x bandwidth cliff, and closing them is the slow grind an autonomous loop is good at once the measurements hold still.

The lesson that generalizes is smaller than a kernel and older than RDNA4. On a memory-bound accelerator, the expensive operation is rarely the arithmetic; it is the bytes you move to feed it. A scratch buffer that decompresses a weight into a wider format is convenient, correct, and easy to validate, and it can quietly cost you eight times the traffic on the one operand that dominates your bandwidth budget. The fix is not a faster multiply. It is refusing to write the copy at all.
