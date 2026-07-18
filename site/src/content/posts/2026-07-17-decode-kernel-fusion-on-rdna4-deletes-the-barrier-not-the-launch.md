---
title: "Decode kernel fusion on RDNA4 deletes the barrier, not the launch"
seoTitle: "RDNA4 Decode Kernel Fusion: Pipeline Barriers, Not Launches, Are the Win"
date: "2026-07-17"
tags:
  - zinc
  - amd
  - rdna4
  - rx-9070-xt
  - decode
  - kernel-fusion
  - vulkan
  - pipeline-barrier
  - rmsnorm
  - rope
  - local-llm
  - llm-inference
keywords:
  - RDNA4 decode kernel fusion
  - Vulkan pipeline barrier decode overhead
  - RMSNorm MUL fusion llama.cpp
  - RoPE fused into QKV projection
  - gated FFN SwiGLU fusion decode
  - pipeline barrier L2 flush L1 invalidate
  - RX 9070 XT decode tokens per second
  - batch one kernel chain latency
  - RADV Vulkan barrier bubble
  - fewer larger kernels decode
excerpt: "Command buffer reuse took the CPU rebuild off yesterday's decode token and left about 4 ms of GPU barrier bubbles. Those bubbles are not kernel launches. Each one is a pipeline barrier that flushes the write cache to L2 and invalidates L1 before the next kernel can read, and at batch one the compute units sit idle through the round trip. Kernel fusion removes them by keeping the intermediate in registers, which is why fusion's real decode win is the barrier it deletes, not the launch it saves."
seoDescription: "A modeled decode profile for Qwen3.5-9B on one RX 9070 XT after command buffer reuse leaves 4 ms of pipeline barrier bubbles and 5.4 ms of small latency-bound kernels per 22.2 ms token. Fusing RMSNorm and the weight multiply into the following projection, folding RoPE into the QKV epilogue, and collapsing gate, up and SwiGLU into one gated-FFN kernel keeps intermediates in registers and deletes the barrier between each pair. The Vulkan docs show relaxing barriers removes pipeline bubbles for a 13 percent frame-time win; llama.cpp already fuses RMS_NORM plus MUL on CUDA. Modeled result: about 55 tok/s, 59 percent of the achievable streaming floor, with the bus finally owning most of the token."
faqs:
  - question: "If command buffer reuse already cut the dispatch cost, what is left for kernel fusion to do?"
    answer: "Different halves of the same 7 ms. Yesterday's command buffer reuse removed the CPU-side rebuild, roughly 3 ms of re-recording the decode command buffer every token. It did not touch the GPU-side bubbles, roughly 4 ms, because those are pipeline barriers the GPU still executes even from a replayed buffer. A barrier flushes the previous kernel's writes to L2 and invalidates L1 before the next kernel reads, and at batch one the compute units idle through that round trip. Fusion is the only lever that removes the barrier, because it keeps the intermediate in registers so no barrier is needed."
  - question: "Why does a pipeline barrier cost so much on a decode step specifically?"
    answer: "Because decode is a chain of tiny dependent kernels with nothing to hide the barrier behind. Each transformer layer runs norm, projection, RoPE, attention, projection, norm, gate and up, SwiGLU, and down, and every kernel reads the previous one's output, so a barrier sits between them. At batch 64 the barrier overlaps with other sequences in flight. At batch one there is no other work, so the barrier is a pure bubble where the CUs wait on an L2 flush and an L1 invalidate. A 40-layer model pays that bubble roughly ten times per layer."
  - question: "Which fusions matter most for RDNA4 decode?"
    answer: "Three. Fuse RMSNorm and its weight multiply into the preamble of the following projection, which is the exact RMS_NORM plus MUL pattern llama.cpp already fuses on CUDA and is porting to CPU. Fold RoPE into the QKV projection epilogue so the rotation happens in registers before the result is written. Collapse the gate projection, up projection and SwiGLU activation into one gated-FFN kernel so the wide intermediate never lands in VRAM. Each removes a standalone kernel and, more importantly, the barrier after it."
  - question: "How much throughput does fusion actually buy?"
    answer: "In the model, decode goes from about 45 tok/s after command buffer reuse to about 55 tok/s after fusion, roughly a fifth. The gain comes from cutting the barrier bubbles from about 4 ms to about 2 ms and shrinking the small-kernel term from 5.4 ms to about 3.2 ms as intermediates stop round-tripping through VRAM. The number is a model, but the direction is not: fewer, larger kernels mean fewer barriers, and on a single-user card the barrier is the part of the token the batch of one exposes."
draft: false
---

Yesterday's post ended one fix short. Command buffer reuse takes the per-token rebuild off the decode path, the roughly [3 ms the CPU spends re-recording the command buffer](/blog/2026-07-16-command-buffer-reuse-is-rdna4s-version-of-the-cuda-graph-decode-win/) every single token, and it projects a jump from 39.6 to about 45 tok/s on one RX 9070 XT. Then it stops, because it cannot touch the other half of the dispatch cost: about 4 ms of GPU-side bubbles that survive even a perfectly replayed buffer.

Those 4 ms are not kernel launches. They are pipeline barriers. A replayed command buffer still contains the barrier between every pair of dependent kernels, and the GPU still executes each one, and at batch one each one is a stall where the compute units sit idle waiting on memory. This post is about deleting them, and the tool that deletes them is kernel fusion.

The claim worth stating up front is narrow and a little counterintuitive. The value of fusing two decode kernels is not mainly the launch you save. Command buffer reuse already amortized the launch. The value is the barrier you delete between them, because on RDNA4 that barrier is a full cache round trip the batch of one has no way to hide.

## What a barrier actually does between two kernels

A pipeline barrier is not free bookkeeping. It is a memory operation with real hardware cost.

When one compute kernel writes a storage buffer and a second kernel needs to read it, Vulkan requires a barrier so the read sees the write. On RDNA4 that barrier does two things. It flushes the first kernel's dirty cache lines out to the globally visible L2, and it invalidates the second kernel's L1 so the fetch units go back to L2 for fresh data. The [RasterGrid write-up on Vulkan memory barriers](https://www.rastergrid.com/blog/gpu-tech/2026/03/vulkan-memory-barriers-and-image-layouts-explained/) lays out this flush-then-invalidate split precisely: the source access mask drives the flush, the destination access mask drives the invalidate.

That round trip takes time, and while it happens the second kernel cannot start. The Khronos performance sample on [using pipeline barriers efficiently](https://docs.vulkan.org/samples/latest/samples/performance/pipeline_barriers/README.html) is blunt about it: a conservative barrier "will force a pipeline flush," and relaxing the barriers so the hardware can overlap work made the "pipeline bubbles disappear" for a 13 percent frame-time improvement in their sample. That sample is a graphics workload with fragment work to hide behind. Decode has nothing to hide behind.

## Decode is where the bubble has nowhere to go

The reason a barrier is cheap during prefill and expensive during decode comes down to what else the GPU is doing.

A decode step for Qwen3.5-9B runs one transformer layer at a time, and each layer is a chain of small dependent kernels: an input RMSNorm, the QKV projection, RoPE on Q and K, attention against the KV cache, the output projection, a second RMSNorm, the gate and up projections, the SwiGLU activation, and the down projection. Every kernel reads the previous kernel's output. That data dependency is exactly what a barrier enforces, so a barrier sits in nearly every gap.

At batch 64 those barriers overlap with other sequences still computing, so the flush of one sequence hides behind the arithmetic of another. At batch one there is no other sequence. The barrier is a pure bubble. The [RX 9070 XT](https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070xt.html) has 64 compute units and they all wait together through the L2 flush, doing nothing. A 40-layer model runs on the order of ten of these barriers per layer, so the token pays a few hundred of them, and none do any arithmetic.

This is the same shape [the llama.cpp maintainer measured years ago](https://github.com/ggml-org/llama.cpp/discussions/3909) when he replaced the Metal kernel bodies with an immediate return and still found significant per-token time he could not eliminate. The kernels were doing no work and the token was still slow. That residual is the barrier chain. It is real, it is stubborn, and command buffer reuse cannot reach it.

## Fusion removes the barrier by removing the write

Fusion attacks the bubble at its root. If two kernels never write their intermediate to VRAM, there is nothing to flush, so there is no barrier to place between them.

Take the most common pattern. Every layer starts by normalizing the hidden state with RMSNorm, multiplying by a learned weight vector, and feeding the result into a projection. Run as three kernels, that is norm, then multiply, then matmul, with a barrier after each. Fuse the norm and the multiply into the preamble of the projection kernel and the normalized vector never leaves registers. This is not hypothetical. The llama.cpp CUDA backend already fuses exactly this, and a [recent discussion on porting it to the CPU backend](https://github.com/ggml-org/llama.cpp/discussions/22315) measured the fused `RMS_NORM + MUL` op running at 20.79 GB/s against 14.49 GB/s unfused, about 1.4 times faster on the op alone, before you even count the deleted barrier.

Two more fusions matter for the decode layer. RoPE folds into the QKV projection epilogue, so the rotation happens on the projected Q and K while they are still in registers, deleting a kernel and its barrier. The gate projection, up projection and SwiGLU activation collapse into a single gated-FFN kernel, so the [wide SwiGLU intermediate](/blog/2026-07-10-down-projection-reads-back-a-swiglu-tensor-wider-than-the-hidden-state/) never materializes in VRAM, which the prefill work already fought on the other side of the roofline. The maintainer thread above notes gated-FFN fusion as the next target precisely because it saves more than the norm fusion.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-07-17-rdna4-decode-fusion-barrier-chain.svg" alt="A two-row diagram on a dark blue background comparing one Qwen3.5-9B decode layer before and after kernel fusion. The top row, labeled today, shows nine small kernel boxes left to right: input RMSNorm, weight multiply, QKV projection, RoPE, attention, output projection, post-attention RMSNorm plus multiply, a combined gate and up projection, SwiGLU, and down projection. Between most adjacent boxes sits a narrow orange vertical bar labeled barrier, marked as an L2 flush and L1 invalidate; ten barriers are shown and annotated as about 4 milliseconds of bubbles per token across 40 layers. The bottom row, labeled after fusion, shows four wider teal boxes: a fused RMSNorm-multiply-QKV-RoPE block, an attention block, a fused RMSNorm-multiply-output block, and a fused gated-FFN block that folds gate, up, SwiGLU and down together. Only three orange barrier bars remain between the fused blocks, annotated as about 2 milliseconds. Curved grey arrows connect the deleted barriers in the top row to a caption reading intermediate now stays in registers, no flush needed. A legend labels teal as a compute kernel, orange as a pipeline barrier, and notes the intermediate that no longer touches VRAM." loading="lazy" />
  <figcaption>One modeled Qwen3.5-9B decode layer before and after fusion on the RX 9070 XT. Each orange bar is a pipeline barrier that flushes to L2 and invalidates L1; fusion deletes a barrier every time it keeps an intermediate in registers.</figcaption>
</figure>

Read the two rows against each other. The top row is today: ten small kernels with a barrier in nearly every gap, and the barriers are the orange bubbles, not the kernels. The bottom row folds the norm, multiply and RoPE into the projections and collapses the FFN, so the same math runs in four kernels with three barriers. The arithmetic is identical. What changed is how many times the layer stops to flush a cache line the next kernel is about to read anyway.

## What it does to the token

Put the fusions into the same modeled per-token budget the last two posts have been carrying, starting from the 22.2 ms token that command buffer reuse leaves behind.

| Segment | After buffer reuse | After fusion | What changed |
| --- | ---: | ---: | --- |
| Weight streaming | 10.7 ms | 10.7 ms | Untouched, bytes are bytes |
| KV read | 0.6 ms | 0.6 ms | Untouched |
| Small kernels | 5.4 ms | 3.2 ms | Intermediates stay in registers |
| Barrier bubbles | 4.0 ms | 2.0 ms | Half the barriers deleted |
| LM head + sampling | 1.5 ms | 1.5 ms | Untouched |
| Token total | 22.2 ms | 18.0 ms | 45 to about 55 tok/s |

The streaming term does not move, because fusion changes nothing about how many weight bytes the token reads. What moves is the overhead: the barrier bubbles roughly halve as the deleted barriers disappear, and the small-kernel term shrinks because the norm, RoPE and SwiGLU work now happens inside a projection kernel instead of as standalone passes that write their results to VRAM and read them back. The token falls from 22.2 ms to about 18.0 ms, which is roughly 55 tok/s.

Set that against the ceiling from two posts ago. The achievable streaming floor for this model on this card is [about 93 tok/s](/blog/2026-07-15-weight-streaming-is-under-half-of-an-rdna4-decode-token/), the honest bandwidth bound after you discount peak. Decode started this arc at 39.6 tok/s, or 43 percent of that floor. Command buffer reuse and fusion together bring it to about 55, or 59 percent. More to the point, weight streaming is now 10.7 of 18.0 ms, so the memory bus finally owns most of the token instead of [under half of it](/blog/2026-07-14-a-qwen3-5-9b-chat-turn-spends-most-of-its-wall-clock-in-decode/). That was the goal the whole time: get the overhead small enough that the bandwidth wall is the thing actually limiting decode.

## Where fusion stops

Fusion is not free and it does not scale forever. Every fused kernel is bigger, holds more intermediate state in registers, and a batch-one decode kernel is already register-pressured before you fold three operations into it. Past a point, adding another fused stage spills registers, drops occupancy, and the barrier you deleted is cheaper than the spill you bought, the same [VGPR pressure ceiling](/blog/2026-07-09-vgpr-pressure-caps-fused-rdna4-prefill-gemm-at-nine-waves/) the prefill GEMM ran into. The three fusions here are the safe ones because the intermediates are small: a normalized hidden vector, a rotated Q and K, a gated activation. The attention step stays its own kernel because a persistent flash-attention kernel is a different and harder fusion, and the [softmax inside it already fights for issue slots](/blog/2026-07-11-softmax-steals-rdna4-wmma-issue-slots-flash-attention/).

So the model keeps three barriers per layer, not zero. That is honest. The point is not to reach a single kernel per layer. It is to notice that the decode token spent a few hundred barriers flushing caches for kernels that read the result immediately, and that most of those flushes were avoidable arithmetic bookkeeping rather than real synchronization.

## What I am building next

The concrete work is to land RMSNorm-plus-multiply fusion and RoPE folding in ZINC's Vulkan decode path first, because they are low-risk and the llama.cpp CUDA backend has already proven the pattern, then measure the barrier count per token before and after with GPU timestamps rather than trusting the model. The gated-FFN fusion comes after, gated on whether the register spill on RDNA4 stays under the barrier it removes.

The decode arc has a clean shape now. Streaming is the floor and it is arithmetic. Command buffer reuse deleted the CPU rebuild. Fusion deletes the barriers the rebuild left behind. Each step took a slice of overhead off a token that the batch of one had exposed, and what remains underneath is the 640 GB/s bus doing the one job decode actually needs it to do.
