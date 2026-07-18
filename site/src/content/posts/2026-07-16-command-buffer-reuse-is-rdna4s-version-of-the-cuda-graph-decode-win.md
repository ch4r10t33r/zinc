---
title: "Command buffer reuse is RDNA4's version of the CUDA graph decode win"
seoTitle: "Vulkan Command Buffer Reuse vs CUDA Graphs: RDNA4 Decode Overhead"
date: "2026-07-16"
tags:
  - zinc
  - amd
  - rdna4
  - rx-9070-xt
  - decode
  - vulkan
  - cuda-graphs
  - command-buffer
  - kernel-launch
  - kernel-fusion
  - local-llm
  - llm-inference
keywords:
  - Vulkan command buffer reuse LLM decode
  - CUDA graphs llama.cpp decode overhead
  - RDNA4 decode kernel launch overhead
  - RX 9070 XT decode tokens per second
  - re-record command buffer per token
  - kernel dispatch overhead batch one
  - RADV Vulkan decode latency
  - pipeline barrier bubbles decode
  - graph capture vs command buffer replay
  - single-user local inference overhead
excerpt: "Yesterday's decode budget left 7 ms of every 25.2 ms token in pure dispatch and barriers. NVIDIA already deleted most of that on CUDA with graph capture, worth 14 percent on a 7B. RDNA4 has no CUDA graphs, so ZINC rebuilds the whole decode command buffer every token on Vulkan. Recording it once and resubmitting it is the cheap half of the fix. The other half, the barrier bubbles, needs kernel fusion, and that is the part command buffer reuse cannot touch."
seoDescription: "A modeled decode profile for Qwen3.5-9B on one RX 9070 XT splits the 7 ms of per-token dispatch overhead into a 3 ms CPU rebuild and 4 ms of GPU barrier bubbles. NVIDIA's CUDA graph work in llama.cpp removed the GPU-side gaps for a 14 percent speedup on a 7B. Vulkan has no graph replay primitive, so ZINC re-records the command buffer every token. Reusing a recorded buffer and patching only the KV-cache parameters takes the 3 ms CPU rebuild off the critical path for a projected 45 tok/s, but the barrier bubbles survive reuse and need kernel fusion to remove."
faqs:
  - question: "What is the decode overhead that command buffer reuse targets?"
    answer: "On Qwen3.5-9B and one RX 9070 XT, a decode token is about 25.2 ms, of which roughly 7 ms is not arithmetic and not weight streaming, it is the cost of preparing and launching the kernel chain. That 7 ms splits into two parts: about 3 ms of CPU work rebuilding the compute graph and re-recording the Vulkan command buffer for every token, and about 4 ms of GPU-side gaps between kernels caused by pipeline barriers. Command buffer reuse targets the 3 ms CPU rebuild."
  - question: "Why can't RDNA4 just use CUDA graphs like NVIDIA?"
    answer: "CUDA graphs are an NVIDIA CUDA feature. ZINC runs on Vulkan through the RADV driver on RDNA4, and Vulkan has no equivalent graph-capture-and-replay primitive that packs a whole token's kernels into one launch. The closest Vulkan mechanism is recording a command buffer once and resubmitting it, which removes the CPU cost of rebuilding the buffer but does not automatically tighten the GPU-side gaps between kernels the way a CUDA graph does."
  - question: "How much did CUDA graphs actually help llama.cpp?"
    answer: "NVIDIA's Alan Gray reported that capturing each token's kernels into a single CUDA graph took llama-2-7B on an H100-PCIe from 143.35 to 163.83 tok/s, a 14 percent speedup, with the graph execution itself about 40 percent faster once launch gaps were removed. The benefit grows as the model shrinks and the GPU gets faster, because that is when per-kernel launch overhead is the largest share of the token."
  - question: "If reuse only removes the CPU rebuild, what removes the barrier bubbles?"
    answer: "Kernel fusion. The GPU gaps come from pipeline barriers between many small kernels, each waiting on the previous one's output. Reusing a recorded command buffer replays the same barriers, so it cannot shrink them. Cutting the number of kernels per layer, by fusing the pre-attention norm and RoPE into the QKV projection and collapsing the gate, up and SwiGLU into one kernel, removes barriers directly. Reuse is the plumbing fix you do first, fusion is the kernel work that follows."
draft: false
---

Yesterday I split a decode token on the RX 9070 XT into five pieces and pointed at the ugliest one. Of the 25.2 milliseconds it takes to generate a token on Qwen3.5-9B, about 7 were [pure dispatch and pipeline barriers](/blog/2026-07-15-weight-streaming-is-under-half-of-an-rdna4-decode-token/), doing no arithmetic at all. I called it the overhead half and said it was the part with the most slack left. This post is about spending some of that slack.

The frustrating thing is that a large chunk of this problem was solved two years ago, on the other vendor's stack. In 2024 NVIDIA landed CUDA graph support in llama.cpp, and on a Llama-2-7B it lifted decode from 143.35 to 163.83 tok/s, a [14 percent speedup](https://github.com/ggml-org/llama.cpp/issues/6763), by collapsing a token's worth of kernel launches into a single graph. RDNA4 gets none of that, because CUDA graphs are a CUDA feature and ZINC runs on Vulkan.

So the question for a local AMD engine is narrow and concrete. What is the Vulkan-shaped version of the CUDA graph win, how much of the 7 ms can it actually reach, and what does it leave behind?

## Why NVIDIA got a free 14 percent

Start with what CUDA graphs did, because the mechanism decides how much of it transfers.

At batch one, generating a token runs a long chain of small kernels, one group per transformer layer, and each kernel is launched separately. In the traditional stream model every launch is scheduled on its own, and the gaps between kernels pile up. NVIDIA's [profile of the pre-graph code](https://developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs/) shows the GPU sitting idle in those gaps, and their note is specific: on that hardware the gaps were mostly GPU-side launch overhead, not the CPU falling behind. The CPU was already running ahead of the GPU.

CUDA graphs fix this by capturing the whole per-token kernel sequence once and replaying it as a single unit. The kernels pack tightly, the idle gaps shrink, and the graph execution itself runs about 40 percent faster. The catch NVIDIA had to solve is that the token's graph is not static: the KV-cache length grows every step, so a handful of kernel parameters change each token. Their answer was to keep the instantiated graph and patch only the KV-related parameters before each replay, recapturing fully only on the rare occasions when the context crosses a size step. Update the small parts cheaply, reuse the expensive structure.

That last idea is the transferable one. The specific API is NVIDIA's, but the pattern, build the launch structure once and patch what little changes per token, is exactly what Vulkan can do with a command buffer.

## What ZINC does today, and why it is wasteful

Here is the part that stings. ZINC, like stock llama.cpp before the graph work, rebuilds the decode graph and re-records the Vulkan command buffer from scratch on every single token.

A decode command buffer for a 40-layer model is a few hundred dispatch calls, each with its pipeline bind, descriptor set, push constants, and a pipeline barrier to order it after the last. Recording that is not free CPU work, and neither is the graph bookkeeping around it. The llama.cpp maintainers found the same thing on their side: the decoder [recomputes the compute graph on every call](https://github.com/ggml-org/llama.cpp/issues/6763) and the schedule reset does real CPU work on the critical path. On a fast GPU with a modest CPU, that record time is not hidden behind anything. The GPU finishes the previous token and waits for the CPU to hand it the next buffer.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-07-16-rdna4-decode-command-buffer-reuse-timeline.svg" alt="A two-panel diagram on a dark background. The top panel shows one decode token as a CPU lane and a GPU lane on a shared 0 to 25 millisecond time axis, drawn twice. In the top 'Today' version, the CPU lane has an orange 'build and record' block of 3.0 milliseconds at the very start, then sits idle and blocked on the GPU; the GPU lane starts only after the CPU block finishes at 3.0 milliseconds and runs a chain of teal weight-streaming and purple attention kernels separated by pink barrier-bubble gaps, ending at 25.2 milliseconds, labeled 18.2 milliseconds of kernels busy plus 4.0 milliseconds of barrier bubbles, giving 39.6 tokens per second. In the bottom 'With command buffer reuse' version, the CPU lane has only a tiny 0.3 millisecond orange 'patch KV params and resubmit' block that is overlapped with the prior token and off the critical path; the GPU lane starts almost immediately and runs the same teal and purple kernel chain with the same pink barrier gaps, ending at 22.2 milliseconds, with a dashed teal bracket at the right marking 3.0 milliseconds removed. A caption notes that the barriers survive reuse and need kernel fusion. The bottom panel is three horizontal bars in tokens per second: ZINC today at 39.6 in solid teal, plus command buffer reuse at 45.0 in faded teal marked projected plus 14 percent, and plus reuse plus kernel fusion at 50.8 in fainter teal marked projected." loading="lazy" />
  <figcaption>A modeled decode token as a CPU and GPU timeline. Reuse removes the 3.0 ms CPU rebuild that blocks the GPU at the start of every token. The pink barrier bubbles between kernels survive reuse untouched.</figcaption>
</figure>

Read the top panel first. Today the GPU cannot start until the CPU has finished recording, so 3 ms of rebuild sits in front of every token before a single weight is read. The bottom panel is the fix: record the command buffer once, and on each token patch only the KV-cache offsets and dispatch dimensions and resubmit the same buffer. The record work drops to a fraction of a millisecond and overlaps with the previous token's GPU tail, so it leaves the critical path entirely. On the modeled budget that turns 25.2 ms into 22.2 ms, which is 39.6 tok/s becoming about 45, a projected 14 percent that lands, not coincidentally, right where NVIDIA's number did.

## The barriers do not care that the buffer was reused

Now the honest part, and it is the reason this post is not just "do what NVIDIA did."

Look at the GPU lane in the bottom panel again. The pink gaps are still there. Reusing the command buffer changed who prepares the work, not what the work is, so every pipeline barrier that was recorded into the buffer gets replayed exactly as before. The 4 ms of GPU-side bubbles, the gaps where the GPU waits for one kernel's writes to be visible before the next kernel reads them, survive reuse completely.

This is where the CUDA graph analogy breaks, and the break is important. NVIDIA's 14 percent came mostly from tightening GPU-side gaps, because on their profile the CPU was already ahead and the gaps were the launch overhead between kernels. A CUDA graph is a GPU-side scheduling object, so packing it helps the GPU directly. A reused Vulkan command buffer is not a rescheduling of the GPU's work, it is a way to skip re-recording on the CPU. The two optimizations attack different halves of the same 7 ms, and which half dominates on RDNA4 is an empirical question I have to measure, not assume.

That is the single most useful thing to take from this: before touching anything, profile which side of the decode token is actually on the critical path. If the GPU is starved waiting for CPU records, reuse is the whole win. If the CPU is already ahead and the token is stalling inside the GPU on barriers, reuse buys almost nothing and the work is elsewhere.

## Two levers, and the order matters

The 7 ms of overhead splits into two costs with two different fixes, and they are not interchangeable.

| Overhead source | Modeled cost | Who pays | The fix | Reuse helps? |
| --- | ---: | --- | --- | :---: |
| Rebuild graph + record command buffer | 3.0 ms | CPU, per token | Record once, patch KV params | yes |
| Pipeline barriers between kernels | 4.0 ms | GPU, per token | Fewer, larger kernels (fusion) | no |

Command buffer reuse is the cheap lever and it comes first, because it is plumbing, not surgery. It does not change a single shader. It changes the lifetime of the command buffer from one token to many, patches the few descriptors and push constants that move with the KV cache, and resubmits. Vulkan's own [command buffer guidance](https://docs.vulkan.org/samples/latest/samples/performance/command_buffer_usage/README.html) is blunt that re-recording and reallocating buffers is the expensive path and recycling them is the cheap one, and it flags a real tradeoff: a buffer you intend to resubmit has to drop the one-time-submit hint, which can cost the driver some optimization. On a decode loop that resubmits the same structure thousands of times, that trade is heavily in reuse's favor.

The barrier lever is the expensive one and it is the same kernel fusion I keep circling back to. Fusing the pre-attention norm and RoPE into the QKV projection removes two kernels and their barriers per layer. Collapsing the gate, up, and SwiGLU into a single kernel removes another launch and the [wide intermediate write](/blog/2026-07-10-down-projection-reads-back-a-swiglu-tensor-wider-than-the-hidden-state/) between them. A single persistent flash-attention kernel that keeps the attention step resident removes several more. Cut a layer from a dozen kernels to a handful and the pink bubbles shrink, no command buffer trick required. The third bar in the figure is that ceiling, a projected 50.8 tok/s once both levers are pulled, and it is honestly the harder half of the two.

None of this touches the memory bus, and that is the point I want to keep straight after a month of roofline posts. Decode is bandwidth-bound in the limit, but ZINC is not in the limit, it is at 39.6 tok/s against a 93 tok/s [streaming floor](/blog/2026-07-15-weight-streaming-is-under-half-of-an-rdna4-decode-token/). Command buffer reuse and kernel fusion are both attacks on the gap between here and that floor, and neither one reads a byte fewer. The bytes-per-token levers, smaller quantization and a leaner KV cache, are a separate axis I will keep pulling on too. This is the overhead axis, and it has been sitting untouched while I chased prefill.

## What I am building next

The concrete next step is not the fusion, it is the measurement. I am going to profile the decode loop the way NVIDIA profiled theirs, with a CPU and GPU timeline side by side, and answer the one question that decides everything: on the RX 9070 XT with RADV, is the GPU waiting for the CPU to record, or is the CPU idle while the GPU stalls on barriers? The model here says the split is roughly 3 ms and 4 ms, but the model is a hypothesis, and the whole reason CUDA graphs worked was that NVIDIA looked at the profile first and found the gaps were where they thought.

If the reuse half is real, it is a few days of buffer-lifetime plumbing for a projected 14 percent, and it is the cheapest decode win on the board. The barrier half is weeks of kernel fusion for another 13. I would rather ship the plumbing, measure the residual, and let the profile tell me how much fusion is actually left to do. The overhead half of a decode token has been the engine talking to itself, and the first fix is just getting it to stop repeating the same sentence every token.
