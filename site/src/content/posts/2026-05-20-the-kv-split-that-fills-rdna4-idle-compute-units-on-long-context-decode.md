---
title: "The KV split that fills RDNA4's idle compute units on long-context decode"
date: "2026-05-20"
tags:
  - zinc
  - rdna4
  - amd
  - vulkan
  - flash-attention
  - flash-decoding
  - long-context
  - decode
  - kv-cache
  - qwen3
  - llm-inference
keywords:
  - Flash-Decoding RDNA4
  - batch=1 decode GPU occupancy
  - Radeon AI PRO R9700 flash attention
  - split-KV attention long context
  - Qwen3-30B-A3B GQA 32 query heads
  - 128k context decode AMD
  - Vulkan flash attention split-K
  - FlashAttention batch size 1 underutilization
  - long-context decode bandwidth R9700
  - YaRN 131k context local inference
excerpt: "At batch size one, FlashAttention uses less than one percent of an A100, and the same starvation happens on a 64-compute-unit Radeon AI PRO R9700. A single-user local engine lives in that regime all day. Flash-Decoding fixes it by splitting the KV cache into chunks so the attention kernel fills every compute unit, and on the published A100 micro-benchmark it cuts batch=1 attention at 128k context from 4592 microseconds to 107. This is the long-context decode win RDNA4 leaves on the table, and the change is one extra parallelization dimension plus a small reduction kernel."
---

The PyTorch team measured something in late 2023 that should shape how a local inference engine writes its attention kernel. At batch size one, [FlashAttention uses less than one percent of an A100](https://pytorch.org/blog/flash-decoding/). The card has 108 streaming multiprocessors and the kernel keeps a handful of them busy. The rest sit idle while the one query token walks the entire key-value cache.

On a data center that runs hundreds of requests at once, this is a non-issue, because the batch dimension fills the machine. On a single-user desktop it is the whole story. A person typing into a chat window generates one token at a time, against one set of weights, with one query row per layer. Batch size one is not an edge case for local inference. It is the default, and it gets worse as the context grows, because long context forces small batches to fit in memory and the attention kernel is the one operation whose cost scales with how many tokens came before.

The Radeon AI PRO R9700 has [64 compute units and 644.6 GB/s of GDDR6](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html). At batch size one it has the same problem the A100 has, scaled down: the standard flash attention kernel cannot find enough independent work to keep all 64 units busy. The fix is a technique called Flash-Decoding, and it is the next attention change worth making on zinc's RDNA4 decode path.

## Why batch=1 long context starves the card

To see why the kernel runs out of work, you have to look at how flash attention is parallelized. The [FlashAttention-2 paper](https://arxiv.org/abs/2307.08691) describes the grid it launches: one thread block per query block, per attention head, per sequence in the batch. During training and prefill that is a large number, because the query length is in the thousands and the batch can be large. The kernel has more independent tiles than the GPU has cores, so occupancy is high and the matrix units stay fed.

Decode breaks that arithmetic. The query length is one. There is exactly one query block. So the grid collapses to one thread block per head per batch element, and at batch size one that is just the number of attention heads.

Take a concrete model. Qwen3-30B-A3B is a grouped-query attention model with [32 query heads and 4 key-value heads across 48 layers](https://huggingface.co/Qwen/Qwen3-30B-A3B), and it supports 131,072 tokens of context through YaRN scaling. Grouped-query attention, or GQA, means several query heads share one key-value head to shrink the cache. At decode time the work is still per query head, so a batch=1 step launches 32 attention thread blocks. The R9700 has 64 compute units. Half of them have nothing to do, and the half that are busy each hold a single workgroup with no second workgroup to hide memory latency behind. The effective utilization is well under the half the head count suggests.

This is the part that does not improve with a better kernel. A faster inner loop makes each of those 32 thread blocks finish sooner, but it does not create the missing 32 thread blocks. The work is simply not there to dispatch. And the longer the context, the longer each of those 32 blocks runs, because each one streams the full key-value history for its head. The card spends the whole attention step lightly loaded and waiting on memory.

## What Flash-Decoding actually changes

Flash-Decoding, published by Tri Dao, Daniel Haziza, Francisco Massa, and Grigory Sizov, adds one new dimension to parallelize over: the key-value sequence length itself. Instead of one thread block streaming all 131k tokens for a head, the kernel splits the cache into chunks and computes the query's partial attention against each chunk in parallel.

The mechanism is three steps, and the [Stanford CRFM writeup](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) lays them out cleanly. First, split the keys and values into smaller chunks, which is free because the chunks are just views into the existing tensors. Second, run a normal flash attention against each chunk in parallel, and for each chunk write one extra scalar per row, the log-sum-exp of that chunk's attention scores. Third, reduce across the chunks, using those log-sum-exp values to rescale each chunk's contribution into the correct global softmax. The softmax is associative when you carry the running max and sum, which is exactly what flash attention already tracks internally, so the cross-chunk reduction is mathematically clean rather than an approximation.

The payoff is occupancy. With 8 splits, the same Qwen3-30B-A3B decode step now launches 32 heads times 8 splits, or 256 thread blocks, against 64 compute units. Every unit gets work, with several workgroups resident to hide latency. The attention step stops being a lightly loaded crawl and starts looking like a workload the card was built to run.

## The shape of the win

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-05-20-flash-decoding-rdna4-cu-occupancy.svg" alt="A two-panel data visualization on a dark slate background. The left panel is titled attention workgroups at batch equals one on a 64 compute unit R9700 and shows two eight by eight grids of square cells, each grid representing the 64 compute units of the Radeon AI PRO R9700. The upper grid is labeled FlashAttention, 32 workgroups, one per query head, and has its top four rows of cells lit in cyan while the bottom four rows are dark and idle, with a caption reading 32 of 64 compute units busy, no latency hiding. The lower grid is labeled Flash-Decoding, 32 heads times 8 KV splits equals 256 workgroups, and has all 64 cells lit in a bright cyan to violet gradient with a caption reading every compute unit busy, about four workgroups resident each. The right panel is a line chart titled A100 attention runtime, batch equals one regime, microseconds on a log scale. The x axis lists ten measured settings from sequence length 256 at batch 256 on the left to sequence length 131072 at batch 1 on the right, with batch size shrinking as sequence length grows. Two lines are plotted. An orange line labeled FlashAttention v2 stays near 350 to 400 microseconds across the short settings then climbs steeply on the right to 1156, then 2301, then 4592 microseconds at sequence length 131072 batch 1. A cyan line labeled Flash-Decoding stays nearly flat along the bottom between 56 and 107 microseconds across every setting. A callout at the right edge reads 43 times faster at 128k context, 4592 microseconds versus 107. A footer credits the PyTorch Flash-Decoding blog micro-benchmark on A100 in float16 with 16 query heads and 2 key-value heads." loading="lazy" />
  <figcaption>Left: the R9700's 64 compute units at batch=1. Standard flash attention lights up only the 32 that match the query-head count; Flash-Decoding's KV split fills all 64. Right: the published A100 micro-benchmark in float16, batch=1 regime. Flash-Decoding holds attention runtime nearly flat as the sequence grows while FlashAttention v2 climbs to 4592 microseconds at 128k context.</figcaption>
</figure>

The left grid is the mechanism and the right chart is the proof. The grid shows why the kernel was idle and how the split fixes it. The chart is the [published A100 micro-benchmark](https://pytorch.org/blog/flash-decoding/), measured in float16 with 16 query heads and 2 key-value heads, batch size one in the two rightmost settings. FlashAttention v2 sits near 360 microseconds at short context and then balloons to 2301 microseconds at 64k and 4592 at 128k. Flash-Decoding stays between 56 and 107 microseconds across the entire range. At 128k context that is a 43x gap, and the authors report the attention itself running up to 50x faster, which translated to up to 8x end-to-end on CodeLlama-34B.

The A100 numbers are not the R9700 numbers, and I will not pretend the multiples carry over unchanged. The point is structural, not numeric. The same starvation that makes FlashAttention v2 climb on the A100 is present on any GPU where the head count is smaller than the compute-unit count at batch one, and on a 64-unit RDNA4 card with a 32-head model it plainly is.

## How this sits above the wave32 and determinism changes

zinc has touched attention twice in the last two weeks, and Flash-Decoding sits above both of those changes rather than replacing either. The [wave32 commit](/blog/2026-05-11-the-wave32-commit-that-closes-rdna4-long-context-flash-attention-gap) was about the SIMD width of the per-chunk kernel: running flash attention in wave32 instead of wave64 so the scalar work at Br=1 stops wasting half of every SIMD cycle. That is the kernel that runs inside each split. Flash-Decoding is the dimension above it, the thing that decides how many copies of that kernel run at once.

The [temperature-zero determinism post](/blog/2026-05-14-temperature-zero-is-not-deterministic-for-local-qwen3-on-rdna4-yet) is the other connection, and it is the one that makes the split design slightly delicate. That post landed on a fixed split-KV reduction strategy so that the order of the cross-chunk reduction does not depend on how the prompt was chunked during prefill. Flash-Decoding introduces exactly that cross-chunk reduction, and floating-point addition is not associative, so the number of splits and the order they reduce in becomes part of what determines the logits. A performance-driven split count that changes with context length and a determinism-driven split count that has to stay fixed are in tension. The resolution is to pick the split count from the context length and the compute-unit count by a deterministic rule, so the same prompt always produces the same split layout, and then reduce in a fixed tree. The split adapts to the workload, but it adapts the same way every time.

## What it does not buy

Flash-Decoding is an attention-kernel win, and attention is not where most of the decode time goes. The bulk of a batch=1 decode step is reading the active weights out of memory, and as the [matrix-cores post](/blog/2026-04-30-rdna4-matrix-cores-sit-out-the-decode-loop) argued, that part lives far on the bandwidth-bound side of the roofline and does not care how many compute units are busy. Filling the card during attention does not speed up the weight read. It speeds up the slice of the step that grows with context, which is precisely the slice that hurts at 128k and is invisible at 2k.

That makes the win conditional on context length, and the conditioning runs the wrong way for short prompts. With a small cache there are not enough key-value tokens to split without the reduction overhead dominating, and the extra reduction kernel plus the log-sum-exp bookkeeping can cost more than they save. The micro-benchmark shows Flash-Decoding already competitive at short settings, but those settings carry large batches that fill the card on their own. On a true batch=1 short-context step the right behavior is to fall back to the plain kernel. The split count should be one below some context threshold and climb from there, capped so the per-split tile never shrinks below a useful matrix size.

The split also does nothing about the bytes themselves. The key-value cache still has to be read in full once per token, and that read is the [bandwidth crossover](/blog/2026-04-27-the-16k-crossover-where-kv-reads-outweigh-active-weights-on-rdna4-decode) that starts to outweigh the active weights past 16k context. Flash-Decoding reads the same bytes; it just reads them in parallel across the card instead of serially down one unit. Cutting the bytes is a separate axis, the one the GQA-shape and KV-quantization work covers, and the [GQA post](/blog/2026-05-03-why-gqa-is-not-the-last-kv-cache-shape-for-local-32gb-long-context) is where that thread lives. The two compose: fewer bytes to read, read across more units.

## What comes next on zinc

The work breaks into four pieces. Add a split-K dimension to the Vulkan flash attention kernel so a single head's key-value range can be partitioned across workgroups, which is the same `VK_KHR_cooperative_matrix` tile the wave32 kernel already uses, just dispatched more times. Write a second small reduction kernel that consumes the per-split partial outputs and per-split log-sum-exp scalars and combines them into the final attention output. Add a deterministic heuristic that picks the split count from the context length and the 64-unit compute budget, so the layout is reproducible. Validate the combined logits against the single-split baseline at 8k, 32k, and 128k context on Qwen3.6-35B-A3B, then measure the decode tokens-per-second curve as context grows and confirm it flattens the way the A100 curve does.

The honest target is not a headline multiple. It is a decode rate that stops sagging as the conversation gets long. Today a long chat on the R9700 slows down token by token as the cache grows, because the attention step gets slower and the card gets emptier at the same time. Flash-Decoding decouples those two: the cache still grows, but the work to read it spreads across all 64 units instead of piling onto 32. The instruction path is the one zinc already wrote for wave32 flash attention. The missing dimension is the split.
