---
title: "Tensor parallelism on two R9700s pays for the Infinity Fabric link RDNA4 left out"
date: "2026-05-21"
tags:
  - zinc
  - rdna4
  - amd
  - multi-gpu
  - tensor-parallelism
  - pipeline-parallelism
  - pcie
  - infinity-fabric
  - xgmi
  - all-reduce
  - llama-70b
  - llm-inference
keywords:
  - tensor parallelism RDNA4
  - two R9700 LLM inference
  - AMD Infinity Fabric XGMI consumer card
  - PCIe 5.0 all-reduce latency
  - pipeline parallelism local 70B
  - Radeon AI PRO R9700 multi-GPU
  - Megatron-LM two all-reduce per layer
  - tensor vs pipeline parallel local inference
  - decode bandwidth bound weight reads
  - GPU-to-GPU interconnect consumer AMD
excerpt: "A dense 70B at 4-bit needs about 40 GB, so it does not fit on one 32 GB Radeon AI PRO R9700 and has to be split across two cards. Tensor parallelism is the only split that makes a single local chat decode faster, because each card then reads half the weights, but it fires two all-reduces per layer, 160 cross-card synchronizations per token on an 80-layer model. On an Instinct card those run over a 128 GB/s Infinity Fabric link built for exactly this. On two R9700s they run over PCIe, the one link AMD did not give the consumer card, and that is what decides whether a tensor split or a layer split is the right default."
---

A dense 70B model in 4-bit weighs about 40 GB. A [Radeon AI PRO R9700](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html) holds 32. The arithmetic is unforgiving: the model does not fit on one card, so to run a Llama-3.1-70B-class model locally you buy a second R9700 and split the weights across both. The moment you do that, you have to answer a question single-card inference never asked. How do you cut the model in half, and what does the cut cost when the two cards can only talk to each other over PCIe?

This is the part of local multi-GPU that the spec sheet hides. Two R9700s give you 64 GB of VRAM and roughly 1.28 TB/s of aggregate memory bandwidth, which reads like a clean doubling. The catch is the wire between the cards. On AMD's data-center accelerators the two GPUs would be joined by Infinity Fabric, a dedicated GPU-to-GPU link. On the consumer and workstation parts they are joined by the PCIe slot, and PCIe is a different kind of link with a different cost model. The split you pick lives or dies on that difference.

## Two ways to cut a model in half

There are two standard ways to spread one model over two GPUs, and they cross the link in completely different patterns.

Tensor parallelism, the approach [Megatron-LM introduced](https://arxiv.org/abs/1909.08053), slices every weight matrix down its columns or rows and puts half on each card. Both GPUs run every layer together, each computing its half of the matmul, and then they exchange and sum partial results so the next operation sees the full activation. That exchange is an all-reduce, a collective both cards must reach at the same time. Megatron's layout is careful to need only two of them per transformer layer: one after the attention block and one after the MLP.

Pipeline parallelism cuts the other way. Card A gets the first 40 layers, card B gets the last 40, and a token walks through A, crosses the link once, and finishes on B. There is no per-layer collective. The activation crosses the boundary a single time on its way through the stack.

Two all-reduces per layer versus one handoff per stack. On an 80-layer model that is 160 cross-card events per token against one. On a single card the difference is invisible because there is no link to cross. On two cards it is the whole decision.

## The all-reduce is tiny in bytes and expensive in trips

It is tempting to reach for bandwidth math here, and bandwidth math says the all-reduce is free. At decode time the model processes one token, so the activation that gets reduced is a single hidden vector. For a 70B with a hidden size of 8192 in FP16 that is 8192 times 2 bytes, about 16 KB. Multiply by 160 all-reduces and a full decode token moves roughly 2.5 MB across the link. At [PCIe 5.0 x16's 63 GB/s per direction](https://en.wikipedia.org/wiki/PCI_Express) that is around 40 microseconds of pure transfer, if it were one transfer.

It is not one transfer. It is 160 of them, and each one is a synchronization, not just a copy. Both cards have to arrive at the same point in the layer, hand their partial sums to each other, wait for the combined result, and only then move on. The bytes are nothing; the round trips are everything. When [we measured the runtime overhead of our Vulkan backend](/blog/2026-05-18-inside-the-decision-to-write-our-own-gpu-runtime-for-local-llm-inference), a single submit-and-wait round trip on RADV came in around 33 microseconds. A cross-device all-reduce needs at least a sync of that order on each side. A hundred and sixty of them, in the worst case, is several milliseconds of a decode token spent waiting on the link rather than doing arithmetic.

That is the shape of the tensor-parallel tax on PCIe: latency-bound, and scaling with the layer count rather than the model's byte size. A deeper model pays more even though each all-reduce stays 16 KB.

## Why tensor parallelism is still the split that helps a single user

Given that tax, the obvious move is to reach for pipeline parallelism, which crosses the link once. But pipeline parallelism has a quiet problem of its own for a single local user: it does not make one chat any faster.

Local decode is bandwidth-bound on weight reads. As [the matrix-cores post argued](/blog/2026-04-30-rdna4-matrix-cores-sit-out-the-decode-loop), the dominant cost of a batch-one decode step is streaming the active weights out of memory, and that work sits far on the bandwidth side of the roofline. Pipeline parallelism does not change how many weight bytes a single token reads. The token still traverses all 80 layers; it just reads layers 0 through 39 from card A and 40 through 79 from card B, in sequence. While card A works, card B is idle, and vice versa. For one request in flight you get the model to fit and nothing more. The throughput only appears when a second chat slot fills the bubble, which is the same multi-slot regime [the chunked-prefill post described](/blog/2026-05-13-how-chunk-size-decides-first-token-latency-on-long-local-qwen3-prompts).

Tensor parallelism is the opposite. Because each card holds half of every weight matrix, each card reads half the weight bytes per token. The bandwidth-bound part of the step roughly halves, and the single token sees close to the combined 1.28 TB/s of both cards. That is the only split that makes one person's chat decode faster on two GPUs. It is also, inconveniently, the split whose communication scales with the layer count. The technique that speeds up a single local stream is the one that fires 160 syncs to do it.

## The link the R9700 does not have

This tension is not fundamental to tensor parallelism. It is fundamental to crossing it over PCIe. The all-reduce storm is exactly the workload AMD's data-center interconnect was built to swallow, and it is the piece the consumer and workstation cards do without.

On an Instinct MI300X, [the ROCm architecture docs spell it out](https://rocm.docs.amd.com/en/docs-6.4.0/conceptual/gpu-arch/mi300.html): each accelerator attaches to the host over a PCIe Gen 5 x16 link, but the GPUs talk to each other over seven dedicated Infinity Fabric links that form a fully connected eight-GPU hive. Each of those links runs at 128 GB/s, with latency far below a PCIe round trip, and the fabric exposes a shared address space so a collective does not bounce through host-managed staging. That is the wire tensor parallelism was designed around.

The R9700 has none of it. It has a PCIe Gen 5 x16 slot, which AMD markets for multi-GPU scalability, and that slot is the only path between two cards. PCIe is a host interconnect doing double duty as a GPU interconnect. It carries the bytes fine. What it does not have is the per-collective latency or the fabric semantics that make 160 small all-reduces per token cheap. The gap between the two cards is not a VRAM gap or a bandwidth gap. It is the GPU-to-GPU link that ships on CDNA and not on RDNA4.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-05-21-tensor-vs-pipeline-pcie-crossing-map.svg" alt="A data visualization on a deep navy background titled where each split crosses the link, per decode token, on an 80-layer 70B. Two horizontal layer rails run left to right from layer 0 to layer 79. The upper rail, labeled tensor parallel, is covered in a dense comb of 160 short amber tick marks, two at every layer, with a label reading 160 all-reduces per token, about 16 kilobytes each, every tick is a cross-card synchronization. The lower rail, labeled pipeline parallel, is mostly empty with a single tall teal tick at the midpoint between layer 39 and layer 40, labeled 1 handoff per token, about 16 kilobytes, one boundary crossing. To the right is a panel titled the link between the two cards comparing two bars. An amber bar labeled PCIe 5.0 x16, the link two R9700s share, reads 63 gigabytes per second per direction and high per-collective latency. A taller teal bar labeled Instinct Infinity Fabric, the link they do not have, reads 128 gigabytes per second per link, seven links, fully connected, low latency. A footer credits Megatron-LM, the AMD ROCm MI300 architecture docs, and the PCI Express specification.">
  <figcaption>Per decode token on an 80-layer 70B, a tensor split crosses the inter-card link 160 times and a pipeline split crosses it once. The right panel is why that matters: the all-reduce storm was designed for the 128 GB/s Infinity Fabric on Instinct cards, and on two R9700s it runs over the shared PCIe 5.0 slot instead.</figcaption>
</figure>

The rail on top is the cost and the rail on the bottom is the alternative. The comb of amber ticks is the tensor-parallel all-reduce pattern, two per layer, all of them landing on a link that was not built for them. The single teal tick is the pipeline handoff. The point of the right panel is that neither split changes the bytes much; what changes is how many times the cards have to stop and agree, and on PCIe each of those stops is expensive.

The two regimes line up like this for a single decode token.

| Split | Cross-card events per token | Bytes per event | Single-stream decode speedup | What it actually buys |
| --- | ---: | ---: | --- | --- |
| Tensor parallel | 160 (2 per layer) | ~16 KB | ~2x, each card reads half the weights | A faster single chat, paid for in per-layer sync latency |
| Pipeline parallel | 1 (one stage boundary) | ~16 KB | none | The model fits; throughput appears only with a second slot |

The table reads as a straight trade. Tensor parallelism is the only column with a single-stream speedup, and it is also the only column whose communication is a per-layer event rather than a one-time one. On a fabric that trade is no contest. On PCIe it is a real fight, and which side wins depends on how cheap your engine can make a cross-card collective.

## Prefill flips the regime but not the verdict

Decode makes the all-reduce look small because the activation is one token wide. Prefill does the opposite. Processing a 2048-token chunk means the reduced activation is 2048 by 8192, about 33.5 MB in FP16 per all-reduce, and 160 of them move on the order of 5 GB across the link for one chunk. At 63 GB/s that is something like 85 milliseconds of communication competing with the prefill compute, and now the cost is bandwidth, not latency.

So tensor parallelism pays at both ends, for different reasons. Decode is gated by the number of synchronizations and prefill by the bytes those synchronizations carry. Infinity Fabric helps both, with lower latency for the decode trips and more than double the bandwidth for the prefill payloads. PCIe helps neither as much, which is the whole reason the data-center cards carry a second interconnect at all.

## What we are going to do on two R9700s

The honest conclusion is that on two R9700s you split a 70B to make it fit, not to make it fly, and the default split should be the cheap one. Pipeline or layer-level partitioning crosses PCIe a handful of times, keeps a single chat at roughly single-card decode latency, and turns into real throughput the moment a second conversation is open. For the common local case, one person with one or two chat tabs, that is the right starting point.

Tensor parallelism stays on the table for the case it uniquely serves, a single long chat that needs to decode faster than one card's bandwidth allows. But it only nets out if the per-layer all-reduce can be hidden, and on PCIe that means fusing the collective into the compute, overlapping it with the next matmul, and keeping the engine's submission granularity well under the 33-microsecond round trip we measure today. That is a runtime problem more than a kernel problem, and it is one more reason the work below the graphics API matters.

What I am watching is whether a tuned peer-to-peer all-reduce over PCIe Gen 5, with the cards mapped into each other's address space, can get the per-collective cost low enough that the tensor split's bandwidth win survives the trip. If it can, two R9700s become a genuine 70B decode machine for a single user. If it cannot, the missing link stays the deciding factor, and the verdict holds: layer split to fit, tensor split only when you can pay the fabric tax without the fabric.
