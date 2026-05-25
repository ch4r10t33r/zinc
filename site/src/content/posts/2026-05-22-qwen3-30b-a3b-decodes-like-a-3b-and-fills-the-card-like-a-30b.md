---
title: "Qwen3-30B-A3B decodes like a 3B and fills the card like a 30B"
date: "2026-05-22"
tags:
  - zinc
  - rdna4
  - amd
  - mixture-of-experts
  - moe
  - qwen3
  - qwen3-30b-a3b
  - decode
  - vram
  - memory-bandwidth
  - expert-routing
  - llm-inference
keywords:
  - Qwen3-30B-A3B local inference
  - MoE active parameters vs total parameters
  - mixture of experts VRAM requirement
  - Radeon AI PRO R9700 MoE decode
  - 8 of 128 experts per token routing
  - MoE decode bandwidth bound batch 1
  - sparse activation memory footprint
  - expert offload PCIe transfer bound
  - MoE decode tokens per second roofline
  - resident weights vs read per token 4-bit
excerpt: "The A3B in Qwen3-30B-A3B means 3.3 billion active parameters, and the story that travels with it is that you get a 30B model at 3B inference cost. Half of that is true. At 4-bit the 3.3B active weights are about 1.9 GB read per token, but the full 30.5B that has to stay resident is about 17 GB, and the card has to hold all 17. Worse, the 1.9 GB is a different 1.9 GB every token, because the router picks a fresh 8 of 128 experts per layer each step. The active-parameter count tells you how fast the model decodes and nothing about how big a card you need, and on a mixture of experts those are very different questions."
---

A single label does most of the reasoning when people talk about Qwen3-30B-A3B. The "A3B" means roughly three billion active parameters, and the story that travels with it is that you get the quality of a thirty billion parameter model while paying the inference cost of a three billion parameter one. Half of that is true. The half that is not true is the half that decides what hardware you need.

Here is the arithmetic the label hides. Qwen3-30B-A3B has [30.5 billion total parameters and 3.3 billion activated per token](https://huggingface.co/Qwen/Qwen3-30B-A3B), spread across 48 layers, with 128 experts in each mixture-of-experts block and 8 of them selected per token. At a typical 4-bit quantization the 3.3 billion active weights come to about 1.9 GB, and the full 30.5 billion come to about 17 GB. The card has to hold all 17. It only reads about 1.9 of them on any given token. The distance between those two numbers, and the fact that it is a *different* 1.9 GB each token, is the whole post.

This matters right now because Qwen3-30B-A3B is one of the most attractive models to run locally on a 32 GB card, and zinc targets exactly that hardware, the [Radeon AI PRO R9700](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html) with 64 compute units and 644.6 GB/s of GDDR6. People reach for the active-parameter count to decide whether the model fits and how fast it will go. It answers the second question well and the first question not at all.

## What the active-parameter count actually measures

A mixture of experts, or MoE, replaces the dense feed-forward block in each transformer layer with a set of smaller feed-forward networks called experts, plus a small router that decides which ones run. The [Hugging Face MoE explainer](https://huggingface.co/blog/moe) walks through the construction. Qwen3-30B-A3B keeps 128 experts per layer and routes each token to 8 of them. The other 120 sit idle for that token. So the matrix multiplies a single token performs add up to roughly a 3.3B dense model's worth of work, and on a card that is bandwidth-bound at decode, the weights those multiplies touch add up to roughly a 3.3B model's worth of bytes.

That is the good news, and it is real. As [we argued in the matrix-cores post](/blog/2026-04-30-rdna4-matrix-cores-sit-out-the-decode-loop), a batch-one decode step on the R9700 spends nearly all of its time streaming weights out of memory rather than doing arithmetic. If you read 1.9 GB per token instead of 17, your bandwidth-bound ceiling moves by almost an order of magnitude. At 644.6 GB/s, reading 1.9 GB is about 3 milliseconds, a ceiling near 340 tokens per second. Reading the full 17 GB of a dense 30B would be about 27 milliseconds, a ceiling near 38. The active-parameter count is an honest description of that gap.

What it does not describe is the 17 GB.

## The 3B is a different 3B every token

The number people forget is that the 1.9 GB an MoE reads is not a fixed 1.9 GB. The router runs per token, and the set of experts it picks changes from one token to the next. The [Mixtral paper](https://arxiv.org/abs/2401.04088) said this plainly when it brought the pattern into the open-weights world: "the selected experts can be different at each timestep," so "each token has access to 47B parameters, but only uses 13B active parameters during inference." Qwen3-30B-A3B is the same idea with more, finer experts, 128 per layer instead of 8 and eight active instead of two.

Walk one decode token through the stack. In each of the 48 layers the router scores 128 experts and keeps the top 8. That is 384 expert feed-forward blocks read for a single token, drawn from a resident pool of 128 times 48. On the next token the router scores again and, in general, lands on a different 384. No working set stays warm across tokens, because batch-one decode processes one token at a time and the routing is content-dependent. The card streams a fresh, scattered 1.9 GB out of the resident 17 every step.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-05-22-moe-active-vs-resident-expert-route.svg" alt="A two-panel data visualization on a deep plum background. The left panel, titled one token's route across the stack, is a grid that is 128 experts wide and 48 layers tall, representing every expert in Qwen3-30B-A3B. Most cells are a dim indigo. Eight cells in each row are lit, and the lit cells are scattered across the width with no clustering, colored on a gradient that runs from magenta in the top layers to amber in the bottom layers. The horizontal axis is labeled expert index 0 to 127 with ticks at 0, 32, 64, 96, and 128; the vertical axis marks layers L0, L12, L24, L36, and L47. A caption below reads 384 expert blocks read this token, 8 times 48 layers, scattered across 128 resident experts, and the next token relights a different 384. The right panel, titled 4-bit weights stored versus read per token, shows three model groups, each with a stored bar in indigo and a read bar in magenta. Dense 3.3B has a short stored bar at 1.9 gigabytes and a short read bar at 1.9 gigabytes, labeled about 350 tokens per second decode ceiling. Qwen3-30B-A3B has a long stored bar at 17.2 gigabytes and a short read bar at 1.9 gigabytes, labeled in amber reads like a 3B, stores like a 30B. Dense 30.5B has a long stored bar at 17.2 gigabytes and a long read bar at 17.2 gigabytes, labeled about 38 tokens per second decode ceiling. A footer notes the architecture comes from the Qwen3-30B-A3B model card and that the ceilings are 644.6 gigabytes per second divided by weight bytes read, upper bounds rather than measured throughput." loading="lazy" />
  <figcaption>Left: one decode token's expert route. Each row is a layer, each column one of the 128 experts, and the eight lit cells per row are the experts that token activated. The scatter is the point, and the next token would light a different set. Right: the consequence for memory. Qwen3-30B-A3B reads like a dense 3.3B and resides like a dense 30.5B.</figcaption>
</figure>

The left panel is one token's route. The lit cells are spread across the width rather than grouped, and the figure would look different for the next token. The right panel is what that costs in memory. A dense 3.3B model stores and reads the same small amount. A dense 30.5B model stores and reads the same large amount. Qwen3-30B-A3B reads like the small one and stores like the large one, which is exactly the corner of the design space that is easy to misjudge.

## Stored versus read, on a 32 GB card

Putting the three models side by side on the R9700 makes the trade concrete.

| Model | Resident weights, 4-bit | Read per token | Weight-bound decode ceiling | On a 32 GB R9700 |
| --- | ---: | ---: | ---: | --- |
| Dense 3.3B | ~1.9 GB | ~1.9 GB | ~340 tok/s | fits with room to spare |
| Qwen3-30B-A3B | ~17 GB | ~1.9 GB, a different 1.9 each token | ~340 tok/s | fits, ~15 GB left for KV cache |
| Dense 30.5B | ~17 GB | ~17 GB | ~38 tok/s | fits, but no decode headroom |

The middle row is the one worth buying hardware for. It reads like the top row, so it decodes nearly as fast, and it scores like the bottom row on quality. The catch is the resident column, which is identical to the bottom row. The "3B" in the name describes the read column and the compute. It says nothing about the resident column. Size a card from the active-parameter count and you will buy a card that cannot load the model.

The ceilings are upper bounds. They count only weight bandwidth and ignore attention, the KV cache read, and runtime overhead, so real decode lands well below them. The ratio between the rows is the part that holds.

## Why you cannot just keep the active experts on the card

The obvious objection is that if a token only reads 1.9 GB, the card should only need to hold 1.9 GB, with the inactive experts parked in system RAM and pulled in on demand. People build exactly this, and for throughput-oriented serving it can pay off. For a single local user at batch one it does not, and the reason is the latency wall that shows up everywhere else in local inference.

When the active experts live in system memory and get fetched to the GPU per token, the fetch is the work. Engines that offload experts report that the majority of inference time goes to host-to-device data movement while the expert math is a small slice, which means the path is bound by the PCIe transfer, not the compute. At batch one there is no second request in flight to hide that transfer behind. You would be trading the 644.6 GB/s of VRAM for the smaller, higher-latency budget of the PCIe link, on the exact 1.9 GB you need this token, and you would pay it again next token because the route changed. The [runtime-overhead numbers we measured](/blog/2026-05-12-the-runtime-below-vulkan-that-local-llms-needed) make the same point from the other side: in the small-transfer regime, round trips dominate and bytes are cheap.

So the experts stay resident. The Hugging Face writeup states the consequence as a flat requirement: although only some parameters run per token, "all parameters need to be loaded in RAM, so memory requirements are high," and for a low-throughput, low-VRAM setting "a dense model will be better." Qwen3-30B-A3B is the case where that requirement is survivable, because 17 GB fits a 32 GB card with margin. The requirement does not go away.

## What this means on the R9700

This is why Qwen3-30B-A3B is close to an ideal local model for the R9700, and why the reason is the opposite of the one the label suggests. The model is good on the card not because three billion parameters is a small number, but because 17 GB of resident weights leaves roughly 15 GB for everything else, and at long context the everything else is mostly the KV cache. A dense model with the same decode speed would be about 3B and far less capable. A dense model with the same capability would be about 30B, would read 17 GB per token, and would crawl at decode. The MoE sits in the seam between those two.

The decode-speed win and the resident-footprint cost then push on two different later problems. The small read per token is what makes long context usable at a real token rate, which runs straight into [the 16k crossover where KV reads start to outweigh the active weights](/blog/2026-04-27-the-16k-crossover-where-kv-reads-outweigh-active-weights-on-rdna4-decode). Past that point the 1.9 GB of expert weights is no longer the largest thing the card reads each token, the cache is, which is why [cutting the KV cache to fp8](/blog/2026-05-19-fp8-kv-cache-is-the-next-decode-bandwidth-cut-rdna4-already-has-the-wmma-for) earns more on an MoE than people expect. The scattered read, meanwhile, is a kernel and scheduling problem. Eight experts per layer means eight separate, data-dependent weight regions to gather, a router to evaluate first, and a grouped matmul that has to serve eight small problems without firing eight serial dispatches. That is the part of the MoE decode path zinc has to get right on RDNA4, and it is invisible if you only read the active-parameter count.

## What comes next

The label is not wrong. It is answering a different question than the one a local user is asking. "3B active" tells you how fast the model decodes and how much arithmetic it does. It does not tell you how big a card you need, and on a mixture of experts those answers diverge hard. Qwen3-30B-A3B reads like a 3B and fills the card like a 30B, and both halves carry weight: the first is why it is fast enough to use locally, the second is why it needs the 32 GB card at all.

What I am watching is the next, finer-grained MoE generation, where the active fraction keeps shrinking while the total keeps growing. Every step in that direction widens the gap between the read column and the resident column, which makes decode faster and the card requirement harder at the same time. The active-parameter count will keep getting more impressive and less informative. The number to set beside it is the resident footprint, because that is the one that decides whether the model runs on your hardware at all.
