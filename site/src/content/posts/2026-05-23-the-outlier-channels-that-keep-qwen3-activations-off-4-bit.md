---
title: "The outlier channels that keep Qwen3 activations off 4-bit"
date: "2026-05-23"
tags:
  - zinc
  - rdna4
  - amd
  - quantization
  - activation-quantization
  - outlier-features
  - massive-activations
  - int8
  - q8-1
  - awq
  - smoothquant
  - qwen3
  - llm-inference
keywords:
  - activation quantization outlier features
  - why activations are harder to quantize than weights
  - W4A16 vs W8A8 vs W4A4 accuracy
  - LLM.int8 emergent outlier features
  - SmoothQuant migrate quantization difficulty
  - AWQ salient weight channels activation aware
  - massive activations attention sink
  - q8_1 per-block scale RDNA4 prefill
  - 4-bit activation quantization perplexity cliff
  - Radeon AI PRO R9700 integer matmul
excerpt: "You can quantize Qwen3's weights to 4 bits and lose a fraction of a perplexity point. Quantize its activations to 4 bits the same naive way and the model falls apart. The reason lies in the activations themselves: a handful of channels carry values tens to thousands of times larger than the rest, and a single quantization scale cannot hold both the giants and everyone else. Those few channels are only about 0.1 percent of the features, but zeroing them degrades perplexity by hundreds of percent. That asymmetry is why zinc's prefill path quantizes activations to 8-bit q8_1 and not lower, why the clever quantization tricks all push difficulty onto the weights, and why the activation floor on RDNA4 sits at 8 bits while the weight floor sits at 4."
---

You can take Qwen3's weights down to 4 bits and barely notice. A good weight-only scheme costs a fraction of a perplexity point, and the model that comes out the other side answers almost exactly like the 16-bit original. Try the same move on the activations, the numbers flowing between the layers, and the model comes apart. Naive 4-bit activations turn a coherent model into one that produces garbage.

That asymmetry is strange the first time you see it. Weights and activations are both just tensors of floating-point numbers that get multiplied together. If 4 bits is enough to describe one side of the matmul, why is it nowhere near enough for the other? The answer is the whole reason zinc's prefill path quantizes activations to 8-bit and not 4-bit, and it is worth being precise about, because the same property shows up again in attention sinks and in the kernels we have to write on RDNA4.

The short version: weights are well-behaved and activations are not. A few activation channels carry values tens to thousands of times larger than everything around them, and a single quantization scale cannot represent the giants and the rest of the field at the same time. Those few channels are a rounding error in count and the entire game in importance.

## Why this matters for a local engine right now

zinc runs Qwen3 on a [Radeon AI PRO R9700](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html), a 32 GB card, and on that hardware quantization is not optional. The weights are stored at 4 bits because that is what makes a 30B-class model fit with room left for the KV cache. The activations, during prefill, get quantized too, but only down to 8-bit, in the q8_1 format that feeds the integer matmul path [we described earlier](/blog/2026-04-19-why-q8-1-activations-are-the-next-rdna4-prefill-unlock).

That 4-versus-8 split looks arbitrary on a spec sheet and it is not. It is the engine respecting a numerical property of the model. If you do not understand why the activations stop at 8 bits, the obvious next optimization, push activations to 4 bits and double the integer-matmul throughput again, looks like free money. It is not free, and the bill comes due as a model that no longer works.

## The activations have a few enormous channels

The property at the center of all this got named precisely in 2022. In [LLM.int8()](https://arxiv.org/abs/2208.07339), Dettmers and coauthors documented what they called emergent outlier features: specific feature dimensions in the hidden state that, past a certain model scale, start carrying values far larger than the rest. They are systematic, not random noise. By around 6.7 billion parameters the same dimensions light up across essentially every layer, and those dimensions dominate the model's predictions.

The numbers are what make it concrete. These outlier dimensions are about 0.1 percent of all features. Set them to zero and the top-1 attention softmax probability mass drops by more than 20 percent, and validation perplexity degrades by 600 to 1000 percent. One feature in a thousand, and removing it costs you the model. Later work pushed the description further. The [Massive Activations](https://arxiv.org/abs/2402.17762) paper found activations up to roughly 100,000 times larger than the median, concentrated in a few fixed dimensions, with values that barely change regardless of the input. They behave less like data and more like a hardwired bias the model installs and then leans on.

That last detail connects straight back to a post from earlier this month. Those massive activations are the mechanism behind [attention sinks](/blog/2026-05-02-attention-sinks-the-four-kv-tokens-local-long-context-cannot-evict), the handful of tokens a long-context model refuses to evict. The same few enormous numbers that make attention pool onto the first tokens are the ones that make the activation tensor impossible to quantize naively. It is one phenomenon wearing two costumes.

## One scale cannot hold a giant and a field

Here is the mechanism, in the plainest terms I can manage. To quantize a tensor to 8-bit integers you pick a scale, divide every value by it, and round to one of 256 levels. The scale has to be large enough that the biggest value in the group does not overflow. So the largest magnitude in the group sets the scale for everyone.

Now put one outlier channel that is 100 times the median into that group. The scale stretches to cover the outlier, and the ordinary values, which are 100 times smaller, now land in the bottom 1 percent of the range. With 256 levels, the bottom 1 percent is two or three levels. Almost every real value in the tensor collapses onto the same two or three integers, and the information they carried is gone. The outlier is represented fine. Everything else is destroyed to make room for it.

At 8 bits this is survivable because real engines do not use one scale for the whole tensor. The q8_1 format quantizes in blocks of 32 values, each block with its own scale. A block-local scale means an outlier only wrecks its own block of 32, not the entire row, and 256 levels leaves enough headroom that even a stretched block keeps a few bits of real signal. Drop to 4-bit activations and you have 16 levels per block. One 100x outlier in a block of 16 leaves the other 15 values fighting over the bottom one or two of those 16 levels. There is no headroom left to give away. That is the cliff: 8-bit blocks can absorb an outlier, 4-bit blocks cannot.

[SmoothQuant](https://arxiv.org/abs/2211.10438) states the asymmetry in one line: weights are easy to quantize while activations are not. Weights, across a layer, tend to sit in a tight, roughly bell-shaped range with no channel screaming above the others. A single scale fits them. Activations have the spikes. That is the entire difference, and every practical fix is built around it.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-05-23-activation-outlier-channels-quant-cliff.svg" alt="A two-panel data visualization on a warm off-white paper background. The left panel, titled per-channel activation magnitude across one hidden state, is a bar chart along a horizontal axis labeled hidden dimension channel. Most channels are a low flat field of small teal bars near the baseline, forming a quiet sea. Three channels stand far above the rest as tall vermilion spikes, annotated as a few fixed channels run tens to one hundred thousand times the median, citing LLM.int8 and Massive Activations. A dashed bracket spans the spikes labeled outliers, about 0.1 percent of features. A small note reads weights, by contrast, fill a tight bell with no spike, so one scale fits them. The right panel, titled the cost of quantizing as if the spikes were not there, shows two horizontal bars. The top vermilion bar, zero the 0.1 percent outlier features, extends far to the right and is labeled perplexity plus 600 to 1000 percent and top-1 attention mass minus 20 percent. The bottom teal bar, zero a random 0.1 percent of features, is a tiny stub labeled negligible. A footer notes magnitudes are illustrative of the documented pattern while the degradation figures are measured in LLM.int8 on transformer language models." loading="lazy" />
  <figcaption>Left: the activation pattern reported across large language models. A flat field of ordinary values with a few fixed channels orders of magnitude larger; the bar heights are illustrative of the shape, the ratios are from LLM.int8 and Massive Activations. Right: the measured consequence. Zeroing the 0.1 percent of outlier features wrecks the model, while zeroing a random 0.1 percent does almost nothing. The outliers are rare and load-bearing, which is exactly what makes a single low-bit scale fail.</figcaption>
</figure>

The left panel is the shape of the problem and the right panel is the proof that the shape matters. The point of the right panel is the gap between the two bars: same number of features removed, wildly different cost, because the 0.1 percent that matters is the spiky 0.1 percent. A quantizer that treats all features equally spends its precision budget on the field and starves the channels that actually carry the model.

## Every fix pushes the difficulty onto the weights

Once you accept that activations have load-bearing outliers, the published solutions all rhyme. They keep the outliers in high precision or move the hard part somewhere easier, and the somewhere easier is always the weights.

LLM.int8() does the literal version. It splits the matmul: the roughly 0.1 percent of outlier dimensions run in 16-bit, the other 99.9 percent run in 8-bit, and the two results are summed. It works and it preserves full accuracy up to 175B parameters, but it pays for it with a mixed-precision decomposition that breaks the clean shape of a single integer matmul. For a latency-bound local decode, splitting every matmul into a fat 8-bit part and a skinny 16-bit part is not where you want to be.

SmoothQuant is the more elegant move. Since the matmul is activations times weights, you can divide the activations by a per-channel factor and multiply the weights by the same factor, and the product is unchanged. So it scales the outlier channels down before quantizing and folds the inverse into the weights offline. The spikes get smoothed into the activation, the weights absorb the variance, and both sides become quantizable to 8-bit. The difficulty did not disappear. It moved to the side that can take it.

[AWQ](https://arxiv.org/abs/2306.00978) is the one zinc's weight path cares about most, and it closes the loop in a satisfying way. It quantizes weights to 4-bit, weight-only, and its key finding is that protecting roughly the top 1 percent of weight channels recovers almost all the accuracy lost to quantization. The twist is how it chooses which channels to protect: it reads the activation distribution, not the weights. The weight channels that line up with large activation magnitudes are the ones it keeps safe. The activation outliers, which made the activations impossible to quantize, turn out to also be the map of which weights you must not break. The same spikes that block low-bit activations tell you how to do low-bit weights well.

## The two floors, side by side

Putting the standard schemes in one place makes the asymmetry concrete. The labels follow the usual convention: W is weight bits, A is activation bits.

| Scheme | Weights | Activations | Reported outcome |
| --- | ---: | ---: | --- |
| FP16 baseline | 16 | 16 | reference quality |
| AWQ (weight-only) | 4 | 16 | near-lossless, protects ~1% salient channels |
| SmoothQuant | 8 | 8 | negligible loss, difficulty migrated to weights |
| Naive per-tensor INT8 | 8 | 8 | large degradation past ~6.7B without outlier handling |
| Naive 4-bit activations | 4 | 4 | model breaks; no headroom for the outlier blocks |

Read the activation column from top to bottom and the floor is obvious. With care, weights go to 4 bits and stay there. Activations go to 8 with block scales or smoothing, and below 8 there is no scheme in this table that holds. The weight floor is 4, the activation floor is 8, and they are different for the single reason that runs through every row: the activations have the outliers and the weights do not.

## What this means on RDNA4 and where it bites

This is why zinc's prefill path stops at q8_1 for activations even though the integer-matmul math would happily run faster on narrower operands. The question of whether RDNA4's matrix units could consume 4-bit activations is separate from the question of whether Qwen3's activations survive 4 bits, and the second question is the binding one. The hardware is not the wall. The model is. You can build the kernel; you cannot build a 4-bit activation tensor that still has the outlier channels in it.

The weight side is where the headroom lives, and it is already being spent. The 4-bit weights that let a 30B-class model fit on the card are the same lever from the [MoE post](/blog/2026-05-22-qwen3-30b-a3b-decodes-like-a-3b-and-fills-the-card-like-a-30b), and AWQ-style activation-aware protection is how you take them to 4 bits without losing quality. Decode is bandwidth-bound, as the [matrix-cores post argued](/blog/2026-04-30-rdna4-matrix-cores-sit-out-the-decode-loop), so every bit shaved off the weights buys real tokens per second. Bits shaved off the activations would help prefill throughput, but that is precisely the door the outliers keep shut.

## What comes next

The activation floor is not a law of nature, it is a property of how today's models are trained. There is active work on knocking it down. Microscaling formats give each small block its own scale at very low bit widths, which is the block-scale idea pushed harder. And the Massive Activations work suggests a cleaner fix at the source: if you give the model an explicit attention bias during training, it no longer needs to grow the giant activations that create the sinks, and the activation tensor that comes out is far easier to quantize. Remove the reason the outliers exist and the 8-bit floor moves.

Until a model ships without the spikes, the rule holds and it is worth stating plainly. Quantize the weights aggressively, protect the channels the activations point to, and leave the activations at 8 bits. The 4-versus-8 split in zinc is not a tuning artifact. It is the shape of the model showing through the engine, and the few outlier channels that set the activation floor are the same few that make long context sticky and attention pool onto the sinks. One property, paid for in three different places.
