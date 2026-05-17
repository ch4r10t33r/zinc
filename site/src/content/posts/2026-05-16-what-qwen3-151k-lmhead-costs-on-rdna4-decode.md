---
title: "What Qwen3's 151k LMHead costs on RDNA4 decode"
date: "2026-05-16"
tags:
  - zinc
  - rdna4
  - amd
  - vulkan
  - qwen3
  - qwen3-next
  - llama-cpp
  - quantization
  - lmhead
  - tied-embeddings
  - llm-inference
keywords:
  - Qwen3 LMHead cost RDNA4
  - tied word embeddings Qwen3 4B 8B
  - llama.cpp output.weight Q6_K Q4_K_M
  - 151936 vocab decode bandwidth
  - Radeon AI PRO R9700 LMHead matmul
  - tie_word_embeddings false Qwen3-Next
  - Qwen3-A3B decode bottleneck
  - output projection memory bandwidth local LLM
excerpt: "Qwen3 ties the input and output embedding weights at 0.6B, 1.7B, and 4B. At 8B it stops. The boundary is also where the LMHead matmul stops being a rounding error and starts costing about half a millisecond per decode token on a Radeon AI PRO R9700, and where the GGUF file size jumps by a separate 510 MB tensor that local engines have to dequantize once per token. After the wave32 attention fix landed, that is the next visible decode tax."
---

A Qwen3-4B GGUF and a Qwen3-8B GGUF look like the same model with twice the budget. They are not. The 4B carries one matrix where the 8B carries two: the input embedding and the LMHead are the same tensor in the 4B file and separate tensors in the 8B file. That is a deliberate Qwen3 design choice, and the cutoff between tied and untied is exactly between 4B and 8B.

On a 32 GB Radeon AI PRO R9700, with the wave32 scalar flash attention fix in, the LMHead matmul is now the largest single non-MoE memory read in the Qwen3 decode loop. It is about half a millisecond per decode token at Q6_K on a hidden size of 4096, and it is the matmul that has to read 510 MB to emit one logit row. The 4B does not pay this. The 8B does. Both share the same 151,936-token vocabulary.

This post is about what that costs on RDNA4, why Qwen3 draws the line where it does, and why the matmul shows up on a profile right after the attention path got fixed.

## What the LMHead actually is

The LMHead is the final matrix multiplication of a decoder-only transformer. It takes the last hidden state, shape `[hidden_size]` for a single decode step, and multiplies it by a weight matrix shape `[hidden_size, vocab_size]` to produce a logit per vocabulary token. The softmax over those logits is the next-token distribution. The sampler picks one.

Qwen3 has a vocabulary of 151,936 tokens. That number is fixed across every Qwen3 variant we have configs for, dense or MoE, 0.6B through 235B-A22B, and it has not changed in Qwen3-Next either. The vocabulary holds Chinese, English, and code tokens at a roughly 3:2 ratio of BPE merges. It is large for an open model. Llama 3 uses 128,256. GPT-2 used 50,257. Mistral 7B uses 32,768.

The size of the LMHead weight matrix is `hidden_size * vocab_size`. For Qwen3, that gives:

| Model | hidden_size | LMHead params | tie_word_embeddings | LMHead at FP16 | LMHead at Q6_K |
| --- | ---: | ---: | :---: | ---: | ---: |
| Qwen3-1.7B | 2048 | 311M | true | 622 MB | shared |
| Qwen3-4B | 2560 | 389M | true | 778 MB | shared |
| Qwen3-8B | 4096 | 622M | false | 1,245 MB | 510 MB |
| Qwen3-32B | 5120 | 778M | false | 1,557 MB | 638 MB |
| Qwen3-Next-80B-A3B | 2048 | 311M | false | 622 MB | 255 MB |
| Qwen3-235B-A22B | 4096 | 622M | false | 1,245 MB | 510 MB |

The `tie_word_embeddings` column comes straight from the published config files: [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B/blob/main/config.json) has it set to `true`, [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json) and [Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/main/config.json) both have it set to `false`. The Q6_K column reflects llama.cpp's Q4_K_M recipe, which keeps `token_embd.weight` and `output.weight` at Q6_K precision even when the rest of the body is Q4_K. The bits-per-weight numbers and recipe defaults are in the [llama.cpp quantize tool README](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md).

The "shared" entries are not zero. They are the same physical tensor as the embedding, used twice. The kernel still has to walk the rows once on input and walk the columns once on output, so the bandwidth on decode is the same as if the tensor were a separate `output.weight`. The savings are on disk and on host RAM, not on per-token decode bandwidth.

## Why tying stops at 4B

The weight-tying trick is older than Llama. The clean reference is [Press and Wolf, 2017](https://arxiv.org/abs/1608.05859), which showed that sharing the input embedding and the output projection of an RNN language model reduces parameters and slightly improves perplexity. GPT-2 tied. The original Llama and Llama 2 untied. Most modern instruct models above 7B parameters untie. Mistral, Llama 3, DeepSeek V3, and Qwen3 above 4B all sit on the same side of the line.

The case for tying is parameter efficiency. A 4B model tying its 389M-parameter LMHead onto the embedding gives back almost ten percent of its total parameter count to actual work in the transformer body. The case for untying is that the embedding wants to capture distributional similarity ("Tokyo" and "Osaka" should land near each other in vector space) while the LMHead wants to be a discriminative classifier ("which of these 151,936 tokens is next"). Forcing those two objectives onto the same matrix has a small cost in perplexity at every scale, and the absolute size of the cost shrinks for larger models because the body has more representational headroom to absorb the mismatch.

In practice the choice is also about quantization. The two passes use the matrix differently. The embedding pass is a single-row gather. The LMHead pass is a full matmul. Modern quantization formats penalize a full matmul more than a gather, because the matmul reads every byte, while the gather only reads one row of 4 KB. Untying lets the two tensors carry different quantization tiers without conflict. The same llama.cpp Q4_K_M recipe quietly keeps both tensors at Q6_K when they are separate, and aggregates them into one Q6_K tensor when they are tied.

The result, for Qwen3 specifically, is the cutoff visible in the config table. The 0.6B, 1.7B, and 4B variants tie. The 8B and every variant larger does not. The Qwen3-Next-80B-A3B model has hidden size 2048, the same as the 1.7B, and a separate `output.weight` anyway. Qwen3-Next decided the perplexity recovered by untying was worth a separate 622 MB of FP16 weight, even though the rest of the body is sparse.

## Where the decode budget goes

A decode step on Qwen3.6-35B-A3B at Q4_K_M on the Radeon AI PRO R9700 takes about 8.5 ms when zinc is running at the [published 117 tok/s decode rate](/blog/2026-05-09-how-we-made-amd-qwen-inference-faster-than-llama-cpp-in-six-weeks-on-the-radeon-ai-pro-r9700). The R9700 has 32 GB of GDDR6 on a 256-bit bus, with [memory bandwidth listed at 644.6 GB/s](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html). At 117 tok/s the engine is reading roughly 5.5 GB per second per decoded token. That is a memory-bound regime, and the LMHead pulls its share.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-05-16-qwen3-lmhead-decode-budget.svg" alt="A two-panel data visualization on a cream-colored background. The upper panel is a horizontal stacked bar titled where one decode token's 8.5 milliseconds go on a Radeon AI PRO R9700 at 117 tokens per second on Qwen3.6-35B-A3B at Q4 K M. The bar runs from 0 milliseconds to 8.5 milliseconds across the page. Reading left to right the segments are: active MoE expert weights at 2.55 milliseconds in deep slate blue, KV cache reads at 1.65 milliseconds at 16k context in medium teal, attention compute at 1.20 milliseconds in lighter teal, FFN shared expert plus router at 0.95 milliseconds in green, RMS norms and residual paths at 0.55 milliseconds in pale green, LMHead matmul at 0.40 milliseconds highlighted in solid orange with a thick black outline, sampler and softmax at 0.30 milliseconds in pale orange, and other overhead at 0.85 milliseconds in light gray. The LMHead segment carries a callout that reads 4.7 percent of decode budget, 255 megabytes read at Q6 K on hidden 2048. The lower panel is a small horizontal bar chart titled LMHead size at Q6 K versus hidden size across Qwen3 variants. Five horizontal bars are stacked. From top: Qwen3-1.7B and Qwen3-Next-80B-A3B at 255 megabytes drawn in pale blue with a label hidden 2048. Qwen3-4B at 319 megabytes drawn in pale blue with a label hidden 2560. Below a thin gray dividing line labeled tying cutoff sits Qwen3-8B and Qwen3-235B-A22B at 510 megabytes drawn in solid orange with a label hidden 4096, and Qwen3-32B at 638 megabytes drawn in deeper orange with a label hidden 5120. A footer note reads the upper four bars use tied weight when tying is enabled in the dense Qwen3 0.6B 1.7B and 4B variants, while every variant at or above 8B carries a separate output dot weight tensor. The chart credits llama.cpp Q4 K M recipe keeping token embd dot weight and output dot weight at Q6 K and the Hugging Face Qwen3 config files for the tie word embeddings flag." loading="lazy" />
  <figcaption>The LMHead at Q6_K on the Qwen3-Next hidden size of 2048 reads 255 MB and lands at about 0.40 ms per decode step on the R9700. The lower panel shows the same Q6_K calculation across every Qwen3 size, with the tied-versus-untied split between 4B and 8B drawn explicitly. Numbers for the upper bar are approximate decode-budget shares for a 16k-context Qwen3.6-A3B trace; the 8.5 ms total reflects the published 117 tok/s figure.</figcaption>
</figure>

The thing to notice in the upper bar is that the LMHead is the smallest of the named segments by time, but it is the only one outside the MoE active path that touches more than 200 MB of weight per token. Active expert weights are bigger only because eight experts of intermediate size 768 sit on top of one shared expert per layer. The KV reads scale with context length but stop scaling once the context is full at 16k. The attention compute is dense work on a small batch and is set by the wave32 FA kernel. The LMHead, on this workload, is roughly 5 percent of the decode budget. It is not free.

The thing to notice in the lower bar is that the only Qwen3 sizes where the LMHead is below 320 MB at Q6_K are the tied ones. The 8B and every variant above it spends at least 510 MB on a separate `output.weight`. That weight has to come off VRAM once per decode token. The wave32 attention fix from [last Wednesday's post](/blog/2026-05-11-the-wave32-commit-that-closes-rdna4-long-context-flash-attention-gap) closed the previous large gap on the same loop. The LMHead is the next-largest single-tensor read.

## Why it shows up now and not before

Until the wave32 commit, the scalar flash attention kernel on RDNA4 was leaving half its SIMD lanes idle and reading the KV cache twice as often as needed. The attention path dominated the profile. Anything else looked rounded off.

After wave32, the attention path drops by enough that the next bottleneck reveals itself. Three candidates compete for that slot: the MoE expert dispatch, the LMHead matmul, and the per-decode RMSNorm pair around the residual. The dispatch is a structural problem that needs queue-level rework, which is the topic of [the ZINC_RT post from May 12](/blog/2026-05-12-the-runtime-below-vulkan-that-local-llms-needed). The RMSNorms are small and already fused. The LMHead is the candidate with the simplest win condition: one big matmul, one quantization choice, one specialization constant.

The reason it is not visible until now is also why it is visible only on bandwidth-bound decode. On prefill, the LMHead runs once per chunk and amortizes across the prompt. On batch-size-one decode at temperature zero with a single chat tab open, the LMHead runs once per token, reads the full output.weight tensor, and contributes the full 0.4 to 0.8 ms slice you see in the budget bar.

## What changes if the matmul gets cheaper

Two levers are worth pulling.

The first is quantization. The Q6_K output.weight in a standard Q4_K_M file can be requantized to Q5_K_M with a measurable but small perplexity cost. That saves about 100 MB on the read and about 80 microseconds per decode step. The opposite direction is the `--leave-output-tensor` flag in the [llama-quantize tool](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md), which keeps the output at FP16 to avoid stacked requantization error. That doubles the read to about 1.0 GB on the 8B and 1.25 GB on the 32B, which would make the LMHead the second-largest non-MoE read in a Q4 model. The default Q6_K recipe sits in the middle for a reason.

The second is shader specialization, and this is the one we are working on for the Vulkan backend. The LMHead is the only dense matmul in the decode loop that has a single fixed shape: `[1, hidden_size] * [hidden_size, vocab_size]`. No batching, no padding tail, no head dimension. A compute shader specialized on `hidden_size = 2048` (for Qwen3-Next and the A3B family) and `vocab_size = 151936` can pre-split the work into tiles that match the LDS budget on gfx1201 with no runtime shape branching. The same trick [specialization constants unlocked for the dmmv kernels in April](/blog/2026-04-23-vulkan-specialization-constants-unlock-rdna4-dmmv-variants) applies here. The expected win is a higher utilized fraction of the 644.6 GB/s peak, not a smaller read.

## The tradeoff

The LMHead is one matmul. It is not where the next ten percent of decode performance is hiding. It is where about three to five percent is, depending on hidden size and context. On a 117 tok/s baseline that is somewhere between three and six tok/s. After three weeks of attention work, that is not nothing.

The interesting part is the architectural lesson, not the kernel one. Qwen3 ties at 4B and unties at 8B because the perplexity hit of tying gets large enough to matter once the body of the model has enough capacity to use a separate classifier head. Qwen3-Next ties nothing despite running at the same hidden size as the 1.7B, because the sparse model decided the small dense matrix was worth its own quantization tier. The local engine inherits both decisions when it loads the GGUF.

The other tradeoff is the one that disappears entirely when the kernel gets specialized: any extra branching inside the LMHead matmul costs more than the matmul saves. A dispatch with three different vocab-size paths on the same shader is slower than three dispatches with one path each. This is the same lesson [the dmmv specialization post](/blog/2026-04-23-vulkan-specialization-constants-unlock-rdna4-dmmv-variants) ended on, applied to a different stage of the pipeline. RDNA4 wants the shader to know its shape at compile time. The LMHead is the cleanest case for that, because the shape is set by the model and never changes during a session.

## What comes next

The LMHead moves up the profile list on the next pass. The work has three parts: a specialized compute shader for the fixed `[1, hidden] * [hidden, 151936]` shape with one variant per `hidden` value across Qwen3-A3B, Qwen3-8B, and Qwen3-32B; a specialization-constant build for each vocabulary so the kernel does not branch on the inner length; and a sanity check against the deterministic flag from [the batch-invariance post](/blog/2026-05-14-temperature-zero-is-not-deterministic-for-local-qwen3-on-rdna4-yet), because the LMHead has to keep the same reduction order regardless of how chunked prefill landed before it.

None of this is exotic. That is also why it is sitting there at three to five percent of every Qwen3 decode token, waiting to be picked up. Qwen3's architects made the tying call deliberately at 4B. The local engine has to do the same kind of deliberate bookkeeping for the kernel that runs on top of it.
