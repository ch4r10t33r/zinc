---
title: "Static YaRN buys Qwen3 128k context by taxing every short prompt"
date: "2026-05-24"
tags:
  - zinc
  - rdna4
  - amd
  - qwen3
  - long-context
  - yarn
  - rope
  - rope-scaling
  - context-extension
  - position-interpolation
  - kv-cache
  - llm-inference
keywords:
  - Qwen3 YaRN 32768 to 131072 context
  - static YaRN degrades short prompts
  - dynamic YaRN per-request rope scaling
  - rope_scaling factor 4.0 original_max_position_embeddings
  - NTK-by-parts interpolation high frequency dimensions
  - YaRN efficient context window extension
  - position interpolation extend context window
  - length-aware rope scaling local inference
  - Qwen3 128k context local engine
  - Radeon AI PRO R9700 long context decode
excerpt: "Qwen3 is trained to 32,768 tokens, and the way you get to 131,072 is a single config field: a YaRN rope_scaling block with factor 4.0. Flip it and the model reaches 128k. The catch is that open-source frameworks implement static YaRN, so that factor of 4 rescales the position math for every request, including the 1,500-token chat that never needed it. Qwen's own documentation says the quiet part out loud: do not enable YaRN unless you are processing long contexts, because it can degrade short text. That makes context length a per-request decision rather than a launch flag, and it is the difference between an engine that serves both the quick question and the 120k-token document well and one that taxes every prompt to keep the rare long one alive."
---

Qwen3 is trained on sequences up to 32,768 tokens. The way you get it to 131,072, the number on the model card, is not a different model or a longer training run. It is a small `rope_scaling` block in `config.json` that turns on [YaRN](https://arxiv.org/abs/2309.00071), a rope-scaling method, with `factor` set to 4.0. Add the block, reload, and the model answers questions about a 120k-token codebase.

Here is the part the model card states plainly and most people skip. That `factor` of 4 is static. It rescales the position math for every request the model ever sees, including the 1,500-token chat turn that never came close to the native window. The [Qwen3-8B model card](https://huggingface.co/Qwen/Qwen3-8B) puts it in one sentence: static YaRN means "the scaling factor remains constant regardless of input length, potentially impacting performance on shorter texts," and "if the average context length does not exceed 32,768 tokens, we do not recommend enabling YaRN in this scenario, as it may potentially degrade model performance."

So reaching 128k is not really a switch you flip once. The factor rides along on every request afterward, paid in full whether the prompt needs it or not. An engine that bakes static YaRN into the loaded config has quietly made every short interaction a little worse to keep the rare long document working. This post is about why that happens at the level of the rotation math, and why a local engine should treat context length as a per-request input instead.

## Why this matters on a 32 GB card

Long context is the reason to run a model like Qwen3 on a [Radeon AI PRO R9700](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html). The 32 GB of VRAM is what lets the KV cache grow far enough to hold a real document, and most of what we have written this month, from [the 16k crossover where KV reads start to dominate decode](/blog/2026-04-27-the-16k-crossover-where-kv-reads-outweigh-active-weights-on-rdna4-decode) to [cutting the cache to fp8](/blog/2026-05-19-fp8-kv-cache-is-the-next-decode-bandwidth-cut-rdna4-already-has-the-wmma-for), exists to make that document usable at a sane token rate.

The trap is that enabling long context is so cheap to configure that it looks free. You set `--max-model-len 131072`, paste in the YaRN block, and ship. Nothing errors. Short prompts still produce reasonable text, so the cost never shows up as a crash or a warning. It shows up as a model that is slightly worse at the thing you actually do most, which is short turns, in exchange for a capacity you reach rarely. That is the worst kind of regression: invisible unless you run the A/B, and nobody runs the A/B.

## What rope scaling is actually doing

Qwen3 uses rotary position embeddings, or RoPE. Instead of adding a position vector to each token, RoPE rotates the query and key vectors by an angle that depends on position, and it does this across a spectrum of frequencies. Some dimensions rotate fast, completing a full turn every few tokens, and they carry fine local order. Others rotate slowly, turning once over thousands of tokens, and they carry long-range position. The attention score between two tokens ends up depending on their relative distance, which is the property that makes RoPE work.

The problem with going past 32,768 is that the slow dimensions were never rotated beyond the angle that the training length produced. Feed the model position 90,000 and the low-frequency dimensions enter angles it has never seen, and attention falls apart. The first fix, [position interpolation](https://arxiv.org/abs/2306.15595) from Chen and coauthors, was blunt: squeeze every position into the trained range by dividing all the angles by the extension factor. It works, but squeezing the fast dimensions too throws away the local resolution the model relied on to tell neighbors apart.

YaRN, from Bowen Peng and coauthors, is the smarter version and the one Qwen validated. Its core move, called NTK-by-parts interpolation, is to treat the frequencies differently. The high-frequency dimensions that encode local order are left almost untouched, and only the low-frequency dimensions that encode long-range position get interpolated, with an extra attention-scaling term to keep the softmax well behaved at the new length. That selectivity is why YaRN beats plain interpolation, and why it was, per the paper, the first method to extend Llama 2 to a 128k context while needing roughly 10x fewer tokens and 2.5x fewer training steps than the approaches before it.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-05-24-yarn-rope-frequency-stretch.svg" alt="A two-panel data visualization on a deep petrol-teal background. The left panel, titled how YaRN stretches the RoPE frequency spectrum, shows two rows of sine waves laid along a horizontal axis that runs from high frequency, fast rotation, local order on the left to low frequency, slow rotation, long range on the right. The top row, labeled native, trained to 32768, shows waves whose period grows steadily from left to right in soft cyan. The bottom row, labeled static YaRN factor 4, shows the leftmost high-frequency waves identical to the native row in cyan and annotated preserved, NTK-by-parts keeps local order, while the rightmost low-frequency waves are stretched to four times their period in warm amber and annotated interpolated 4x for positions up to 131072. A vertical dashed line in the middle marks the by-parts boundary. The right panel, titled the static tax, effective rope scaling versus prompt length, is a line chart whose horizontal axis is sequence length from 0 to 131072 with ticks at 8192, 32768, 65536, and 131072, and whose vertical axis is the effective scaling factor from 1 to 4. A flat amber line sits at factor 4 across the whole axis, labeled static YaRN, applied to every prompt. A cyan line labeled length-aware, dynamic stays at factor 1 until 32768 then ramps up to 2 at 65536 and 4 at 131072. The region between the two lines from 0 to 32768 is shaded amber and labeled the tax, short prompts pay full stretch for context they never use. A footer notes that wave periods are illustrative of the NTK-by-parts mechanism and that the dynamic curve is factor equals max of 1 and length divided by 32768." loading="lazy" />
  <figcaption>Left: the mechanism. YaRN leaves the fast, local-order dimensions alone and interpolates only the slow, long-range ones, which is why it degrades short context less than plain interpolation. Right: the cost of doing that statically. A constant factor of 4 applies the full long-context stretch to every prompt, while a length-aware factor stays at 1 until the prompt actually exceeds the trained window. The shaded region is the quality short prompts give up for context they never use.</figcaption>
</figure>

The left panel is the reason YaRN is the good method and the right panel is the reason static is the wrong way to deploy it. NTK-by-parts is careful, but careful is not the same as free. Even with the fast dimensions preserved, a short prompt running under `factor` 4 has its slow dimensions interpolated and its attention rescaled for a 131k regime it is nowhere near, and Qwen's "may potentially degrade" is the measured residue of exactly that.

## The static part is the whole problem

Read Qwen's deployment guidance and the design intent is unmistakable. The [Qwen vLLM docs](https://qwen.readthedocs.io/en/latest/deployment/vllm.html) note that the default `max_position_embeddings` is 40,960, not 131,072, and explain the split: 32,768 reserved for output and 8,192 for a typical prompt. The shipping default is tuned so that ordinary use never touches YaRN at all. You have to opt into the long-context behavior, and when you do, you should match the factor to your real workload, not the maximum. The docs are explicit that if your typical context is 65,536, you should set `factor` to 2.0 rather than 4.0, because a smaller stretch costs less on everything below it.

| Configuration | Reaches | Best when typical context is | What every shorter prompt pays |
| --- | ---: | ---: | --- |
| Native, no YaRN | 32,768 | up to ~32k | nothing; trained range |
| YaRN `factor` 2.0 | 65,536 | around 64k | the 2x stretch, on all prompts |
| YaRN `factor` 4.0 | 131,072 | around 128k | the 4x stretch, on all prompts |
| Length-aware (dynamic) | 131,072 | mixed short and long | nothing below 32k; ramps only past it |

The first three rows are the static options, and the column that matters is the last one. Pick `factor` 4.0 because you occasionally paste a long document, and the bottom-of-the-table stretch lands on the 2,000-token questions that make up most of your traffic. The fourth row is the one that breaks the tradeoff, and it is not a different model. It is the same YaRN math with the factor computed per request instead of frozen at load time.

## Make context length a per-request decision

Dynamic, or length-aware, rope scaling sets the effective factor from the actual sequence length: roughly `max(1, length / 32768)`. A 4k prompt runs at factor 1, which is to say native, untouched, exactly the model Qwen trained. A 60k prompt runs near factor 2. A 128k document runs at factor 4. Nobody pays the long-context stretch except the requests that are actually long, and the short-prompt tax goes to zero.

This is not an exotic idea, and one provider already ships it by default. Qwen's own model card says that "the endpoint provided by Alibaba Model Studio supports dynamic YaRN by default and no extra configuration is needed." The first-party hosted endpoint does not ship the static behavior that the open-source configs default to. The gap between those two is precisely the gap a local engine has to close on its own, because the `config.json` you download encodes the static version.

For an engine that already computes RoPE on the GPU, closing it is cheap. zinc applies the rotation in its own kernels, so the scaling factor is an input to that step, a value that rides along in the per-dispatch constants, not a new code path. Deciding the factor from the prompt length is a few lines on the host side before the prefill launches. The model file can keep its honest native window, and the engine stretches the frequencies only when a request crosses 32,768. The capability and the default stop fighting each other.

## The tradeoff dynamic scaling actually carries

Per-request scaling is not free either, and the cost is in the cache, not the math. A KV entry is written with the rotation that was in effect when its token was processed. If the factor depends on sequence length, then the rotation applied to token 5,000 in a request that ends at 8k is different from the rotation applied to that same token in a request that ends at 100k. The cached keys are not interchangeable across scales.

That collides directly with [prefix KV reuse](/blog/2026-05-06-why-prefix-kv-reuse-is-the-cheapest-five-x-left-on-local-qwen3-chat), the cheapest speedup we have on local chat. A prefix cached at one effective factor cannot be reused by a later request that lands at a different factor without recomputing its rotations. The clean answer is to make the scaling factor part of the cache key, so a prefix is only reused when the regime matches, and to pick factor boundaries that are coarse enough that most same-shaped requests share one. It is real plumbing, and it is the honest price of not taxing short prompts. The static config dodges this problem by making every request wrong in the same way, which is a consistency you do not want.

## What comes next

The thing to take away is that "131,072 tokens" on the Qwen3 card is a ceiling, not a setting. Reaching it costs short-prompt quality if you reach for it the way the default configs invite, by freezing a factor of 4 across every request. The math that gets you there, YaRN's selective interpolation, is genuinely good, and the [paper](https://arxiv.org/abs/2309.00071) earns its place as the method Qwen validated. The deployment default is what is wrong, not the technique.

What I am watching is the next generation of models trained natively at long context, where the scaling factor at typical lengths is 1 because the trained window already covers the workload, and this entire tradeoff evaporates. Until those are the models people run locally, the rule holds and it is simple to state. Keep the model's native window in the config, decide the rope-scaling factor from the length of the prompt in front of you, and let the 1k chat be a 1k chat. Context length is not a property you turn on. It is a question the engine should answer one request at a time.
