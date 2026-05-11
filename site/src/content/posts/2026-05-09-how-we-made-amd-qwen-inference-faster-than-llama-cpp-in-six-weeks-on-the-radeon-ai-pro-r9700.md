---
title: "How we made AMD Qwen inference faster than llama.cpp in six weeks on the Radeon AI PRO R9700"
date: "2026-05-09"
tags:
  - zinc
  - rdna4
  - amd
  - amd-inference
  - amd-qwen
  - amd-llm
  - retrospective
  - llama-cpp
  - qwen3
  - qwen-3-6
  - vulkan
  - performance
  - benchmark
  - radeon-ai-pro-r9700
  - rx-9070
  - local-llm
  - llm-inference
keywords:
  - AMD Qwen inference
  - AMD LLM inference
  - AMD inference benchmark
  - AMD Radeon AI PRO R9700 LLM
  - AMD RDNA4 Qwen 3.6
  - AMD vs llama.cpp Qwen
  - Qwen 35B AMD GPU benchmark
  - Qwen3.6-35B-A3B local inference AMD
  - Radeon AI PRO R9700 Qwen 3.6 tok/s
  - faster than llama.cpp AMD
  - local Qwen 3.6 on AMD
  - AMD GPU local LLM 2026
  - zinc vs llama.cpp benchmark
  - Vulkan inference RDNA4 Qwen
  - prefix KV cache TTFT AMD
  - AMD inference engine Qwen
  - 7 tok/s to 33 tok/s AMD
  - RX 9070 XT Qwen 35B
faqs:
  - question: "What exactly beat llama.cpp in this benchmark?"
    answer: "The scoped result is Qwen3.6-35B-A3B UD Q4_K_XL on one Radeon AI PRO R9700 in an 8k multi-turn chat workload. ZINC measured 30.4 tok/s decode versus llama.cpp's 22.1, 198 tok/s on the new 4k user-message prefill versus 184, and 0.4 seconds second-turn TTFT versus 5.9 when the shared prefix was cached."
  - question: "Is ZINC faster than llama.cpp everywhere on AMD now?"
    answer: "No. This post is deliberately scoped to one flagship Qwen 3.6 chat-shaped workload on the 32 GB RDNA4 card. The broader benchmark suite still includes prompt shapes and model families where llama.cpp wins, especially short-context prefill on hybrid MoE plus SSM models."
  - question: "Why is the result significant if the scope is narrow?"
    answer: "Because the workload is the one a local AMD chat server actually feels: a long system prompt, repeated turns, a 35B resident model with 3B active parameters, and a single 32 GB card. Beating llama.cpp there means the AMD path is not doomed by hardware. The difference came from correctness, cache semantics, SSM state handling, and Vulkan dispatch shape."
  - question: "What was the biggest lesson from the failed attempts?"
    answer: "Porting a kernel is not the same as making it hot. Several llama.cpp-inspired pieces produced correct shaders but no throughput because they were wired only into cold call sites or because the buffer layout around them was still serial. The winning work changed where repeated work happened."
excerpt: "Six weeks ago the first end-to-end zinc run on AMD RDNA4 produced four tokens per second of confident gibberish with 97 percent of the vocabulary logits sitting at zero. Today on the same Radeon AI PRO R9700, zinc decodes Qwen3.6-35B-A3B UD Q4_K_XL at 30.4 tok/s against llama.cpp's 22.1, prefills a 4k user message at 198 tok/s against 184, and serves a multi-turn second-turn TTFT of 0.4 seconds against 5.9. The interesting part is not one magic shader. It is six weeks of correctness fixes, deleted dead ends, cache semantics, and Vulkan work that turned a 32 GB AMD card from a curiosity into a serious local Qwen 3.6 target."
---

The first RDNA4 run we trusted enough to inspect produced four tokens per second of confident gibberish. The card was a Radeon AI PRO R9700. The trace that made the bug undeniable was Qwen3.5-35B-A3B-UD Q4_K_XL. The output read like Markov-chain English. The tokenizer was fine. The sampling was fine. The LM head was reading 7,760 rows out of a 248,320-row vocabulary matrix and the other 240,560 rows were exactly the value they were initialized to: zero.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/zero-logits-meme.svg" alt="A two-panel postcard. The left panel labelled 'what it felt like' shows a rising orange curve and the caption 'Interesting output, maybe we are close.' The right panel labelled 'what was actually happening' shows a vocabulary-row histogram with three tall orange bars in the middle of a sea of flat grey ticks, annotated '248,320 vocab rows · 7,760 non-zero', and a tagline noting that only about three percent of the rows ever got computed." loading="lazy" />
  <figcaption>The bug postcard from the early days. 97 percent of the vocabulary rows never got computed, and the three percent that did happened to look varied enough to fool us for two days.</figcaption>
</figure>

The dispatch table for the matrix-vector kernel had one guard condition that returned early if the column index exceeded a hardcoded ceiling somebody had set during a debugging session and forgot to delete. The bug took two days to find and twenty seconds to fix. [The longer write-up of that period](/blog/2026-03-27-what-broke-first-when-we-built-zinc-on-amd-rdna4) is honest about how cheerful we were before we noticed.

Today, on the same Radeon AI PRO R9700 — 32 GB GDDR6, 128 AI accelerators, gfx1201, the card AMD shipped as the consumer-facing answer to local AI in 2026 — zinc decodes Qwen3.6-35B-A3B UD Q4_K_XL at 30.4 tok/s against llama.cpp's 22.1, prefills the 4k user message at 198 tok/s against 184, and the multi-turn second-turn TTFT lands at 0.4 seconds against llama.cpp's 5.9. The numbers are below.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-05-09-zinc-vs-llamacpp-qwen36-35b-a3b.svg" alt="Three side-by-side panels comparing zinc against llama.cpp on Qwen3.6-35B-A3B UD Q4_K_XL on a Radeon AI PRO R9700 with an 8k context window and a 4k system prompt plus 4k of multi-turn history. The left panel shows decode tok/s: zinc 30.4 versus llama.cpp 22.1, a 37 percent zinc lead. The middle panel shows prefill tok/s on the 4k user message after the system prompt is prefix-cached: zinc 198 versus llama.cpp 184, an 8 percent zinc lead. The right panel shows second-turn time-to-first-token in seconds with the system prompt and turn-1 history radix-cached: zinc 0.4 seconds versus llama.cpp 5.9, a 14.8x advantage for zinc. A footer notes the llama.cpp baseline is build b8967 with ROCm 7.2.1 plus Vulkan and that six weeks ago zinc was at 7.6 tok/s decode on the same model." loading="lazy" />
  <figcaption>Same model, same card, same Q4_K_XL file, same seed. Decode is the largest win, TTFT is the most useful one for chat workloads.</figcaption>
</figure>

This is not a broad "zinc beats llama.cpp everywhere" claim. The honest scope is narrow: Qwen3.6-35B-A3B UD Q4_K_XL, the Radeon AI PRO R9700, 8k to 64k context, and multi-turn chat with a prefix-cached system prompt plus turn-1 history. The broader benchmark suite still includes prompt shapes and model families where llama.cpp wins, especially short-context prefill on hybrid MoE plus SSM models. This post is about the chat-shaped workload where the cache, the recurrent state, and the decode loop are all part of what the user experiences.

## Why this result matters

The result matters because it moves the AMD conversation out of the usual fit-versus-fast tradeoff. A 32 GB RDNA4 card can fit a 35B-A3B model, but fitting the model is not the hard bar for local inference. The hard bar is whether a real chat session feels interactive once the system prompt, tool definitions, prior turns, KV cache, sampler, and long-context attention all show up at the same time.

These three numbers measure different waits:

| Measurement | zinc | llama.cpp | Why it matters |
| --- | ---: | ---: | --- |
| 1,200-token decode | **30.4 tok/s** | 22.1 tok/s | The answer streams fast enough that the model feels alive. |
| New 4k user-message prefill | **198 tok/s** | 184 tok/s | Long user turns stop being a separate pause after the cached prefix. |
| Second-turn TTFT | **0.4 s** | 5.9 s | The old "re-read the whole chat" tax disappears from the user path. |

The 0.4 second TTFT is the number I care about most. A lot of local-engine benchmarks end at decode tok/s because it is easy to measure. Users do not experience a decode kernel in isolation. They experience the blank time before the first token, then the stream. On a tool-using Qwen session, a few seconds of repeated prefill can make a 30 tok/s model feel slower than a 20 tok/s model with the right cache.

The story of how those numbers happened is a six-week step-line.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-05-09-zinc-six-week-journey.svg" alt="A step-line chart showing zinc's decode throughput on the flagship Qwen model across six weeks on the Radeon AI PRO R9700, from March 25 to May 9. The Y-axis is decode tokens per second from 0 to 35. The X-axis runs week by week. A flat dashed orange line at 22.1 tok/s marks the llama.cpp baseline. A green step-line starts at 0.8 on March 27, jumps to 7.6 on March 28 (first coherent run), leaps to 33 on March 30 (the seven-to-thirty-three Vulkan jump), drops to 18 on April 5 when Qwen 3 lands and the engine has to learn MoE plus gated DeltaNet, holds around 20 to 22 through April 26 (the SSM bottleneck gate), steps up to 26 on April 23 after Vulkan specialization constants, to 27 on April 25, to 28 on May 1, to 29 on May 4 after the min-p sampler, and reaches 30.4 on May 9 after the May 6 prefix KV reuse work. The crossing of the green line over the orange baseline is highlighted with a yellow burst around April 23 labeled parity. Major milestones along the line are tagged: first coherent at 7.6, 33 tok/s on Qwen3.5-35B, Qwen 3 lands (MoE plus gated DeltaNet to learn), Qwen 3.6 GA, Vulkan spec constants, prefix KV reuse, and 30.4 today. A footer notes that the flagship model rolls forward from Qwen 2.5 to Qwen 3 to Qwen 3.5-35B-A3B to Qwen 3.6-35B-A3B as the family ships." loading="lazy" />
  <figcaption>The whole story in one chart. The dip on April 5 is Qwen 3 landing and the engine needing to learn MoE plus gated DeltaNet from scratch. The crossing in late April is where AMD Qwen inference on zinc passed llama.cpp on the same card.</figcaption>
</figure>

The chart is not monotonic because the target kept getting harder. The model family moved from Qwen 2.5 to Qwen 3 to Qwen3.5-35B-A3B to Qwen3.6-35B-A3B while the engine was being written under it. The useful version of the history is the proof each phase gave us:

| Date | State | What it proved |
| --- | --- | --- |
| Mar 27 | 0.8 tok/s and mostly wrong | Correctness had to come before optimization. |
| Mar 30 | 33.58 tok/s on Qwen3.5-35B-A3B | The Vulkan path could move real weight bandwidth once dispatch and memory placement were sane. |
| Apr 5 | Qwen 3 landed and the number fell | MoE plus gated DeltaNet was not a Llama-shaped porting problem. |
| Apr 26 | 90.24 tok/s prefill versus about 180 | The remaining gap was structural: SSM state and MoE batching, not one bad shader. |
| May 6 | Second-turn prefill fell under half a second | Cache semantics beat another round of micro-kernel tuning for chat. |
| May 9 | 30.4 tok/s decode and 0.4 s TTFT | The narrow AMD Qwen chat workload crossed llama.cpp on the same card. |

The rest of the post is the longer version of each step and the dead ends we stopped carrying.

## The first ten days

The first run that produced any plausible English was on March 30, five days after we started. The path from there is documented in [the 33 tok/s recap](/blog/2026-03-30-how-we-moved-zinc-from-7-tok-s-to-33-tok-s-on-amd-rdna4). The shortest version:

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/zinc-7-to-33-rdna4.gif" alt="A sped-up screen recording of zinc decoding a Qwen3.5-35B-A3B response on the Radeon AI PRO R9700 after the late-March throughput jump, with the tok/s counter holding steady around 33 in the terminal status bar." loading="lazy" />
  <figcaption>March 30 on the same card. The 33 tok/s number was the first clean baseline that made the RDNA4 path feel credible.</figcaption>
</figure>

Three changes mattered. We switched from `VK_BUFFER_USAGE_TRANSFER_DST_BIT` host-visible staging buffers to device-local memory with a single `vkCmdCopyBuffer` per layer, which killed a PCIe round trip per dot product. We wrote a fused dmmv kernel for the common case of a vector-by-matrix multiply where the matrix is Q4_K_M, replacing the dequant-then-matmul pipeline that was the inherited Vulkan baseline. And we collapsed the `vkQueueSubmit` per pipeline stage into one command buffer per layer.

None of that was new. All of it was unimplemented in the Vulkan backend we forked from. The 33.58 tok/s number was on Qwen3.5-35B-A3B-UD Q4_K_XL, batch one, decode-only. It was also the moment the project stopped feeling like an interesting bring-up and started feeling like an actual inference engine.

## The dead ends were useful

The most useful work in April was not always the work that stayed in the tree. The optimization loop tried a lot of ideas that sounded like "what llama.cpp does" and then measured them as flat or negative because the call site was wrong, the batch shape was wrong, or the engine was still paying a larger state-management tax somewhere else.

| Tried | What happened | What it taught |
| --- | --- | --- |
| [Port llama.cpp-style tiled GEMM foundations](/blog/2026-04-26-the-gate-that-keeps-qwen-35b-prefill-at-half-of-llama-cpp-on-rdna4) | `mul_mm_q4k` was bit-identical in the LM head, measured 78.14 tok/s against a 78.55 baseline, and 1,470 lines of dormant infrastructure were later reverted. | A correct kernel is not a win until it is wired into the hot path. |
| [Chase 32-column DMMV before full GEMM](/blog/2026-04-22-why-rdna4-prefill-wants-a-32-column-dmmv-before-a-gemm) | The weight-read math was right, but shared register budgets made the decode path pay for prefill's column count. | Variant-specific pipelines matter on RDNA4 because VGPR pressure is throughput. |
| [Plan Q8_1 activation quantization](/blog/2026-04-19-why-q8-1-activations-are-the-next-rdna4-prefill-unlock) | The direction stayed right, but a standalone shader port was not enough without the layer-level reuse and buffer lifecycle. | Activation quantization is a systems change, not one shader. |
| [Try a vocab-matched speculative draft](/blog/2026-04-28-why-speculative-decoding-does-not-net-out-on-qwen-35b-a3b) | Public 3090 tests found no net win across nineteen configs, matching the 0.96x worst-case formula once SSM rewind cost was included. | MoE lowers verifier cost enough that the classic dense-model speculation math stops applying. |
| [Follow the FP4 wave](/blog/2026-05-09-the-fp4-wave-breaks-at-rdna4-and-fp8-wmma-already-does-what-local-qwen3-needs) | FP4 saves footprint, but gfx1201 has no FP4 WMMA instruction, so the compute path dequantizes back toward FP16-class math. | The ISA decides which quantization fashions are real throughput wins. |

This table is why the May result does not feel like one trick. The winning path was not "copy llama.cpp." It was copy the idea only after understanding which part of the idea was doing the work: fewer repeated reads, fewer launches, better cache keys, and state that lives at the right boundary.

## April: this is fine

The dark days started immediately after. Qwen 3 landed on April 5, and Qwen 3 was structurally different from Llama 3 in five places the 7→33 work had not touched. Qwen 3.5/3.6 35B-A3B activates 3B parameters per token through MoE routing. Three out of four attention blocks are replaced by gated DeltaNet linear attention with a recurrent state. The KV cache shape, the sampler chain, and the prefill batching path all need different code from the dense Llama 3 path.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/design-this-is-fine.svg" alt="A stylized debugging room scene with a calm cartoon developer sitting at a desk while small flames lick at the edges of the floor. Four red debug cards float around the scene: SSM state NaN at layer 12 token 47, delta-net Q norm drift of 0.003 per token, MoE router logits all zeros, and flash attention negative infinity in softmax. The developer's mug reads 'this is fine.'" loading="lazy" />
  <figcaption>April was four specific debugs, each found by binary-searching forward passes for four to twenty hours and fixed in one or two hundred lines.</figcaption>
</figure>

The SSM state NaN was a gated DeltaNet decay-gate underflow in FP16, fixed by clamping the gate before the recurrence. The MoE router-logits-all-zeros bug was a top-k=2 selection happening before softmax in a fused path, fixed by reordering ops. The delta-net Q-norm drift was a missing RMSNorm on the recurrent state, compounding into a 30 percent error by token 100. The flash-attention negative-infinity-in-softmax was a cold pipeline state with an uninitialized FlashAttention scale.

The cumulative effect over April was zinc going from "runs Qwen 3 dense" to "runs Qwen 3.5-35B-A3B at roughly 50 percent of llama.cpp on prefill and 90 percent on decode." [The April 26 gate post](/blog/2026-04-26-the-gate-that-keeps-qwen-35b-prefill-at-half-of-llama-cpp-on-rdna4) is the honest snapshot at the bottom of that valley: zinc 90.24 tok/s prefill against llama.cpp 180 on Qwen 3.5-35B-A3B, with the SSM bucket sitting at 925 ms out of 2,110 ms of GPU phase time.

## Late April: closing the prefill gap

The first structural win against llama.cpp on RDNA4 was [Vulkan specialization constants for the dmmv variants](/blog/2026-04-23-vulkan-specialization-constants-unlock-rdna4-dmmv-variants). Letting the kernel decide at compile time whether the matrix was a 32-column, 64-column, or full-tile shape, instead of branching at runtime, was worth +18 percent on prefill at 4k context. Two days later [the `vkQueueSubmit`-per-prompt change](/blog/2026-04-25-why-one-vkqueuesubmit-per-prompt-is-the-next-quiet-rdna4-prefill-unlock) was worth another +11 percent. The single largest unlock was block-resident gated DeltaNet state: keep the recurrent state in registers across the whole token loop instead of re-reading and re-writing 2 MB per layer per token, which dropped the SSM bucket from 925 ms to 165 ms.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-05-09-prefill-bucket-before-after.svg" alt="Two horizontal stacked bars comparing the prefill GPU phase budget on Qwen 3.5-35B-A3B on a Radeon AI PRO R9700, before and after the late-April fixes. The top bar labeled BEFORE is 2,110 ms long and split into colored segments: SSM 925 ms (purple, 44%), MoE 739 ms (teal, 35%), attention 333 ms (orange, 16%), and GEMM plus other 113 ms (grey). The bottom bar labeled AFTER is 1,096 ms long: SSM shrunk to 165 ms, MoE held at 720 ms, attention reduced to 165 ms, GEMM plus other 46 ms. Annotations between the bars name the responsible fixes: block-resident gated DeltaNet state under SSM, MUL_MAT_ID dispatch coalescing under MoE, and a FA softmax fix under attention. A green annotation in the empty space to the right of the after bar reads 1,014 ms returned to decode budget, SSM dropped 5.6 times, attention dropped 2.0 times. The total elapsed prefill drops from 2,110 ms to 1,096 ms, a 1.93 times speedup overall." loading="lazy" />
  <figcaption>Where the prefill time went, before and after. The SSM bucket carried 44 percent of the prefill on April 26. Block-resident state cut it 5.6x. The total halves and the surplus is what makes the decode loop faster downstream.</figcaption>
</figure>

By the May 1 [attention-not-GEMM post](/blog/2026-05-01-why-rdna4-long-prefill-plateaus-on-attention-not-gemm), the question had changed. The chat-shaped long-prefill case was no longer a generic "we need a GEMM" problem. The remaining profile had named buckets: MoE cohort grouping, attention shape, SSM state, and cache reuse. Decode was still behind.

## Early May: closing the decode gap

Decode is harder because there is less to optimize. The matmul is bandwidth-bound by the KV reads at long context, the sampler is on the hot path, and the attention has to actually be correct. The May posts walk through each fix, one per day, and the shape of where they landed in the decode loop is below.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-05-09-zinc-decode-pipeline-flow.svg" alt="A vertical flow diagram of zinc's decode loop on Qwen3.6-35B-A3B. A new input token enters at the top and hits a prefix radix tree decision. On a cache hit the flow short-circuits down a dashed green line straight to the LM head, labeled prefix KV reuse. On a miss the flow descends through a representative 3:1 hybrid layer stack: three pale-purple gated DeltaNet linear-attention blocks followed by one teal full softmax attention block. The full-attention block reads from a small KV cache tile annotated INT8 plus 4 attention sinks. Below the layer stack a wide orange MoE block shows top-k=2 routing into 8 expert tiles, two highlighted in bright yellow as active and six dimmed. After 'repeated 48 times' the flow goes through the LM head matmul over the 151,936-row vocabulary, then a horizontal sampler chain shown as three green pill boxes in order: temperature, DRY penalty, min-p. The final output is a single next token at the bottom. To the right of each block, small annotations name the May post that landed each fix: May 6 prefix KV reuse on the radix tree, April 26 plus May 7 block-resident state on the GDN layer, April 26 INT8 KV plus May 2 attention sinks on the attention block, April MUL_MAT_ID dispatch coalescing on the MoE block, and May 5 DRY plus May 4 min-p on the sampler chain." loading="lazy" />
  <figcaption>One step of the decode loop. Five of the seven highlighted boxes are May fixes. The radix-tree short-circuit at the top is what produces the 14.8x TTFT advantage on multi-turn chat.</figcaption>
</figure>

[INT8 KV cache by default](/blog/2026-04-26-why-fp16-kv-cache-is-the-wrong-default-for-128k-context-on-32gb-rdna4) doubled the per-token KV bandwidth budget at 32k context. [Attention sinks resident in KV](/blog/2026-05-02-attention-sinks-the-four-kv-tokens-local-long-context-cannot-evict) stopped the long-context perplexity blowup that was forcing aggressive eviction. [Min-p before temperature](/blog/2026-05-04-why-min-p-is-the-right-default-sampler-for-local-qwen3-decode) and [DRY repetition penalty](/blog/2026-05-05-why-dry-earns-the-slot-before-min-p-on-qwen3-long-context-decode) replaced the sampler chain that had been on by default. [Prefix KV reuse](/blog/2026-05-06-why-prefix-kv-reuse-is-the-cheapest-five-x-left-on-local-qwen3-chat) eliminated the 27-second re-prefill of the 4,800-token system prompt on every turn, which is what produced the 14.8x TTFT advantage in the chart.

None of these were inventions. All of them were already known in the research and the local-inference community. The work was reading the literature, implementing the primitives correctly for gfx1201, and shipping them one per day until the decode loop was no longer leaving anything on the table.

## What today's number is and is not

The configuration: Qwen3.6-35B-A3B UD Q4_K_XL on a Radeon AI PRO R9700, 32 GB VRAM, ROCm 7.2.1 + Vulkan, kernel 6.16.6, 8k context with a 4k system prompt and 4k of multi-turn history. Decode is averaged over a 1,200-token reply across nine MT-Bench prompt categories. Prefill is the 4k user message after the 4k system prompt is prefix-cached. Second-turn TTFT measures the wall-clock seconds from prompt submission to the first decoded token for a turn whose 4k system prompt and 4k turn-1 history are radix-cached on both sides. llama.cpp baseline is build b8967 with `cache_prompt=true` and `--cache-reuse 256`. Same GGUF, same prompt, same seed.

zinc is not faster than llama.cpp on a 5090. zinc is not faster on every dense model in the 7B to 14B class where llama.cpp's Vulkan path has been polished for a year. zinc is not done with short-context prefill on hybrid MoE plus SSM models. zinc is not faster on the Qwen3-Next gated DeltaNet path because [the state-checkpoint plane is half-written](/blog/2026-05-07-why-qwen3-next-gated-deltanet-breaks-the-prefix-cache-local-engines-just-built) and [the Vulkan MTP-head kernel has not landed](/blog/2026-05-08-why-mtp-heads-are-the-speculative-decode-draft-qwen3-a3b-deserves). zinc is also not the right engine if you want to run [an FP4 quantized model](/blog/2026-05-09-the-fp4-wave-breaks-at-rdna4-and-fp8-wmma-already-does-what-local-qwen3-needs), because the RDNA4 silicon does not accelerate that format.

The honest scope of "we beat llama.cpp" is Qwen3.6-35B-A3B UD Q4_K_XL on the R9700 at 32 GB and 8k to 64k context with prefix-cached chat workloads. It is a narrow configuration. It is also the one the R9700 was made for, and it is the configuration that lets a local engine feel like a real assistant instead of a curiosity.

## What comes next

Three items are open and roughly sized. Native INT4 WMMA on the iu4 path is the largest unclaimed throughput on the card; the Vulkan kernel is half-written and the calibration story is the harder part. The Qwen3-Next state-checkpoint plane is the cheapest path to long multi-turn chat on hybrid models and depends on one Vulkan compute shader for the gated DeltaNet partial rollback. FP8 weights via the MI350X-shared Triton path on vLLM is a separate stack but the same hardware, covered in [today's other post](/blog/2026-05-09-the-fp4-wave-breaks-at-rdna4-and-fp8-wmma-already-does-what-local-qwen3-needs).

The shape of the work is the same as the first six weeks. Read the relevant paper. Implement the primitive correctly for gfx1201. Measure honestly. Ship one fix per day. Try not to chase the next quantization fashion before squeezing the throughput the silicon already pays for.

Six weeks ago zinc was 97 percent zero logits. Six weeks from now we should be either honestly faster than llama.cpp across the consumer model line on the R9700, or we should know why we are not. The work is small in lines of code and large in compound interest. The next post is tomorrow.
