---
title: "How AMD Qwen decode passed llama.cpp in six weeks on the Radeon AI PRO R9700"
seoTitle: "AMD Qwen Inference vs llama.cpp"
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
  - AMD decode faster than llama.cpp
  - local Qwen 3.6 on AMD
  - AMD GPU local LLM 2026
  - zinc vs llama.cpp benchmark
  - Vulkan inference RDNA4 Qwen
  - Qwen3.6 RDNA4 decode benchmark
  - AMD inference engine Qwen
  - 7 tok/s to 33 tok/s AMD
  - RX 9070 XT Qwen 35B
faqs:
  - question: "What exactly beat llama.cpp in this benchmark?"
    answer: "The scoped result is decode throughput on Qwen3.6-35B-A3B UD Q4_K_XL on one Radeon AI PRO R9700. In the latest published May 10 suite, ZINC measured 117.07 tok/s decode versus llama.cpp's 104.47 on the same RDNA node and model file."
  - question: "Is ZINC faster than llama.cpp everywhere on AMD now?"
    answer: "No. This post is deliberately scoped to the flagship Qwen 3.6 decode result on the 32 GB RDNA4 card. The same published suite still shows a real prefill gap: 88.08 tok/s in ZINC versus 181.95 tok/s in llama.cpp."
  - question: "Why is the result significant if prefill is still behind?"
    answer: "Because decode is the loop users sit in during long answers, and Qwen3.6-35B-A3B is a difficult hybrid MoE plus SSM model that fits on one 32 GB AMD card. Crossing llama.cpp on decode proves the RDNA4 path is no longer just a bring-up exercise, while the remaining prefill gap tells us exactly where the next structural work lives."
  - question: "What was the biggest lesson from the failed attempts?"
    answer: "Porting a kernel is not the same as making it hot. Several llama.cpp-inspired pieces produced correct shaders but no throughput because they were wired only into cold call sites or because the buffer layout around them was still serial. The winning work changed where repeated work happened."
excerpt: "The first serious zinc trace on AMD RDNA4 looked almost good, which was exactly the problem: it was English-shaped nonsense from a model whose LM head computed only three percent of the vocabulary rows. Six weeks later, on the same Radeon AI PRO R9700, the latest published suite shows zinc decoding Qwen3.6-35B-A3B UD Q4_K_XL at 117.07 tok/s against llama.cpp's 104.47. Prefill is not won yet: zinc is at 88.08 tok/s against llama.cpp's 181.95. The interesting part is not one magic shader. It is six weeks of correctness fixes, deleted dead ends, and Vulkan work that turned a 32 GB AMD card from a curiosity into a serious local Qwen 3.6 decode target."
seoDescription: "AMD Qwen inference on Radeon AI PRO R9700: how ZINC decode crossed llama.cpp on Qwen3.6 and where prefill still lags."
---

Quick answer: ZINC crossed llama.cpp on the scoped Qwen3.6 decode benchmark by fixing correctness first, then removing dead paths and moving repeated work into the hot Vulkan decode path. The remaining gap is prefill, not whether AMD RDNA4 can be a serious local Qwen inference target.

The first serious RDNA4 trace looked almost good, which is worse than looking broken. zinc was emitting English-shaped text at four tokens per second on the Radeon AI PRO R9700. Then the LM-head dump showed the truth: on Qwen3.5-35B-A3B-UD Q4_K_XL, 240,560 of the 248,320 vocabulary rows were still zero. The model was sampling from the three percent of logits our dispatcher happened to compute.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/zero-logits-meme.svg" alt="A two-panel postcard. The left panel labelled 'what it felt like' shows a rising orange curve and the caption 'Interesting output, maybe we are close.' The right panel labelled 'what was actually happening' shows a vocabulary-row histogram with three tall orange bars in the middle of a sea of flat grey ticks, annotated '248,320 vocab rows · 7,760 non-zero', and a tagline noting that only about three percent of the rows ever got computed." loading="lazy" />
  <figcaption>The bug postcard from the early days. 97 percent of the vocabulary rows never got computed, and the three percent that did happened to look varied enough to fool us for two days.</figcaption>
</figure>

The fix was not glamorous. The host-side dispatch math for the matrix-vector kernel used the wrong rows-per-workgroup formula. The Q8_0 shader processes two output rows per workgroup, but the dispatcher launched it as if each workgroup covered 64 rows. On the LM head, that meant only 7,760 of 248,320 vocabulary rows were computed. The bug took two days to find and twenty seconds to fix. [The longer write-up of that period](/blog/2026-03-27-what-broke-first-when-we-built-zinc-on-amd-rdna4) is honest about how cheerful we were before we noticed.

Six weeks later, the headline is narrower and stronger than the old draft made it sound. On the same Radeon AI PRO R9700 — 32 GB GDDR6, 128 AI accelerators, gfx1201 — the latest published benchmark artifact shows zinc decoding Qwen3.6-35B-A3B UD Q4_K_XL at 117.07 tok/s against llama.cpp's 104.47 on the same machine and model file. The same artifact also shows the unfinished part: prefill is 88.08 tok/s in zinc against 181.95 in llama.cpp. The numbers are below.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-05-09-zinc-vs-llamacpp-qwen36-35b-a3b.svg" alt="Three side-by-side panels comparing zinc against llama.cpp on Qwen3.6-35B-A3B UD Q4_K_XL on a Radeon AI PRO R9700 in the May 10 published suite. The left panel shows decode tok/s: zinc 117.07 versus llama.cpp 104.47, a 12 percent zinc lead. The middle panel shows prefill tok/s: zinc 88.08 versus llama.cpp 181.95, so zinc is at 48 percent of llama.cpp on prefill. The right panel summarizes the suite verdict: decode crossed, prefill remains the active structural gap, and the result is progress rather than a clean sweep." loading="lazy" />
  <figcaption>Same model, same RDNA node, same Q4_K_XL file. Decode crossed llama.cpp in the published suite; prefill is still the open structural gap.</figcaption>
</figure>

This is not a broad "zinc beats llama.cpp everywhere" claim. The honest scope is narrower and more useful: Qwen3.6-35B-A3B UD Q4_K_XL decode on the Radeon AI PRO R9700 has crossed llama.cpp in the public suite, while hybrid MoE plus SSM prefill is still behind. That is a real milestone and a real caveat at the same time.

## Why this result matters

The result matters because it moves the AMD conversation out of the usual fit-versus-fast tradeoff. A 32 GB RDNA4 card can fit a 35B-A3B model, but fitting the model is not the hard bar for local inference. The hard bar is whether the decode loop can stream a long answer at competitive speed once MoE routing, gated DeltaNet state, KV cache reads, sampler work, and Vulkan dispatch overhead all show up at the same time.

These three numbers measure different waits:

| Measurement | zinc | llama.cpp | Why it matters |
| --- | ---: | ---: | --- |
| Published-suite decode | **117.07 tok/s** | 104.47 tok/s | The hot answer loop is now ahead on the flagship model. |
| Published-suite prefill | 88.08 tok/s | **181.95 tok/s** | Prompt ingestion is still the biggest Qwen gap. |
| Journey delta | **0.8 tok/s broken -> 117.07 tok/s decode** | 107 tok/s March llama.cpp reference | The path went from barely coherent to competitive on the same card class. |

The middle row is as important as the first. If a post says only "decode crossed," it can be true while still hiding the user-visible pause before a long first token. If it says "inference is faster" without the prefill caveat, it is too broad. The interesting engineering state is the combination: decode is now good enough to take seriously, and prefill has a named structural backlog.

The story is not a clean upward chart because the target kept getting harder. The model family moved from Qwen 2.5 to Qwen 3 to Qwen3.5-35B-A3B to Qwen3.6-35B-A3B while the engine was being written under it. The useful version of the history is the proof each phase gave us:

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-05-09-zinc-six-week-journey.svg" alt="A seven-step timeline of zinc's RDNA4 Qwen work from March 27 to May 10. The milestones are: 97 percent zero logits and about 0.8 tok/s during Qwen3.5 bring-up; 33.58 tok/s plain decode on Qwen3.5 after memory and dispatch cleanup; Qwen 3 changing the target with MoE plus gated DeltaNet; the April 26 prefill snapshot at 90.24 tok/s versus about 180 in llama.cpp with SSM at 925 ms of 2,110 ms; the May 1 profile naming MoE grouping, attention, SSM state, and cache reuse as buckets; May 6 prefix KV reuse as a serving primitive; and the May 10 Qwen3.6 artifact showing decode 117.07 versus 104.47 while prefill remained 88.08 versus 181.95." loading="lazy" />
  <figcaption>The journey as milestones rather than one fake y-axis. The model and harness changed, so the honest through-line is what each checkpoint proved.</figcaption>
</figure>

| Date | State | What it proved |
| --- | --- | --- |
| Mar 27 | 0.8 tok/s and mostly wrong | Correctness had to come before optimization. |
| Mar 30 | 33.58 tok/s on Qwen3.5-35B-A3B | The Vulkan path could move real weight bandwidth once dispatch and memory placement were sane. |
| Apr 5 | Qwen 3 landed and the number fell | MoE plus gated DeltaNet was not a Llama-shaped porting problem. |
| Apr 26 | 90.24 tok/s prefill versus about 180 | The remaining gap was structural: SSM state and MoE batching, not one bad shader. |
| May 6 | Prefix KV reuse moved from idea to serving primitive | Cache semantics beat another round of micro-kernel tuning for repeated chat prefixes. |
| May 10 | 117.07 tok/s decode, 88.08 tok/s prefill | Decode crossed llama.cpp; prefill remained the active gap. |

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
| [Try a vocab-matched speculative draft](/blog/2026-04-28-why-speculative-decoding-does-not-net-out-on-qwen-35b-a3b) | Nineteen public draft/verifier configs found no net win once SSM rewind cost was included. | MoE lowers verifier cost enough that the classic dense-model speculation math stops applying. |
| [Follow the FP4 wave](/blog/2026-05-09-the-fp4-wave-breaks-at-rdna4-and-fp8-wmma-already-does-what-local-qwen3-needs) | FP4 saves footprint, but gfx1201 has no FP4 WMMA instruction, so the compute path dequantizes back toward FP16-class math. | The ISA decides which quantization fashions are real throughput wins. |

This table is why the May result does not feel like one trick. The winning path was not "copy llama.cpp." It was copy the idea only after understanding which part of the idea was doing the work: fewer repeated reads, fewer launches, better cache keys, and state that lives at the right boundary.

## April: this is fine

The dark days started immediately after. Qwen 3 landed on April 5, and Qwen 3 was structurally different from Llama 3 in five places the 7→33 work had not touched. Qwen 3.5/3.6 35B-A3B activates 3B parameters per token through MoE routing. Three out of four attention blocks are replaced by gated DeltaNet linear attention with a recurrent state. The KV cache shape, the sampler chain, and the prefill batching path all need different code from the dense Llama 3 path.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/design-this-is-fine.svg" alt="A stylized debugging room scene with a calm cartoon developer sitting at a desk while small flames lick at the edges of the floor. Four red debug cards float around the scene: SSM state NaN at layer 12 token 47, delta-net Q norm drift of 0.003 per token, MoE router logits all zeros, and flash attention negative infinity in softmax. The developer's mug reads 'this is fine.'" loading="lazy" />
  <figcaption>April was four specific debugs, each found by binary-searching forward passes for four to twenty hours and fixed in one or two hundred lines.</figcaption>
</figure>

The SSM state NaN was a gated DeltaNet decay-gate underflow in FP16, fixed by clamping the gate before the recurrence. The MoE router-logits-all-zeros bug was a top-k=2 selection happening before softmax in a fused path, fixed by reordering ops. The delta-net Q-norm drift was a missing RMSNorm on the recurrent state, compounding into a 30 percent error by token 100. The flash-attention negative-infinity-in-softmax was a cold pipeline state with an uninitialized FlashAttention scale.

The cumulative effect over April was zinc going from "runs Qwen 3 dense" to "runs Qwen 3.5-35B-A3B, but the hybrid prefill path is structurally behind." [The April 26 gate post](/blog/2026-04-26-the-gate-that-keeps-qwen-35b-prefill-at-half-of-llama-cpp-on-rdna4) is the honest snapshot at the bottom of that valley: zinc 90.24 tok/s prefill against llama.cpp 180 on Qwen 3.5-35B-A3B, with the SSM bucket sitting at 925 ms out of 2,110 ms of GPU phase time.

## Late April: naming the prefill gap

The first structural prefill wins were not enough to declare victory, but they changed the shape of the problem. [Vulkan specialization constants for the dmmv variants](/blog/2026-04-23-vulkan-specialization-constants-unlock-rdna4-dmmv-variants) let kernels specialize the matrix shape at compile time instead of branching at runtime. Two days later, [the `vkQueueSubmit`-per-prompt change](/blog/2026-04-25-why-one-vkqueuesubmit-per-prompt-is-the-next-quiet-rdna4-prefill-unlock) attacked launch overhead. The larger lesson from the phase profile was that gated DeltaNet state had to become block-resident: keep the recurrent state in registers across the token loop instead of re-reading and re-writing 2 MB per layer per token.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-05-09-prefill-bucket-before-after.svg" alt="Two horizontal stacked bars comparing the prefill GPU phase budget on Qwen 3.5-35B-A3B on a Radeon AI PRO R9700, before and after the late-April fixes. The top bar labeled BEFORE is 2,110 ms long and split into colored segments: SSM 925 ms (purple, 44%), MoE 739 ms (teal, 35%), attention 333 ms (orange, 16%), and GEMM plus other 113 ms (grey). The bottom bar labeled AFTER is 1,096 ms long: SSM shrunk to 165 ms, MoE held at 720 ms, attention reduced to 165 ms, GEMM plus other 46 ms. Annotations between the bars name the responsible fixes: block-resident gated DeltaNet state under SSM, MUL_MAT_ID dispatch coalescing under MoE, and a FA softmax fix under attention. A green annotation in the empty space to the right of the after bar reads 1,014 ms returned to decode budget, SSM dropped 5.6 times, attention dropped 2.0 times. The total elapsed prefill drops from 2,110 ms to 1,096 ms, a 1.93 times speedup overall." loading="lazy" />
  <figcaption>Where the April prefill time went in the phase profile. The SSM bucket carried 44 percent of the measured GPU phase time. The point is not that prefill was solved; the May 10 artifact still shows zinc at 48 percent of llama.cpp. The point is that the remaining gap finally had named buckets.</figcaption>
</figure>

By the May 1 [attention-not-GEMM post](/blog/2026-05-01-why-rdna4-long-prefill-plateaus-on-attention-not-gemm), the question had changed. The chat-shaped long-prefill case was no longer a generic "we need a GEMM" problem. The remaining profile had named buckets: MoE cohort grouping, attention shape, SSM state, and cache reuse. The public May 10 number says the same thing more bluntly: Qwen3.6 prefill is the biggest gap left.

## Early May: making decode credible

Decode is harder because there is less batch to hide behind. The matmul is bandwidth-bound by weight and KV reads, the sampler is on the hot path, and the attention has to be correct at every generated token. The May posts walk through each fix, one per day, and the shape of where they landed in the decode loop is below.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-05-09-zinc-decode-pipeline-flow.svg" alt="A vertical flow diagram of zinc's decode loop on Qwen3.6-35B-A3B. A new input token enters at the top and hits a prefix radix tree decision. On a cache hit the flow short-circuits down a dashed green line straight to the LM head, labeled prefix KV reuse. On a miss the flow descends through a representative 3:1 hybrid layer stack: three pale-purple gated DeltaNet linear-attention blocks followed by one teal full softmax attention block. The full-attention block reads from a small KV cache tile annotated INT8 plus 4 attention sinks. Below the layer stack a wide orange MoE block shows top-k=2 routing into 8 expert tiles, two highlighted in bright yellow as active and six dimmed. After 'repeated 48 times' the flow goes through the LM head matmul over the 151,936-row vocabulary, then a horizontal sampler chain shown as three green pill boxes in order: temperature, DRY penalty, min-p. The final output is a single next token at the bottom. To the right of each block, small annotations name the May post that landed each fix: May 6 prefix KV reuse on the radix tree, April 26 plus May 7 block-resident state on the GDN layer, April 26 INT8 KV plus May 2 attention sinks on the attention block, April MUL_MAT_ID dispatch coalescing on the MoE block, and May 5 DRY plus May 4 min-p on the sampler chain." loading="lazy" />
  <figcaption>One step of the decode loop. Five of the seven highlighted boxes are May fixes. The radix-tree short-circuit at the top is why prefix reuse remains the right next user-visible target.</figcaption>
</figure>

[INT8 KV cache by default](/blog/2026-04-26-why-fp16-kv-cache-is-the-wrong-default-for-128k-context-on-32gb-rdna4) doubled the per-token KV bandwidth budget at 32k context. [Attention sinks resident in KV](/blog/2026-05-02-attention-sinks-the-four-kv-tokens-local-long-context-cannot-evict) stopped the long-context perplexity blowup that was forcing aggressive eviction. [Min-p before temperature](/blog/2026-05-04-why-min-p-is-the-right-default-sampler-for-local-qwen3-decode) and [DRY repetition penalty](/blog/2026-05-05-why-dry-earns-the-slot-before-min-p-on-qwen3-long-context-decode) replaced the sampler chain that had been on by default. [Prefix KV reuse](/blog/2026-05-06-why-prefix-kv-reuse-is-the-cheapest-five-x-left-on-local-qwen3-chat) remains the highest-leverage chat-serving direction because it removes repeated prefix work rather than making one more single-token dispatch marginally faster.

None of these were inventions. All of them were already known in the research and the local-inference community. The work was reading the literature, implementing the primitives correctly for gfx1201, and shipping them one per day until the decode loop was no longer leaving anything on the table.

## What today's number is and is not

The configuration for the headline numbers is the published May 10 RDNA artifact: Qwen3.6-35B-A3B UD Q4_K_XL on a Radeon AI PRO R9700 with 32 GB VRAM, ZINC CLI against llama.cpp on the same RDNA node, same model file, one discarded warmup, and three measured runs. The numbers quoted here are the top-line reference entry for that model, each reported as the median of the measured samples; the artifact also includes longer-context and extended-decode scenarios. It records ZINC commit `321309a2fe8b` and llama.cpp commit `9725a313b`.

zinc is not faster than llama.cpp on a 5090. zinc is not faster on every dense model in the 7B to 14B class where llama.cpp's Vulkan path has been polished for a year. zinc is not done with short-context prefill on hybrid MoE plus SSM models; the current public number is 88.08 tok/s against llama.cpp's 181.95. zinc is not faster on the Qwen3-Next gated DeltaNet path because [the state-checkpoint plane is half-written](/blog/2026-05-07-why-qwen3-next-gated-deltanet-breaks-the-prefix-cache-local-engines-just-built) and [the Vulkan MTP-head kernel has not landed](/blog/2026-05-08-why-mtp-heads-are-the-speculative-decode-draft-qwen3-a3b-deserves). zinc is also not the right engine if you want to run [an FP4 quantized model](/blog/2026-05-09-the-fp4-wave-breaks-at-rdna4-and-fp8-wmma-already-does-what-local-qwen3-needs), because the RDNA4 silicon does not accelerate that format.

The honest scope of "we beat llama.cpp" is Qwen3.6-35B-A3B UD Q4_K_XL decode on the R9700 at 32 GB in the published RDNA suite. It is a narrow configuration. It is also the one that tells us the RDNA4 path is no longer a correctness demo. The remaining question is whether we can make prefill equally boring.

## What comes next

Three items are open and roughly sized. First, close Qwen3.6 prefill by wiring the batched path through hybrid MoE plus SSM instead of falling back to per-token dispatch when `n_experts > 0` or `ssm_d_inner > 0`. Second, keep the decode lead through the longer-context cases, where llama.cpp still has mature scheduling and cache behavior. Third, finish the Qwen3-Next state-checkpoint plane so prefix reuse can handle gated DeltaNet rollback cleanly instead of only helping transformer-shaped prefixes. FP8 weights via the MI350X-shared Triton path on vLLM is a separate stack but the same hardware, covered in [today's other post](/blog/2026-05-09-the-fp4-wave-breaks-at-rdna4-and-fp8-wmma-already-does-what-local-qwen3-needs).

The shape of the work is the same as the first six weeks. Read the relevant paper. Implement the primitive correctly for gfx1201. Measure honestly. Ship one fix per day. Try not to chase the next quantization fashion before squeezing the throughput the silicon already pays for.

Six weeks ago zinc was 97 percent zero logits. Today, decode on the flagship Qwen3.6 model is fast enough that the interesting question moved to prefill and long-context serving. That is the kind of progress that matters: not a clean sweep, not a victory lap, but a benchmark artifact that narrows the next week of work to the right bottleneck.
