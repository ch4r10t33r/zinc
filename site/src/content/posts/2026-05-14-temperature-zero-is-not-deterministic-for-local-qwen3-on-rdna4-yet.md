---
title: "Temperature zero is not deterministic for local Qwen3 on RDNA4 yet"
date: "2026-05-14"
tags:
  - zinc
  - rdna4
  - amd
  - vulkan
  - determinism
  - batch-invariance
  - reproducibility
  - qwen3
  - llama-cpp
  - sglang
  - chunked-prefill
  - flash-attention
  - llm-inference
keywords:
  - batch invariance local LLM
  - deterministic Qwen3 inference
  - llama.cpp PR 16016 deterministic mode
  - SGLang deterministic inference 2025
  - Thinking Machines defeating nondeterminism
  - fixed split KV flash attention
  - chunked prefill reduction order
  - temperature zero non-deterministic
  - RDNA4 Vulkan deterministic inference
  - Radeon AI PRO R9700 reproducible decode
  - on-policy RL local Qwen3
  - GGML_DETERMINISTIC flag
excerpt: "Same Qwen3 prompt. Same seed. Temperature zero. One user, one chat tab, one local GPU. The completion changes between runs anyway. Thinking Machines named the mechanism in September 2025: kernels are run-to-run deterministic, but they are not batch-invariant, and the chunk size that decides first-token latency also decides which reduction tree fires. SGLang and llama.cpp now ship a batch-invariant path on CUDA. The Vulkan path on RDNA4 does not, and that is the next correctness fight for local inference."
---

Run the same Qwen3 prompt twice, with temperature zero, top_k one, top_p one, on a single Radeon AI PRO R9700 with no concurrent traffic. You will sometimes get two different answers. This is not a server-load story and it is not an RNG bug. It is supposed to be impossible.

Horace He and Thinking Machines Lab dropped the cleanest demonstration of this in September 2025. They sampled `Qwen/Qwen3-235B-A22B-Instruct-2507` 1000 times at temperature zero with the prompt "Tell me about Richard Feynman". The first 102 tokens were bit-identical across every run. At the 103rd token the completions split. 992 of the runs went "...born on May 11, 1918, in Queens, New York". 8 of the runs went "...born on May 11, 1918, in New York City". Over the full 1000-token windows, 80 distinct completions came out of a single fixed prompt at temperature zero. [The Thinking Machines blog post](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/) lays out the experiment in full.

The hypothesis most engineers reach for first is the "concurrency plus floating point" story: GPUs are parallel, floating-point addition is non-associative, atomic adds finish in race order. The Thinking Machines team showed that hypothesis is wrong for LLM inference. The forward pass of a modern transformer has no atomic adds. Run a matmul on the same inputs a thousand times and every result is bitwise identical. The real culprit is one step deeper.

## What "deterministic" actually means here

The cleanest framing in the post is that several true statements look contradictory: some GPU kernels are nondeterministic, but the forward pass of an LLM uses none of them; the forward pass is deterministic given the exact inputs it receives; the inference server, given the exact same set of user requests, is deterministic too. From the perspective of any individual user, the system still feels nondeterministic, because the input they cannot see is the other requests batched with theirs.

That last hop is what fails on a local box, even with one user.

The forward pass is not "batch-invariant". When the batch size of a kernel changes, the kernel quietly switches its reduction strategy. A small batch with not enough work per output tile triggers a split-K matmul, splitting the inner reduction across multiple SMs that combine partial sums at the end. A flash attention kernel with a very short query length switches to flash-decoding's split-KV strategy, splitting the KV reduction so it can keep cores busy. RMSNorm with a tiny batch flips from one-row-per-SM to split-reduction. None of these are bugs. They are how you keep a GPU saturated across a wide batch range.

The price is that the reduction order changes with batch size. Floating-point addition is non-associative. The sum `(a + b) + c` and the sum `a + (b + c)` are not the same number. So the logits change, and the argmax tips to a different token. The same prompt, given a different batch size, produces a different completion.

## Three kernels that break batch invariance

The Thinking Machines analysis ordered the offenders by difficulty: RMSNorm, matrix multiplication, attention.

RMSNorm is the easiest to fix. The trick is to pick a reduction strategy that has enough parallelism even at the smallest batch sizes you care about, then refuse to change it. You give up peak performance on very small batches and you gain a reduction order that stops drifting. This is what the reference implementation in [thinking-machines-lab/batch_invariant_ops](https://github.com/thinking-machines-lab/batch_invariant_ops) does for RMSNorm.

Matrix multiplication is harder because tensorcore instructions themselves have internal reduction orders that depend on tile shape. The fix is to pick one kernel configuration, with one tile size, and compile against it for every shape that matters. You lose about 20 percent against cuBLAS on a Triton implementation. On real LLM inference workloads with reasonably large model dimensions, that 20 percent is mostly recoverable.

Attention is where it gets ugly. Decode at long context cannot be parallelized along the query length, because the query length is one. The only saturating axis is the KV cache, which means you have to split the reduction across the KV dimension. FlashInfer's "balanced scheduling" picks the largest split size that still saturates the cores, which means the split count depends on the KV length, which means the reduction order depends on how many decode requests are in flight. The fix is a fixed split-KV size, set in the kernel configuration, applied regardless of batch state. Sometimes the splits stay underutilized. That cost is what local engines have to swallow.

The bookkeeping detail that nobody loves is that chunked prefill and prefix caching both quietly change which key/value tokens are "in cache" versus "in the current pass" at attention time. The fix is to update the KV cache and page table before the attention kernel runs, so the keys and values are always laid out in the same place regardless of how the request got chunked.

## Why a single-user local box still hits this

The argument that "this only matters for multi-tenant servers" is wrong, and it is wrong in exactly the place where the local engine landscape is most active right now.

A local chat workload looks single-user only on the slowest timescale. On the dispatch timescale it looks like this. Tab A is streaming a long answer at 117 tok/s. Tab B is being typed and submits a new 8k prompt. The new prefill chunks at ubatch 512 and lands on top of whatever decode iteration is in flight. The KV cache is at 12k tokens at the moment the first chunk lands. By the time the third chunk runs, the cache is at 13.5k. The flash attention kernel's split count is a function of the KV length and the SM count. The split count is therefore different on chunks one, two, and three. So is the reduction order, and so are the logits.

This was the topic of [yesterday's post on chunk size and first-token latency](/blog/2026-05-13-how-chunk-size-decides-first-token-latency-on-long-local-qwen3-prompts). The same chunk size that determines TTFT also determines which reduction tree the attention kernel walks. If a user changes `--ubatch-size` between two runs to chase a better TTFT, they are also changing the floating-point sum order for the entire decode that follows. This is invisible in the visible API. It shows up as a different answer at token 103.

Local fine-tuning makes this worse. The "on-policy RL" claim in any local RL recipe depends on the assertion that the sampling forward pass produces the same logits the trainer's forward pass would. When the sampler is non-batch-invariant, that assertion is false. The KL divergence between sampler and trainer drifts upward and the policy spikes. The Thinking Machines runs showed a clean reward collapse around training step 318 without batch invariance, and a flat-zero KL divergence with it.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-05-14-qwen3-batch-invariance-divergence.svg" alt="A two-panel diagram on a dark navy background. The upper panel labeled non-deterministic forward pass shows a horizontal strip of about 110 small pale-blue token squares. Tokens one through 102 are uniform and labeled identical across 1000 runs. At token 103 the strip forks into two orange branches. The thicker upper branch leads to a label that reads quotes Queens, New York, end quotes, 992 of 1000 runs. The thinner lower branch leads to a label that reads quotes New York City, end quotes, 8 of 1000 runs. Below the token strip a small histogram shows 80 unique completions over 1000 sampling trials, with one tall orange bar at 78, a shorter bar at 8, and a long tail of single-occurrence completions. The lower panel labeled batch-invariant forward pass shows two stacked horizontal prefill timelines for the same 6000-token prompt. The first row labeled ubatch equals 512 is broken into 11 blue chunks of equal width plus a partial twelfth chunk. The second row labeled ubatch equals 1024 is broken into 5 wider indigo chunks plus a partial sixth chunk. A third row below them labeled fixed split-KV equals 256 shows 24 green vertical ticks of identical width, aligned at the same boundaries underneath both prefill timelines, so the reduction landing pattern is the same regardless of which chunk schedule fired. A green check mark on the right reads identical output, 1000 of 1000 runs, with a note that SGLang reports about 34 percent throughput cost. The footer credits the September 2025 Thinking Machines analysis, the LMSYS SGLang deterministic-inference blog, and llama.cpp PR 16016." loading="lazy" />
  <figcaption>The first divergence in 1000 temperature-zero runs lands at token 103. A fixed split-KV reduction strategy makes the reduction order of the attention kernel independent of how the prompt got chunked. The chunks change. The logits do not.</figcaption>
</figure>

The reader should notice two things. One: bit-identical completions for the first 102 tokens means the model's behavior is not "stochastic" in any meaningful sense at this point in the decode. The divergence is a kernel artifact. Two: the fix does not remove split-KV. Split-KV stays; what changes is that the split size is fixed in the kernel configuration and held constant across every batch state. The chunked-prefill schedule above still varies. The reduction landing pattern below does not.

## What the deterministic path costs

[The LMSYS team's September 22 follow-up](https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/) ported Thinking Machines' batch-invariant ops into SGLang, kept compatibility with chunked prefill, CUDA graphs, and the radix cache, and reported the cost. With CUDA graphs enabled, deterministic mode is about 34 percent slower than non-deterministic mode on Qwen3-8B at TP1 on an H200. Without CUDA graphs it is about 60 percent slower. Without batch-invariant kernels and with deterministic flags set, 50 sampling trials on a fixed prompt give them between four and eighteen unique outputs depending on the prefix length. With batch-invariant kernels on, every trial of every prefix length lands the same answer.

The same trajectory is now active in llama.cpp. [PR 16016](https://github.com/ggml-org/llama.cpp/pull/16016) adds an opt-in deterministic mode on the CUDA backend, gated by `-DGGML_DETERMINISTIC=ON` at build time and `--deterministic` (or `GGML_DETERMINISTIC=1`) at runtime. The PR description is precise about what changes: deterministic RMSNorm with a fixed per-row reduction order; deterministic FP16 and BF16 MatMul with fixed tiling, no split-K, and FP32 accumulation; deterministic attention with a fixed split-size over KV, a stable softmax reduction, and a unified KV path so that chunked and one-shot prefill are numerically identical. The MoE `mul_mat_id` path is in scope too. The default fast paths are unchanged when the flag is off.

That is a lot of careful work. None of it is free, and all of it is gated to CUDA today.

## What this means for zinc on RDNA4

The Vulkan inference path on RDNA4 does not have batch-invariant kernels. The flash attention path that landed in [the wave32 commit](/blog/2026-05-11-the-wave32-commit-that-closes-rdna4-long-context-flash-attention-gap) is now correctly utilizing the 32-wide SIMD lanes on gfx1201, but the scalar FA kernel still picks a split count that depends on KV length and query length. Decode-time attention on a Qwen3 prompt with 14k cached tokens uses a different split count than decode-time attention on the same prompt with 16k cached tokens. The reductions tip.

That is the next correctness chunk for the Vulkan backend. The work is not exotic. The pieces are: pick a fixed split-KV size at kernel planning time, expose it through a specialization constant so the shader compiles once per size, align the chunked-prefill truncation to multiples of the split size, and make sure the KV cache is padded to that alignment so the kernel does not have to handle ragged tails differently between configurations. The matmul side wants one tile shape per quantization format and an FP32 accumulator that does not change with batch size. The RMSNorm side wants the same data-parallel reduction at every batch.

The performance cost is real but not catastrophic. SGLang's numbers are the closest reference we have for a tuned local engine: about 34 percent slower with CUDA graphs, on a workload where the non-deterministic baseline was already heavily optimized. A first cut on Vulkan will be worse before it gets better, because the kernel zoo currently selects between several specialization variants based on shape, and a single-variant policy gives up some of those wins.

The other half of this is the test surface. Once a deterministic flag exists, the bar for any kernel change is that it should not move logits with the flag on. That is a strong invariant. The Thinking Machines test does it the hard way: sample 1000 completions, count unique outputs, demand one. The SGLang test does it incrementally: same prompt, different batch sizes, same output. We will steal both.

## What changes when the contract holds

The contract is small. The same prompt and the same seed return the same completion, regardless of what the engine is doing in another tab, in another chat slot, or in a benchmark that happens to sit on a different prefill chunk boundary. Today, on Vulkan and RDNA4, that contract does not hold. With the work that Thinking Machines, SGLang, and llama.cpp have already laid out, it can.

Local users do not usually notice this directly, because chat outputs vary in ways that look "like sampling" even when the temperature is zero. What they notice instead is that an eval they ran on Tuesday cannot be reproduced on Wednesday, that two checkpoints in a local fine-tune compare differently every time the benchmark runs, that an RL run on a single 32 GB card spikes its KL divergence at the wrong step. None of those failures look like batch invariance until you measure for it.

The chunk size from yesterday's post and the reduction order from today's are the same lever. The cheapest local correctness fix on RDNA4 right now is to commit to a fixed split-KV size, hold it across every prefill schedule, and let the rest of the engine route around it. Twelve weeks ago that would have been an esoteric request. After Thinking Machines published, SGLang shipped, and llama.cpp opened PR 16016, it is the table-stakes feature local inference engines have to pay for in order to be taken seriously for anything beyond chat.
