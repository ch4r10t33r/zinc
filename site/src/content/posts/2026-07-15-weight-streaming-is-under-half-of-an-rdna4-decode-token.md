---
title: "Weight streaming is under half of a ZINC decode token on the RX 9070 XT"
seoTitle: "RDNA4 Decode Latency Breakdown: Where the RX 9070 XT Bandwidth Ceiling Leaks"
date: "2026-07-15"
tags:
  - zinc
  - amd
  - rdna4
  - rx-9070-xt
  - decode
  - memory-bandwidth
  - kv-cache
  - gemv
  - roofline
  - kernel-launch
  - local-llm
  - llm-inference
keywords:
  - RDNA4 decode latency breakdown
  - memory bandwidth utilization local LLM
  - RX 9070 XT 640 GB/s decode ceiling
  - batch one GEMV memory bound
  - kernel launch overhead decode
  - achievable vs peak memory bandwidth GPU
  - tokens per second ceiling Qwen3.5-9B
  - decode time per token budget
  - llama.cpp irreducible kernel overhead
  - weight streaming fraction decode
excerpt: "Yesterday's post said a chat turn is 96 percent decode and left decode stuck at 39.6 tok/s, a third of the RX 9070 XT's 116 tok/s bandwidth ceiling. This is where the other two thirds go. Streaming the weights is only 10.7 ms of a 25.2 ms decode token. The rest is latency the batch of one cannot hide, and the memory bus is innocent for most of it."
seoDescription: "A modeled per-token budget for Qwen3.5-9B decode on one RX 9070 XT: 10.7 ms streaming Q4_K_M weights, 5.4 ms of latency-bound norm, RoPE, attention and SwiGLU kernels, 7.0 ms of kernel dispatch and pipeline barriers, and 1.5 ms of LM head plus sampling, for 25.2 ms and 39.6 tok/s. Peak bandwidth caps decode near 116 tok/s but no GEMV kernel sustains peak, so the real streaming floor is about 93 tok/s. The gap from 93 to 39.6 is not the 640 GB/s bus, it is overhead the batch of one exposes, and it is the part ZINC can still win."
faqs:
  - question: "If decode is memory-bound, why is ZINC only at a third of the bandwidth ceiling?"
    answer: "Because most of a decode token is not spent moving weights. On Qwen3.5-9B and one RX 9070 XT, streaming the 5.5 GB of Q4_K_M weights takes about 10.7 ms at a realistic 80 percent of the 640 GB/s bus. The measured token takes 25.2 ms. The other 14.5 ms is latency-bound work: small norm, RoPE, attention and SwiGLU kernels, plus kernel dispatch and the pipeline barriers between them. Decode is bandwidth-bound in the limit, but only about 42 percent of the current token is actually bandwidth."
  - question: "Is the 116 tok/s decode ceiling real?"
    answer: "It is a peak-bandwidth ceiling, not an achievable one. 5.5 GB divided by 640 GB/s is 8.6 ms, which is 116 tok/s, but no real streaming kernel sustains peak bandwidth. The STREAM benchmark tradition puts sustained bandwidth at roughly 80 percent of peak on good hardware, which drops the honest streaming floor to about 10.7 ms and 93 tok/s. ZINC's 39.6 tok/s should be measured against 93, not 116, and it is at 43 percent of that."
  - question: "Where does the kernel dispatch overhead come from?"
    answer: "At batch one, each transformer layer runs a chain of small kernels that cannot overlap because each reads the previous one's output. A 40-layer model launches several hundred kernels per token, and every launch carries queue submission and a pipeline barrier. llama.cpp's maintainer measured this directly: replacing the kernel bodies with an immediate return still leaves significant per-token time that cannot be eliminated. That fixed cost is invisible at batch 64 and dominant at batch 1."
  - question: "So what actually raises decode throughput?"
    answer: "Two different levers for two different halves of the token. The streaming half falls with fewer weight bytes: smaller quantization, MoE, or speculative decoding. The overhead half falls with fewer, larger, better-overlapped kernels: fusing norm and RoPE into the projection, a single persistent attention kernel, and cutting pipeline barriers. The prefill month was one lever. The overhead half is a different engineering problem, and it is the one with the most slack left on a single-user local card."
draft: false
---

Yesterday's post ended on a number I did not explain. A chat turn on Qwen3.5-9B is 96 percent decode, decode runs at 39.6 tok/s, and the RX 9070 XT's memory bandwidth caps a perfect implementation near [116 tok/s](/blog/2026-07-14-a-qwen3-5-9b-chat-turn-spends-most-of-its-wall-clock-in-decode/). I said ZINC sits at about a third of that ceiling and that there was headroom in the implementation, then moved on.

This post is the accounting. If decode is a bandwidth wall, and ZINC is at a third of the wall, the interesting question is what the other two thirds are doing, because "memory-bound" is the reason people stop looking and it turns out to be only half true.

Here is the short version. Streaming the weights, the part the bus is actually responsible for, is 10.7 ms of a 25.2 ms decode token. Everything else is latency, and latency is not a bandwidth problem.

## The ceiling is lower than the spec sheet

Start by fixing the ceiling, because the 116 tok/s figure is optimistic in a specific way.

Qwen3.5-9B in Q4_K_M is about 5.5 GB. The [RX 9070 XT](https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070xt.html) moves up to 640 GB/s across a 256-bit GDDR6 bus at 20 Gbps. Divide 5.5 GB by 640 GB/s and you get 8.6 ms, which is 116 tokens per second. That is the number a roofline gives you, and it assumes the weight-reading kernel sustains the full rated bandwidth of the card.

No kernel does. The [STREAM benchmark](https://www.cs.virginia.edu/stream/), the thirty-year-old standard for sustained memory bandwidth, exists precisely because peak is not achievable: McCalpin's own summary notes that good machines reach roughly 80 to 85 percent of rated bandwidth on a simple streaming loop and many reach far less. A decode GEMV is a streaming loop with worse access structure than STREAM's, because it also reads quantization scales on a different stride and writes partial tiles. Take a generous 80 percent and the honest streaming floor is 5.5 GB over 512 GB/s, which is 10.7 ms and about 93 tok/s.

So the ceiling worth chasing is 93, not 116. ZINC's 39.6 tok/s is 43 percent of that floor, which is a more useful framing than "a third," because it says the remaining gap is not the bus. The bus, running flat out at a realistic sustained rate, only accounts for 10.7 of the 25.2 milliseconds a token actually costs.

## What the other 14.5 milliseconds are

A decode step is not one big matmul. It is a chain of small ones. Each transformer layer normalizes the hidden state, projects it to Q, K and V, applies rotary position encoding, runs attention against the KV cache, projects the result back, normalizes again, runs the gate and up projections, applies the SwiGLU nonlinearity, and runs the down projection. At batch one, every one of those steps is a matrix-vector product or an elementwise pass over a single 4096-wide vector, and none of them can start until the previous one's output lands in memory.

The big projections are where the weight bytes live, and their time is already counted in the 10.7 ms streaming term, because a batch-one GEMV is bound by reading its weight matrix and nothing else. What is not counted is everything between them.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-07-15-rdna4-decode-token-time-budget.svg" alt="A two-panel diagram on a dark background. The top panel is a single horizontal stacked bar representing one 25.2 millisecond decode token on a 0 to 27 millisecond axis. From left to right the segments are: weight streaming in teal, 10.7 milliseconds; a thin blue KV cache read, 0.6 milliseconds; attention, norms, RoPE and SwiGLU in purple, 5.4 milliseconds; kernel dispatch and pipeline barriers in orange, 7.0 milliseconds; and LM head over the 151k vocabulary plus sampling in pink, 1.5 milliseconds. A white dashed vertical line inside the teal streaming segment marks the peak-bandwidth ideal at 8.6 milliseconds and 116 tokens per second. A solid teal line at the right edge of the streaming segment marks the achievable streaming floor at 10.7 milliseconds and 93 tokens per second. Everything to the right of that line is overhead. The bottom panel is three horizontal bars in tokens per second: the peak-bandwidth ideal at 116 in grey, the achievable streaming floor at 93 in faded teal, and ZINC measured at 39.6 in solid teal, annotated as 34 percent of peak and 43 percent of the achievable floor. A legend labels the four cost categories: bandwidth-bound weight read, latency-bound small kernels, kernel dispatch and pipeline barriers, and LM head plus sampling." loading="lazy" />
  <figcaption>A modeled per-token decode budget for Qwen3.5-9B on one RX 9070 XT. The teal streaming segment is the only part the 640 GB/s bus governs; it ends exactly at the achievable floor. The 14.5 ms to its right is latency the batch of one exposes.</figcaption>
</figure>

Read the bar left to right. The teal block is the weight read, the only segment the bus governs, and it fills the track right up to the 93 tok/s floor. The three blocks after it are the story. Attention, the norms, RoPE and the SwiGLU activation are small kernels that touch little memory but each pay a full round trip of launch latency and a barrier before the next kernel can read their output. The orange block is pure dispatch: queue submission and pipeline barriers, hundreds of them per token, doing no arithmetic at all. The pink sliver is the LM head reading a 151k-row output embedding plus the sampler, a cost [this blog has measured before](/blog/2026-05-16-what-qwen3-151k-lmhead-costs-on-rdna4-decode/).

The segment sizes are a model, not a profiler trace, but the total is pinned to the measured 25.2 ms and the split is the shape every batch-one engine shows. The point does not depend on whether dispatch is 7.0 ms or 6.0. It depends on the streaming term being about 10.7, which is arithmetic, and the remainder being large, which is the observation.

## This overhead is known, and it is stubborn

The clearest evidence that decode overhead is real and irreducible comes from the reference engine. In a [llama.cpp discussion on memory bandwidth utilization](https://github.com/ggml-org/llama.cpp/discussions/3909), the project's maintainer describes replacing the Metal kernel bodies with an immediate `return` on the first line, so the kernels do no work, and finding that a significant per-token time remains. His words: an overhead that, at the time, they could not figure out how to eliminate. That residual is the orange block. It is the cost of launching and synchronizing the kernel chain, and it exists whether or not the kernels compute anything.

The reason it hurts decode specifically is arithmetic intensity. As Finbarr Timbers lays out in [a well-known explainer](https://finbarr.ca/how-is-llama-cpp-possible/), batch-one generation reads every weight to produce a single token, so latency is the max of compute time and memory time, and memory wins by two orders of magnitude. That is true, and it is why people call decode memory-bound. But the same batch of one that makes the GEMV memory-bound also makes the kernel chain latency-bound, because there is no second sequence in flight to hide the launch gaps behind. At batch 64 the dispatch cost amortizes across 64 tokens and disappears into the noise. At batch one it is a fixed tax on every token, and on a single-user local engine batch one is the whole workload.

The matrix core makes this vivid. The RX 9070 XT is rated at 195 TFLOPs of FP16 matrix throughput and 389 INT8 TOPS, both on the same [product page](https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070xt.html) as the 640 GB/s. During decode that entire matrix engine produces a few million FLOPs of useful work per step and spends the rest of the token idle, waiting on memory and on barriers. The month of prefill work doubled a rate that decode was never using, and the overhead half of the token does not touch the matrix core at all.

## Two halves, two different levers

Splitting the token in two also splits the roadmap in two, and the halves do not respond to the same fix.

| Half of the token | Cost | Bound by | The lever |
| --- | ---: | --- | --- |
| Weight streaming | 10.7 ms | Memory bandwidth | Fewer weight bytes |
| KV read | 0.6 ms | Memory bandwidth | Fewer KV bytes |
| Small kernels | 5.4 ms | Kernel latency | Fuse into the projections |
| Dispatch + barriers | 7.0 ms | Launch overhead | Fewer, larger kernels |
| LM head + sampling | 1.5 ms | Mixed | Vocab-parallel head |

The streaming half is the one every "bytes per token" post is about, and it is real: smaller quantization, an [IQ3 build](/blog/2026-05-29-the-imatrix-pass-that-decides-which-qwen3-weights-survive-iq3-m/), an [int8 KV cache](/blog/2026-05-19-fp8-kv-cache-is-the-next-decode-bandwidth-cut-rdna4-already-has-the-wmma-for/), or a mixture-of-experts model that reads only the active experts. Every one of those lowers the teal block, and none of them touches the 14.5 ms to its right.

The overhead half is a different craft. Fusing the pre-attention norm and RoPE into the QKV projection removes two kernel launches and two barriers per layer. A single persistent flash-attention kernel that keeps the whole attention step resident removes several more. Collapsing the gate, up and SwiGLU into one kernel removes a launch and an intermediate write, which is the same [wide-intermediate tensor](/blog/2026-07-10-down-projection-reads-back-a-swiglu-tensor-wider-than-the-hidden-state/) the prefill work already fought on the other side of the roofline. Cut the layer's kernel count from roughly a dozen to a handful and the orange block shrinks directly, no new bandwidth required.

That is the part I got backwards for a month. I chased the streaming ceiling because it was the honest bound in the limit, and in the limit it is. But ZINC is not in the limit. It is at 39.6 tok/s against a 93 tok/s floor, and more than half the gap is kernels and barriers, not the 640 GB/s bus. The bus is running at a realistic sustained rate for the fraction of the token it owns. The rest is the batch of one showing its cost, and the batch of one is the whole point of a local card.

## What I am building next

The concrete change is to profile the decode kernel chain the way we profiled the prefill GEMM, and to report two decode numbers instead of one: the streaming floor, which is bytes and quantization, and the achieved fraction of it, which is kernels and barriers. Quoting 39.6 tok/s against a 116 tok/s peak hid the fact that the peak is unreachable and that the real slack is in the overhead, not the bus.

Decode is memory-bound. That sentence is true and it is also where I stopped thinking for too long. On a single-user RDNA4 card the memory bus is doing its job for 10.7 milliseconds of every token, and the other 14.5 are the engine talking to itself. The next posts are going to be about making it talk less.
