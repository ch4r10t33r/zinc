---
title: "A day later: batched prefill, CUDA graphs, and the catalog re-measured on RTX 5090"
seoTitle: "RTX 5090: Batched Prefill & CUDA Graphs in ZINC"
date: "2026-06-13"
tags:
  - zinc
  - nvidia
  - cuda
  - rtx-5090
  - blackwell
  - tensor-cores
  - cuda-graphs
  - prefill
  - moe
  - qwen3
  - gemma
  - local-llm
  - llm-inference
  - gpu-kernels
  - kernel-fusion
  - llama-cpp
keywords:
  - RTX 5090 LLM prefill
  - tensor core prefill fp16 wmma
  - CUDA graphs LLM decode
  - CUDA graph replay inference
  - Qwen3.6 prefill NVIDIA
  - Gemma MoE prefill CUDA
  - batched prefill GEMM
  - launch-latency bound decode
  - ZINC vs llama.cpp prefill
  - Blackwell sm_120 tensor cores
  - GPU-side embedding lookup
  - local LLM RTX 5090 benchmark
faqs:
  - question: "How much faster is prefill in ZINC's CUDA backend now, and was it the tensor cores?"
    answer: "On the RTX 5090, catalog prefill roughly doubled on average (31 to 71 tok/s), with the largest jump on the Gemma-4-26B Mixture-of-Experts model at 8.3 to 106 tok/s — about 12.8x. It was not the tensor cores: that came from batching the routed-expert and dense GEMMs across all prompt tokens at once instead of a per-token, per-expert dispatch. The fp16 wmma tensor-core kernels now compile but stay opt-in, because prefill is launch-bound (the GPU sits near 10 percent utilization), so a faster matmul is an end-to-end wash until CUDA graphs cut the per-launch overhead. On dense Qwen prefill ZINC now edges ahead of llama.cpp (Qwen3.6-27B: 47.8 vs 28.8 tok/s)."
  - question: "What do CUDA graphs do for LLM decode?"
    answer: "Decode issues a long chain of tiny kernels per token — on a 60-layer model, hundreds of launches whose per-launch overhead and inter-kernel bubbles dominate when each kernel is small. A CUDA graph captures that whole per-token chain once and replays it as a single submission, so the driver stops paying launch cost per kernel. In ZINC it is an opt-in mode worth about 8 to 12 percent on the small dense Qwen3.5-9B, and it is size-gated: the win shrinks to nothing on larger models where each matvec is big enough to hide the launch bubble, and it cannot capture the mixed-quantization MoE path at all yet."
  - question: "Does moving the per-token embedding lookup onto the GPU speed up decode?"
    answer: "On its own, no — and that null result is the interesting part. Dequantizing the token's embedding row on the GPU instead of the CPU, and shrinking the host-to-device copy from a full row to four bytes, is bit-for-bit correct but measured perf-neutral on Qwen3.5-9B, because decode there is bound by GPU launch latency, not host work. Removing host work cannot move a wall clock the host is not on. Its value is as a building block: with the token id and embedding already GPU-resident, multiple decode steps can eventually be chained into one CUDA graph with no per-token host round-trip."
  - question: "Where does ZINC's RTX 5090 decode stand against llama.cpp now?"
    answer: "Across the five-model catalog it averages about 70 percent of llama.cpp decode, up from 51 percent on the previously published snapshot. The dense models are close — Qwen3.6-27B at 91 percent, Gemma-4-31B at 82 percent, Qwen3.5-9B at 75 percent — while the Mixture-of-Experts models still trail at 31 to 42 percent, where llama.cpp's years-tuned expert kernels keep the lead."
  - question: "Why were the published benchmark numbers so much lower than the current ones?"
    answer: "The published catalog snapshot was a correctness-first build from before the optimization work landed — no batched MoE experts, no kernel fusion, no tensor-core prefill. This post is the first full re-measurement after merging two parallel optimization lines, so the dashboard now reflects the engine as it actually runs rather than as it first booted. The merge was gated on a 5-of-5 token-for-token correctness check against llama.cpp before any number was trusted."
excerpt: "Yesterday's post ended with a to-do list: wire the tensor-core prefill, move the per-token glue onto the GPU, fuse more of Gemma. A day later the list is mostly shipped — and a lever that wasn't even on it, CUDA graphs. The honest scorecard: prefill roughly doubled (Gemma-26B MoE 8 to 106 tok/s), the catalog decode average went from 51 to 70 percent of llama.cpp, ZINC now beats llama on dense Qwen prefill — and one of the 'wins' measured as a perfectly correct no-op that taught us where the real next lever is."
seoDescription: "A day of NVIDIA work on ZINC's CUDA backend: batched prefill (Gemma-26B MoE prefill 12.8x), why fp16 tensor cores are an end-to-end wash until CUDA graphs land (prefill is launch-bound at ~10% util), CUDA-graph decode replay, full Gemma attention fusion, and an honest null result on GPU-side embedding — re-measured on RTX 5090 vs llama.cpp."
draft: false
---

[Yesterday's post](/blog/2026-06-12-four-bottlenecks-one-cuda-backend-moe-gemma-tensor-cores-rtx-5090-4090) was a tour of four different decode bottlenecks and ended, like most honest engineering posts, with a to-do list:

> Wire the fp16 tensor-core prefill · move the dense per-token glue onto the GPU · more Gemma fusion · close the MoE gap.

A day later, three of those are shipped, a fourth lever that *wasn't* on the list landed too — CUDA graphs — and the whole five-model catalog has been re-measured on the RTX 5090. This is what a day looks like when the levers were already sized: mostly wiring, one genuine surprise, and one "win" that turned out to be a perfectly correct no-op.

It also landed in an unusual shape. The prefill work and the decode work had been growing on **two separate branches** — one productionizing batched/tensor-core prefill, one chasing decode launch latency — and the first job today was merging them. That's an eleven-hunk collision across the CUDA kernel files (two refactors editing the same functions), so the merge was gated the only way a kernel merge can be honestly gated: rebuild and re-run the catalog token-for-token against llama.cpp. **5 of 5 models matched** before a single throughput number was trusted.

## The scoreboard, re-measured

Same box as before — an RTX 5090 (Blackwell, sm_120, 32 GB, 1792 GB/s) under WSL2, `ReleaseFast`, NVRTC-compiled kernels, the catalog GGUF files, greedy decode, measured over SSH against llama.cpp on the same hardware and files, medians of three runs.

First, the headline the to-do list was really about — **prefill**:

<figure class="diagram-card diagram-wide">

| RTX 5090 prefill (tok/s) | published | today | gain | vs llama.cpp |
| --- | ---: | ---: | ---: | ---: |
| **Gemma-4-26B-A4B** (MoE) | 8.3 | **106.3** | **12.8x** | 0.26x |
| **Qwen3.6-35B-A3B** (MoE) | 14.8 | **45.5** | 3.1x | 0.94x |
| **Gemma-4-31B** (dense) | 29.9 | **57.3** | 1.9x | 0.15x |
| **Qwen3.5-9B** (dense) | 64.6 | **97.6** | 1.5x | **1.14x** |
| **Qwen3.6-27B** (dense) | 39.2 | **47.8** | 1.2x | **1.66x** |

  <figcaption>Prefill throughput, RTX 5090. "published" is the previously-live correctness-first snapshot; "today" is the current build with the tensor-core / batched prefill path wired in. The "vs llama.cpp" column is the honest ceiling check: ZINC now <em>beats</em> llama.cpp on dense Qwen prefill, and is still a fraction of its heavily-tuned Gemma prefill even after a 12.8x self-improvement.</figcaption>
</figure>

And decode, the metric this series tracks, with the catalog now reflecting reality instead of a months-old boot:

<figure class="diagram-card diagram-wide">

| RTX 5090 decode (tok/s) | published | today | gain | vs llama.cpp |
| --- | ---: | ---: | ---: | ---: |
| **Gemma-4-26B-A4B** (MoE) | 8.3 | **47.5** | 5.7x | 31% |
| **Qwen3.6-35B-A3B** (MoE) | 16.3 | **52.9** | 3.2x | 42% |
| **Gemma-4-31B** (dense) | 33.9 | **46.9** | 1.4x | 82% |
| **Qwen3.5-9B** (dense) | 92.0 | **120.8** | 1.3x | 75% |
| Qwen3.6-27B (dense) | 47.7 | **50.5** | 1.06x | 91% |

  <figcaption>Decode throughput, RTX 5090. The catalog average moved from 39.6 to 63.7 tok/s — from 51% to 70% of llama.cpp on the same hardware. Read the multiples as <em>cumulative</em>: the published snapshot predated batched MoE experts and kernel fusion, so this is the gap between "as it first booted, correctly" and "as it runs today," not a single day's decode delta.</figcaption>
</figure>

<img class="diagram-visual" src="/blog/2026-06-13-rtx-5090-prefill-decode.svg" alt="Two stacked horizontal bar charts for the RTX 5090. Top, prefill tok/s published versus today: Gemma-4-26B MoE 8.3 to 106.3 (12.8x), Qwen3.6-35B-A3B 14.8 to 45.5, Gemma-4-31B 29.9 to 57.3, Qwen3.5-9B 64.6 to 97.6, Qwen3.6-27B 39.2 to 47.8. Bottom, decode tok/s published versus today: Gemma-4-26B 8.3 to 47.5, Qwen3.6-35B-A3B 16.3 to 52.9, Gemma-4-31B 33.9 to 46.9, Qwen3.5-9B 92 to 120.8, Qwen3.6-27B 47.7 to 50.5." loading="lazy" />

Two honest framings on top of that. The win: prefill roughly doubled on average (31 → 71 tok/s), the dense decode models are all within striking distance of llama.cpp, and on dense Qwen *prefill* ZINC is now ahead. The gap: the MoE models still decode at 31–42% of llama.cpp, and Gemma prefill — even up 12.8x — is a quarter of llama's number, because the tensor-core path is *wired*, not yet *tuned*.

## What actually landed

### Prefill — batching shipped the jump; tensor cores were a red herring

Yesterday's post measured an fp16 `wmma` prefill GEMM at ~2.2x the fp32 register-blocked kernel, blocked only on NVRTC's missing `-I/usr/local/cuda/include`. That flag landed, so the tensor-core kernels compile in-tree now — but the prefill jump did **not** come from them. It came from **batching**.

What went default-on is the batched (not tensor-core) path: one register-tiled GEMM over **all prompt tokens at once** (`Y[T,M] = A[T,K]·W[M,K]ᵀ`) instead of a per-token matvec, and for the MoE models the routed-expert matmuls batched across tokens with a GPU-side work list. Gemma-4-26B went **8.3 → 106 tok/s** because the old path paid per-token *and* per-expert dispatch on a 128-expert model; batching collapses both. Dense Qwen prefill already skipped the wasted LM-head matvec on prompt-internal tokens, and with the tiled GEMM it now clears llama.cpp (**47.8 vs 28.8 tok/s on the 27B**).

The fp16 `wmma` tensor-core GEMM is ~2.2x faster *in isolation* — and it stays **opt-in, off by default, because end-to-end it's a wash.** Profiling a batched gemma-31B prefill shows why: the GPU sits at **~10% utilization**. Prefill is **launch-bound**, not compute-bound — idle between kernels across a ~960-launch-per-prompt chain, not crunching the GEMM. A faster matmul can't speed up a GPU that's mostly idle, which is also why gemma prefill is still 0.15x of llama's tuned number: the gap is launch overhead, not the math. The real prefill lever is the one that won decode — **CUDA graphs over the prefill chain** — and that effort is now running. Tensor cores will matter once the GPU is busy; first it has to stop being idle.

### CUDA graphs — the structural answer to launch latency

This is the lever that wasn't on yesterday's list, and it's aimed squarely at yesterday's most useful finding: that Gemma decode is **launch-latency bound**, idle inside the GPU waiting on its own launch queue across ~180 tiny serial dispatches per token, where removing host syncs does nothing.

A CUDA graph attacks that directly. Capture the entire per-token kernel chain — embed, every layer, the final norm/LM-head/argmax — into a `CUgraphExec` **once**, then replay it as a single submission. The driver stops paying launch overhead per kernel; only the per-token push-constant scalars change, which is a cheap in-place exec update. An isolated proof clocked the replay at ~9x the cost of relaunching the chain at the real ~60-layer length.

Wired into decode (behind `ZINC_CUDA_GRAPH`), interleaved A/B puts it at **~8–12% on the small dense Qwen3.5-9B**, with the embedding upload and the argmax readback folded into the graph as pinned-memory copy nodes so the whole token drains on one sync. And then it's honest about its ceiling: the win is **size-gated**. On the 27B it's a measured no-op — the matvecs are big enough that the launch bubble is already negligible — and it can't capture the catalog's MoE path at all, because those experts carry mixed quantization and the captured topology has to be identical every step. So it ships opt-in, a real win exactly where the model is small and the launches dominate, and nowhere else. That's a more useful result than a universal speedup would have been: it tells you precisely which models are launch-bound.

### Gemma fusion — the stacked 1%s, completed

The full attention-fusion stack from yesterday's "more Gemma fusion" line is in: the three per-head V/Q/K RMS-norms collapsed into one launch, the same-input Q4_K matvec pairs fused, and the per-layer pre-norms folded into the *preceding* block's post-norm+residual so the four per-layer norm boundaries land at two fused launches. Each is a 1–2% kernel win that only clears this box's wandering-boost noise floor when you stack them and require the fused build to win *every* interleaved round — which is how a 1.5% fusion becomes a number you publish instead of boost noise you regret.

### GPU-side embed — a correct no-op, and why it matters anyway

The last to-do item, "move the dense per-token glue onto the GPU," shipped — and measured as a **perfect no-op**, which is the most instructive result here. Dequantizing the token's Q4_K embedding row on the GPU (reading the id from a 4-byte device buffer) instead of on the CPU, shrinking the host→device copy from a full embedding row to four bytes, is bit-for-bit identical output and **perf-neutral** on the 9B: a 400-token interleaved A/B landed inside ±2.4% boost noise.

It's neutral for the same reason the CUDA graph is *not* — decode here is bound by GPU launch latency, not host work, so removing host work can't move a wall clock the host isn't standing on. The value isn't the kernel; it's the **primitive**. With the token id and its embedding already GPU-resident, the real next lever becomes possible: chaining several decode steps into a *single* CUDA graph with no per-token host round-trip at all. The no-op is the groundwork for the lever that isn't a no-op.

## The honest part: a wandering clock and a stale scoreboard

Two caveats keep this post from overclaiming. First, the published numbers it improves on were a **correctness-first snapshot** — the engine as it first booted, before any optimization — so the big decode multiples are the cumulative gap, not a single day's decode delta. The clean *day-over-day* win is prefill; decode's day-over-day movement is real but partly inside the 5090's boost variance, which on this box is wide enough that a single before/after reading on a 1.5% change is worthless. Every sub-noise number here came from interleaved A/B, not a naive comparison.

Second, the gap that's still open is the one yesterday named and this day didn't close: **MoE decode**. The router still keeps a per-layer sync, the expert matvecs are small-M and occupancy-starved, and llama.cpp's expert kernels are years ahead. 31–42% of llama is up from where it was, and it's still the frontier.

## What's next

- **Close the MoE decode gap** — a GPU-side gather that drops the last router round-trip, and a multi-row expert kernel for the small-M matvecs.
- **Tune Gemma prefill** — the tensor-core path is wired and 12.8x up on MoE, but dense Gemma prefill is still a fraction of llama's; the kernel needs the occupancy and pipelining work, not just the wiring.
- **Cross-step graph chaining** — now that embed and argmax are GPU-resident, chain N decode steps into one graph and delete the per-token host round-trip entirely. The neutral GPU-embed result is the door to this one.
- **MoE under graphs** — capture needs uniform expert quantization; a single-quant MoE build would unlock the launch-latency win for the experts too.

None of it needs new hardware. The recurring method from both posts held: a llama.cpp reference to diff every token against, a profiler to say *which* wall — sync, launch, dispatch, dequant — you're standing in front of, and boost-aware interleaved A/B so a real 1.5% win isn't drowned by a clock that won't sit still. A day's worth of that, and the scoreboard finally tells the truth.

*ZINC is a from-scratch local inference engine with Vulkan (AMD RDNA), Metal (Apple Silicon), and CUDA (NVIDIA) backends — one engine, hand-written kernels, no heavyweight frameworks. The CUDA backend runs all five catalog models coherently on RTX 4090 and 5090, validated token-for-token against llama.cpp.*
