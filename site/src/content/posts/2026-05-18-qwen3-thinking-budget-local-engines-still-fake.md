---
title: "Qwen3's thinking budget is a cloud feature local engines still have to fake"
date: "2026-05-18"
tags:
  - zinc
  - rdna4
  - amd
  - vulkan
  - qwen3
  - reasoning
  - thinking-budget
  - thinking-mode
  - llama-cpp
  - chain-of-thought
  - sampler
  - local-llm-inference
keywords:
  - Qwen3 thinking budget
  - enable_thinking false llama.cpp
  - Considering the limited time by the user
  - Qwen3 reasoning mode local inference
  - thinking budget two-pass workaround
  - ThinkingTokenBudgetProcessor LogitsProcessor
  - Qwen3 chat template kwargs
  - llama.cpp issue 20182
  - reasoning-parser qwen3
  - Qwen3 technical report thinking mode fusion
excerpt: "Alibaba Cloud Model Studio caps a Qwen3 reasoning trace at the token budget you ask for, inserts the same fixed sentence the technical report names, and resumes decoding into the answer. Open-source engines do not. llama.cpp cannot reliably turn thinking off, has no budget, and exposes neither the splice nor the logit warp that would emulate one. On a 117 tok/s Radeon AI PRO R9700 that is fifteen seconds of avoidable reasoning every time the model picks a math problem to chew on."
---

A Qwen3 user types "what is 52 squared" into a local chat tab. Qwen3-8B at temperature zero, decoding at 117 tok/s on a Radeon AI PRO R9700, opens a `<think>` block and reasons about whether 2,722 might be a perfect square. It walks the multiplication table from 50 squared to 54 squared, picks the right answer, double-checks with a linear approximation, and finally closes the `</think>`. The number was 2,704. The model spent 1,800 tokens, fifteen seconds of wall clock, getting to it.

If the same prompt hits Alibaba Cloud Model Studio with `thinking_budget=256`, the reasoning trace stops at 256 tokens, an early-stopping sentence is appended, and the model writes "2704" in another twenty tokens. The user gets the answer in two seconds. The local engine has no equivalent. That is the gap this post is about.

The interesting part is that the budget is not a model feature. It is a runtime pattern that Qwen3 happens to support naturally, that the [Qwen3 Technical Report](https://arxiv.org/html/2505.09388v1) names in one specific sentence, and that no open-source local engine has fully wired in yet.

## What the technical report actually says

Section 4.3 of the Qwen3 paper introduces Thinking Mode Fusion: the post-training stage that takes a long-CoT reasoning model and teaches it to also answer in non-thinking mode in the same checkpoint. The fusion is what makes the same Qwen3-8B able to answer "hello" in 5 tokens and "prove that the square root of 2 is irrational" in 4,000.

The paper notes a side effect of that fusion. "Once the model learns to respond in both non-thinking and thinking modes, it naturally develops the ability to handle intermediate cases, generating responses based on incomplete thinking." Then it gives the operational recipe in one sentence:

> when the length of the model's thinking reaches a user-defined threshold, we manually halt the thinking process and insert the stop-thinking instruction: "Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>.\n\n".

That is the budget controller. A token threshold, a fixed splice string, and a continuation. The paper is careful to say the splice was not explicitly trained for; it emerges from the fusion. Figure 2 of the same report shows AIME'24, GPQA, MATH, and LiveCodeBench accuracy climbing roughly log-linearly with the budget across the Qwen3-32B and Qwen3-235B-A22B configurations.

The Alibaba Cloud Model Studio API exposes that knob as a `thinking_budget` integer. Local engines do not.

## What local engines have right now

There are three switches local engines surface today, none of which is a budget.

The first switch is the `enable_thinking` chat template kwarg. It is what Hugging Face's tokenizer exposes through `apply_chat_template`, and what [the Qwen3 announcement post](https://qwenlm.github.io/blog/qwen3/) prints in its `transformers` quickstart. Setting it to false concatenates an empty `<think></think>` block onto the assistant's start-of-response, which tells the fused model to skip reasoning. It works in `transformers` and in vLLM. In llama.cpp it is, currently, [open issue 20182](https://github.com/ggml-org/llama.cpp/issues/20182): a user passed `--chat-template-kwargs '{"enable_thinking": false}'` to `llama-cli`, typed "hello", and got a thinking trace anyway. The fix is queued in [PR 20329](https://github.com/ggml-org/llama.cpp/pull/20329) but the soft switch and the hard switch are not fully aligned in mainline as of this week.

The second switch is the `/no_think` and `/think` flags Qwen3 has been trained to obey inside the user message. They flip the same template-level toggle as `enable_thinking`. They are also all-or-nothing. No budget.

The third switch is the one nobody actually wants: cap the assistant's total output length, hope the model finishes thinking before it hits the cap, and accept a truncated answer when it does not. SGLang's `--reasoning-parser qwen3` and vLLM's `--enable-reasoning --reasoning-parser deepseek_r1` separate the reasoning tokens from the final content in the response stream, but neither caps the reasoning length on its own.

The only ways to actually enforce a budget in open source today are workarounds, and there are exactly two of them.

## The two workarounds, and what each one costs

The official Qwen workaround is documented in [thinking_budget.md](https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/thinking_budget.md) in the Qwen3 repo. It runs two API calls. The first call sends the user message with `max_tokens=thinking_budget`. If the reasoning trace closes with `</think>` before the cap, the response is already done. If the cap is reached without a close, the client appends the fixed sentence from the technical report, wraps it in `<think>...</think>`, and replays the prompt with `continue_final_message=True` for a second call. The second call prefills the early-stopping splice on top of the cached prefix and decodes the answer.

The unofficial workaround is the one Zach Mueller wrote up in [a TIL post](https://muellerzr.github.io/til/end_thinking.html) the week Qwen3 first dropped. It is a Hugging Face `LogitsProcessor` named `ThinkingTokenBudgetProcessor` that watches generated tokens and, at the budget boundary, sets every logit except `\n` and `</think>` to negative infinity. The model is forced to emit the close, the `<think>` block ends, and decoding continues into the answer. The 95% mark gets a gentle prior bump toward closing tokens so the transition does not feel like a hard slam.

Each path has a different bill.

The two-pass splice pays for an extra prefill of the early-stopping sentence on the second call. The fixed splice is 28 to 32 tokens depending on tokenizer rounding, plus the boilerplate from the chat template. On a Qwen3-A3B Vulkan prefill at the [88 tok/s zinc reports against its own benchmark suite](/blog/2026-05-09-how-we-made-amd-qwen-inference-faster-than-llama-cpp-in-six-weeks-on-the-radeon-ai-pro-r9700), that splice is around 350 ms before the answer starts to stream. The KV cache from the first pass is reused for everything before the splice, so the marginal cost is just the splice and whatever fragment of `<think>` body was already in the cache.

The logit warp pays nothing in extra prefill. It runs as a single forward pass and the cache state is whatever the natural decode produced through token N-1. The hidden cost is in the model's own gradient: the warp does not just close the thinking, it overrides whatever the model would have done next, which on a long trace can be subtly different from the "natural" close the two-pass approach gets after seeing the early-stopping instruction. Mueller's writeup includes an honest counter-example where capping a 100-token reasoning trace produced a worse answer than 600 tokens of natural thinking on a square-root prompt. The budget is not a free win.

## What the splice looks like in flight

The two-pass and single-pass paths produce the same final token stream from the model's point of view but route through different parts of the local engine's pipeline.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-05-18-qwen3-thinking-budget-paths.svg" alt="A three-row pipeline diagram on a warm amber background. Each row is a horizontal token timeline for the same prompt 'what is 52 squared' rendered as a sequence of small tiles colored by role. Row one is labeled 'Alibaba Cloud Model Studio native budget' and shows from left to right: a short violet user-prompt tile, a medium pale-yellow think-open tile labeled less-than think greater-than, a band of 256 medium-yellow thinking tiles capped by a vertical orange line labeled budget 256, a compact gold splice tile labeled 'Considering the limited time' tucked inside the think band, then a darker amber think-close tile labeled less-than slash think greater-than, then a short olive answer tile labeled '52 squared = 2704'. A note above the row reads 'single API call, server-side splice, total decoded tokens 280, wall clock 2.4 s at 117 tok per s'. Row two is labeled 'Local two-pass workaround per Qwen3 docs' and shows the same prompt, two API calls divided by a vertical dashed line. The left half shows the user prompt, think-open, and 256 thinking tiles ending at the budget line, with a red x labeled 'first call returned without a close-think token'. The right half shows the same prompt re-played from the left, the cached KV bar drawn as a thin green ribbon underneath, then the splice tile prefilled in cyan to mark a new prefill chunk, then the close-think, then the answer. A note above the row reads 'two API calls, extra 32-token prefill, wall clock 2.7 s, extra 0.35 s for the splice prefill at 88 tok per s'. Row three is labeled 'Local logit-warp workaround per Mueller TIL' and shows a single timeline matching row one's geometry but with the budget line drawn as a soft yellow gradient where the warp pushes toward the close-think token starting at 95 percent of the budget. A small ramp icon above the budget line is labeled 'logit prior bump toward newline and close-think'. The close-think and answer continue inline with no splice tile. A note above the row reads 'single forward pass, zero extra prefill, may emit a slightly different close than the two-pass route'. A footer caption reads 'budget 256 tokens, prompt 8 tokens, splice 32 tokens, answer 20 tokens, decoded at 117 tokens per second and prefilled at 88 tokens per second on the Radeon AI PRO R9700 in the May 10 Qwen3.6-35B-A3B benchmark suite'." loading="lazy" />
  <figcaption>The native budget closes the trace server-side and never round-trips. The two-pass route is one extra prefill, the splice prefill itself is cheap, and the cache from the first pass is reused. The logit warp avoids the extra prefill but only sees the close, not the early-stopping prompt the model was trained to respond to.</figcaption>
</figure>

Two things are worth noticing in the diagram. One: the two-pass route's extra prefill is small in absolute terms because the splice is only 32 tokens long, but it is exactly the kind of cost that becomes structurally visible once an engine is running thousands of bounded reasoning queries a day. Two: the logit warp does not interact with the KV cache at all, which makes it the lowest-overhead path on a Vulkan engine that is already tight on host-GPU sync points, but it also changes which tokens the model emits when the budget bites. Those two paths trade against each other.

## Where this lives on RDNA4

The splice point is a small prefill chunk in the middle of an active decode. The wave32 attention fix from [the May 11 commit](/blog/2026-05-11-the-wave32-commit-that-closes-rdna4-long-context-flash-attention-gap) is the kernel doing the work on the splice chunk; the Q4_K_M decode path runs the answer that follows. The KV cache up to the splice is the cache the first-pass decode just produced, which means the splice does not pay for a recompute of the user prompt or any of the partial thinking trace.

The logit warp does not need a kernel at all. It is a sampler-side modification: take the logits the LMHead just emitted, set every token outside `{nl, </think>}` to negative infinity at the budget boundary, and let the existing argmax or min-p sampler pick the close. The [LMHead matmul cost from yesterday's post](/blog/2026-05-16-what-qwen3-151k-lmhead-costs-on-rdna4-decode) is unchanged. The sampler runs on the CPU side in most local engines and the warp is a half-page of code.

The hard part is not the kernel. It is the contract. A budget controller has to commit to one of the two paths, expose it through the chat API, and document what happens at the boundary. The cleanest open answer is probably "logit warp by default, with a `--thinking-splice-prompt` flag that switches to the two-pass route for parity with the Alibaba cloud behavior". Neither llama.cpp nor SGLang ships that today. The Hugging Face `transformers` stack ships only the LogitsProcessor variant, and only because individual users have written it. Recent academic work on dynamic budgets, including [BudgetThinker](https://arxiv.org/abs/2508.17196), points to a more sophisticated controller that watches the trace and stops when entropy or self-confidence crosses a threshold, but the static budget is the right first cut for local.

## The tradeoff

The case for shipping a budget controller is that the local reasoning experience is currently disproportionately bad on the slow end. A 117 tok/s decode at temperature zero on a 32 GB R9700 is competitive with what an H100 will give a single user from a hosted Qwen3-30B-A3B endpoint. The thing the H100 endpoint can do that the R9700 can today not do is cap the reasoning. On a workload of arithmetic, factual recall, and short summarization, half of every Qwen3 response is reasoning that the user did not ask for and will not read.

The case against is the accuracy floor. Bounding the budget shaves seconds off easy prompts and shaves correctness off the hard ones. The Qwen3 technical report's own figures show the AIME'24 score climbing from roughly 50 to 85 as the budget grows from 1k to 32k tokens. A user who sets a global cap of 1k tokens is asking for the bottom of that curve on the prompts that need the top.

The right interface is therefore not a single global cap. It is a default cap, an opt-out for hard problems, and a way for the model to signal that it could not finish. The `<think>...</think>` block already gives the engine a clean signal. A budget hit that lands inside the block can be exposed as a `truncated_thinking=true` flag on the response, the way Anthropic exposes max-tokens truncation today. None of that is exotic. None of it is shipping in open source.

## What comes next

For zinc specifically, the path is clear and small. Implement the logit warp in the sampler as a `ThinkingBudget` mode behind a flag, expose the budget through the existing chat completion API, and validate that the budget hit still produces a closed `<think>` block in the parsed response. Add the two-pass splice as a server-side option for parity with the Alibaba endpoint. Run the [batch-invariance tests from May 14](/blog/2026-05-14-temperature-zero-is-not-deterministic-for-local-qwen3-on-rdna4-yet) against the budget mode so that the budget choice does not silently change the reduction order downstream.

The bigger story is that local inference has now closed the kernel-side gap with the cloud for Qwen3 decode and is starting to open up an interface gap. Cloud APIs ship features like the thinking budget, structured grammars, and prompt caching that local engines treat as adjacent concerns. They are not adjacent. They are the next set of things users want, and they are the next set of things the engine has to own. The thinking budget is the smallest of the three and the easiest to ship. It is also the one Qwen3 went out of its way to make trivially controllable. The line the model was trained to respond to is in the technical report, in plain English. The work is to wire it up.
