---
title: "Timeline semaphores let zinc queue decode steps without a fence wait per token"
date: "2026-05-30"
tags:
  - zinc
  - rdna4
  - amd
  - vulkan
  - timeline-semaphore
  - vkfence
  - synchronization
  - decode-loop
  - llama-cpp
  - llm-inference
keywords:
  - Vulkan timeline semaphore local LLM decode
  - VK_KHR_timeline_semaphore inference loop
  - per-token vkWaitForFences cost
  - Radeon AI PRO R9700 Vulkan submission overhead
  - llama.cpp Vulkan fence per decode step
  - Vulkan 1.2 monotonically increasing counter
  - host roundtrip between Vulkan submissions
  - wait-before-signal Vulkan timeline
  - vkSignalSemaphore host kick GPU
  - VkSemaphoreType TIMELINE for decode batching
excerpt: "A 100 tok/s decode loop on the Radeon AI PRO R9700 leaves about ten milliseconds per token, and llama.cpp's Vulkan backend currently spends a measurable slice of that on a host-side vkWaitForFences between every submission. The fence is doing what a fence has to do: hold the CPU on the device until the GPU is idle, so the next submission can be safely recorded and queued. Timeline semaphores, core in Vulkan 1.2 since 2020, replace that with a single 64-bit counter the GPU bumps as each submission completes and the host queries when it wants the next result. zinc uses the counter to record three or four decode submissions ahead, hand them to the queue in one burst, and let the GPU pull them back-to-back while the CPU prepares the sampler input for whichever token finishes next. The host roundtrip disappears, and the decode loop turns into a pipeline rather than a ping-pong."
---

A decode step on Qwen3-30B fits in roughly ten milliseconds on the [Radeon AI PRO R9700](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html). The forward pass is GPU work, the sampler is mostly CPU work, and the seam between them is the part of the loop that has quietly become the most expensive thing nobody is measuring. Every token in a vanilla Vulkan inference engine pays for the GPU to drain, the host fence to wake up, and the next command buffer to land before the GPU sees anything to do.

That seam is a binary `VkFence` and a host call to `vkWaitForFences`. It is the synchronization primitive llama.cpp's Vulkan backend reaches for by default, and it is the right answer for a backend that has to support eight different vendor drivers without surprises. It is also a host-to-device roundtrip per submission, which on a fast decode loop is the kind of cost that grows as the GPU gets faster.

Timeline semaphores are the part of Vulkan 1.2 that lets a local engine stop paying it. They were introduced in early 2020 by [Khronos as VK_KHR_timeline_semaphore](https://www.khronos.org/blog/vulkan-timeline-semaphores) and promoted into core, and the API they expose is small enough to fit into one zinc commit. The interesting part is what they let the decode loop look like once the host fence comes out.

## What the fence is buying

A fence is a one-shot binary signal the GPU sets when a submission finishes, which the host can wait on with `vkWaitForFences`. llama.cpp's Vulkan backend creates one per batched submission, signals it as the last act of the queue, and blocks the worker thread on it before recording the next command buffer. That is the safe shape, because the recorded buffer references descriptor sets and intermediate buffers that the GPU might still be reading.

The cost is a host-to-device roundtrip per submission. On a typical desktop with a Vulkan driver that uses `KHR_external_fence_fd` on Linux or `D3DKMTWaitForSynchronizationObjectFromCpu` on Windows, that roundtrip is a sub-millisecond hop, but it is one whole CPU-context-switch and one whole kernel wake on every decode token. On the [Radeon AI PRO R9700](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html) at 100 tok/s, that is once every ten milliseconds, and the gap between "GPU drained" and "next submit visible" is empty time the GPU could have been computing the next forward pass.

The pattern shows up in every local Vulkan inference loop, including llama.cpp's [Vulkan multi-GPU implementation in PR #5321](https://github.com/ggml-org/llama.cpp/pull/5321) and any single-card variant of the same code, because the upstream `ggml_backend_vk_graph_compute` waits on a fence before letting the host record the next graph. The fence is not bug, it is the API doing what it was specified to do. The fix is to ask for a different API.

## What the counter changes

A timeline semaphore is a `VkSemaphore` with a 64-bit monotonically increasing counter instead of a binary state. The GPU bumps the counter when a submission completes, the host queries it with `vkGetSemaphoreCounterValue` or blocks on it with `vkWaitSemaphores`, and the host can also bump it directly with `vkSignalSemaphore`. The Khronos [Vulkan-Samples timeline_semaphore walkthrough](https://docs.vulkan.org/samples/latest/samples/extensions/timeline_semaphore/README.html) describes it as "viewing a `VkQueue` as a sequence" of monotonic counter values, which is the framing that makes the rest fall out.

Three properties of the counter matter for a decode loop. The first is that a submission can wait on a value the GPU has not produced yet, which the spec calls wait-before-signal. The second is that one signal can be waited on by many things without consuming it, so the host and any other queue can both wait on counter `N` without racing. The third is that a `VkSemaphore` of `TIMELINE` type is a superset of `VkFence`, so the host wait that the fence used to do is now a `vkWaitSemaphores` against a value.

Put those together and the decode loop has a different shape. The host records command buffers for submissions one through four, queues them with `vkQueueSubmit` using signal values one through four, and starts preparing the sampler input for token one. When `vkGetSemaphoreCounterValue` reports `>= 1`, token one is ready and the host runs its sampler on it. While the host samples, the GPU is already executing submission two, because submission two does not wait on a host wakeup; it only waits on the queue ordering. The CPU work and GPU work overlap, instead of taking turns.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-05-30-vulkan-timeline-semaphore-decode-loop.svg" alt="A two-band timeline schematic on a deep indigo background titled one fence per token versus one timeline counter for the whole decode burst, with a subtitle noting that a 100 tok/s decode budget on the Radeon AI PRO R9700 leaves 10 ms per step. The top band, binary VkFence, one wait per decode step, shows a CPU lane and a GPU lane sharing a time axis. Each decode step is a short amber rec+submit block on the CPU lane, a long pink hatched vkWaitForFences block covering the same time the GPU is busy, and a cyan GPU forward and sample block. Between each pair of GPU blocks a dashed pink marker labels an idle gap, GPU drained, next submit not yet visible. A grey bracket reads one token, record, submit, wait, read, repeat. The bottom band, VK_KHR_timeline_semaphore, one counter for N decode steps, shows the same lanes but the CPU lane starts with a single amber block, queue 4 submits signal vals 1 through 4. The GPU lane is four cyan forward and sample blocks placed back to back with no gaps, each labeled GPU step k arrow signal k. A small gold staircase line below the GPU lane steps from counter equals 0 up through 1, 2, 3, 4 as each block completes. A small pink hatched block at the right reads wait greater than or equal to 4, labeled host roundtrip happens once at the end of the burst, not between every step. A footer caption notes the layout is schematic with proportional lane widths, and credits the Khronos introduction of VK_KHR_timeline_semaphore in Vulkan 1.2 and the Vulkan-Samples timeline_semaphore walkthrough for the mechanism." loading="lazy" />
  <figcaption>Top: a binary fence forces a host wait between every submission, and the GPU lane shows a small gap at every boundary while the next submit is not yet visible. Bottom: the counter lets the host queue the whole burst, the GPU drains it back-to-back, and the only host roundtrip is the single wait at the end. The chart is schematic; relative durations match the typical CPU and GPU split of a 10 ms decode step on RDNA4.</figcaption>
</figure>

The story the chart tells is the boundary between the two GPU blocks on the top band. That sliver is empty because the host has not gotten the next submission to the queue yet. On the bottom band the sliver is gone, because the queue already had the next submission pending when the previous one signaled its counter value.

## What this looks like in zinc

The version zinc has been carrying lives in the same place as the prior fence wait. The command-buffer recorder is unchanged, the descriptor sets are unchanged, the actual compute shaders that move tensors around RDNA4 are unchanged. The only edits are the synchronization primitives, and the structure is essentially the example out of the Khronos blog.

```zig
// Decode-loop sync, abbreviated. One timeline semaphore for the whole session.
const timeline = try createTimeline(dev, .{ .initial = 0 });

var submit_value: u64 = 0;
var reaped_value: u64 = 0;

// Pre-record a small ring of command buffers, one per in-flight decode step.
// `inflight_depth` is the number of submissions we let the GPU stay ahead.
var depth: usize = 0;
while (decoding) : (depth += 1) {
    if (depth >= inflight_depth) {
        reaped_value += 1;
        try waitTimeline(dev, timeline, reaped_value); // host blocks here, once per N
        const tok = sampleFromKvSlot(reaped_value);
        try feedNextToken(tok);
        depth -= 1;
    }
    submit_value += 1;
    try recordDecodeStep(cmd_ring.next(), submit_value);
    try submitSignaling(queue, cmd_ring.last(), timeline, submit_value);
}
```

What matters is that the `while` body has no `vkWaitForFences` in it on the common path. The wait only happens when the host is `inflight_depth` submissions behind, which is the natural backpressure: the GPU is allowed to stay a few steps ahead of the host without the host blocking, and the host blocks exactly long enough to keep that depth steady.

`inflight_depth` is the parameter to tune, and the right value is small. Two is enough to hide the host roundtrip. Three covers most jitter in the host scheduler. Four is the point where command-buffer pool memory and KV slot fragmentation start to cost more than the latency hiding saves, and is where zinc lands by default. Anything past four is producing tokens the user has not seen yet and might never see if the chat gets cancelled, which means wasted GPU work on a single-user engine and exactly the wrong tradeoff. The depth that worked for batched serving is not the depth that works for a chat window.

## What the host stops doing

The host roundtrip is the visible piece, and it is not the only one the counter removes. Three smaller things go with it.

The first is the command-buffer reset. Binary fences gate command buffer recycling, and a typical Vulkan engine resets and rerecords each buffer after its fence signals. With timeline semaphores the same logic applies, but the host has a 64-bit ordering to reason about, so a ring of buffers can be reset in any order the counter says is safe without per-buffer fences to track. The bookkeeping shrinks to one comparison per ring slot, which is the same observation the Khronos blog makes about [object bloat](https://www.khronos.org/blog/vulkan-timeline-semaphores) being one of the original binary-semaphore pain points.

The second is the cross-queue handoff. A decode loop that wants to overlap LMHead sampling on a compute queue with the next forward pass on a graphics queue can do it with a binary semaphore, but only at the cost of allocating one binary semaphore per submission pair. With a timeline the two queues share one counter, the compute queue waits on value `N` to start sampling, and the graphics queue waits on value `N+1` to start the next forward pass. The two queues are wired by an integer rather than by objects.

The third is the wait-before-signal that the spec calls out as the structurally hardest fix to do with binary primitives. A speculative-decode draft path that submits a verify step before the draft step has signaled, for example, deadlocks under binary semaphores and works under a timeline because the verify submission is allowed to be queued before the value it waits on exists. This is the path zinc would take if it ever revisits the [speculative decoding question we shelved on Qwen3-A3B](/blog/2026-05-25-speculative-decoding-on-qwen3-a3b-loses-even-at-100-percent-draft-acceptance), since the loop shape under speculation needs exactly this property.

## Where it does not buy anything

There are three places the timeline semaphore is a wash or worse, and the honest version of the story names them.

It does not help on the swapchain. The Vulkan window-system integration APIs still take binary semaphores, as the Khronos blog and [Vulkan-Samples documentation](https://docs.vulkan.org/samples/latest/samples/extensions/timeline_semaphore/README.html) both note. zinc has no presentation path in its inference loop, so this caveat does not apply, but anyone porting the trick to a renderer should expect to keep a binary semaphore for `vkQueuePresentKHR` and pay the rules for mixing the two.

It does not help if the per-step GPU work is large enough that the host fence is already overlapped with it. A prefill graph that runs for 60 ms per chunk leaves so much GPU time that the host fence wake comes back before the GPU needs anything new, and the counter saves nothing measurable. Prefill is not where this lives. Decode is, because decode is short enough per step that the host roundtrip is a meaningful slice of the budget.

It also does not help on drivers that emulate `VK_KHR_timeline_semaphore` rather than implement it natively. The [Vulkan-ExtensionLayer](https://github.com/KhronosGroup/Vulkan-ExtensionLayer) repository ships exactly that emulation as `VK_LAYER_KHRONOS_timeline_semaphore`, and it is a faithful API but still backs the counter with binary fences under the hood, so the host roundtrip is still happening, just hidden. All the drivers zinc cares about, recent Mesa RADV on RDNA4, recent AMDGPU-Pro, and the Intel Linux driver in the Arc B70 timeline we wrote about in the [Intel Arc Pro B70 deep dive](/blog/2026-05-18-intel-arc-pro-b70-deep-dive-and-zincs-t-intel-plan), support `VK_KHR_timeline_semaphore` as a native feature, so the layer is only relevant as a fallback for embedded targets.

## What zinc actually does with it

The thing zinc carries forward is the framing. A local engine on a fast card spends most of its decode budget on the GPU, but the host work between GPU bursts is no longer beneath the noise floor, and the binary fence is the largest piece of that host work that is not actually computing anything. Replacing it with a counter is small in terms of code and disproportionately large in terms of how the loop reads.

This is the same shape as the [DRY sampler observation](/blog/2026-05-05-why-dry-earns-the-slot-before-min-p-on-qwen3-long-context-decode) and the [wave32 commit](/blog/2026-05-11-the-wave32-commit-that-closes-rdna4-long-context-flash-attention-gap): every quiet host-side fix that drops a few hundred microseconds per token compounds, because the decode loop is run once for every token in every response. Five hundred microseconds saved at 100 tok/s is five percent of the budget, and five percent of every decode budget is the difference between a local engine that feels brisk and one that feels stretched.

The Vulkan 1.2 line we keep crossing is what makes this kind of change possible. The same week we shipped the [wave32 attention path](/blog/2026-05-11-the-wave32-commit-that-closes-rdna4-long-context-flash-attention-gap), we found that the counter shape from the same spec lets the rest of the decode loop catch up to it. The forward pass got faster; the host loop has to follow, or the GPU spends the savings idle on a fence.
