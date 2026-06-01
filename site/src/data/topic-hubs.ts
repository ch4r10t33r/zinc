export type TopicHubLink = {
  title: string;
  href: string;
  description: string;
};

export type TopicHubFaq = {
  question: string;
  answer: string;
};

export type TopicHubBriefItem = {
  label: string;
  detail: string;
};

export type TopicHub = {
  slug: string;
  title: string;
  shortTitle: string;
  description: string;
  keywords: string;
  summary: string;
  practicalAnswer: string;
  bestUse: string;
  actionPlan: TopicHubBriefItem[];
  pitfalls: TopicHubBriefItem[];
  whatMatters: string[];
  readNext: TopicHubLink[];
  docs: TopicHubLink[];
  related: string[];
  faqs: TopicHubFaq[];
};

export const topicHubs: TopicHub[] = [
  {
    slug: 'gemma-local-inference',
    title: 'Gemma 4 Local Inference',
    shortTitle: 'Gemma 4',
    description: 'A clean guide to Gemma 4 local inference: model fit, MoE vs dense variants, sliding-window attention, asymmetric GQA, Vulkan, Metal, and ZINC.',
    keywords: 'Gemma 4 local inference, Gemma 4 AMD GPU, Gemma 4 RDNA4, Gemma 4 Metal, Gemma 4 Vulkan, Gemma MoE inference, local LLM Gemma',
    summary: 'Gemma 4 is a useful local inference target because it stresses the parts of an engine that simple Llama-shaped models do not: sliding-window attention, asymmetric grouped-query attention, Gemma-specific normalization, and MoE routing on the A4B variant.',
    practicalAnswer: 'If you want to run Gemma locally, treat Gemma 4 as an architecture port, not just another GGUF file. Dense Gemma 4 is mostly a memory and attention-shape problem. Gemma 4 26B-A4B adds sparse routing and Gemma-specific FFN behavior. On ZINC, the practical path is to use the managed Gemma model ids, verify the benchmark dashboard for the current backend, and expect the RDNA4 and Metal paths to improve as the Gemma prefill work lands.',
    bestUse: 'Use this page when you are deciding whether Gemma is a model-family port, a benchmark target, or a writing cluster. The useful reader intent is specific: what breaks differently from Qwen, what fits locally, and what an AMD or Metal engine has to optimize next.',
    actionPlan: [
      {
        label: 'Start from the model shape',
        detail: 'Separate dense Gemma runs from A4B MoE runs before comparing numbers. They stress different kernels and memory paths.',
      },
      {
        label: 'Measure prefill first',
        detail: 'Gemma can look healthy in decode while still losing badly on time-to-first-token because prompt batching and attention shape are the hard parts.',
      },
      {
        label: 'Keep Gemma posts practical',
        detail: 'The strongest future posts should answer a concrete local-running question: head dimensions, sliding windows, MoE routing, Metal parity, or RDNA4 prefill.',
      },
      {
        label: 'Link back to benchmarks',
        detail: 'Readers landing from search need the current ZINC result, the llama.cpp baseline if available, and the exact hardware class.',
      },
    ],
    pitfalls: [
      {
        label: 'Assuming Qwen kernels are enough',
        detail: 'Gemma changes attention dimensions, activation behavior, norm placement, and sometimes MoE layout. A generic transformer path is not a full answer.',
      },
      {
        label: 'Publishing decode-only conclusions',
        detail: 'Large Gemma prompts are often prefill-bound. Decode tokens per second alone can make the local experience look better than it is.',
      },
      {
        label: 'Writing generic model coverage',
        detail: 'Searchers do not need another broad Gemma overview. The opportunity is local inference mechanics on real hardware.',
      },
    ],
    whatMatters: [
      'Gemma 4 uses sliding-window attention for most layers, so long-context memory does not grow the same way on every layer.',
      'Full-attention Gemma layers can use different Q and KV head dimensions, which breaks kernels that assume one shared head_dim.',
      'Gemma MoE is not identical to Qwen MoE: activations, norms, and residual placement differ.',
      'Prefill performance depends on batching the prompt correctly; a per-token path wastes bandwidth on large Gemma models.',
    ],
    readNext: [
      {
        title: 'The single push constant blocking Gemma 4 prefill on RDNA4',
        href: '/blog/2026-04-24-the-single-push-constant-blocking-gemma-4-prefill-on-rdna4',
        description: 'Why Gemma 4 forces a split between Q and KV head dimensions in Vulkan flash attention.',
      },
      {
        title: 'Why one vkQueueSubmit per prompt matters for Gemma 4',
        href: '/blog/2026-04-25-why-one-vkqueuesubmit-per-prompt-is-the-next-quiet-rdna4-prefill-unlock',
        description: 'How 60-layer Gemma prefill exposes command submission overhead on AMD.',
      },
      {
        title: 'How MoE models work in ZINC',
        href: '/blog/2026-04-04-how-moe-models-work-in-zinc',
        description: 'The shared routing ideas and the Gemma-specific differences in sparse inference.',
      },
      {
        title: 'Why FP16 KV cache is the wrong default for 128k context',
        href: '/blog/2026-04-26-why-fp16-kv-cache-is-the-wrong-default-for-128k-context-on-32gb-rdna4',
        description: 'How Gemma sliding-window attention changes, but does not remove, the KV memory problem.',
      },
    ],
    docs: [
      {
        title: 'Getting Started with ZINC',
        href: '/zinc/docs/getting-started',
        description: 'Build ZINC, pull a managed model, and run local inference on AMD or Apple Silicon.',
      },
      {
        title: 'ZINC Benchmarks',
        href: '/zinc/benchmarks',
        description: 'Current same-machine benchmark results for Gemma, Qwen, AMD, Metal, and baseline comparisons.',
      },
      {
        title: 'Running ZINC',
        href: '/zinc/docs/running-zinc',
        description: 'CLI and server usage, managed model ids, backend selection, and runtime flags.',
      },
    ],
    related: ['qwen3-6-local-inference', 'amd-rdna4-llm-inference', 'kv-cache-quantization'],
    faqs: [
      {
        question: 'Can Gemma 4 run locally in ZINC?',
        answer: 'Yes. ZINC has managed Gemma 4 targets in the catalog, including dense and MoE variants. The exact performance depends on backend, memory budget, and whether the path is using per-token fallback or batched prefill.',
      },
      {
        question: 'Why is Gemma harder than a normal dense transformer?',
        answer: 'Gemma 4 combines sliding-window attention, asymmetric grouped-query attention on full-attention layers, Gemma-specific norms, and in some variants MoE routing. Each of those changes assumptions inside inference kernels.',
      },
      {
        question: 'Should future blog posts focus on Gemma?',
        answer: 'Yes. Gemma is a strong follow-up cluster because it is technically distinct from Qwen and creates useful comparisons around sliding-window attention, memory, MoE routing, Vulkan, and Metal.',
      },
    ],
  },
  {
    slug: 'qwen3-6-local-inference',
    title: 'Qwen3.6 Local Inference',
    shortTitle: 'Qwen3.6',
    description: 'A practical guide to Qwen3.6 local inference: architecture details, GGUF status, sparse MoE, SSM, MTP, speculative decoding, RDNA4, and Metal.',
    keywords: 'Qwen3.6 local inference, Qwen3.6 architecture, Qwen3.6 GGUF, Qwen 3.6 AMD GPU, Qwen3.6 RDNA4, Qwen3.6 speculative decoding, Qwen3.6 MTP',
    summary: 'Qwen3.6 is the core search cluster for ZINC because it combines model-architecture curiosity with practical local inference intent. Readers want to know what the model is, whether it exists as GGUF, and what an engine has to do to run it well.',
    practicalAnswer: 'For local Qwen3.6 inference, the important distinction is between model availability and engine readiness. ZINC already supports managed Qwen3.6 GGUF targets, but the fastest path depends on the variant. Dense Qwen3.6 is a batched-prefill and attention problem. A3B-style Qwen3.6 is a hybrid MoE plus SSM problem, where router scheduling, recurrent state, MTP, and KV memory all matter.',
    bestUse: 'Use this page as the entry point for people searching "can I run Qwen3.6 locally" or "what does Qwen3.6 require from an inference engine." It should turn model-name curiosity into exact local-running guidance.',
    actionPlan: [
      {
        label: 'Identify the variant',
        detail: 'Dense, A3B MoE, and Next-style hybrid models do not have the same bottlenecks. Name the exact GGUF or managed model id before discussing performance.',
      },
      {
        label: 'Split readiness from availability',
        detail: 'A model file can exist before the fast path is ready. State what runs today, then call out which kernels still decide production quality.',
      },
      {
        label: 'Track prefill, decode, and context',
        detail: 'Qwen can win in one phase and lose in another. The useful comparison shows prompt throughput, decode throughput, latency, and context length together.',
      },
      {
        label: 'Treat MTP as the speculation path',
        detail: 'Generic draft-model speculation is fragile on sparse MoE. Target-attached MTP is the more credible story for Qwen A3B models.',
      },
    ],
    pitfalls: [
      {
        label: 'Mixing Qwen names',
        detail: 'Qwen3, Qwen3.5, Qwen3.6, and Qwen3-Next attract overlapping searches. The page should keep model family, checkpoint, and architecture separate.',
      },
      {
        label: 'Overpromising long context',
        detail: 'A model-card context length is not the same as useful local context. KV memory, recurrent state, and prefill time still decide the run.',
      },
      {
        label: 'Assuming speculation is free',
        detail: 'Sparse expert verification can wake more experts than the draft saved. Any speedup claim needs acceptance rate and verifier cost.',
      },
    ],
    whatMatters: [
      'Qwen3.6 interest is split between architecture details, GGUF availability, local runtime support, and performance.',
      'Sparse A3B variants decode like small active models but fill memory like large total-parameter models.',
      'Speculative decoding is not automatically useful on sparse MoE models because verification can wake many experts.',
      'MTP-style drafts are more promising than generic draft models when hidden-state alignment is available.',
    ],
    readNext: [
      {
        title: 'Qwen 3.6 architecture and local inference in ZINC',
        href: '/blog/2026-04-05-qwen-3-6-architecture-and-what-it-would-take-to-bring-it-into-zinc',
        description: 'The main architecture explainer for Qwen3.6, MoE, SSM, context, and local engine implications.',
      },
      {
        title: 'Qwen3.6-35B-A3B GGUF on AMD and Metal',
        href: '/blog/2026-04-17-qwen-3-6-is-now-generally-available-in-zinc',
        description: 'Managed model support, AMD RDNA4 and Apple Silicon notes, and practical run guidance.',
      },
      {
        title: 'Why speculative decoding does not net out on Qwen 35B-A3B',
        href: '/blog/2026-04-28-why-speculative-decoding-does-not-net-out-on-qwen-35b-a3b',
        description: 'Why generic draft-model speculation can lose on sparse Qwen verification.',
      },
      {
        title: 'Why MTP heads are the speculative decode draft Qwen3 A3B deserves',
        href: '/blog/2026-05-08-why-mtp-heads-are-the-speculative-decode-draft-qwen3-a3b-deserves',
        description: 'Why target-attached multi-token prediction is a better fit than separate draft models.',
      },
      {
        title: 'The gate that keeps Qwen 35B prefill at half of llama.cpp on RDNA4',
        href: '/blog/2026-04-26-the-gate-that-keeps-qwen-35b-prefill-at-half-of-llama-cpp-on-rdna4',
        description: 'The structural MoE plus SSM prefill work needed for flagship Qwen performance.',
      },
    ],
    docs: [
      {
        title: 'Getting Started with ZINC',
        href: '/zinc/docs/getting-started',
        description: 'Pull a managed Qwen model and run local inference through the CLI or chat path.',
      },
      {
        title: 'ZINC Benchmarks',
        href: '/zinc/benchmarks',
        description: 'Current Qwen3.6 results against same-machine llama.cpp baselines.',
      },
      {
        title: 'Running ZINC',
        href: '/zinc/docs/running-zinc',
        description: 'Model ids, CLI flags, chat mode, server mode, and KV quantization options.',
      },
    ],
    related: ['amd-rdna4-llm-inference', 'kv-cache-quantization', 'gemma-local-inference'],
    faqs: [
      {
        question: 'Can Qwen3.6 run locally?',
        answer: 'Yes, when a supported GGUF target exists and the machine has enough memory. In ZINC, the managed model catalog is the preferred path because it keeps local files, defaults, and backend support aligned.',
      },
      {
        question: 'Why is Qwen3.6 hard for local inference engines?',
        answer: 'The hard cases combine sparse experts, recurrent or state-space blocks, large context, and a large vocabulary. That pushes work into routing, recurrent state updates, KV memory, sampling, and prefill scheduling.',
      },
      {
        question: 'Is speculative decoding a clear win on Qwen3.6?',
        answer: 'Not with a generic draft model. Sparse expert verification can erase the win. MTP-style target-attached drafts are the more credible direction for Qwen A3B models.',
      },
    ],
  },
  {
    slug: 'amd-rdna4-llm-inference',
    title: 'AMD RDNA4 LLM Inference',
    shortTitle: 'AMD RDNA4',
    description: 'A practical guide to local LLM inference on AMD RDNA4 GPUs: R9700, RX 9070 XT, Vulkan, llama.cpp comparisons, prefill, decode, and ZINC.',
    keywords: 'AMD RDNA4 LLM inference, Radeon AI PRO R9700 inference, RX 9070 XT LLM, AMD GPU local LLM, Vulkan LLM inference, llama.cpp RDNA4, ROCm alternative',
    summary: 'RDNA4 is the default hardware story for ZINC: useful consumer and workstation AMD GPUs, strong memory bandwidth, Vulkan support, and no dependence on ROCm for local LLM inference.',
    practicalAnswer: 'If you want local LLM inference on AMD RDNA4, use Vulkan-first software and treat ROCm as optional rather than required. ZINC targets Radeon AI PRO R9700 and RX 9070-class hardware directly with Vulkan compute. The important performance split is decode versus prefill: decode is mostly memory and scheduling; prefill needs batched kernels, command-buffer discipline, and model-aware routing.',
    bestUse: 'Use this page for readers choosing AMD hardware or validating whether Vulkan local inference is real on RDNA4. The page should answer the hardware, driver, benchmark, and baseline questions before sending them to deep posts.',
    actionPlan: [
      {
        label: 'Pick the memory budget first',
        detail: 'The R9700-class 32 GB card changes which Qwen and Gemma models are realistic. RX 9070-class cards share RDNA4 behavior but fit fewer large targets.',
      },
      {
        label: 'Benchmark against llama.cpp',
        detail: 'Use the same machine, model file, quantization, prompt, output cap, warmup policy, and backend residency. Cross-machine screenshots are not evidence.',
      },
      {
        label: 'Tune the platform before the shader',
        detail: 'Driver version, GECC, cooperative matrix flags, ASPM, and stale GPU processes can move results enough to hide real kernel work.',
      },
      {
        label: 'Report prefill and decode separately',
        detail: 'RDNA4 decode can look strong while prefill remains the user-visible latency limiter. A good post shows both phases.',
      },
    ],
    pitfalls: [
      {
        label: 'Making ROCm the story',
        detail: 'The useful ZINC angle is Vulkan-first AMD inference. Mention ROCm only to clarify that this path does not require it.',
      },
      {
        label: 'Comparing dirty runs',
        detail: 'Background GPU users, cold builds, one-token completions, and changed prompts will swamp the signal.',
      },
      {
        label: 'Treating RDNA4 as one number',
        detail: 'R9700, RX 9070 XT, and future workstation cards share architecture but not memory capacity, clocks, thermals, or deployment fit.',
      },
    ],
    whatMatters: [
      'RDNA4 can run useful local LLMs without ROCm when the engine uses Vulkan compute directly.',
      'The R9700 is the strongest ZINC tuning target because 32 GB VRAM changes which Qwen and Gemma models fit.',
      'Prefill and decode are different workloads; a fast decode loop does not imply fast time-to-first-token.',
      'llama.cpp is the right baseline, but ZINC is deliberately optimizing the AMD path as a first-class target.',
    ],
    readNext: [
      {
        title: 'How we made AMD LLM inference 4x faster on a single GPU',
        href: '/blog/2026-03-30-how-we-moved-zinc-from-7-tok-s-to-33-tok-s-on-amd-rdna4',
        description: 'The early RDNA4 speedup story and what changed in the decode path.',
      },
      {
        title: 'What broke first in local LLM inference on AMD RDNA4',
        href: '/blog/2026-03-27-what-broke-first-when-we-built-zinc-on-amd-rdna4',
        description: 'The first correctness failures: attention, KV cache, RoPE, MoE, SSM, and tokenizer drift.',
      },
      {
        title: 'The broken Vulkan shaders keeping AMD RDNA4 inference stuck at 4 tok/s',
        href: '/blog/2026-03-29-the-shaders-standing-between-4-tok-s-and-27-tok-s',
        description: 'How shader debugging moved work back onto the GPU.',
      },
      {
        title: 'Why RDNA4 prefill for Qwen3.5-35B is stuck at 25 tok/s',
        href: '/blog/2026-04-18-why-rdna4-prefill-for-qwen-3-5-is-stuck-at-25-tok-s',
        description: 'Why prefill needs different kernels than decode on AMD.',
      },
      {
        title: 'How AMD Qwen decode passed llama.cpp in six weeks',
        href: '/blog/2026-05-09-how-we-made-amd-qwen-inference-faster-than-llama-cpp-in-six-weeks-on-the-radeon-ai-pro-r9700',
        description: 'The public comparison point against llama.cpp on the R9700.',
      },
    ],
    docs: [
      {
        title: 'Run LLMs on AMD GPUs Without ROCm',
        href: '/zinc/docs/getting-started',
        description: 'The fastest path from clone to local AMD inference.',
      },
      {
        title: 'AMD RDNA3/RDNA4 GPU Reference',
        href: '/zinc/docs/amd-gpu-reference',
        description: 'Hardware, memory, wave execution, and Vulkan compute details.',
      },
      {
        title: 'RDNA4 Tuning Guide',
        href: '/zinc/docs/rdna4-tuning',
        description: 'Driver, shader, cooperative matrix, and benchmark tuning notes.',
      },
      {
        title: 'ZINC Benchmarks',
        href: '/zinc/benchmarks',
        description: 'Current public measurements across AMD, Metal, and baseline engines.',
      },
    ],
    related: ['qwen3-6-local-inference', 'gemma-local-inference', 'kv-cache-quantization'],
    faqs: [
      {
        question: 'Can AMD RDNA4 run local LLMs without ROCm?',
        answer: 'Yes. ZINC uses Vulkan compute on AMD RDNA4, so the local inference path does not depend on ROCm support for consumer cards.',
      },
      {
        question: 'Which RDNA4 GPU is the best target for ZINC?',
        answer: 'The Radeon AI PRO R9700 is the primary tuning target because it combines RDNA4 behavior with a 32 GB memory budget. RX 9070-class cards share the architecture but fit fewer large models.',
      },
      {
        question: 'Why compare to llama.cpp?',
        answer: 'llama.cpp is the strongest widely used local baseline with Vulkan and Metal support. ZINC compares against it on the same machine to avoid misleading cross-hardware numbers.',
      },
    ],
  },
  {
    slug: 'kv-cache-quantization',
    title: 'KV Cache Quantization for Local LLMs',
    shortTitle: 'KV Cache',
    description: 'A practical guide to KV cache quantization for local LLM inference: FP16 vs INT8/Q4, 128k context, 32 GB GPUs, TurboQuant, and ZINC.',
    keywords: 'KV cache quantization, local LLM long context, TurboQuant, FP16 KV cache, INT8 KV cache, Q4 KV cache, 128k context, RDNA4 VRAM',
    summary: 'KV cache quantization is the long-context memory lever. Once a model fits, the prompt length and concurrent sessions are usually limited by K/V bytes per token, not by the static weight file.',
    practicalAnswer: 'For local long-context inference, FP16 KV cache is a debug-friendly default, not the right long-term default. INT8 K/V, asymmetric Q4 K plus higher-precision V, and TurboQuant-style compression all attack the same bottleneck: K/V memory grows linearly with tokens. On 16 GB and 32 GB GPUs, that growth decides whether 128k context is real or just a model-card number.',
    bestUse: 'Use this page for readers trying to understand why a model that fits in VRAM still fails at long context. The search intent is practical memory math: bytes per token, which precision is safe, and what an engine must implement.',
    actionPlan: [
      {
        label: 'Compute bytes per token',
        detail: 'Start from layers, KV heads, head dimension, and precision. That number explains context limits better than model size alone.',
      },
      {
        label: 'Prefer asymmetric precision',
        detail: 'K often tolerates lower precision than V. INT8 or Q4 K with higher-precision V is usually a better first target than uniformly shrinking both.',
      },
      {
        label: 'Read compressed KV directly',
        detail: 'A long-context speedup disappears if the attention path dequantizes the whole cache into FP16 scratch before every read.',
      },
      {
        label: 'Validate at long context',
        detail: 'Short prompts rarely reveal KV quantization regressions. Test perplexity, retrieval, and decode throughput at 16k, 32k, and beyond.',
      },
    ],
    pitfalls: [
      {
        label: 'Calling FP16 the default answer',
        detail: 'FP16 is useful for correctness bring-up, but it is too expensive for serious 128k local context on 16 GB and 32 GB cards.',
      },
      {
        label: 'Ignoring page layout',
        detail: 'Quantization, paging, and prefix reuse interact. A smaller cache still needs an allocation layout that serves the workload.',
      },
      {
        label: 'Optimizing memory but losing bandwidth',
        detail: 'The format only helps if the kernel spends less time reading memory after scale loads, unpacking, and correction are included.',
      },
    ],
    whatMatters: [
      'Weights are static after load; KV cache grows with every prompt token and generated token.',
      'Grouped-query attention reduces KV cost, but long context still makes KV memory large.',
      'K vectors usually tolerate lower precision than V vectors, so asymmetric formats are attractive.',
      'A good attention kernel should read quantized KV directly rather than dequantizing into a full FP16 scratch copy.',
    ],
    readNext: [
      {
        title: 'Why FP16 KV cache is the wrong default for 128k context',
        href: '/blog/2026-04-26-why-fp16-kv-cache-is-the-wrong-default-for-128k-context-on-32gb-rdna4',
        description: 'The memory math for long context on a 32 GB RDNA4 card.',
      },
      {
        title: 'The 16k crossover where KV reads outweigh active weights',
        href: '/blog/2026-04-27-the-16k-crossover-where-kv-reads-outweigh-active-weights-on-rdna4-decode',
        description: 'Why decode becomes KV-bandwidth dominated at long context.',
      },
      {
        title: 'Paged KV cache is the serving fix a single-user local engine can mostly skip',
        href: '/blog/2026-05-26-paged-kv-cache-is-the-serving-fix-a-single-user-local-engine-can-mostly-skip',
        description: 'Where paging matters, where contiguous cache is simpler, and what prefix sharing changes.',
      },
      {
        title: 'FP8 KV cache is the next decode bandwidth cut RDNA4 already has the WMMA for',
        href: '/blog/2026-05-19-fp8-kv-cache-is-the-next-decode-bandwidth-cut-rdna4-already-has-the-wmma-for',
        description: 'Why lower-precision K/V storage also becomes a throughput optimization.',
      },
      {
        title: 'Why GQA is not the last KV cache shape for local 32 GB long context',
        href: '/blog/2026-05-03-why-gqa-is-not-the-last-kv-cache-shape-for-local-32gb-long-context',
        description: 'Why the next reductions are architectural and memory-layout driven.',
      },
    ],
    docs: [
      {
        title: 'TurboQuant KV Cache Compression',
        href: '/zinc/docs/turboquant-spec',
        description: 'ZINC design notes for 2-4 bit K/V pages and QJL correction.',
      },
      {
        title: 'ZINC Technical Specification',
        href: '/zinc/docs/spec',
        description: 'Paged KV cache, request scheduling, and attention engine architecture.',
      },
      {
        title: 'Running ZINC',
        href: '/zinc/docs/running-zinc',
        description: 'Runtime flags and model-running details, including KV-related options.',
      },
    ],
    related: ['amd-rdna4-llm-inference', 'qwen3-6-local-inference', 'gemma-local-inference'],
    faqs: [
      {
        question: 'Why does KV cache matter for local LLMs?',
        answer: 'KV cache stores the keys and values needed for attention over prior tokens. It grows linearly with context length and can become the largest moving memory cost after the model weights fit.',
      },
      {
        question: 'Is FP16 KV cache still useful?',
        answer: 'Yes, as a simple correctness baseline and debug mode. For long context on 16 GB or 32 GB cards, it is usually too expensive to be the practical default.',
      },
      {
        question: 'What is TurboQuant doing differently?',
        answer: 'TurboQuant combines vector quantization with residual correction for keys so attention inner products stay low-bias at very low bit widths. It is a design target for future compressed KV paths in ZINC.',
      },
    ],
  },
];

export function getTopicHub(slug: string) {
  return topicHubs.find((hub) => hub.slug === slug);
}
