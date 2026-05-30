---
title: "The imatrix pass that decides which Qwen3 weights survive IQ3_M"
date: "2026-05-29"
tags:
  - zinc
  - rdna4
  - amd
  - quantization
  - imatrix
  - importance-matrix
  - iq3
  - iq3-m
  - codebook
  - gguf
  - qwen3
  - llama-cpp
  - llm-inference
keywords:
  - llama.cpp imatrix Qwen3 quantization
  - importance matrix IQ3_M calibration
  - GGUF i-quant codebook lookup
  - bits per weight perplexity IQ3
  - imatrix.dat calibration dataset
  - Qwen3-30B IQ3_M 32 GB
  - PR 4861 importance matrix llama.cpp
  - PR 5676 IQ3_S IQ3_M Kawrakow
  - imatrix vs no imatrix perplexity
  - Radeon AI PRO R9700 IQ3 Qwen3
excerpt: "Quantize Qwen3-30B to IQ3_M without an imatrix and the model ships at roughly 3.63 bits per weight, which on a 32 GB Radeon AI PRO R9700 looks like a free win against Q4_K_M. Skip the calibration pass that produced the codebook lookup, though, and the same file gives a coherent assistant at the start of a chat and a confused one by turn three. IQ3_S and IQ3_M are not 3-bit quants in the way Q3_K is. They are 512-entry codebooks the quantizer searches against a weighted RMSE loss, and the only thing supplying the weights is the diagonal of a per-layer activation Hessian computed over a calibration corpus. Without it, the codebook is doing approximately a uniform fit, and the quality curve below 4 bpw collapses back onto k-quants. This is the offline pass nobody talks about and the reason every IQ-quant on Hugging Face has an `imatrix` step in its lineage."
---

The cheapest way to ship Qwen3-30B on a 32 GB card is to take it past 4 bits, and the cheapest place to land past 4 bits is `IQ3_M`. The file is roughly 3.63 bits per weight, which leaves comfortable headroom for KV cache on the [Radeon AI PRO R9700](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html), and the perplexity gap to fp16 is smaller than `Q3_K_M` at 0.26 bpw less. On paper it is the dominant option in that bit range.

It is dominant on a specific condition that nobody puts in the filename. `IQ3_M`, like every other quant in the IQ family, is built around an importance matrix. Strip the imatrix step out of the pipeline, ship the file anyway, and the quality curve in that bit range collapses back to where the k-quants live. The codebook is doing the work of 3 bits and the imatrix is telling it where to spend them. This post is about that calibration pass, what it actually computes, and why it is the part of the GGUF supply chain that turns IQ3 from a research curiosity into the default sub-4-bit quant for a single-user local engine.

The reason this matters now is that the IQ3 family is the operating point for 30B-class models on consumer 32 GB cards. We sized [Qwen3 activations off 4 bits](/blog/2026-05-23-the-outlier-channels-that-keep-qwen3-activations-off-4-bit) last week and argued that the activation floor is 8-bit, which leaves all the bit-saving on the weight side. IQ3_M is the result of that arithmetic, and it only exists because of an offline pass that runs on CPU for an hour and is then never spoken of again.

## What an importance matrix actually is

The standard reference is ikawrakow's [PR #4861, where llama.cpp learned to compute an importance matrix](https://github.com/ggml-org/llama.cpp/pull/4861), and the math in the PR description is short enough to repeat in plain English. Pick a row of weights `w_j` in a tensor that the model is going to multiply against an activation column `a_j`. The dot product produces one number that the next layer reads. If we quantize the weights to `q_j` and want to minimize the impact on that downstream value, the right loss is the squared error of the dot product after averaging over a corpus of activations, `F = (Σ (q_j − w_j) a_j)^2`. Take the derivative with respect to each `q_i`, and the cross terms involve `<a_i a_j>`. For most pairs of activation channels the cross-correlation is small relative to `<a_i^2>`, and the loss collapses to a weighted RMSE in which each weight is scaled by the expected squared activation in its channel.

That diagonal of squared activations is the importance matrix. It is one float per input channel per tensor, which is a tiny fraction of the model size, and it is computed by running a forward pass over a calibration corpus and accumulating `<a_i^2>`. The PR ships this as the standalone `llama-imatrix` tool. The usage line in the [Qwen llama.cpp guide](https://qwen.readthedocs.io/en/latest/quantization/llama.cpp.html) is the entire user interface most people see, `./llama-imatrix -m Qwen3-30B-F16.gguf -f calibration.txt --chunk 512 -o Qwen3-30B.imatrix.dat`, after which the quantizer is fed the matrix with `--imatrix`.

The cost is real but small enough to ignore once. The original PR cites about an hour on a 32-core CPU for a 70B model with 50k calibration tokens, and a [later GPU path in PR #4957](https://github.com/ggml-org/llama.cpp/pull/4957) brings that down by an order of magnitude. For a 30B Qwen3 on a single R9700 it is a coffee-break offline pass, not a workflow.

## Codebooks are the other half

The reason this calibration matters more for IQ-quants than for k-quants is the shape of the encoder. k-quants are scalar: each weight is stored as a few bits inside a super-block with a shared scale, and `Q3_K_M` quantizes most rows to 3-bit values with the scale absorbing the dynamic range. The imatrix changes which scale the quantizer picks per column, but the underlying scheme already covers the value range without further help.

IQ-quants are vector. The introduction in ikawrakow's [PR #5676 for IQ3_S](https://github.com/ggml-org/llama.cpp/pull/5676) is the cleanest description. A group of 4 or 8 weights is encoded as an index into a fixed codebook of 256 or 512 entries, where each entry holds a tuple of quantized values drawn from a small set. For IQ3_S the codebook has 512 entries over groups of 4 weights, and the 8 possible quant values per weight are `{4, 12, 20, 28, 36, 44, 52, 62}` with a sign stored separately. The codebook itself is not random. It was chosen by quantizing real models with all 4096 possible combinations, counting which combinations show up, and keeping the 512 that minimize a weighted-distance loss against the rest. The IQ4_NL and IQ2 families are built the same way at different bit depths, and they all share the property that decoding a tile of weights requires a lookup into a kilobyte-scale table to recover the actual values for the matmul. ikawrakow's [SOTA quants discussion](https://github.com/ggml-org/llama.cpp/discussions/4852) is the long form of that design choice.

What makes the codebook fit hard is also what makes it fit well. Because each 4-weight tile has only 512 legal combinations, the encoder is solving a search problem: which codebook entry minimizes the loss for these four weights. The loss it is minimizing is the weighted RMSE described above, weighted by the diagonal of the activation Hessian. Without the imatrix, the codebook gets a uniform loss and the search collapses to "which entry is closest in plain L2 distance." That is enough to recover the most common combinations, which is why IQ2 and IQ3 produce something the model can still execute even without calibration, but it is not enough to land where the published quality numbers live.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-05-29-imatrix-iq3-bits-per-weight-perplexity-curve.svg" alt="A dark-navy scatter plot titled the IQ curve sits on top of an importance matrix. The x axis runs from 2 to 7 bits per weight, the y axis is ln of PPL of the quant divided by PPL of fp16 on Mistral-7B, from zero at the bottom to one at the top. A faint cyan vertical band labeled IQ-family marks the region from about 1.7 to 4.7 bpw. Inside that band, cyan circles plot IQ1_S high up at 0.92, IQ2_XXS near 0.30, IQ2_XS near 0.20, IQ3_XXS at 0.06, IQ3_S at 0.03, IQ3_M as a highlighted gold dot at 3.63 bpw and 0.027, and IQ4_NL near 0.007. Copper square markers plot the k-quant family at the same bit range, sitting visibly higher than the IQ points at matching bpw: Q2_K_S at 0.16, Q2_K at 0.11, Q3_K_S at 0.05, Q3_K_M at 0.026, Q3_K_L at 0.02, Q4_K_M at 0.006, Q5_K_M near zero, Q6_K at zero. A dashed cyan line labeled fp16 floor runs along the x axis. A gold annotation arrow from the IQ3_M point reads IQ3_M sits below Q3_K_M at 0.26 bpw less and below Q3_K_L at 0.59 bpw less; without an imatrix the cyan band collapses and the curve looks like the copper one shifted up. A footer note credits Artefact2's table from llama.cpp PR 5676 for the Mistral-7B numbers and notes IQ-family values are not achievable without an importance matrix." loading="lazy" />
  <figcaption>Each cyan circle is an imatrix-calibrated IQ-quant on Mistral-7B. Each copper square is a k-quant at a nearby bpw. Below about 4 bpw the IQ band is meaningfully under the k-quant band, which is the whole reason IQ3_M is the operating point on a 32 GB card. The numbers reproduce Artefact2's KL-divergence table from llama.cpp PR #5676; the x axis excludes the higher-bit `output.weight` tensor that all mixes promote.</figcaption>
</figure>

The story the chart tells is the one that decides the workflow. Pick any horizontal line below `ln(ratio) = 0.05` and the cyan circle on that line lives noticeably to the left of the copper square. At IQ3_M the lead is about 0.26 bpw against Q3_K_M and 0.59 bpw against Q3_K_L for the same quality, which on a 30B model is a 1 to 2 GB savings that goes straight into the KV cache budget. Erase the calibration step and the cyan circles slide up the y axis by enough to lose the lead.

## What the calibration data has to look like

The choice of calibration corpus is the most argued-about part of the pipeline, and ikawrakow's [discussion on near-random data](https://github.com/ggml-org/llama.cpp/discussions/5006) is the canonical place to read about it. The naive expectation is that the calibration text should match the deployment task, the same instinct from GPTQ and AWQ where the calibration data sometimes has to be in-domain to recover accuracy. The empirical finding for imatrix is closer to the opposite. Highly structured corpora can underfit the importance matrix because they exercise too few channels, while a mixed corpus of code, prose, and language fragments excites more of the activation space and tends to produce a better matrix even on prose-heavy evaluations. The community consensus that has formed since the PR is that any reasonable mixed corpus beats vanilla quantization by enough that the calibration choice is secondary to whether the step is run at all.

The size of the corpus turns out to matter less than the variety. Fifty thousand tokens is the figure in the original PR and it is still the figure most public quantizers use. The reason is that the matrix only stores per-channel squared expectations, and those expectations converge fast.

## Where it does not help

The same PR carries an option, `--output-weight 0`, that disables imatrix use on the language model head, and the default is off because Kawrakow's experiments showed that the head quantizes more reliably without weighted reconstruction. The output tensor is special, every quantization mix in llama.cpp's [quantize tool README](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md) promotes it to a higher precision than the body, and the imatrix discussion is not the lever that helps there. The head is reflected in the chart only as the disclaimer in the footer that the x axis is bits per weight excluding `output.weight`.

The other place imatrix does not buy much is in the 5 to 6 bpw range. The chart shows Q5_K_M and Q6_K already sitting at the fp16 floor, and there is no room left for a weighted RMSE to do better than uniform. The imatrix helps where the bit budget is too small for the codebook to fit blindly. Past about 5 bpw the codebook can absorb the variance on its own.

## What zinc does with this

The first thing this changes is the supply chain question. The HF community's IQ3 GGUFs are produced with imatrix calibration as a matter of course, and reading the model card for any reasonable Qwen3 quant will show the calibration dataset and the command line. A local engine that takes those files at face value is taking the calibration on faith, which is fine, because the alternative is rerunning the pass per model on the dev's machine for no benefit.

The second thing it changes is what zinc does when it can no longer take the files at face value. The case that comes up is a new model architecture where no good IQ3 quants exist yet, which happened recently with Qwen3-Next's hybrid cache and will happen again every time a model ships a new layout that exposes new activation statistics. In those cases the runtime needs its own imatrix path, which is the same `llama-imatrix` flow plus the GPU acceleration from PR #4957 so that the offline pass does not take a workstation overnight. Ikawrakow's [ik_llama.cpp fork](https://github.com/ikawrakow/ik_llama.cpp) is also worth tracking, because its IQ-K series adds further codebook variants in the same family and the imatrix step is shared.

The piece worth carrying forward is that the bit budget below 4 bpw is not a property of the encoder alone. It is a property of the encoder plus the calibration that picks which codebook entries the encoder is allowed to spend. The chart only looks the way it does because there is an importance matrix sitting under every cyan point. Without it, the file still exists, but the quality story is the orange curve everywhere.
