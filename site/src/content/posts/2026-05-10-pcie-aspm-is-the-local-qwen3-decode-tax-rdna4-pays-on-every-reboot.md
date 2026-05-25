---
title: "PCIe ASPM is the local Qwen3 decode tax RDNA4 pays on every reboot"
date: "2026-05-10"
tags:
  - zinc
  - rdna4
  - amd
  - pcie
  - aspm
  - power-management
  - qwen3
  - llama-cpp
  - vulkan
  - llm-inference
keywords:
  - PCIe ASPM RDNA4 inference
  - pcie_aspm.policy performance local LLM
  - L1 exit latency dense Qwen3 decode
  - amdgpu power profile COMPUTE no effect
  - power_dpm_force_performance_level profile_peak
  - Radeon AI PRO R9700 dense decode tuning
  - RADV Vulkan Qwen3 27B tok/s
  - llama.cpp discussion 21043 RDNA4
  - PCIe L1 substate inference latency
  - kernel boot parameter pcie_aspm policy
excerpt: "Every public RDNA4 tuning thread starts at the GPU. Set the AMD power profile to COMPUTE, force the SMU to profile_peak, lock the SCLK, write a new kernel config. None of those move dense Qwen3 decode on a Radeon AI PRO R9700. The one knob that does live outside the GPU entirely. PCIe ASPM resets to the kernel's default powersave-leaning policy on every boot, and on dense decode that costs 10.8 percent of tokens per second on a 27B model under llama.cpp Vulkan. The fix is one line, the cost is no extra watts that the GPU is already not drawing, and the conversation has been pointed at the wrong subsystem."
---

The most expensive single knob on a Radeon AI PRO R9700 running local Qwen3 decode has nothing to do with the GPU. It is one of the kernel parameters that gets set during PCIe enumeration at boot, falls back to the firmware default on every reset, and has no representation in `amd-smi`, `rocm-smi`, `radeontop`, `amdgpu_top`, `nvtop`, LACT, or any of the tools an RDNA4 user has open during a benchmark run. The knob is `pcie_aspm.policy`. On a stock Ubuntu kernel its default is `default`, which honors the firmware's preference, which on every modern desktop board is some flavor of powersave. Flip it to `performance` and dense Qwen3.5-27B Q4_K_M decode on a single R9700 jumps from 29.30 tok/s to 32.46 tok/s under RADV.

That is not the result of a kernel patch, a driver fork, a custom shader, a recompile, or a reflash. It is one line of shell:

```bash
echo performance | sudo tee /sys/module/pcie_aspm/parameters/policy
```

And it resets at the next reboot unless you make it permanent with `pcie_aspm.policy=performance` on the kernel command line. The hostile reading of that is uncharitable to the kernel maintainers; the structural reading is that PCIe Active State Power Management was designed for laptops and HEDT idle, and local inference is the workload that finally makes the design tradeoff visible.

This post is the structural reason a 10.8% local-inference decode tax lives on the PCIe link instead of on the GPU, why every GPU-side tuning knob people reach for first does roughly nothing on this workload, and what the right operating point on a 32 GB RDNA4 card actually is.

## What ASPM is, and what it costs

PCIe Active State Power Management is a link-state feature the [PCIe specification adopted from the laptop power-management world](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/7/html/power_management_guide/aspm). When both ends of a link agree the device is not actively transacting, the link drops from L0, the active state, into L0s or L1, progressively deeper low-power states. The endpoint and the upstream port have to renegotiate the link before any new transaction can go through, and the time that renegotiation takes is the L1 exit latency.

The Red Hat power-management documentation [is unusually clean on this](https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/7/html/power_management_guide/aspm): "When ASPM is enabled, device latency increases because of the time required to transition the link between different power states." The Linux kernel exposes three policies through `/sys/module/pcie_aspm/parameters/policy`: `default` defers to the BIOS, `powersave` aggressively favors low-power link states everywhere, and `performance` disables ASPM entirely to let PCIe links operate at maximum throughput. The relevant source is in [drivers/pci/pcie/aspm.c in the mainline kernel tree](https://github.com/torvalds/linux/blob/master/drivers/pci/pcie/aspm.c), and the design is explicit: latency is the price the link pays for power savings.

The exit latency itself is not advertised in a way the kernel can trust. L1 substate exit times are not in the PCIe configuration space directly, so the kernel falls back to the value the device reports for plain L1 exit. The result is that the actual L1-to-L0 transition on consumer cards is somewhere in the 4 to 16 microsecond range depending on the substate and platform. That sounds small. It is small. It just happens hundreds of times per token.

## Why dense decode pays for ASPM and MoE does not

The shape of an LLM decode step on a Vulkan backend is bursty. Each decoder layer issues a small set of compute dispatches against the GPU command processor, each dispatch is announced to the hardware through a doorbell write across the PCIe link, and each dispatch settles back through a completion fence. On a dense Qwen3.5-27B forward pass with several dozen layers and a handful of dispatches per layer for attention, the FFN projections, RoPE, and norm fusions, a single decode token produces a few hundred PCIe round-trips between the host driver and the GPU command ring.

In between those round-trips, the GPU is doing real bandwidth-bound work pulling 15.6 GB of model weights through the GDDR6 bus. The PCIe link is idle relative to its own time scale, which means the AS power manager on the upstream root port has time to drop the link into L1 substate. Every dispatch then pays an L1 exit latency before its doorbell write reaches the GPU. None of that latency shows up in `rocm-smi gpu`. None of it shows up in the per-shader trace. It shows up as a slightly larger inter-dispatch gap, and on a workload that issues a few hundred dispatches per token, the integral is real.

MoE decode does not pay the same tax. On Qwen3.5-35B-A3B with roughly 3.5B active parameters per token, the expert-routing path batches its dispatches and the per-token PCIe transaction count drops, even though the total compute is similar. The empirical version of this argument is in the [llama.cpp RDNA4 Llama Experiments discussion](https://github.com/ggml-org/llama.cpp/discussions/21043) by `BSpasov`, which ran more than fifty optimization combinations on a single R9700 and isolated the ASPM effect cleanly: dense 27B decode gains 10.8% on RADV with `pcie_aspm.policy=performance`, dense decode on AMDVLK gains 1.3%, and the 35B MoE workload gains 0% on both drivers. The bandwidth-utilization math in the same discussion lands the dense-decode improvement from 71% of the R9700's 640 GB/s peak to 79%, which is exactly where you would expect the inter-dispatch overhead to show up.

## What the GPU-side knobs actually do

The same discussion is also the cleanest public refutation I have seen of every GPU-side knob the consumer-tuning community reaches for first. The full grid of negative results on dense 27B decode at `tg128` is the part of the writeup that does not show up in social-media summaries but is the load-bearing piece of the argument.

| Tuning intervention | Where it lives | RADV dense decode |
| --- | --- | ---: |
| `pcie_aspm.policy=performance` | PCIe root complex | **+10.8%** |
| `rm_kq=1` one-line VGPR-pressure patch | RADV kernel source | +0.8% |
| `-ub 2048` micro-batch flag | llama.cpp build | −1% |
| `pp_power_profile_mode = COMPUTE` | amdgpu SMU | 0% |
| `power_dpm_force_performance_level = profile_peak` | amdgpu SMU | 0% |
| `power_dpm_force_performance_level = high` | amdgpu SMU | 0% |
| Hugepages, CPU pinning, `nice -n -20` | Linux scheduler | 0% |

The first row is the headline. The next two are the second-tier optimizations that are real but small. The bottom block is the set of interventions every RDNA4 tuning guide tells you to try first, and they all measure inside the noise floor of the benchmark. The two power-state interventions are particularly worth a look. They are the things that "obviously" should help when the workload looks bursty, because the user model is that the GPU is downclocking between bursts. On RDNA4 it is not. The R9700's GPU clock stays pegged near 2350 MHz under active inference because the [PowerPlay DPM heuristic on gfx1201](https://rocm.docs.amd.com/projects/amdsmi/en/develop/conceptual/perf-determinism.html) classifies decode as a continuous workload, not as a burst. Setting `profile_peak` does not change anything because the GPU is already at peak.

The PCIe link is the thing that is not at peak. The PCIe link has its own power-state machine, run by the root complex chipset and the kernel's ASPM policy, completely outside the amdgpu driver's view. Setting an AMD-specific power profile does not propagate to the upstream port because it cannot.

## Where the tax shows up

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-05-10-rdna4-decode-tuning-knob-impact.svg" alt="A vertical bar chart comparing the measured impact of six tuning interventions on dense Qwen3.5-27B Q4_K_M decode tokens per second on the AMD Radeon AI PRO R9700 under llama.cpp with the Vulkan RADV backend. The baseline at the left is the stock build at 29.30 tokens per second. Four neutral interventions follow: the minus ub 2048 flag at 29.01, the rm_kq equals 1 source patch at 29.31, writing COMPUTE to pp_power_profile_mode at 29.30, and forcing power_dpm_force_performance_level to profile_peak at 29.30. The rightmost bar shows pcie_aspm.policy set to performance at 32.46 tokens per second, a 10.8 percent gain, filled solid green and taller than the others. A horizontal dashed orange reference line at 29.30 marks the stock baseline. An inset on the right shows three PCIe doorbell transactions across a per-decode-step timeline, each followed by an L1 exit latency strip annotated as the tax that ASPM performance mode removes. A footer cites llama.cpp discussion 21043 and gives the one-line shell command to apply the fix." loading="lazy" />
  <figcaption>Six knobs, one winner. The bar chart is the GPU-side tuning consensus measured against itself; the only intervention that crosses the baseline lives outside the GPU entirely.</figcaption>
</figure>

The chart is the argument in shape. The four interventions everyone tries first sit flat against the baseline. The rm_kq patch is the only GPU-side change with a measurable signal at all, and it costs less than a percent. The one big bar belongs to a kernel parameter for the wrong device family.

The inset is the structural reason. A single decode step issues hundreds of small PCIe transactions, each paying an L1 exit latency if the link has been allowed to drop into a low-power substate. With ASPM at performance, the link stays in L0 across the entire decode step and every dispatch reaches the GPU's command ring without a renegotiation tax. With ASPM at the default, every transaction after a short idle interval pays the tax. The total cost is the integral of those microseconds over the dispatch count, which for dense decode on a transformer of this size is enough to move tok/s by double digits.

## Why this is not solved upstream

The honest answer is that nobody has reason to solve it. The kernel default leans powersave because PCIe ASPM matters for laptop battery life and is invisible on every workload that does not issue hundreds of small transactions per millisecond. The amdgpu driver cannot speak for the upstream port. The Vulkan and HIP runtimes have no API surface for PCIe link power management. The AMD SMI library's [performance-determinism and performance-level controls](https://rocm.docs.amd.com/projects/amdsmi/en/develop/conceptual/perf-determinism.html) end at the GPU's own power state machine. The user is left to set a sysfs parameter on the root complex, which is something most local-inference users have never had reason to learn exists.

The [GPU sysfs power-state documentation in the kernel tree](https://dri.freedesktop.org/docs/drm/gpu/amdgpu/thermal.html) is unusually thorough about every knob amdgpu owns, and the existence of `power_dpm_force_performance_level` and `pp_power_profile_mode` is the reason every RDNA4 tuning guide starts by writing to those files. They are the visible knobs. The invisible knob is the one with the largest effect on the one workload the GPU is currently most used for.

There is also a structural reason this gets harder on multi-GPU rigs. The R9700 in the discussion's reference machine sits behind PCIe 5.0 x16 lanes off the CPU. A second card on the same machine often gets x8 or x4 lanes off the chipset, with an additional PCIe switch in the path. Every hop in that chain has its own ASPM state machine, and the `pcie_aspm.policy=performance` setting disables ASPM globally across all of them. The power cost of that is not zero on a desktop with a dozen NVMe drives and several network controllers, but it is small relative to the GPU under load, and it is the right tradeoff when the workload is local inference.

## What the right RDNA4 local-inference baseline looks like

The actionable summary is short. On a Radeon AI PRO R9700 running llama.cpp with Vulkan RADV for dense local Qwen3, the kernel command line should include `pcie_aspm.policy=performance`. The sysfs equivalent is a one-line write to `/sys/module/pcie_aspm/parameters/policy`. The amdgpu power profile should be left at its default; the GPU is already at peak during decode and the COMPUTE profile costs power without buying throughput. The micro-batch size should be 2048 for prefill workloads but does not move decode. The rm_kq=1 patch is worth carrying as a one-line local diff against the Vulkan backend for the 0.8% it buys on dense and the 13% it buys on AMDVLK. Nothing else in the long tail of tuning lore matters at the noise floor of the benchmark.

For MoE workloads on the same card, the picture inverts. The 35B-A3B target gains nothing from the ASPM fix and a few percent from `GGML_VK_ALLOW_GRAPHICS_QUEUE=1` on AMDVLK. The right move is to switch drivers based on the model shape, which is the opposite of the consensus that one driver is universally faster.

The general lesson is the one the [past two weeks of the chart sequence](/blog/2026-05-09-the-fp4-wave-breaks-at-rdna4-and-fp8-wmma-already-does-what-local-qwen3-needs) has been gesturing at from a different angle. Local-inference performance on consumer hardware is increasingly bounded by interfaces between subsystems, not by the subsystems themselves. The matrix engines on the R9700 are not the bottleneck; the GDDR6 bandwidth is not the bottleneck for dense decode either; the bottleneck is the small overhead the host pays each time it asks the GPU to do something. PCIe ASPM is one of those overheads, and the kernel's default policy was written for a workload that does not exist on a local-inference machine. Flipping it costs nothing, fixes a measurable problem, and survives until the next reboot.

The next thing zinc is watching on this card is the long tail of the same argument. Driver-side fence-coalescing in RADV, command-buffer reuse across decode steps, and the still-unmerged work on persistent kernel launches all attack the same per-dispatch overhead from inside the GPU stack. The PCIe knob is the cheapest version of that work; the rest of it will be where the next double-digit wins come from on the same hardware.
