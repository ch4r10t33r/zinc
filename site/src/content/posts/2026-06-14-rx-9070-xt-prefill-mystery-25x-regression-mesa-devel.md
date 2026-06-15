---
title: "The 9070 XT prefill mystery: a 25× regression hiding in a Mesa devel build"
seoTitle: "RX 9070 XT: finding a 25× prefill regression in Mesa 26"
date: "2026-06-14"
tags:
  - zinc
  - amd
  - rdna4
  - rx-9070-xt
  - gfx1201
  - mesa
  - radv
  - vulkan
  - prefill
  - qwen3.5
  - dp4a
  - integer-dot-product
  - cooperative-matrix
  - local-llm
  - llm-inference
  - gpu-kernels
  - debugging
  - llama-cpp
keywords:
  - RX 9070 XT LLM prefill
  - Mesa 26 RADV regression
  - RDNA4 GFX1201 Vulkan compute
  - DP4a shader slow Mesa devel
  - ZINC vs llama.cpp RDNA4
  - Qwen3.5-9B prefill tok/s
  - integer dot product Vulkan
  - cooperative matrix RDNA4
  - GPU clock ramp benchmark noise
  - autonomous LLM optimization loop
  - opencode GLM-5.2 agent
faqs:
  - question: "Why was ZINC prefill on the RX 9070 XT 143× slower than llama.cpp when decode was fine?"
    answer: "It was a driver-level shader regression, not a ZINC bug. The 9070 XT is brand-new RDNA4 hardware that requires Mesa 26; on that Mesa devel build, the int8 DP4a (shaderIntegerDotProduct) dense gate+up+SwiGLI kernel compiled by RADV ran about 25× slower than the non-DP4a branchless path — a single 64-token prefill spent 97% of its wall clock (9.1 seconds) inside that one kernel. Decode was unaffected because decode uses per-token matvec dispatch, not the batched DP4a GEMM. The fix is to default that model off the DP4a path on RDNA4 when cooperative matrix is exposed, falling back to a branchless Q4_K kernel that is already loaded. 6.8 → ~170 tok/s, identical output."
  - question: "How did you isolate a driver regression from a ZINC code bug?"
    answer: "A natural experiment: the project has a second RDNA4 node, a Radeon AI PRO R9700, which is the same GFX1201 architecture but pinned to Mesa 25.0.7 (the older, stable driver). Same ZINC code, same Qwen3.5-9B weights, same Vulkan path — the R9700 prefilled at 392 tok/s, the 9070 XT at 6.8. Then the decisive toggle: ZINC_QWEN_DENSE_FFN_DP4A=0 on the 9070 XT took it straight to 169 tok/s. Same shader, two driver versions, 25× apart — that isolates the cause to the compiler, not the kernel logic."
  - question: "Does decode on the 9070 XT beat llama.cpp?"
    answer: "Yes. ZINC decodes the Qwen3.5-9B at about 39.6 tok/s versus llama.cpp's 21.7 tok/s (tg256) on the same card — a wider lead than on the R9700. The low absolute number is a card/driver characteristic that hits both engines equally on this node; it is not a ZINC regression, which is why the fix targets prefill only and explicitly leaves decode alone."
  - question: "Why did the autonomous optimization loop keep reverting real improvements at first?"
    answer: "Benchmark noise from GPU clock ramp. In dynamic power mode the 9070 XT drops its shader clock to 0 MHz between runs and ramps back to 2.5 GHz over about 0.4 seconds — and a 64-token prefill burst only lasts about 0.37 seconds. So each short sample caught a different point on the ramp and the median swung between 160 and 210 tok/s on identical code, drowning any real 1–2% win. Pinning the card to high-performance mode collapsed that to a <0.6% variance baseline, after which keeps started landing."
  - question: "How far behind llama.cpp is ZINC prefill on the 9070 XT now?"
    answer: "About 4.5×. After the regression fix the small autonomous loop (GLM-5.2) stacked another ~6.5% on top, taking prefill from 205 to ~219 tok/s; llama.cpp measures 973 tok/s (pp512) on the same card. The remaining gap is structural kernel efficiency — llama fuses weight dequantization into the GEMM (MMQ), while ZINC's batched path round-trips through a wider scratch — so closing it is real GEMM work, not another toggle."
excerpt: "A brand-new RX 9070 XT, ZINC prefilled the Qwen3.5-9B at 6.8 tok/s while llama.cpp ran 973 — a 143× gap, with decode perfectly fine. The cause turned out to be a 25× shader regression in a Mesa 26 devel build, isolated by a natural experiment across two nearly-identical RDNA4 cards. Then: a GPU clock-ramp noise trap, a mid-cycle crash, an opencode/GLM-5.2 optimization loop, and a stack of incremental keeps taking prefill from 205 to 219 tok/s. Still 4.5× behind llama — honestly — but the detective story is the point."
seoDescription: "Debugging a 143× ZINC prefill gap on a new RX 9070 XT (RDNA4, GFX1201, Mesa 26 devel): a 25× DP4a shader regression isolated by comparing against a Mesa 25.0.7 R9700, the clock-ramp benchmark noise trap, an opencode/GLM-5.2 autonomous loop, and the incremental keeps that took Qwen3.5-9B prefill from 6.8 to 219 tok/s."
draft: false
---

A new GPU arrived in the rack — a consumer **Radeon RX 9070 XT**, Navi 48, RDNA4, GFX1201, 16 GB. Brand-new silicon. The kind of card a Vulkan LLM engine wants to be good on, because it is the cheapest way to put a fast RDNA4 compute die on a desk. So the first thing to do is the obvious one: point ZINC at the smallest Qwen, the 9B, and read the number.

The number was **6.8 tok/s prefill**.

On the same card, on the same model, `llama-bench` reads **973 tok/s**. That is a **143× gap**, and the truly strange part: **decode was fine** — ZINC was decoding at 39.6 tok/s, comfortably *ahead* of llama's 21.7. So the engine works. The forward pass is correct. The GPU is being used. And yet prefill — the part where you feed the whole prompt in one batched shot — is off by two orders of magnitude.

This is the story of finding the 25× regression hiding inside that 143×, why it took a *second* GPU to see it, and then everything that came after: a clock-ramp noise trap that fooled the optimizer for half a day, a crash, an autonomous loop driven by GLM-5.2, and the stack of incremental keeps that finally got the 9B from a flatline to a real curve.

## The scene, and the first (wrong) theory

The 9070 XT is an awkward card to benchmark for one specific reason: it is so new that the only driver that knows its device ID is **Mesa 26.0.0-devel**. The project's reference RDNA4 node — a Radeon AI PRO R9700 — is deliberately pinned to **Mesa 25.0.7**, because 25.2.8 already cost it ~14% in a known RADV regression. So the two RDNA4 cards in the rack run *different Mesa major versions*, and there is no way around it: older Mesa literally does not enumerate GFX1201.

The first theory was the boring one and, like most boring theories in kernel work, it was wrong. The Qwen3.5-9B is a dense SSM+attention hybrid, and an older optimization effort had already documented that its fast prefill path was **shape-locked to the 27B model** — a predicate literally named `isQwen36DenseHybrid27B` gated the whole layer-major machinery on `hidden_dim == 5120`, which the 9B (4096) fails. The hypothesis wrote itself: the 9B was falling through to `prefillBatch`, the per-token "token-major" path, and token-major is just slow.

Easy to check. Probe the GPU while prefill runs:

```
sample 1: gpu= 18%  cpu= 0.5%   (warming up)
sample 2: gpu=100%  cpu= 4.2%
sample 3: gpu=100%  cpu= 3.2%
sample 4: gpu=100%  cpu= 4.8%
```

The GPU is **pinned at 100% busy** for the entire prefill, CPU nearly idle. This is not a fallback to a slow host path. The GPU is hard at work — it is just doing the *wrong work*, slowly. Theory one: dead.

So the profiler. ZINC has a built-in `ZINC_PREFILL_PROFILE=1` that bucketizes the prefill wall clock into GPU phases. For a 64-token prompt (the "decode-extended" site scenario), the totals over a 9.5-second prefill:

```
Prefill GPU phases totals: attn=43.1  ssm=172.3  dense_ffn=9223.6  tail=2.3 ms
dense_ffn subphases: gateup_matmul=9123.4 (generic=9123.4  q4=0.0  q6=0.0)
```

**Ninety-seven percent of the prefill is in one sub-phase**, the dense feed-forward gate+up matmul, and it is running the *generic* path — not the specialized `q4` path — for the full 9.1 seconds. Everything else (attention, the entire SSM stack) is a rounding error next to it. Whatever is wrong is wrong in exactly one matmul.

## The two-GPU natural experiment

Here is the move that cracked it, and it is a move that only works because the rack happens to have *two* RDNA4 cards.

The Radeon AI PRO R9700 node is the same architecture — GFX1201, 64 CUs, same wave size, same cooperative-matrix silicon — but pinned to the stable Mesa 25.0.7. Same ZINC binary. Same Qwen3.5-9B Q4_K_M weights. Same Vulkan path. The only things that differ between the two cards are the things you cannot make equal: the Mesa version, and one capability bit.

Run the identical 64-token prefill on both:

| | R9700 (Mesa 25.0.7) | 9070 XT (Mesa 26.0-devel) |
| --- | ---: | ---: |
| Prefill, 64 tok | **160 ms / 392 tok/s** | 9479 ms / 6.8 tok/s |
| gateup matmul phase | **~0 ms** | 9095 ms |
| `cooperative_matrix` exposed | no | yes |

Same shader. Same dispatch. Same model. **Fifty-eight times apart.** And the profiler's smoking gun: the gateup matmul time on the R9700 is *zero* (it runs in the fused path, attributed elsewhere), while on the 9070 XT the *same code path* logs 9.1 seconds under the `generic` phase label.

There is exactly one knob that distinguishes "use the int8 DP4a (`shaderIntegerDotProduct`) kernel" from "use the branchless Q4_K kernel": a predicate called `qwenDenseFfnDp4aEnabled`. The decisive experiment takes one line:

```
ZINC_QWEN_DENSE_FFN_DP4A=0  ./zig-out/bin/zinc --model-id qwen35-9b-q4k-m ...
```

Disabling DP4a on the 9070 XT takes prefill from **6.8 → 169 tok/s**. Twenty-five times faster. **Identical output** (`Command shape: benchmark --preset ...`, the correct Long Coding Plan answer). The regression is not in ZINC's logic at all. It is in the **compiler**: on Mesa 26.0.0-devel, when the device also exposes `VK_KHR_cooperative_matrix`, RADV emits a catastrophically slow SPIR-V→AMDGPU lowering for the int8 dot-product accumulate kernel (`mul_mm_q4k_gate_up_swiglu_full_dp4a`). The same kernel is fast on RDNA4 without cooperative matrix (Mesa 25.0.7). It is a driver regression on brand-new hardware, and it was quietly absorbing 97% of the prefill wall clock.

This is why the two-node trick mattered. On a single card, "the DP4a kernel is slow" looks identical to "my kernel is badly tuned" or "the 9B is hitting a slow path" — and you can spend a week retuning a shader that was never the problem. The R9700 was the control that proved the shader logic was correct and the *environment* was the variable.

## The fix, and the scoreboard

The fix is small and deliberately narrow. `qwenDenseFfnDp4aEnabled` now defaults the Qwen3.5-9B to the **branchless Q4_K path** when `amd_rdna4` *and* `cooperative_matrix` are both exposed — i.e. exactly this configuration — and lets the already-loaded `mul_mm_q4k_gate_up_swiglu_full` kernel run instead. It is scoped to the 9B so the 27B and Gemma paths are untouched, and `ZINC_QWEN_DENSE_FFN_DP4A=1` forces the DP4a path back on for re-testing the day Mesa 26 ships stable.

<figure class="diagram-card diagram-wide">

| Qwen3.5-9B prefill on the 9070 XT | before | after | |
| --- | ---: | ---: | ---: |
| 64-token (decode-extended) | 6.8 tok/s | **~170 tok/s** | 25× |
| 326-token (context-long) | 5.8 tok/s | **~153 tok/s** | 26× |
| 781-token | — | ~142 tok/s | — |
| **decode** (256 tok) | 39.6 tok/s | 39.6 tok/s | unchanged — still beats llama's 21.7 |

  <figcaption>The single-gate fix. Decode is deliberately untouched (it already led llama). The R9700 node — cooperative matrix off, Mesa 25.0.7 — is <strong>also untouched at 421 tok/s</strong>, still on the fast DP4a path. One config, one model, one regression.</figcaption>
</figure>

So the 143× gap was really a 25× regression sitting on top of an honest ~6× structural gap to llama's batched GEMM — and the regression was the part you could fix in an afternoon, once you could see it.

## Then: a noise trap, and a clock that would not sit still

With the regression fixed, the obvious next move is to point an autonomous optimization loop at the remaining gap and let it grind. The project already had a harness for exactly this — `optimize_perf.ts` — but it ran `claude`/`codex` agents. The first harness work was adding **opencode + GLM-5.2** as a first-class agent (`opencode run --model zai-coding-plan/glm-5.2 --variant max`), plus a crash-restart supervisor, plus an RDNA2-targeted effort doc seeding the loop with everything above so it would not re-tread the DP4a discovery.

The loop started up, ran the baseline at a healthy 212 tok/s, and then… **reverted every single cycle**. Cycle 1: reverted. Cycle 2: reverted. Cycle 3: reverted. Each with a hopeful-looking change that came back inside the noise floor. This is the optimizer's job — it should not keep a change that does not measurably win — but the *shape* of the reverts was wrong: the samples inside one cycle were swinging from **160 to 210 tok/s on identical code**.

The 9070 XT's `pp_dpm_sclk` reads three states: `500 MHz`, `0 MHz`, `2520 MHz`. In dynamic (`auto`) power mode the card drops to that **0 MHz** idle state the moment a kernel stops issuing, and ramps back to 2.5 GHz over **~0.4 seconds** when work arrives. A 64-token prefill burst lasts **~0.37 seconds**. The burst and the ramp are the same length. So whether a given sample catches the clock cold (mid-ramp, ~160 tok/s) or warm (already boosted, ~210 tok/s) is a coin flip, and the loop's median-of-five flipped with it. Three cycles of "real" improvements were drowned in that coin flip.

The fix is the sysfs one-liner the harness now runs before every benchmark:

```
echo high > /sys/class/drm/card1/device/power_dpm_force_performance_level
```

Pin the card to high-performance and the ramp disappears — consecutive samples come back at **205.23, 205.97, 205.74 tok/s**, a variance under 0.6%. (And `high` mode survives a 30-second idle that drops the reading back to 0 MHz: the *ramp* is what it kills, so even a cold-start burst now lands at full boost.) With a stable floor, the loop's keep threshold — `max(+3 tok/s, +2%)` — finally means something. As a bonus, the old 212 tok/s "baseline" turned out to be a lucky-warm outlier from the noisy era; the real stable speed was 205, and a fresh re-baseline reset the target to something achievable.

## The crash, and the supervisor

Of course the loop then crashed.

Midway through cycle 1 of the clean run — agent reading the dense-down shader, heartbeat ticking at seven minutes — the process simply vanished. No error. No stack trace. No OOM kill, no `SIGKILL` in the system log, no crash report. One transient exit, and the entire multi-hour effort was dead. The existing harness treated any exit as terminal.

The lesson, learned the hard way, is that a long autonomous loop needs to be **restartable by construction**, the way the project's overnight `zinc_rt-autopilot` loop already is. A small supervisor (`loops/supervise_perf.sh`) now wraps the effort: any exit — crash, timeout, or just cycles-exhausted — is followed by a `--resume` restart from the last committed cycle boundary (state is written only after a keep/revert decision, so a mid-cycle crash never corrupts it). Run until killed. The next crash will cost one cycle, not the whole run.

(Along the way, two genuinely funny macOS-as-host footguns bit the GLM-5.2 agent: it kept wrapping commands in `timeout` — which does not exist on macOS — and one launch used `setsid`, which also does not exist. The effort doc now carries an "Agent environment" section: *no `timeout`, no `setsid`, raise the shell-tool per-command timeout for builds, the model is remote-only, the controller does the authoritative build so a local one is just a sanity check*. Small frictions, but they were each costing the agent a full turn.)

## The keeps, honestly

Stable floor, restartable loop, max-reasoning agent. From there the loop did what loops are supposed to do: grind.

<figure class="diagram-card diagram-wide">

| cycle | change | prefill tok/s |
| --- | --- | ---: |
| baseline | (regression fix, stable floor) | 205.44 |
| 1 | dense-FFN tuning | **208.10** |
| 2 | +1.88% | **212.02** |
| 3 | foundation: SSM conv1d recurrent state → shared memory | — |
| 4 | +2.75% | **217.84** |
| 5 | foundation: profiling timestamp artifact fix | — |
| 8 | foundation: fused SSM gated-norm shader (one WG/head, all tokens internal) | — |
| 9 | +0.40% | **218.71** |

  <figcaption>Nine cycles: four perf keeps, three "foundation" (perf-neutral enablement) keeps, three reverts, zero broken. The mix is what you want — the foundation keeps are the agent banking groundwork that later cycles build on.</figcaption>
</figure>

**205 → 218.71 tok/s, about +6.5%**, stacked entirely on top of the 25× regression fix. The keeps are real — each cleared the `max(+3, +2%)` bar against a tight noise floor — and they are the kind of incremental, profile-driven work (cache the conv1d state in shared memory, fuse the gated-norm dispatch, fix a profiling-timestamp artifact that was mis-attributing time) that an autonomous loop is genuinely good at once its measurements are trustworthy.

## The honest ceiling

The 9070 XT went from **6.8 to ~219 tok/s** on the Qwen3.5-9B. Decode was never the problem and still leads llama (39.6 vs 21.7). And yet — `llama-bench pp512` on this same card reads **973 tok/s**. ZINC is still **~4.5× behind** on prefill, and that last gap is the hard one: it is structural kernel efficiency. llama.cpp fuses weight dequantization *into* the GEMM (MMQ), so each weight block is read once and consumed on the fly; ZINC's batched path dequantizes into a wider scratch and reads it back. Closing that is real GEMM work — a persistent fp16 weight cache, or an MMQ-style fused-dequant kernel — not another predicate flip. The loop is now grinding on exactly that, on a node whose measurements finally hold still long enough to trust a 2% win.

Two lessons, neither new, both worth retelling. **A single GPU cannot diagnose its own driver regression** — "my kernel is slow" and "my compiler is broken" are indistinguishable without a control, and the control here was a second card running an older Mesa. And **a noisy benchmark is worse than no benchmark**: it does not just fail to find improvements, it actively manufactures confident-looking reverts of real wins. Fix the floor before you trust the scoreboard. After that, an autonomous loop with a max-reasoning model and a restartable harness will grind out the 6% you would have found yourself, and keep going through the crash you would not have.
