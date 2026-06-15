# Effort 15 - ZINC_RT direct decode path to 120+ tok/s on RDNA4

Created: 2026-05-16

This effort exists because the ZINC_RT autopilot reached about 80 tok/s, then
stalled. The important finding is that this is not yet a GPU-kernel tuning
plateau. It is a host-assisted shortcut path with small direct PM4/CS probes.

## Current Standing

Cycle 64 reported:

```
vulkan:  115.0 tok/s
zinc_rt:  80.2 tok/s
ratio:     69.7%
```

That number is not directly comparable to Vulkan yet. The zinc_rt run logs:

```
forward_zinc_rt M1 host-assisted path with direct token-boundary gate
decode LM-head row scan capped at 4096/248320 rows
decode MoE top-k lowered to 0 after prefill
path clamped decode budget from 256 to 8 tokens
direct_compute_ops=2 direct_compute_kind=argmax_rms_norm_elem0
```

So the 80 tok/s result is a bring-up metric, not proof that M2/M3 decode is
near Vulkan. It exercises CPU-side decode with two tiny direct compute probes.

## 2026-06-14 Re-baseline (shortcut-free) + resident-weight pivot

Hands-on re-baseline this session (`--dry-run`, runs/bin=3, `/root/zinc-rdna1-m1`,
Qwen3.6-35B-A3B, full LM head, no row cap / no top-k clamp):

```
vulkan:  109.8 tok/s   (samples 128.4 / 109.8 / 97.5, coherent)
zinc_rt:  34.3 tok/s   (34.3 / 35.4 / 34.1, coherent)
ratio:    0.312
```

The 80 tok/s / 0.70 in "Current Standing" was shortcut-inflated. The honest
MIGRATE standing is **~0.31**.

**Root cause (verified in `src/compute/forward_zinc_rt.zig`):** the bulk Q4_K
matvecs (`matvecRaw`/`matvecTensor`/`matvecFusedTensors` — MoE gate/up/down,
attn projections, LM head) run on the CPU thread pool (`state.pool`). The GPU
only runs tiny LM-head argmax validation slices. Decode is CPU-DRAM-bandwidth
bound (~3B active Q4_K bytes/token → ~35 tok/s ≈ CPU BW ceiling).

**DO NOT (new anti-pattern):** route bulk decode matvecs through the existing
`dmmvQ4_0RowRangeParallel` as-is. It re-stages the weight bytes into `input_map`
**every call** (`cs.zig:1866`) and is capped at **64 rows per submit-and-wait**.
Per token that is a full weight re-upload + ~33 µs submit × ~1400 matvecs →
**slower than the CPU pool**. This is exactly the scalar-plateau trap; verified
by code inspection.

**The lever = resident weights + batched submit:**
1. Weights resident in VRAM (own BO, uploaded once at load — not re-staged per call).
2. Grid-over-rows: workgroup `g` handles rows `[g*64, (g+1)*64)` reading
   `weight_va + g*64*row_bytes`; dispatch `grid=(ceil(rows/64),1,1)` → ONE submit
   per matvec. (The parallel template avoided multi-WG TGID delivery — validate
   it on gfx1201; if TGID is unreliable, fall back to a few single-WG chunk
   dispatches, still one upload.)
3. Per token: upload only the input activation (`cols*4` B), read back outputs.

**First target = LM head.** ~half the per-token DRAM bytes (design doc §0), one
matvec/token (amortizes the single submit cleanly), `Model.lm_head_q4_0` blob
already built at load, already partially GPU-wired (`dmmvQ4_0ArgmaxRowRange`).
Rough budget: 384 MB Q4_0 head ≈ 7.7 ms on CPU (~50 GB/s) → ~0.7 ms on GPU
(576 GB/s) → ~+30% decode if the submit is amortized.

**Cycle-1 steps:** (a) add a resident weight BO to `TokenBoundary` + upload the
head once; (b) new `dmmv_q4_0_resident_grid.s` (parallel template + resident
weight ptr in user_data + `workgroup_id_x` row-block offset), embed in `cs.zig`;
(c) `dmmvQ4_0Resident(input, rows, cols, output)` boundary method; (d) validate
**bit-exact vs the CPU LM-head matvec** before consuming; (e) consume in decode,
keep coherence, A/B vs Vulkan, keep iff ratio↑ + coherent. Cycle 2 = native
K-parallel **Q4_K** resident kernel (§806) for the gate/up experts that stay Q4_K.

### 2026-06-15 de-risk probe results (`zinc --probe-gpu`)

Built a reproducible GPU probe (`cs.zig` `sgprTgidProbe`/`sgprTgidDump`/
`probeVramMappable`, `main.zig --probe-gpu`, kernels `tgid_probe.s` +
`sgpr_tgid_dump.s`, llvm-mc-assembled) to validate the two gating assumptions
before building the resident matvec:

- **VRAM residency: ✅ WORKS.** A 400 MB `AMDGPU_GEM_DOMAIN_VRAM` BO (with
  `CPU_ACCESS_REQUIRED`) is CPU-mmap'able and round-trips a host write/read →
  large-BAR is available, the LM-head weight can live in VRAM at 576 GB/s. The
  perf premise is sound.
- **Multi-WG TGID delivery: ❌ BROKEN (root blocker, reproduced).** With 8 user
  SGPRs + `ENABLE_SGPR_WORKGROUP_ID_X` (rsrc2=0x90) and `dispatchDirectInitiator(4,1,1)`,
  the kernel RUNS (signature `0x53475052` written) but `workgroup_id_x` reaches
  **none of s8..s19** for any of the 4 workgroups (exhaustive value-as-index
  dump, all slots stay sentinel). User-data delivery is fine (output ptr in s0/s1
  works), so the issue is specifically the *system* SGPR (workgroup id), not the
  ABI offset. This is the same wall that stalled cycles 66–82.

**Consequence:** without the workgroup id in-shader, the matvec can't be spread
across the ~64 CUs in one submit (1 WG = 1 CU ≈ 1/64 bandwidth → loses to CPU),
and per-WG host re-submit is too many submits/token. So the direct GPU matvec is
blocked on gfx12 TGID delivery, NOT on VRAM or the kernel itself.

**Next attempts to crack it (untried here):** (1) sweep s20..s31 / ttmp regs;
(2) confirm whether `dispatchDirectInitiator(N)` actually launches N workgroups
(atomic-count probe) vs a grid-dims register issue; (3) MIMIC RADV's exact gfx12
compute COMPUTE_PGM_RSRC2 + COMPUTE_DISPATCH_INITIATOR + grid register setup
(Vulkan/RADV dispatches multi-WG compute fine at 110 tok/s, so the correct
register recipe exists — copy it). Probe lives on branch
`zinc-rt-rdna1-lmhead-gpu-matvec`.

## Profiling Snapshot

Remote node: R9700 RDNA4 / gfx1201, Linux 6.17, Zig 0.15.2. Measurements were
run on a clean export of committed HEAD under `/root/zinc_profile_analysis`, not
on the interrupted local worktree patch.

### A/B environment matrix

Three samples per case, one run per binary shape:

| Case | Median tok/s | Notes |
|---|---:|---|
| default | 79.38 | 4096 LM-head rows, 3 workers, top-k=0 after prefill |
| `ZINC_RT_CPU_WORKERS=4` | 80.45 | Slightly better, mostly noise |
| `ZINC_RT_CPU_WORKERS=2` | 73.63 | Too few workers |
| `ZINC_RT_FAST_POOL=0` | 76.18 | FastPool is worth about 2-5 tok/s |
| `ZINC_RT_LM_HEAD_ROWS=8192` | 78.99 | 4096 -> 8192 is basically noise |
| `ZINC_RT_LM_HEAD_ROWS=2048` | 79.83 | Tiny gain, worse output |
| `ZINC_RT_LM_HEAD_ROWS=0` | 59.66 | Full 248320-row LM head costs about 3.5-4.3 ms/token |

The full-vocab result matters: even on the current CPU path, the shortcut LM
head is hiding a real correctness/performance cost.

### Whole-process perf

Whole-process `perf record` mostly profiles model load and re-quantization:

```
71.91%  forward_zinc_rt.buildRequantizedQ4_0
11.53%  isa.cpu_zig.dequant.row
 4.41%  forward_zinc_rt.matvecRawDirectSerial
```

Do not use whole-process perf to reason about decode until the process is held
inside the decode loop long enough.

### Decode-only perf

A remote scratch build changed only `m0_max_decode_tokens` from 8 to 512. The
CLI still generated 256 tokens, enough to sample decode:

```
Generated 256 tokens in 3756.4 ms - 68.15 tok/s (14.7 ms/tok)
Output text: Paris, and the city of the city of the city ...
```

Decode-only `perf record -p <pid>` for five seconds:

```
78.09%  forward_zinc_rt.matvecRawDirectSerial
11.38%  Thread.PosixThreadImpl.spawn__anon_11386.Instance.entryFn
 4.26%  forward_zinc_rt.runSsmHeadRange
 0.96%  forward_zinc_rt.scalarEvalToken
 0.52%  forward_zinc_rt.matvecFused
```

The call stacks put most samples in `matvecFusedWorkerTask` and
`matvecDirectWorkerTask`. This is CPU matrix-vector work. Profiling does not
show a hot native GPU model kernel because there is not one in the exercised
path yet.

## GPU Opcode Verdict

The native gfx1201 kernels currently exercised from `src/zinc_rt/ring/cs.zig`
are validation probes:

- `argmax_top2_gfx1201`: one wave, two scalar scores, select one of two token
  IDs, `global_store_b32` the result.
- `rms_norm_elem0_gfx1201`: one wave, three `global_load_b32`, two `v_mul_f32`,
  one `global_store_b32`.

They prove PM4 dispatch, SGPR user data, memory visibility, and fence ordering.
They do not touch the model's DMMV workload, router row ranges, SSM projections,
MoE expert matrices, or full LM head. Hand-tuning these two opcode blobs cannot
bridge the gap from 80 tok/s to 120 tok/s.

The first useful opcode/kernel work is a real model slice:

- a Q4_0/Q8_0 DMMV row-range kernel reachable through T1/T2,
- a router row-range whose GPU output changes the selected expert,
- or an LM-head row range whose GPU score participates in the sampled token.

## Performance Target

To beat 120 tok/s, decode must be below 8.33 ms/token. The long scratch run is
14.7 ms/token, so the remaining gap is at least 6.4 ms/token. Since about 78%
of sampled decode cycles are CPU `matvecRawDirectSerial`, the only credible path
is to remove CPU matvec work from the token path.

Small CPU wins are exhausted:

- worker count is within noise around 4 workers,
- FastPool already contributes a small win,
- 4096 -> 8192 LM-head rows is noise,
- 2048 LM-head rows degrades output,
- top-k=0 and an 8-token clamp make the current metric easier than the real
  decode problem.

## Source Material To Use

Use the existing Vulkan path as the correctness and dataflow reference:

- `src/compute/forward.zig` - real layer/token dataflow.
- `src/compute/dmmv.zig` - matrix-vector dispatch ABI and shape selection.
- `src/shaders/dmmv_q4k.comp`, `src/shaders/dmmv_q8_0.comp` - row-oriented
  quantized DMMV logic.
- `src/shaders/softmax_topk.comp` - GPU router/top-k reference.
- `src/zinc_rt/ring/cs.zig` - current PM4 dispatch, SGPR ABI, fence, and GTT
  buffer plumbing.
- `docs/GPU_REFERENCE.md` - RDNA4 wave64/cache/occupancy facts. Use it after a
  real model kernel exists; it is not the next bottleneck by itself.

## Recommended Next Moves

1. Build a minimal direct DMMV row-range kernel.

   Start with the re-encoded Q4_0 path used by host-assisted decode. Use the
   `cs.zig` ABI pattern: user SGPRs for input/output pointers and scalar shape
   constants, direct `DISPATCH_DIRECT`, explicit signal write, CPU compare.
   The first version may compute 1-8 rows; it must consume the result in the
   current prompt path.

2. Prefer an LM-head row-range or router row-range as the first consumed slice.

   Router row-range is small and easier to validate but has limited perf upside.
   LM-head row-range has direct quality/perf significance because full-vocab CPU
   LM head costs several ms/token. A partial LM-head row-range is acceptable
   only if the output explicitly reports the covered rows and whether the GPU
   value affected the selected token.

3. Remove benchmark shortcuts before claiming an optimization win.

   A real win should survive:

   ```
   ZINC_RT_LM_HEAD_ROWS=0
   ZINC_QWEN36_DECODE_TOPK=<metadata/default or explicit nonzero>
   generated tokens >= 128
   ```

   If those knobs are not ready, label the result as M1 validation only.

4. If direct DMMV cannot advance, write a measured-dead report.

   The report must include the exact remote UAPI or hardware blocker: failed
   ioctl, invalid PM4 packet, memory visibility failure, shader fault, missing
   address space, or unsupported queue feature. Do not spend another cycle on
   admission markers without a consumed model value.

## Do Not Repeat

- Do not tune the `argmax_top2` or `rms_norm_elem0` opcode blobs as a path to
  120 tok/s. They are not the hot path.
- Do not call `model_execution=hybrid_direct_compute` a direct model path when
  the run also says `host-assisted`.
- Do not treat an 8-token clamped output as evidence of steady-state decode.
- Do not chase continuous batching, paged KV metadata, descriptor churn, or
  submission-count tuning while the hot profile is CPU `matvecRawDirectSerial`.
- Do not make more CPU worker, dequant, or FastPool changes unless they are a
  prerequisite for validating a direct GPU slice.

## Harness Implications

The autopilot should keep the current state in MIGRATE/M1 until the run is both:

- shortcut-free enough for quality comparison, and
- backed by a real GPU-produced model value that is consumed by generation.

The loop may still print tok/s for visibility, but it should not let the 80 tok/s
shortcut number push the prompt into M2/M3 work.
