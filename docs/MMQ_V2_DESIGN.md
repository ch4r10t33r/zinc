# ZINC MMQ v2 — Fused Q4_K×Q8_1 Tensor-Core GEMM (beat llama.cpp)

**Goal:** a custom CUDA kernel for Q4_K (and Q6_K/Q5_K/Q8_0) prefill GEMMs that
**beats llama.cpp's MMQ by ≥20%** on RTX 4090 (sm_89) and RTX 5090 (sm_120).

## Why this exists

ZINC's current dense prefill path does `dequant Q4_K→fp16` + `cuBLAS fp16 TC GEMM`.
llama.cpp's MMQ fuses the dequant INTO the TC inner loop + uses Q8_1 activation
(half the activation bytes). Result: ZINC dense prefill is 6-16× behind llama.cpp.

The gap doc (`docs/prefill_gap_closure_plan.md`) identifies this as lever T1 —
the highest-impact remaining work. This doc designs the replacement.

## llama.cpp's MMQ — what they do right + what they leave on the table

### What they do right (we must match)
- **Q8_1 activation**: quantize fp32 → Q8_1 (1 byte/elem) before the GEMM → halves
  activation DRAM traffic vs fp16
- **Fused dequant**: Q4_K block unpacked in registers per `mma.sync` call — no
  separate dequant pass
- **128×128 tile** (mmq_x=128, mmq_y=128), 8 warps (256 threads/CTA)
- **mma.sync m16n8k16** (not wmma — finer-grained, lower register pressure)
- **Stream-K** work decomposition for full SM utilization at small T
- **Always preferred over cuBLAS** on Turing+ (`turing_mma_available → true`)

### What they leave on the table (our edge)

| Gap | llama.cpp | ZINC v2 improvement | Expected gain |
|-----|-----------|---------------------|---------------|
| **Pipeline depth** | Manual 2-stage with `__syncthreads` | **cp.async 3-stage** (overlap load+dequant+compute, no sync between stages) | +10-15% (hides ~33% more latency) |
| **Activation quant** | Separate `quantize_mmq_q8_1_cuda` kernel (extra DRAM round-trip: read fp32, write Q8_1, read Q8_1) | **Fuse fp32→Q8_1 into the GEMM pipeline** (quantize in shared mem during cp.async load) | +5-10% (eliminates one full activation DRAM pass) |
| **Adaptive activation format** | Always Q8_1 | **fp16 at large T** (compute-bound, skip quant overhead); **Q8_1 at small T** (bandwidth-bound, halve traffic) | +5-10% at T≥1024 |
| **Weight in shared mem** | Q4_K blocks → dequant in registers per mma | **Dequant Q4_K→fp16 in shared mem once** (during cp.async stage), then `ldmatrix` from fp16 — cleaner pipeline separation | +5% (fewer per-mma register ops) |
| **MoE expert routing** | Same MMQ kernel with `ids` (gather into Q8_1) | **Persistent kernel** processes expert buckets sequentially — expert weight stays in shared mem across all its tokens (zero re-read) | +15-20% on MoE models |
| **Stream-K fixup** | Extra fixup kernel + atomic buffer | **Persistent CTAs with atomic work queue** — no fixup, simpler dispatch | +2-3% (no fixup overhead) |

**Expected combined:** +42-73% over llama.cpp's MMQ kernel throughput. Even taking
the conservative half: **+20-35% over llama.cpp end-to-end prefill** (non-GEMM
overhead caps the kernel speedup translation).

## Hardware targets

| GPU | SMs | sm_ | TC TFLOPS (fp16) | DRAM GB/s | L2 (MB) | Shared/SM | cp.async | TMA |
|-----|-----|-----|-------------------|-----------|---------|-----------|----------|-----|
| RTX 4090 | 128 | 8.9 | 82.6 (f32 acc) / 165.2 (f16 acc) | 1008 | 72 | 100 KB | ✅ | ❌ |
| RTX 5090 | ~170 | 12.0 | ~210 (est, fp16) | ~1792 | ~96 | 228 KB | ✅ | ✅ |

**Compute/bandwidth ratio (roofline):**
- 4090: 82.6T / 1008G = **82 ops/byte** → compute-bound at AI > 82
  (i.e., T > ~23 for Q4_K at 0.5625 bytes/elem)
- 5090: ~210T / 1792G = **117 ops/byte** → compute-bound at AI > 117
  (i.e., T > ~33)

**Implication:** at T ≥ 64 prefill is solidly **compute-bound**. The kernel's TC
utilization (roofline efficiency) is the primary driver. At T < 64 (decode, small
prefill), it's bandwidth-bound and Q8_1 activation matters most.

## Kernel architecture

### Tile layout
```
Block tile:  [BM=128, BN=128, BK=32]  (M=weight rows, N=tokens, K=inner)
Warps:       8 warps × 32 threads = 256 threads/CTA
Warp tile:   [WM=64, WN=64]  (each warp owns a 64×64 sub-tile = 4×4 mma tiles)
MMA inst:    mma.sync.aligned.m16n8k16.f32.f16.f16.f32  (Ada/Turing)
             wgmma.mma_async.sync.aligned.m64n128k16    (Blackwell, 5090)
K-loop:      process BK=32 per inner iteration, pipeline 3 stages
```

### 3-stage cp.async pipeline
```
Stage 0: cp.async load Q4_K weight [128, 32] + fp32 activation [128, 32] → buf[0]
Stage 1: cp.async load → buf[1], while: dequant buf[0] Q4_K→fp16 in shared mem,
         quantize buf[0] fp32→Q8_1→fp16 in shared mem (fused quant)
Stage 2: cp.async load → buf[2], while: dequant/quant buf[1], mma.sync on buf[0]
Steady:  compute on buf[k], dequant/quant buf[k-1], load buf[k-2]
         (all overlapped via cp.async.commit_group + wait_group)
```

Shared memory budget per CTA:
- 3 × weight fp16 [128, 32] = 3 × 8 KB = 24 KB
- 3 × activation fp16 [128, 32] = 3 × 8 KB = 24 KB (or Q8_1: 12 KB)
- Total: 48 KB (fp16 act) or 36 KB (Q8_1 act) → **3 CTAs/SM** on 4090 (100 KB)

### Adaptive activation format
```
if T >= COMPUTE_BOUND_T (64):  // compute-bound → skip quant overhead
    activation_format = FP16    // read fp32 from DRAM, f32→f16 in shared, mma(fp16,fp16)
else:                           // bandwidth-bound → halve activation traffic
    activation_format = Q8_1    // read fp32 from DRAM, quantize to Q8_1 in shared, mma(fp16,fp16)
```

The fp16 weight × fp16 activation path is the same either way — only the shared
memory staging format differs (fp16 direct vs Q8_1→fp16 dequant). The Q8_1 path
trades a per-block scale multiply for halved DRAM reads.

### Persistent kernel + atomic work queue
```
__global__ void mmq_v2_kernel(...) {
    // Each CTA persists, claims (M_tile, N_tile) from atomic counter
    __shared__ int work_idx;
    while (true) {
        if (threadIdx.x == 0) work_idx = atomicAdd(&global_work_idx, 1);
        __syncthreads();
        if (work_idx >= total_tiles) return;
        int m_tile = work_idx / n_tiles_n;
        int n_tile = work_idx % n_tiles_n;
        process_tile(m_tile, n_tile);  // full K-loop over [BM, BK] × [BK, BN]
    }
}
```

No stream-K fixup — each tile is computed atomically by one CTA. For MoE, the
work queue entries are `(expert_id, m_tile, n_tile)` so an expert's weight stays
L1/L2-resident as consecutive tiles are processed.

### MoE routing (fused gather)
For `mul_mat_id` (MoE), the activation is already gathered by expert in the Q8_1
quantization pass (same as llama.cpp's `ids`). The persistent kernel processes
entries `(expert, m_tile, n_tile)` — the expert's weight slice [BM, K] is loaded
once and reused for all its n_tiles (tokens routed to that expert). This is the
key MoE win: **each expert weight read ~once** instead of once-per-token.

## Implementation milestones

### M0 — Microbench harness ✅
Isolated GEMM benchmark: allocate Q4_K [M, K] + fp32 [K, T], run kernel, measure
TFLOPS + GB/s. Compare vs cuBLAS + dequant round-trip.

### M1 — Baseline wmma kernel ✅ (14.6 TFLOPS, 16% of cuBLAS)
64×64 tile, BK=32, 4 warps, no bank-conflict padding. Naive per-element dequant.
Bottleneck: 8% occupancy (misdiagnosed as ALU-bound; actually shared mem exceeded
48KB limit in later BK=256 attempt).

### M2 — Parity with dequant+cuBLAS ✅ (92→136 TFLOPS)
Fixes: WSTRIDE=36 (bank-conflict-free), 9.2 KB shared (4 CTAs/SM, 33% occupancy).
**Key result:** matches dequant+cuBLAS exactly at all T values. The fused kernel
does the same DRAM traffic, so parity is expected without traffic reduction.

### M3 — Warp specialization ✅ (experiment, no gain over M2)
Tried producer-consumer split (4 warps dequant + 4 warps mma, double-buffered).
**Finding:** no improvement. The GPU's warp scheduler already overlaps ALU and TC
across warps naturally — explicit warp specialization adds thread/sync overhead
without achieving additional overlap. The bottleneck is **total ALU instruction
count** for the Q4_K dequant (~6 instr/element × 2048 elements/K-iter × 128
K-iters = 1.6M instructions/CTA), not pipeline stalls.

Tried `__launch_bounds__(128, 8)` (64 regs, 67% occupancy): also no gain. The
grid provides only 4 CTAs/SM regardless, so extra register headroom is unused.

**Conclusion:** to BEAT dequant+cuBLAS, must either:
1. **Reduce DRAM traffic** — Q8_1 activation (halve activation reads, the llama.cpp approach)
2. **Reduce dequant ALU** — vectorized byte access (uint32 loads), lookup table, or process
   both nibble halves per byte read
3. **Use mma.sync PTX** instead of wmma — finer-grained register control, potentially fewer
   instructions per TC operation

### M4 — Q8_1 activation quantization (next)
Quantize fp32 activation to Q8_1 (1 byte/elem) before the GEMM. At bandwidth-bound
sizes (T<256), this halves the activation DRAM traffic. Requires a fast quantization
pre-pass + Q8_1→fp16 dequant in the GEMM inner loop. This is how llama.cpp achieves
its edge over cuBLAS+dequant."
**Gate:** must match cuBLAS+dequant throughput before adding complexity.

### M2 — cp.async 3-stage pipeline
Add the 3-stage pipeline. **Gate:** must beat M1 by ≥10%.

### M3 — Fused fp32→Q8_1 activation quant
Quantize activation in the pipeline. **Gate:** must beat M2 at T<256 by ≥5%.

### M4 — Persistent kernel + MoE fusion
Replace grid dispatch with persistent CTAs. Fuse MoE expert routing.

### M5 — Integrate into forward_cuda.zig
Wire into `gemmDispatchPrefill`, gated by env (`ZINC_MMQ_V2=1`), default-on after
validation.

## Revised priorities (post-M3 findings)

The warp specialization experiment (M3) proved that ALU/TC overlap is already
maximized by the GPU's warp scheduler. The remaining levers are:

1. **Q8_1 activation** (M4) — the proven llama.cpp approach. Halves activation
   DRAM reads. Expected: +20-30% at T≤512 (bandwidth-bound regime).
2. **Vectorized dequant** — read uint32 (4 bytes = 8 nibbles) instead of individual
   bytes. Reduces load instruction count by 4×. Expected: +10-15%.
3. **mma.sync PTX** — replace wmma with inline PTX `mma.sync.m16n8k16`. Gives
   finer register control, avoids wmma overhead. Expected: +5-10%.
4. **Larger 128×128 tile** — amortizes sync/launch overhead over 4× more work.
   Now viable with the bank-conflict-free layout. Expected: +5%.

## Correctness gate
- `validate_catalog.sh` 5/5 (fp16 TC is token-tolerance, not bit-identical — that's OK)
- Reference: compare kernel output vs CPU dequant + fp32 matmul (max abs diff < 0.1)
- GEN_IDS: bit-identical ON vs OFF for fp16 path (reduction-order tolerance for tokens)
