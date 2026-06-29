# Delta-Net: llama.cpp vs ZINC Analysis

## Executive Summary (Updated 2026-06-29)

Through profiling and A/B experiments on the RTX 5090, we discovered that the
**SSM GEMMs (not the delta-net scan) are the prefill bottleneck**. Routing
prefill GEMMs through ZINC's existing batched tiled GEMM kernel (`gemm_*_tiled_v2`)
instead of the per-token DMMV loop yielded a **24% overall prefill speedup**.

### Verified Results (Qwen3.5-9B, RTX 5090, T=101)

| Config | attn | ssm | ffn | total | vs baseline |
|--------|------|-----|-----|-------|-------------|
| Baseline (per-token DMMV) | 61ms | 254ms | 250ms | 565ms | — |
| **Batched tiled GEMM** | 37ms | 171ms | 223ms | **431ms** | **24% faster** |
| Batched GEMM + chunked SSM | 47ms | 201ms | 240ms | 488ms | 14% faster |
| cuBLAS + fp16 cache | 95ms | 876ms | 633ms | 1604ms | 2.8× slower |

### Root Cause Discovery

The `ZINC_SSM_PROFILE` sub-phase profiling revealed that within each SSM layer:
- **Pre-scan GEMMs** (qkv, z, alpha, beta + conv1d): ~60-70% of SSM time
- **Delta-net scan**: ~10% of SSM time (only 1ms/layer!)
- **Post-scan** (gated norm + output GEMM): ~20% of SSM time

The per-token DMMV fallback path (used when T < `cublas_min_t=128`) reads the
quantized weight matrix **T times** (once per token). The batched tiled GEMM
reads it only **ceilDiv(T, 64)** times — a 46× reduction in weight traffic for
T=92.

### Changes Applied

1. **`gemmDispatchPrefill` in `forward_cuda.zig`**: Route T≥2 prefill GEMMs
   through `pipes.gemm[idx]` (batched tiled GEMM) instead of the per-token
   DMMV loop. T=1 (decode) still uses DMMV.

2. **`ZINC_CUBLAS_MIN_T` env var**: Allows overriding the cuBLAS threshold.
   Verified that cuBLAS + fp16 cache is SLOWER than DMMV for these GEMM sizes.

3. **`ZINC_SSM_PROFILE` env var**: Sub-phase profiling (pre-scan / scan /
   post-scan) for SSM layers.

4. **Chunked delta-net kernel** (`ssm_delta_net_chunked`): Correct (matches
   sequential output) but not needed — the scan is only 10% of SSM time.
   Gated behind `ZINC_SSM_CHUNKED=1`.

### What Didn't Work

- **State layout transposition** (Gap #2): 0% benefit — state fits in L2 cache
- **Q/K norm removal** (Gap #3): Not a real gap — both implementations normalize
- **Chunked delta-net**: Correct but adds overhead — scan isn't the bottleneck
- **cuBLAS + fp16 cache**: 3× slower due to fp16 dequant round-trip traffic

---

## Gap #1: Chunking Algorithm (THE BIG ONE)

### What llama.cpp does

llama.cpp has three delta-net paths, dispatched in `build_delta_net()`
(`src/models/delta-net-base.cpp:423-445`):

```cpp
if (n_seq_tokens == 1) {
    // Decode: fused or autoregressive
    return build_delta_net_fused/autoregressive(q, k, v, g, b, s, il);
}
// Prefill (n_tokens > 1):
if (cparams.fused_gdn_ch) {
    return build_delta_net_fused(q, k, v, g, b, s, il);  // opt-in
}
return build_delta_net_chunking(q, k, v, g, b, s, il);   // DEFAULT
```

**The default prefill path is `build_delta_net_chunking`** (lines 15-286), NOT
the sequential fused kernel.

### How chunking works

```
Input: Q, K, V, gate, beta for T tokens (e.g., T=512)
Chunk size: CS = 64 (non-KDA) or 16 (KDA)

1. Precompute:
   - gate cumsum: g_cs = cumsum(g)                         [parallel, O(T)]
   - v_b = v * beta, k_b = k * beta                        [parallel, O(T)]
   - q_scaled = q * (1/sqrt(S_k))                          [parallel, O(T)]

2. Intra-chunk (parallel within each 64-token chunk):
   - decay_mask = exp(tril(g_cs_j - g_cs_i))               [CS x CS matrix]
   - kb = k_b @ k^T * decay_mask                           [GEMM: CS x CS]
   - kq = k @ q * decay_mask                               [GEMM: CS x CS]
   - Solve: attn = (I + tril(kb, -1))^{-1} * (-tril(kb)) + I
                                                           [triangular solve]
   - v_new = v_b - k_cumdecay @ S_prev                     [GEMM]
   - output = attn_inter + v_new @ kq                      [GEMM]

3. Inter-chunk state propagation (sequential, but only T/CS = 8 steps):
   for chunk in 0..n_chunks:
     s = s * g_last_exp + kg_t @ v_new                     [GEMM + element-wise]
```

**Key insight**: The sequential dependency chain is reduced from T=512 to
T/CS=8. Each "step" is a cuBLAS GEMM (parallel within the chunk), not a
scalar recurrence.

### What ZINC does

ZINC uses a single sequential kernel (`ssm_delta_net_warp`,
`src/shaders/cuda/kernels.cu:1257`) for ALL prefill:

```
for t in 0..n_tok:                                        ← 512 sequential steps
    load Q[t], K[t], V[t]
    state *= gate[t]
    sk = dot(state, K[t])
    d = beta[t] * (V[t] - sk)
    state += K[t] * d
    o[t] = dot(state, Q[t])
```

Every token depends on the previous token's state update. This is an
irreducible O(T) sequential dependency. For T=512, that's 512 steps.

### Impact Estimate

At ~50us per sequential step (ZINC's profiled cost), 512 steps = ~25.6ms per
SSM layer. With 30 SSM layers, that's ~768ms total.

llama.cpp's chunking: 8 sequential steps per layer, each a cuBLAS GEMM
(~0.1ms), total ~0.8ms per layer, ~24ms total.

**Speedup: ~32x on SSM phase.**

### Implementation Plan

The chunking algorithm is implemented as a **graph of standard operations**
(GEMMs, element-wise, triangular solve) — NOT a custom kernel. In ZINC's
architecture, this maps to:

1. **cuBLAS GEMMs**: Already available (`gemmDispatchPrefill`)
2. **Element-wise ops**: Need `exp`, `cumsum`, `tri`, `mul`, `sub`, `add`
   kernels — some exist, some need porting
3. **Triangular solve**: Need a new operation. Options:
   - cuBLAS `cublasStrtri` (batched matrix inverse) + matmul
   - Custom CUDA kernel for small (64x64) triangular solve
   - `cublasSgetrfBatched` + `cublasSgetrsBatched` (LU + solve)

**Effort**: 2-3 days for a working implementation, 1-2 days for tuning.

**Risk**: The triangular solve is the most complex piece. For CS=64, the
matrix is 64x64 — small enough that a custom batched kernel may be faster
than cuBLAS overhead.

---

## Gap #2: State Layout (7x within sequential kernel)

### The Problem

Both ZINC and llama.cpp's fused kernel use the same warp-level algorithm:
each warp owns one COLUMN of the state matrix, and each lane owns ROWS of
that column. The state is S[row][col], a 128x128 matrix per head.

**ZINC** stores state in row-major (`kernels.cu:1278`):
```c
s_shard[r] = state[((size_t)h * hv + row) * hv + col];
//                               ^^^ stride = hv = 128 between consecutive rows
```

For a fixed `col`, consecutive `row` values (lane=0..31) have stride 128:
```
Lane 0 reads state[col + 0*128]     → cache line A
Lane 1 reads state[col + 1*128]     → cache line B
Lane 2 reads state[col + 2*128]     → cache line C
...
Lane 31 reads state[col + 31*128]   → cache line AF
```
**32 separate cache lines per warp per state read.**

**llama.cpp** stores state transposed / column-major (`gated_delta_net.cu:42,49`):
```c
// state is stored transposed: M[col][i] = S[i][col], row col is contiguous
curr_state += col * S_v;
s_shard[r] = curr_state[r * warp_size + lane];
// = state[col * S_v + r * 32 + lane]
```

For a fixed `col`, consecutive `row` values are contiguous:
```
Lane 0 reads state[col*128 + 0]     → cache line A (bytes 0-127)
Lane 1 reads state[col*128 + 1]     → cache line A (bytes 0-127)
...
Lane 31 reads state[col*128 + 31]   → cache line A (bytes 0-127)
```
**1 cache line per warp per state read. 32x fewer cache line fetches.**

### Impact

Each SSM iteration does 4 state reads (decay, sk dot, update, o dot).
At ~300 cycles per L2 cache line fetch:
- ZINC: 4 * 32 = 128 cache line fetches → ~38,400 cycles per iteration
- llama.cpp: 4 * 1 = 4 cache line fetches → ~1,200 cycles per iteration
- **Ratio: ~7x** (matches observed per-iteration gap)

### Fix

Swap row/col in state indexing in ALL three CUDA kernels:

| Kernel | Current | Fixed |
|--------|---------|-------|
| `ssm_delta_net` (block) | `state[(h*hv + row)*hv + col]` | `state[(h*hv + col)*hv + row]` |
| `ssm_delta_net_warp` | `state[(h*hv + row)*hv + col]` | `state[(h*hv + col)*hv + row]` |
| `ssm_delta_net_seq` (decode) | `state[(h*hv + row)*hv + col]` | `state[(h*hv + col)*hv + row]` |

State zeroing is layout-independent (bulk memset). The Metal backend has its
own separate state buffers — unchanged.

**Effort**: ~1 hour (swap indices in 6 places + verify correctness).

**Important**: All three kernels must change simultaneously. If prefill
writes col-major but decode reads row-major, the state is garbage.

---

## Gap #3: Q/K Normalization (Potentially a Bug)

### The Discrepancy

**ZINC** (`kernels.cu:1320-1354`): Inline L2 normalization of Q and K inside
the delta-net kernel:
```c
// Q/K L2 norm (2 warp_reduce_sum_all + 2 rsqrt per iteration)
sumq = zinc_warp_reduce_sum_all(sumq);
sumk = zinc_warp_reduce_sum_all(sumk);
float q_rinv = rsqrtf(fmaxf(sumq, 1e-12f)) * inv_sqrt_d_state;
float k_rinv = rsqrtf(fmaxf(sumk, 1e-12f));
for (r ...) {
    q_reg[r] *= q_rinv;
    k_reg[r] *= k_rinv;
}
```

**llama.cpp** (`delta-net-base.cpp:44-46`, `gated_delta_net.cu:79-87`):
No Q/K normalization. Just scales Q upstream:
```c
const float scale = 1.0f / sqrtf(S_k);  // S_k = head_dim = 128
q = ggml_scale(ctx0, q, scale);         // applied BEFORE the kernel
```

Inside the kernel, raw Q and K are used:
```c
float k_reg[rows_per_lane];
float q_reg[rows_per_lane];
for (r ...) {
    k_reg[r] = k_t[i];   // raw K, no normalization
    q_reg[r] = q_t[i];   // raw Q, already scaled by 1/sqrt(S_k)
}
```

### Impact

ZINC's Q/K normalization adds per iteration:
- 2 `zinc_warp_reduce_sum_all` (each = 5 `__shfl_sync` + 1 broadcast)
- 2 `rsqrtf`
- 2 * 4 = 8 multiplies (applying rinv to 4 rows each)

This is ~40% of the per-iteration FLOP count, based on profiling.

### Is ZINC's Normalization Correct?

**Unknown.** Two possibilities:

1. **Bug**: ZINC incorrectly adds L2 norm that the model doesn't require.
   The model would still "work" because the normalization is approximately
   scale-invariant for the attention mechanism, but outputs would differ
   from the reference implementation.

2. **Architecture-specific**: ZINC's model variant (Qwen3.5-A3B) may require
   Q/K normalization that llama.cpp's delta-net-base.cpp handles differently
   (e.g., in a separate graph node before the delta-net op).

**Action**: Run a token-level comparison between ZINC and llama.cpp on the
same model with the same prompt. If outputs match, the normalization is
harmless (but wasteful). If they differ, investigate which is correct.

### Quick Win

If the normalization is unnecessary, removing it saves ~40% per-iteration
cost on the sequential kernel. This is a one-line change (comment out the
norm block + add `scale = 1/sqrt(d_state)` to the output).

---

## Per-Iteration Cost Breakdown (Sequential Kernel)

For Qwen3.5-A3B (head_v_dim=128, rows_per_lane=4, T tokens):

| Operation | ZINC (cycles) | llama.cpp (cycles) | Ratio |
|-----------|--------------|-------------------|-------|
| State read (4 reads) | 4 * 32 * 300 = 38,400 | 4 * 1 * 300 = 1,200 | 32x |
| Q/K L2 norm (2 reduces + 2 rsqrt) | ~800 | 0 | inf |
| Gate compute (expf/logf) | 0 (precomputed in smem) | ~100 (expf inline) | - |
| State decay + sk dot (1 reduce) | ~900 | ~600 | 1.5x |
| State update + o dot (1 reduce) | ~900 | ~600 | 1.5x |
| **Total per iteration** | **~41,000** | **~2,500** | **~16x** |

Observed gap: ~16x (ZINC ~50us/iter vs llama.cpp ~3us/iter). Matches.

After fixing gaps #2 + #3 (but still sequential):
- ZINC improved: ~2,500 cycles/iter → ~3us/iter → **parity with llama.cpp fused**

After fixing gap #1 (chunking):
- Sequential steps reduced from T to T/64
- Each step is a GEMM (fully parallel)
- **10-50x on SSM phase** (depends on T)

---

## Prioritized Implementation Plan

### Phase 1: Quick Wins (1 day)

1. **Transpose state layout** (Gap #2)
   - Swap `(h*hv+row)*hv+col` → `(h*hv+col)*hv+row` in all 3 CUDA kernels
   - Verify correctness: `validate_catalog.sh` must pass 5/5
   - Expected: ~7x speedup on sequential kernel

2. **Investigate Q/K norm** (Gap #3)
   - Generate 10 tokens from both ZINC and llama.cpp on same prompt
   - If outputs match → remove Q/K norm for ~40% speedup
   - If outputs differ → investigate which is correct

### Phase 2: Chunking (3-5 days)

3. **Implement chunked delta-net** (Gap #1)
   - Port the algorithm from `delta-net-base.cpp:15-286`
   - Implement as a sequence of existing operations:
     - cuBLAS GEMMs (available)
     - Element-wise kernels (some need porting: cumsum, tri, solve_tri)
     - Triangular solve (new — custom kernel or cuBLAS batched)
   - Gate behind `ZINC_SSM_CHUNKED=1` env flag
   - Verify: `validate_catalog.sh` + token-level comparison with llama.cpp

4. **Decode path**: Keep sequential kernel (T=1, no benefit from chunking)
   - But apply Gap #2 (transpose) and Gap #3 (remove norm) fixes

### Phase 3: Metal (optional, 1-2 days)

5. **Port fixes to Metal backend**
   - Transpose state layout in `ssm_delta_net_prefill_warp.metal`
   - Remove Q/K norm if confirmed unnecessary
   - Chunking: defer (Metal prefill is not the bottleneck for local use)

---

## Code References

### llama.cpp

| File | Lines | What |
|------|-------|------|
| `src/models/delta-net-base.cpp` | 15-286 | **Chunking algorithm** (default prefill) |
| `src/models/delta-net-base.cpp` | 288-370 | Autoregressive (decode, decomposed ops) |
| `src/models/delta-net-base.cpp` | 372-410 | Fused (opt-in, uses CUDA kernel) |
| `src/models/delta-net-base.cpp` | 423-445 | Dispatch logic |
| `ggml/src/ggml-cuda/gated_delta_net.cu` | 1-273 | Fused CUDA kernel (transposed state, no Q/K norm) |

### ZINC

| File | Lines | What |
|------|-------|------|
| `src/shaders/cuda/kernels.cu` | 1162-1235 | `ssm_delta_net` (block kernel, row-major state, Q/K norm) |
| `src/shaders/cuda/kernels.cu` | 1237-1395 | `ssm_delta_net_warp` (warp kernel, same issues) |
| `src/shaders/cuda/kernels.cu` | 1465+ | `ssm_delta_net_seq` (decode, same state layout) |
| `src/compute/forward_cuda.zig` | 1600+ | `ssmLayerBatched` (dispatches warp or block kernel) |

### Key Differences Summary

```
                    ZINC                    llama.cpp (chunking)     llama.cpp (fused)
Algorithm:          Sequential O(T)         Chunked O(T/64)          Sequential O(T)
State layout:       Row-major (strided)     N/A (GEMMs)              Col-major (coalesced)
Q/K norm:           Inline L2              None (scale on Q)         None (scale on Q)
Per-iter reduces:   4 (sumq, sumk, sk, o)  N/A                       2 (kv, attn)
Gate compute:       Precomputed (smem)      Precomputed (graph)      Inline (expf)
Iterations (T=512): 512                    8                         512
```
