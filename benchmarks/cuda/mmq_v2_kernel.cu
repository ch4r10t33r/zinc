// mmq_v2_kernel.cu — M4: 128×128 tile with 8 warps + Q8_1 activation path + f16 diagnostic.
//
// Key insight from f16-only diagnostic: the wmma TILE CONFIG is the bottleneck,
// not the dequant. BK=32 with 64×64 tile gives only 4 mma/sync/warp → sync overhead
// dominates. Fix: 128×128 tile with 8 warps (WM=32, WN=64) → 8 mma/sync/warp,
// same 33% occupancy. This matches llama.cpp's mmq_x=128, mmq_y=128.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>

#define QK4_K 256
#define Q4_K_BLOCK_BYTES 176
#define QK8_1 32
#define Q8_1_BLOCK_BYTES 34

#define BM 128   // 2× larger
#define BN 128   // 2× larger
#define BK 32
#define PAD 4
#define WSTRIDE (BK + PAD)
#define NWARPS 8
#define WM 32    // warp M tile
#define WN 64    // warp N tile (4×2 wmma = 8 frags/warp)
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

static __device__ __forceinline__ float h2f_u16(uint16_t h) {
    __half_raw r; r.x = h; return __half2float(*(__half*)&r);
}

struct block_q8_1 { __half d, s; int8_t qs[QK8_1]; };

// ---- Q8_1 quantization ----
extern "C" __global__ void quantize_q8_1_kernel(
    const float* __restrict__ x, block_q8_1* __restrict__ qx, int K, int n_rows)
{
    const int n_blocks_total = n_rows * (K / QK8_1);
    const int tid_global = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid_global >= n_blocks_total) return;
    const int row = tid_global / (K / QK8_1);
    const int blk_idx = tid_global % (K / QK8_1);
    const int k0 = blk_idx * QK8_1;
    const float* xp = x + (size_t)row * K + k0;
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < QK8_1; i++) { float ax = fabsf(xp[i]); if (ax > amax) amax = ax; }
    float d = amax / 127.0f;
    float id = (d > 0.0f) ? 1.0f / d : 0.0f;
    block_q8_1 bq;
    bq.d = __float2half(d);
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < QK8_1; i++) {
        int qi = (int)roundf(xp[i] * id);
        qi = qi > 127 ? 127 : (qi < -127 ? -127 : qi);
        bq.qs[i] = (int8_t)qi;
        sum += qi;
    }
    bq.s = __float2half(sum);
    qx[(size_t)row * (K / QK8_1) + blk_idx] = bq;
}

// ---- Q4_K dequant helper ----
static __device__ __forceinline__ void dequant_q4k_subblock(
    const unsigned char* blk, int sub_in_block, int col,
    float& val)
{
    float d = h2f_u16((uint16_t)(blk[0] | (blk[1] << 8)));
    float dmin = h2f_u16((uint16_t)(blk[2] | (blk[3] << 8)));
    const unsigned char* scales = blk + 4;
    const unsigned char* qh = blk + 16;
    const unsigned char* qs = blk + 48;
    int chunk = sub_in_block / 2, half_ = sub_in_block % 2;
    int j = chunk * 2 + half_;
    uint8_t sc, mn;
    if (j < 4) { sc = scales[j] & 63u; mn = (scales[j] >> 6) | ((scales[j + 4] << 2) & 0xC0); }
    else { sc = ((scales[j - 4] >> 4) | ((scales[j] << 2) & 0x3C)) & 63u; mn = scales[j] >> 6; }
    int l = col;
    uint8_t ql = qs[chunk * 32u + l];
    uint32_t nib = (half_ == 0u) ? (ql & 0xFu) : (uint32_t)(ql >> 4);
    uint32_t bit = (qh[l] >> (2u * chunk + half_)) & 1u;
    uint32_t q5 = nib + (bit ? 16u : 0u);
    val = d * (float)sc * (float)q5 - dmin * (float)mn;
}

// ============================================================================
// 128×128 tile kernel — fp32 activation
// ============================================================================
extern __global__ void mmq_v2_kernel_q4k(
    const unsigned char* __restrict__ W_q4k,
    const float* __restrict__ X_f32,
    float* __restrict__ Y_f32,
    int M, int K, int T)
{
    const int blocks_per_row = K / 256;
    const int k_chunks = K / BK;
    const int n_threads = NWARPS * 32;

    // Shared: 128×36×2 + 36×128×2 = 9216 + 9216 = 18432 bytes → 2 CTAs/SM
    __shared__ __half Ws[BM * WSTRIDE];
    __shared__ __half Xs[WSTRIDE * BN];

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    // 4 warps in M (128/32=4), 2 in N (128/64=2)
    const int warp_m = warp_id % 4;   // 0..3
    const int warp_n = warp_id / 4;   // 0..1
    const int bm0 = blockIdx.x * BM;
    const int bn0 = blockIdx.y * BN;

    using namespace nvcuda::wmma;
    // 2×4 = 8 accumulator fragments per warp (WM=32 → 2 m-groups, WN=64 → 4 n-groups)
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[2][4];
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            fill_fragment(acc[mi][ni], 0.0f);

    for (int kc = 0; kc < k_chunks; kc++) {
        const int k0 = kc * BK;
        const int superblock = k0 / 256;
        const int sub_in_block = (k0 % 256) / 32;

        // Dequant Q4_K [BM, BK] → Ws
        {
            const int n_vals = BM * BK;
            #pragma unroll 1
            for (int v = 0; v < (n_vals + n_threads - 1) / n_threads; v++) {
                int idx = tid + v * n_threads;
                if (idx >= n_vals) break;
                int row = idx / BK, col = idx % BK;
                int gr = bm0 + row;
                if (gr >= M) { Ws[row * WSTRIDE + col] = __float2half(0.0f); continue; }
                const unsigned char* blk = W_q4k + (size_t)gr * blocks_per_row * Q4_K_BLOCK_BYTES + superblock * Q4_K_BLOCK_BYTES;
                float val;
                dequant_q4k_subblock(blk, sub_in_block, col, val);
                Ws[row * WSTRIDE + col] = __float2half(val);
            }
        }

        // Load fp32 activation [BK, BN] → Xs col_major
        {
            const int n_vals = BK * BN;
            #pragma unroll 1
            for (int v = 0; v < (n_vals + n_threads - 1) / n_threads; v++) {
                int idx = tid + v * n_threads;
                if (idx >= n_vals) break;
                int kk = idx % BK, nn = idx / BK;
                int gt = bn0 + nn, gk = k0 + kk;
                float val = (gt < T && gk < K) ? X_f32[gt * K + gk] : 0.0f;
                Xs[kk + nn * WSTRIDE] = __float2half(val);
            }
        }

        __syncthreads();

        // mma: 2 K-steps × 8 frags = 16 mma per warp
        #pragma unroll
        for (int ki = 0; ki < BK / WMMA_K; ki++) {
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                #pragma unroll
                for (int ni = 0; ni < 4; ni++) {
                    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
                    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, col_major> b_frag;
                    load_matrix_sync(a_frag, Ws + (warp_m*WM + mi*WMMA_M) * WSTRIDE + ki*WMMA_K, WSTRIDE);
                    load_matrix_sync(b_frag, Xs + ki*WMMA_K + (warp_n*WN + ni*WMMA_N) * WSTRIDE, WSTRIDE);
                    mma_sync(acc[mi][ni], a_frag, b_frag, acc[mi][ni]);
                }
            }
        }

        __syncthreads();
    }

    // Store output
    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            int out_row = bm0 + warp_m*WM + mi*WMMA_M;
            int out_col = bn0 + warp_n*WN + ni*WMMA_N;
            if (out_row < M && out_col < T)
                store_matrix_sync(Y_f32 + out_row*T + out_col, acc[mi][ni], T, mem_row_major);
        }
    }
}

// ============================================================================
// fp16-only diagnostic (no dequant)
// ============================================================================
extern __global__ void mmq_v2_kernel_f16_only(
    const __half* __restrict__ W_f16,
    const __half* __restrict__ X_f16,
    float* __restrict__ Y_f32,
    int M, int K, int T)
{
    const int k_chunks = K / BK;
    const int n_threads = NWARPS * 32;

    __shared__ __half Ws[BM * WSTRIDE];
    __shared__ __half Xs[WSTRIDE * BN];

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int warp_m = warp_id % 4;
    const int warp_n = warp_id / 4;
    const int bm0 = blockIdx.x * BM;
    const int bn0 = blockIdx.y * BN;

    using namespace nvcuda::wmma;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[2][4];
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            fill_fragment(acc[mi][ni], 0.0f);

    for (int kc = 0; kc < k_chunks; kc++) {
        const int k0 = kc * BK;
        // Load fp16 weight
        {
            const int n_vals = BM * BK;
            for (int v = 0; v < (n_vals + n_threads - 1) / n_threads; v++) {
                int idx = tid + v * n_threads;
                if (idx >= n_vals) break;
                int row = idx / BK, col = idx % BK;
                int gr = bm0 + row;
                Ws[row * WSTRIDE + col] = (gr < M) ? W_f16[gr * K + k0 + col] : __float2half(0.0f);
            }
        }
        // Load fp16 activation
        {
            const int n_vals = BK * BN;
            for (int v = 0; v < (n_vals + n_threads - 1) / n_threads; v++) {
                int idx = tid + v * n_threads;
                if (idx >= n_vals) break;
                int kk = idx % BK, nn = idx / BK;
                int gt = bn0 + nn;
                Xs[kk + nn * WSTRIDE] = (gt < T) ? X_f16[gt * K + k0 + kk] : __float2half(0.0f);
            }
        }
        __syncthreads();
        #pragma unroll
        for (int ki = 0; ki < BK / WMMA_K; ki++) {
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                #pragma unroll
                for (int ni = 0; ni < 4; ni++) {
                    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> a_frag;
                    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, col_major> b_frag;
                    load_matrix_sync(a_frag, Ws + (warp_m*WM + mi*WMMA_M) * WSTRIDE + ki*WMMA_K, WSTRIDE);
                    load_matrix_sync(b_frag, Xs + ki*WMMA_K + (warp_n*WN + ni*WMMA_N) * WSTRIDE, WSTRIDE);
                    mma_sync(acc[mi][ni], a_frag, b_frag, acc[mi][ni]);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            int out_row = bm0 + warp_m*WM + mi*WMMA_M;
            int out_col = bn0 + warp_n*WN + ni*WMMA_N;
            if (out_row < M && out_col < T)
                store_matrix_sync(Y_f32 + out_row*T + out_col, acc[mi][ni], T, mem_row_major);
        }
    }
}
