// mmq_v2_kernel.cu — M2: BK=32 + bank-conflict-free stride + occupancy check.
//
// Key insight from M1 profiling: the kernel is ALU-bound on the per-element
// Q4_K dequant (~10 instructions/element × 2K elements/K-iter × 128 K-iters
// = 2.6M instructions/CTA, dominating the ~65K FLOPS of wmma compute).
//
// This version adds bank-conflict padding (WSTRIDE=BK+4=36) and measures
// occupancy. M3 will address the dequant ALU bottleneck via warp specialization
// or a separate dequant warp group.
//
// Computes: Y[M, T] = W[M, K] × X[T, K]^T

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>

#define QK4_K 256
#define Q4_K_BLOCK_BYTES 176

#define BM 64
#define BN 64
#define BK 32
#define PAD 4
#define WSTRIDE (BK + PAD)  // 36 — not a multiple of 32 banks → conflict-free
#define NWARPS 4
#define WM 32
#define WN 32
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

static __device__ __forceinline__ float h2f_u16(uint16_t h) {
    __half_raw r; r.x = h; return __half2float(*(__half*)&r);
}

extern __global__ void mmq_v2_kernel_q4k(
    const unsigned char* __restrict__ W_q4k,
    const float* __restrict__ X_f32,
    float* __restrict__ Y_f32,
    int M, int K, int T)
{
    const int blocks_per_row = K / 256;
    const int k_chunks = K / BK;  // = K/32

    // Shared memory: weight fp16 [BM, WSTRIDE] + activation fp16 [WSTRIDE, BN]
    // 64×36×2 + 36×64×2 = 4608 + 4608 = 9216 bytes → high occupancy
    __shared__ __half Ws[BM * WSTRIDE];
    __shared__ __half Xs[WSTRIDE * BN];

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int warp_m = warp_id / 2;
    const int warp_n = warp_id % 2;

    const int bm0 = blockIdx.x * BM;
    const int bn0 = blockIdx.y * BN;

    using namespace nvcuda::wmma;
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[2][2];
    for (int mi = 0; mi < 2; mi++)
        for (int ni = 0; ni < 2; ni++)
            fill_fragment(acc[mi][ni], 0.0f);

    const int n_threads = NWARPS * 32;

    for (int kc = 0; kc < k_chunks; kc++) {
        const int k0 = kc * BK;
        const int superblock = k0 / 256;        // which 256-elem Q4_K superblock
        const int sub_in_block = (k0 % 256) / 32; // which 32-elem sub-block (0..7)

        // ---- Dequant Q4_K weight [BM, BK] → Ws ----
        {
            const int n_vals = BM * BK;
            for (int v = 0; v < (n_vals + n_threads - 1) / n_threads; v++) {
                int idx = tid + v * n_threads;
                if (idx >= n_vals) break;
                int row = idx / BK;
                int col = idx % BK;
                int global_row = bm0 + row;
                if (global_row >= M) { Ws[row * WSTRIDE + col] = __float2half(0.0f); continue; }

                const unsigned char* blk = W_q4k +
                    (size_t)global_row * blocks_per_row * Q4_K_BLOCK_BYTES +
                    superblock * Q4_K_BLOCK_BYTES;
                float d = h2f_u16((uint16_t)(blk[0] | (blk[1] << 8)));
                float dmin = h2f_u16((uint16_t)(blk[2] | (blk[3] << 8)));
                const unsigned char* scales = blk + 4;
                const unsigned char* qh = blk + 16;
                const unsigned char* qs = blk + 48;

                int chunk = sub_in_block / 2;
                int half_ = sub_in_block % 2;
                int j = chunk * 2 + half_;
                uint8_t sc, mn;
                if (j < 4) { sc = scales[j] & 63u; mn = (scales[j] >> 6) | ((scales[j+4] << 2) & 0xC0); }
                else { sc = ((scales[j-4] >> 4) | ((scales[j] << 2) & 0x3C)) & 63u; mn = scales[j] >> 6; }

                int l = col;
                uint8_t ql = qs[chunk * 32u + l];
                uint32_t nib = (half_ == 0u) ? (ql & 0xFu) : (uint32_t)(ql >> 4);
                uint32_t bit = (qh[l] >> (2u * chunk + half_)) & 1u;
                uint32_t q5 = nib + (bit ? 16u : 0u);
                Ws[row * WSTRIDE + col] = __float2half(d * (float)sc * (float)q5 - dmin * (float)mn);
            }
        }

        // ---- Load activation [BK, BN] → Xs (col_major) ----
        {
            const int n_vals = BK * BN;
            for (int v = 0; v < (n_vals + n_threads - 1) / n_threads; v++) {
                int idx = tid + v * n_threads;
                if (idx >= n_vals) break;
                int kk = idx % BK;
                int nn = idx / BK;
                int gt = bn0 + nn, gk = k0 + kk;
                float val = (gt < T && gk < K) ? X_f32[gt * K + gk] : 0.0f;
                Xs[kk + nn * WSTRIDE] = __float2half(val);
            }
        }

        __syncthreads();

        // ---- mma: 2 K-steps × 2×2 fragments = 8 mma calls ----
        for (int ki = 0; ki < BK / WMMA_K; ki++) {
            for (int mi = 0; mi < 2; mi++) {
                for (int ni = 0; ni < 2; ni++) {
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

    // ---- Store output ----
    for (int mi = 0; mi < 2; mi++) {
        for (int ni = 0; ni < 2; ni++) {
            int out_row = bm0 + warp_m * WM + mi * WMMA_M;
            int out_col = bn0 + warp_n * WN + ni * WMMA_N;
            if (out_row < M && out_col < T)
                store_matrix_sync(Y_f32 + out_row * T + out_col, acc[mi][ni], T, mem_row_major);
        }
    }
}
