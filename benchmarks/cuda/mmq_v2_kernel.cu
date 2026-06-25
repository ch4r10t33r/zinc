// mmq_v2_kernel.cu — M5: mma.sync PTX kernel (replaces wmma).
//
// Uses inline PTX: mma.sync.aligned.m16n8k16 + ldmatrix.x4/x2.trans
// to bypass the wmma C++ wrapper overhead (proven to cap at 67% of cuBLAS).
//
// Block tile: 64×64, 4 warps, each warp computes [32,32] = 2×4 mma tiles.
// Shared mem: same layout as M2 (9.2 KB, bank-conflict-free).
//
// Computes: Y[M, T] = W[M, K] × X[T, K]^T (fp16 weight × fp16 activation → fp32 output)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#define QK4_K 256
#define Q4_K_BLOCK_BYTES 176

#define BM 64
#define BN 64
#define BK 32
#define PAD 4
#define WSTRIDE (BK + PAD)  // 36
#define NWARPS 4
#define WM 32    // warp M tile (2 mma groups of 16)
#define WN 32    // warp N tile (4 mma groups of 8)
#define MMA_M 16
#define MMA_N 8
#define MMA_K 16

static __device__ __forceinline__ float h2f_u16(uint16_t h) {
    __half_raw r; r.x = h; return __half2float(*(__half*)&r);
}

// ---- PTX helpers ----

// ldmatrix.x4: load 4 × 8×8 half matrices from shared → 4 uint32 regs
static __device__ __forceinline__ void ldmatrix_x4(
    uint32_t (&r)[4], const void* smem_ptr)
{
    uint32_t addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(addr));
}

// ldmatrix.x2.trans: load 2 × 8×8 half matrices (transposed) → 2 uint32 regs
static __device__ __forceinline__ void ldmatrix_x2_trans(
    uint32_t (&r)[2], const void* smem_ptr)
{
    uint32_t addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(r[0]), "=r"(r[1])
        : "r"(addr));
}

// mma.sync m16n8k16: D[16,8] = A[16,16] × B[16,8] + C[16,8]
// A: 4 uint32 (row-major fp16), B: 2 uint32 (col-major fp16), C/D: 4 float
static __device__ __forceinline__ void mma_m16n8k16(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2], const float (&c)[4])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// ---- The kernel: fp16 weight × fp16 activation → fp32 output ----
// (No Q4_K dequant — diagnostic to see if mma.sync matches cuBLAS)
extern __global__ void mmq_v2_kernel_f16_only(
    const __half* __restrict__ W_f16,
    const __half* __restrict__ X_f16,
    float* __restrict__ Y_f32,
    int M, int K, int T)
{
    const int k_chunks = K / BK;
    const int n_threads = NWARPS * 32;

    __shared__ __half Ws[BM * WSTRIDE];   // 4.6 KB
    __shared__ __half Xs[WSTRIDE * BN];   // 4.6 KB

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;
    const int warp_m = warp_id / 2;  // 0..1
    const int warp_n = warp_id % 2;  // 0..1
    const int bm0 = blockIdx.x * BM;
    const int bn0 = blockIdx.y * BN;

    // Accumulators: 2 M-groups × 4 N-groups × 4 floats = 32 floats/thread
    float acc[2][4][4];
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            #pragma unroll
            for (int i = 0; i < 4; i++)
                acc[mi][ni][i] = 0.0f;

    for (int kc = 0; kc < k_chunks; kc++) {
        const int k0 = kc * BK;

        // ---- Load weight [BM, BK] → Ws (row-major, direct copy) ----
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

        // ---- Load activation [BK, BN] → Xs (col_major) ----
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

        // ---- mma.sync compute ----
        // 2 K-steps (BK=32, MMA_K=16)
        #pragma unroll
        for (int ki = 0; ki < BK / MMA_K; ki++) {
            // Load A fragments for all 2 M-groups
            uint32_t a_frag[2][4];  // [mi][4 regs]
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                // ldmatrix.x4 loads a [16,16] tile from Ws
                // Each thread provides a row address within the [16,16] tile
                int group = lane / 8;       // 0-3
                int row_in_grp = lane % 8;  // 0-7
                int tile_row = row_in_grp + (group >= 2 ? 8 : 0);
                int tile_col = (group % 2) * 8;
                int abs_row = warp_m * WM + mi * MMA_M + tile_row;
                int abs_col = ki * MMA_K + tile_col;
                const __half* addr = &Ws[abs_row * WSTRIDE + abs_col];
                ldmatrix_x4(a_frag[mi], addr);
            }

            // Load B fragments for all 4 N-groups
            uint32_t b_frag[4][2];  // [ni][2 regs]
            #pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                // ldmatrix.x2.trans loads a [16,8] tile from Xs (col_major)
                // Thread t in group g provides a column address
                int group = lane / 8;       // 0 or 1
                int col_in_grp = lane % 8;  // 0-7 (the N index)
                int k_local = group * 8;    // 0 or 8
                int abs_k = ki * MMA_K + k_local;
                int abs_n = warp_n * WN + ni * MMA_N + col_in_grp;
                const __half* addr = &Xs[abs_k + abs_n * WSTRIDE];
                ldmatrix_x2_trans(b_frag[ni], addr);
            }

            // Compute all 2×4 = 8 mma operations
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                #pragma unroll
                for (int ni = 0; ni < 4; ni++) {
                    mma_m16n8k16(acc[mi][ni], a_frag[mi], b_frag[ni], acc[mi][ni]);
                }
            }
        }

        __syncthreads();
    }

    // ---- Store output ----
    // D fragment [16,8] mapping: thread t holds 4 elements
    // D[0] = result[t/4*2, t%4*2], D[1] = result[t/4*2, t%4*2+1]
    // D[2] = result[t/4*2+8, t%4*2], D[3] = result[t/4*2+8, t%4*2+1]
    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 4; ni++) {
            int base_row = bm0 + warp_m * WM + mi * MMA_M;
            int base_col = bn0 + warp_n * WN + ni * MMA_N;
            int r = (lane / 4) * 2;
            int c = (lane % 4) * 2;
            if (base_row + r < M && base_col + c < T)
                Y_f32[(base_row + r) * T + base_col + c] = acc[mi][ni][0];
            if (base_row + r < M && base_col + c + 1 < T)
                Y_f32[(base_row + r) * T + base_col + c + 1] = acc[mi][ni][1];
            if (base_row + r + 8 < M && base_col + c < T)
                Y_f32[(base_row + r + 8) * T + base_col + c] = acc[mi][ni][2];
            if (base_row + r + 8 < M && base_col + c + 1 < T)
                Y_f32[(base_row + r + 8) * T + base_col + c + 1] = acc[mi][ni][3];
        }
    }
}

// ---- Q4_K dequant + mma.sync kernel ----
extern __global__ void mmq_v2_kernel_q4k(
    const unsigned char* __restrict__ W_q4k,
    const float* __restrict__ X_f32,
    float* __restrict__ Y_f32,
    int M, int K, int T)
{
    const int blocks_per_row = K / 256;
    const int k_chunks = K / BK;
    const int n_threads = NWARPS * 32;

    __shared__ __half Ws[BM * WSTRIDE];
    __shared__ __half Xs[WSTRIDE * BN];

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;
    const int warp_m = warp_id / 2;
    const int warp_n = warp_id % 2;
    const int bm0 = blockIdx.x * BM;
    const int bn0 = blockIdx.y * BN;

    float acc[2][4][4];
    #pragma unroll
    for (int mi = 0; mi < 2; mi++)
        #pragma unroll
        for (int ni = 0; ni < 4; ni++)
            #pragma unroll
            for (int i = 0; i < 4; i++)
                acc[mi][ni][i] = 0.0f;

    for (int kc = 0; kc < k_chunks; kc++) {
        const int k0 = kc * BK;
        const int superblock = k0 / 256;
        const int sub_in_block = (k0 % 256) / 32;

        // Dequant Q4_K [BM, BK] → Ws
        {
            const int n_vals = BM * BK;
            for (int v = 0; v < (n_vals + n_threads - 1) / n_threads; v++) {
                int idx = tid + v * n_threads;
                if (idx >= n_vals) break;
                int row = idx / BK, col = idx % BK;
                int gr = bm0 + row;
                if (gr >= M) { Ws[row * WSTRIDE + col] = __float2half(0.0f); continue; }
                const unsigned char* blk = W_q4k +
                    (size_t)gr * blocks_per_row * Q4_K_BLOCK_BYTES + superblock * Q4_K_BLOCK_BYTES;
                float d = h2f_u16((uint16_t)(blk[0] | (blk[1] << 8)));
                float dmin = h2f_u16((uint16_t)(blk[2] | (blk[3] << 8)));
                const unsigned char* scales = blk + 4, *qh = blk + 16, *qs = blk + 48;
                int chunk = sub_in_block/2, half_ = sub_in_block%2;
                int j = chunk*2+half_;
                uint8_t sc, mn;
                if (j<4) { sc=scales[j]&63u; mn=(scales[j]>>6)|((scales[j+4]<<2)&0xC0); }
                else { sc=((scales[j-4]>>4)|((scales[j]<<2)&0x3C))&63u; mn=scales[j]>>6; }
                int l = col;
                uint8_t ql = qs[chunk*32u+l];
                uint32_t nib=(half_==0u)?(ql&0xFu):(uint32_t)(ql>>4);
                uint32_t bit=(qh[l]>>(2u*chunk+half_))&1u;
                uint32_t q5=nib+(bit?16u:0u);
                Ws[row*WSTRIDE+col] = __float2half(d*(float)sc*(float)q5-dmin*(float)mn);
            }
        }

        // Load fp32 activation → Xs (col_major, fp16)
        {
            const int n_vals = BK * BN;
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

        // mma.sync compute
        #pragma unroll
        for (int ki = 0; ki < BK / MMA_K; ki++) {
            uint32_t a_frag[2][4];
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                int group = lane / 8, row_in_grp = lane % 8;
                int tile_row = row_in_grp + (group >= 2 ? 8 : 0);
                int tile_col = (group % 2) * 8;
                int abs_row = warp_m * WM + mi * MMA_M + tile_row;
                int abs_col = ki * MMA_K + tile_col;
                ldmatrix_x4(a_frag[mi], &Ws[abs_row * WSTRIDE + abs_col]);
            }

            uint32_t b_frag[4][2];
            #pragma unroll
            for (int ni = 0; ni < 4; ni++) {
                int group = lane / 8, col_in_grp = lane % 8;
                int k_local = group * 8;
                int abs_k = ki * MMA_K + k_local;
                int abs_n = warp_n * WN + ni * MMA_N + col_in_grp;
                ldmatrix_x2_trans(b_frag[ni], &Xs[abs_k + abs_n * WSTRIDE]);
            }

            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                #pragma unroll
                for (int ni = 0; ni < 4; ni++) {
                    mma_m16n8k16(acc[mi][ni], a_frag[mi], b_frag[ni], acc[mi][ni]);
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
            int base_row = bm0 + warp_m * WM + mi * MMA_M;
            int base_col = bn0 + warp_n * WN + ni * MMA_N;
            int r = (lane / 4) * 2, c = (lane % 4) * 2;
            if (base_row+r < M && base_col+c < T) Y_f32[(base_row+r)*T+base_col+c] = acc[mi][ni][0];
            if (base_row+r < M && base_col+c+1 < T) Y_f32[(base_row+r)*T+base_col+c+1] = acc[mi][ni][1];
            if (base_row+r+8 < M && base_col+c < T) Y_f32[(base_row+r+8)*T+base_col+c] = acc[mi][ni][2];
            if (base_row+r+8 < M && base_col+c+1 < T) Y_f32[(base_row+r+8)*T+base_col+c+1] = acc[mi][ni][3];
        }
    }
}
