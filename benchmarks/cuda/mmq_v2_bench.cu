// mmq_v2_bench.cu — Isolated microbench: cuBLAS vs dequant+cuBLAS vs mmq_v2.
// Compile: nvcc -O3 -arch=sm_89 -std=c++17 -o mmq_v2_bench mmq_v2_bench.cu mmq_v2_kernel.cu -lcublas
// Usage:   ./mmq_v2_bench [M] [K] [T] [iters]
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#define QK4_K 256
#define Q4_K_BLOCK_BYTES 176

struct DequantQ4KPush { uint32_t M, K, a_offset; };

// Dequant kernel (standalone copy from ZINC's kernels.cu)
static __device__ __forceinline__ float h2f_u16(uint16_t h) {
    __half_raw r; r.x = h; return __half2float(*(__half*)&r);
}
extern "C" __global__ void dequant_q4k_to_f16_bench(
    const unsigned char* a, __half* Wf16, DequantQ4KPush pc)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)pc.M * pc.K;
    if (i >= total) return;
    uint32_t row = (uint32_t)(i / pc.K), k = (uint32_t)(i % pc.K);
    uint32_t bpr = pc.K >> 8, within = k & 255u, b = k >> 8;
    const unsigned char* blk = a + pc.a_offset + (size_t)row * bpr * 176u + (size_t)b * 176u;
    float d = h2f_u16((uint16_t)(blk[0] | (blk[1] << 8)));
    float dmin = h2f_u16((uint16_t)(blk[2] | (blk[3] << 8)));
    const unsigned char* scales = blk + 4, *qh = blk + 16, *qs = blk + 48;
    uint32_t chunk = within >> 6, half_ = (within & 63u) >> 5, l = within & 31u;
    uint8_t ql = qs[chunk * 32u + l];
    uint32_t nib = (half_ == 0u) ? (ql & 0xFu) : (uint32_t)(ql >> 4);
    uint32_t bit = (qh[l] >> (2u * chunk + half_)) & 1u;
    uint32_t q5 = nib + (bit ? 16u : 0u);
    int j = chunk * 2 + half_;
    uint8_t sc, mn;
    if (j < 4) { sc = scales[j] & 63u; mn = (scales[j] >> 6) | ((scales[j+4] << 2) & 0xC0); }
    else { sc = ((scales[j-4] >> 4) | ((scales[j] << 2) & 0x3C)) & 63u; mn = scales[j] >> 6; }
    Wf16[i] = __float2half(d * (float)sc * (float)q5 - dmin * (float)mn);
}

// f32→f16 convert kernel
__global__ void f32_to_f16_kernel(const float* in, __half* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

extern __global__ void mmq_v2_kernel_f16_only(const __half*, const __half*, float*, int, int, int);

// MMQ v2 kernel (in mmq_v2_kernel.cu)
extern __global__ void mmq_v2_kernel_q4k(
    const unsigned char* W_q4k, const float* X_f32, float* Y_f32, int M, int K, int T);

static cublasHandle_t g_cublas;

// Global buffers (set in main, used by bench functions)
static unsigned char* g_W_q4k;
static __half* g_W_f16;
static float* g_X_f32;
static __half* g_X_f16;
static float* g_Y;

static void bench(const char* label, int M, int K, int T, int iters,
                  void (*launch)(int, int, int, cudaStream_t))
{
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    cudaStream_t stream; cudaStreamCreate(&stream);
    for (int i = 0; i < 3; i++) launch(M, K, T, stream);
    cudaStreamSynchronize(stream);
    cudaEventRecord(s, stream);
    for (int i = 0; i < iters; i++) launch(M, K, T, stream);
    cudaEventRecord(e, stream);
    cudaStreamSynchronize(stream);
    float ms; cudaEventElapsedTime(&ms, s, e);
    double t = ms / 1000.0 / iters;
    double flops = 2.0 * M * K * T;
    double wbytes = (double)M * (K / 256) * 176;
    double xbytes = (double)K * T * 4;
    double ybytes = (double)M * T * 4;
    printf("  %-35s %8.2f ms  %6.1f TFLOPS  %7.1f GB/s\n",
           label, ms / iters, flops / t / 1e12, (wbytes + xbytes + ybytes) / t / 1e9);
    cudaEventDestroy(s); cudaEventDestroy(e); cudaStreamDestroy(stream);
}

int main(int argc, char** argv) {
    int M = argc > 1 ? atoi(argv[1]) : 4096;
    int K = argc > 2 ? atoi(argv[2]) : 4096;
    int T = argc > 3 ? atoi(argv[3]) : 512;
    int iters = argc > 4 ? atoi(argv[4]) : 20;

    // Align K to 256 for Q4_K
    K = (K + 255) & ~255;

    printf("=== MMQ v2 microbench: M=%d K=%d T=%d iters=%d ===\n\n", M, K, T, iters);

    size_t w_q4k_bytes = (size_t)M * (K / 256) * 176;
    cudaMalloc(&g_W_q4k, w_q4k_bytes);
    cudaMalloc(&g_W_f16, (size_t)M * K * 2);
    cudaMalloc(&g_X_f32, (size_t)K * T * 4);
    cudaMalloc(&g_X_f16, (size_t)K * T * 2);
    cudaMalloc(&g_Y,     (size_t)M * T * 4);

    // Init
    srand(42);
    {   unsigned char* h = (unsigned char*)malloc(w_q4k_bytes);
        for (size_t i = 0; i < w_q4k_bytes; i++) h[i] = rand() & 0xFF;
        for (size_t b = 0; b < w_q4k_bytes; b += 176) { h[b] = 0; h[b+1] = 0x3C; h[b+2] = 0; h[b+3] = 0; }
        cudaMemcpy(g_W_q4k, h, w_q4k_bytes, cudaMemcpyHostToDevice); free(h);
    }
    {   float* h = (float*)malloc((size_t)K * T * 4);
        for (size_t i = 0; i < (size_t)K * T; i++) h[i] = (float)(rand() % 100) / 100.0f - 0.5f;
        cudaMemcpy(g_X_f32, h, (size_t)K * T * 4, cudaMemcpyHostToDevice); free(h);
    }
    // Pre-dequant W to fp16 (for cuBLAS baselines)
    {   DequantQ4KPush pc{(uint32_t)M, (uint32_t)K, 0};
        dequant_q4k_to_f16_bench<<<(M*K+255)/256, 256>>>(g_W_q4k, g_W_f16, pc); }
    // Pre-convert X to fp16 (for cuBLAS baselines)
    {   f32_to_f16_kernel<<<(K*T+255)/256, 256>>>(g_X_f32, g_X_f16, K * T); }
    cudaDeviceSynchronize();

    cublasCreate(&g_cublas);

    // 1. cuBLAS fp16×fp16 (upper bound — pre-dequanted weight)
    bench("1. cuBLAS fp16 (pre-dequant)", M, K, T, iters,
          [](int M, int K, int T, cudaStream_t s) {
              cublasSetStream(g_cublas, s);
              float alpha = 1.0f, beta = 0.0f;
              cublasGemmEx(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, M, T, K,
                           &alpha, g_W_f16, CUDA_R_16F, K, g_X_f16, CUDA_R_16F, K,
                           &beta, g_Y, CUDA_R_32F, M,
                           CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
          });

    // 2. dequant + cuBLAS (ZINC's current path)
    bench("2. dequant_q4k + cuBLAS (current)", M, K, T, iters,
          [](int M, int K, int T, cudaStream_t s) {
              DequantQ4KPush pc{(uint32_t)M, (uint32_t)K, 0};
              dequant_q4k_to_f16_bench<<<(M*K+255)/256, 256, 0, s>>>(g_W_q4k, g_W_f16, pc);
              cublasSetStream(g_cublas, s);
              float alpha = 1.0f, beta = 0.0f;
              cublasGemmEx(g_cublas, CUBLAS_OP_T, CUBLAS_OP_N, M, T, K,
                           &alpha, g_W_f16, CUDA_R_16F, K, g_X_f16, CUDA_R_16F, K,
                           &beta, g_Y, CUDA_R_32F, M,
                           CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
          });

    // 3. MMQ v2 (fused Q4_K dequant + wmma TC GEMM)
    bench("3. mmq_v2 fp32 act (fused Q4_K)", M, K, T, iters,
          [](int M, int K, int T, cudaStream_t s) {
              dim3 grid((M + 127) / 128, (T + 127) / 128);
              mmq_v2_kernel_q4k<<<grid, 256, 0, s>>>(g_W_q4k, g_X_f32, g_Y, M, K, T);
          });

    // 4. DIAGNOSTIC: fp16-only wmma GEMM (no dequant)
    bench("4. wmma fp16-only (no dequant)", M, K, T, iters,
          [](int M, int K, int T, cudaStream_t s) {
              dim3 grid((M + 127) / 128, (T + 127) / 128);
              mmq_v2_kernel_f16_only<<<grid, 256, 0, s>>>(g_W_f16, g_X_f16, g_Y, M, K, T);
          });

    // Roofline
    double ai = 2.0 * M * K * T / ((double)M * (K/256) * 176 + (double)K * T * 4 + (double)M * T * 4);
    printf("\n  Roofline AI=%.1f ops/byte (compute-bound at AI>82 for 4090)\n", ai);

    cublasDestroy(g_cublas);
    cudaFree(g_W_q4k); cudaFree(g_W_f16); cudaFree(g_X_f32); cudaFree(g_X_f16); cudaFree(g_Y);
    return 0;
}
