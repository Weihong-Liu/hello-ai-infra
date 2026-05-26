// matmul_bench.hip
// 单版本 benchmark 可执行文件，由 bench_matmul.py 调用
// 用法：./matmul_bench <version> <M> <N> <K> <warmup> <repeat>
// 输出格式（最后一行）：avg_ms=<float>
// 编译：hipcc matmul_bench.hip -O3 -o matmul_bench

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

#define HIP_CHECK(call)                                                         \
    do {                                                                        \
        hipError_t err = (call);                                                \
        if (err != hipSuccess) {                                                \
            fprintf(stderr, "HIP error %s:%d: %s\n",                          \
                    __FILE__, __LINE__, hipGetErrorString(err));                \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// ---- 所有 kernel 定义（复制自 matmul.hip） ----

__global__ void gemm_v0_naive(
    const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ C, int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; k++) acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}

#define V1_BM 16
#define V1_BN 16
#define V1_BK 16
__global__ void gemm_v1_lds(
    const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ C, int M, int N, int K)
{
    __shared__ float As[V1_BM][V1_BK];
    __shared__ float Bs[V1_BK][V1_BN];
    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * V1_BM + ty, col = blockIdx.x * V1_BN + tx;
    float acc = 0.0f;
    for (int kb = 0; kb < (K + V1_BK - 1) / V1_BK; kb++) {
        As[ty][tx] = (row < M && kb*V1_BK+tx < K) ? A[row*K + kb*V1_BK+tx] : 0.f;
        Bs[ty][tx] = (kb*V1_BK+ty < K && col < N) ? B[(kb*V1_BK+ty)*N + col] : 0.f;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < V1_BK; k++) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row*N+col] = acc;
}

#define V2_BM 16
#define V2_BN 16
#define V2_BK 32
__global__ void gemm_v2_larger_bk(
    const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ C, int M, int N, int K)
{
    __shared__ float As[V2_BM][V2_BK];
    __shared__ float Bs[V2_BK][V2_BN];
    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * V2_BM + ty, col = blockIdx.x * V2_BN + tx;
    float acc = 0.0f;
    for (int kb = 0; kb < (K + V2_BK - 1) / V2_BK; kb++) {
        for (int i = 0; i < V2_BK / V2_BN; i++) {
            int a_col = kb*V2_BK + tx + i*V2_BN;
            As[ty][tx + i*V2_BN] = (row < M && a_col < K) ? A[row*K + a_col] : 0.f;
        }
        for (int i = 0; i < V2_BK / V2_BM; i++) {
            int b_row = kb*V2_BK + ty + i*V2_BM;
            Bs[ty + i*V2_BM][tx] = (b_row < K && col < N) ? B[b_row*N + col] : 0.f;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < V2_BK; k++) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N) C[row*N+col] = acc;
}

#define V3_BM 64
#define V3_BN 64
#define V3_BK 16
#define V3_TM  4
#define V3_TN  4
__global__ void gemm_v3_reg_blocking(
    const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ C, int M, int N, int K)
{
    __shared__ float As[V3_BM][V3_BK];
    __shared__ float Bs[V3_BK][V3_BN];
    int thread_row = threadIdx.y * V3_TM, thread_col = threadIdx.x * V3_TN;
    int block_row = blockIdx.y * V3_BM, block_col = blockIdx.x * V3_BN;
    float acc[V3_TM][V3_TN] = {};
    int num_threads = blockDim.x * blockDim.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    for (int kb = 0; kb < (K + V3_BK - 1) / V3_BK; kb++) {
        for (int i = tid; i < V3_BM * V3_BK; i += num_threads) {
            int r = i / V3_BK, c = i % V3_BK;
            int gr = block_row + r, gc = kb * V3_BK + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.f;
        }
        for (int i = tid; i < V3_BK * V3_BN; i += num_threads) {
            int r = i / V3_BN, c = i % V3_BN;
            int gr = kb * V3_BK + r, gc = block_col + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.f;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < V3_BK; k++) {
            float a_frag[V3_TM], b_frag[V3_TN];
            #pragma unroll
            for (int tm = 0; tm < V3_TM; tm++) a_frag[tm] = As[thread_row + tm][k];
            #pragma unroll
            for (int tn = 0; tn < V3_TN; tn++) b_frag[tn] = Bs[k][thread_col + tn];
            #pragma unroll
            for (int tm = 0; tm < V3_TM; tm++)
                #pragma unroll
                for (int tn = 0; tn < V3_TN; tn++)
                    acc[tm][tn] += a_frag[tm] * b_frag[tn];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int tm = 0; tm < V3_TM; tm++)
        #pragma unroll
        for (int tn = 0; tn < V3_TN; tn++) {
            int gr = block_row + thread_row + tm, gc = block_col + thread_col + tn;
            if (gr < M && gc < N) C[gr * N + gc] = acc[tm][tn];
        }
}

#define V4_BM 128
#define V4_BN 128
#define V4_BK  32
#define V4_TM   8
#define V4_TN   8
__global__ void gemm_v4_combined(
    const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ C, int M, int N, int K)
{
    __shared__ float As[V4_BM][V4_BK + 1];
    __shared__ float Bs[V4_BK][V4_BN + 1];
    int thread_row = threadIdx.y * V4_TM, thread_col = threadIdx.x * V4_TN;
    int block_row = blockIdx.y * V4_BM, block_col = blockIdx.x * V4_BN;
    float acc[V4_TM][V4_TN] = {};
    int num_threads = blockDim.x * blockDim.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    for (int kb = 0; kb < (K + V4_BK - 1) / V4_BK; kb++) {
        for (int i = tid; i < V4_BM * V4_BK; i += num_threads) {
            int r = i / V4_BK, c = i % V4_BK;
            int gr = block_row + r, gc = kb * V4_BK + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.f;
        }
        for (int i = tid; i < V4_BK * V4_BN; i += num_threads) {
            int r = i / V4_BN, c = i % V4_BN;
            int gr = kb * V4_BK + r, gc = block_col + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.f;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < V4_BK; k++) {
            float a_frag[V4_TM], b_frag[V4_TN];
            #pragma unroll
            for (int tm = 0; tm < V4_TM; tm++) a_frag[tm] = As[thread_row + tm][k];
            #pragma unroll
            for (int tn = 0; tn < V4_TN; tn++) b_frag[tn] = Bs[k][thread_col + tn];
            #pragma unroll
            for (int tm = 0; tm < V4_TM; tm++)
                #pragma unroll
                for (int tn = 0; tn < V4_TN; tn++)
                    acc[tm][tn] += a_frag[tm] * b_frag[tn];
        }
        __syncthreads();
    }
    #pragma unroll
    for (int tm = 0; tm < V4_TM; tm++)
        #pragma unroll
        for (int tn = 0; tn < V4_TN; tn++) {
            int gr = block_row + thread_row + tm, gc = block_col + thread_col + tn;
            if (gr < M && gc < N) C[gr * N + gc] = acc[tm][tn];
        }
}

// ---- 主函数 ----
int main(int argc, char** argv)
{
    if (argc < 7) {
        fprintf(stderr, "用法：%s <version 0-4> <M> <N> <K> <warmup> <repeat>\n", argv[0]);
        return 1;
    }
    int version = atoi(argv[1]);
    int M = atoi(argv[2]), N = atoi(argv[3]), K = atoi(argv[4]);
    int warmup = atoi(argv[5]), repeat = atoi(argv[6]);

    size_t bytes_A = (size_t)M * K * sizeof(float);
    size_t bytes_B = (size_t)K * N * sizeof(float);
    size_t bytes_C = (size_t)M * N * sizeof(float);

    float *h_A = (float*)malloc(bytes_A);
    float *h_B = (float*)malloc(bytes_B);
    srand(42);
    for (int i = 0; i < M * K; i++) h_A[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)rand() / RAND_MAX - 0.5f;

    float *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, bytes_A));
    HIP_CHECK(hipMalloc(&d_B, bytes_B));
    HIP_CHECK(hipMalloc(&d_C, bytes_C));
    HIP_CHECK(hipMemcpy(d_A, h_A, bytes_A, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B, bytes_B, hipMemcpyHostToDevice));

    auto launch = [&]() {
        switch (version) {
        case 0: {
            dim3 block(16, 16);
            dim3 grid((N+15)/16, (M+15)/16);
            gemm_v0_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
            break;
        }
        case 1: {
            dim3 block(V1_BN, V1_BM);
            dim3 grid((N+V1_BN-1)/V1_BN, (M+V1_BM-1)/V1_BM);
            gemm_v1_lds<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
            break;
        }
        case 2: {
            dim3 block(V2_BN, V2_BM);
            dim3 grid((N+V2_BN-1)/V2_BN, (M+V2_BM-1)/V2_BM);
            gemm_v2_larger_bk<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
            break;
        }
        case 3: {
            dim3 block(V3_BN/V3_TN, V3_BM/V3_TM);
            dim3 grid((N+V3_BN-1)/V3_BN, (M+V3_BM-1)/V3_BM);
            gemm_v3_reg_blocking<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
            break;
        }
        case 4: {
            dim3 block(V4_BN/V4_TN, V4_BM/V4_TM);
            dim3 grid((N+V4_BN-1)/V4_BN, (M+V4_BM-1)/V4_BM);
            gemm_v4_combined<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
            break;
        }
        default:
            fprintf(stderr, "未知版本 %d\n", version);
            exit(1);
        }
        HIP_CHECK(hipDeviceSynchronize());
    };

    // Warmup
    for (int i = 0; i < warmup; i++) launch();

    // 计时
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < repeat; i++) launch();
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double avg_ms = elapsed_ms / repeat;

    printf("version=%d M=%d N=%d K=%d warmup=%d repeat=%d\n",
           version, M, N, K, warmup, repeat);
    printf("avg_ms=%.6f\n", avg_ms);

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
    free(h_A); free(h_B);
    return 0;
}
