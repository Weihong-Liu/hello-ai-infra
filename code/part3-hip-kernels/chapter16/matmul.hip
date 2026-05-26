// matmul.hip
// 教学版 GEMM：v0 Naive → v1 LDS-tiled → v2 BLOCK_K=32 → v3 Register Blocking → v4 组合版
// 硬件目标：AI MAX 395 (gfx1151), ROCm 7.12.0
// 编译：hipcc matmul.hip -O3 -o matmul

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// ============================================================
// 通用宏：HIP 错误检查
// ============================================================
#define HIP_CHECK(call)                                                         \
    do {                                                                        \
        hipError_t err = (call);                                                \
        if (err != hipSuccess) {                                                \
            fprintf(stderr, "HIP error %s:%d: %s\n",                          \
                    __FILE__, __LINE__, hipGetErrorString(err));                \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// ============================================================
// v0: Naive — 一线程计算一个输出元素，不用 LDS，不做 tiling
// ============================================================
__global__ void gemm_v0_naive(
    const float* __restrict__ A,   // [M, K] 行主序
    const float* __restrict__ B,   // [K, N] 行主序
    float*       __restrict__ C,   // [M, N] 行主序
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}

// ============================================================
// v1: LDS-tiled — BLOCK_M=BLOCK_N=BLOCK_K=16
//     每个 block 协作把 A-tile 和 B-tile 加载进 LDS，
//     再从 LDS 读做点积。
// ============================================================
#define V1_BM 16
#define V1_BN 16
#define V1_BK 16

__global__ void gemm_v1_lds(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N, int K)
{
    __shared__ float As[V1_BM][V1_BK];
    __shared__ float Bs[V1_BK][V1_BN];

    int ty  = threadIdx.y, tx  = threadIdx.x;
    int row = blockIdx.y * V1_BM + ty;
    int col = blockIdx.x * V1_BN + tx;

    float acc = 0.0f;

    for (int kb = 0; kb < (K + V1_BK - 1) / V1_BK; kb++) {
        // 协作加载 A tile：每线程负责一个元素
        int a_col = kb * V1_BK + tx;
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        // 协作加载 B tile：每线程负责一个元素
        int b_row = kb * V1_BK + ty;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();   // 确保 LDS 写完再读

        #pragma unroll
        for (int k = 0; k < V1_BK; k++) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();   // 确保本轮读完再进入下一轮写
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}

// ============================================================
// v2: BLOCK_K=32 — 增大 K 方向的 tile，减少同步次数
//     线程块仍为 16×16，每线程需额外加载 A/B 各 2 个元素
// ============================================================
#define V2_BM 16
#define V2_BN 16
#define V2_BK 32

__global__ void gemm_v2_larger_bk(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N, int K)
{
    __shared__ float As[V2_BM][V2_BK];
    __shared__ float Bs[V2_BK][V2_BN];

    int ty  = threadIdx.y, tx  = threadIdx.x;
    int row = blockIdx.y * V2_BM + ty;
    int col = blockIdx.x * V2_BN + tx;

    float acc = 0.0f;

    for (int kb = 0; kb < (K + V2_BK - 1) / V2_BK; kb++) {
        // 每线程加载 A tile 中 2 列（BK/BN = 2）
        for (int i = 0; i < V2_BK / V2_BN; i++) {
            int a_col = kb * V2_BK + tx + i * V2_BN;
            As[ty][tx + i * V2_BN] =
                (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        }

        // 每线程加载 B tile 中 2 行（BK/BM = 2）
        for (int i = 0; i < V2_BK / V2_BM; i++) {
            int b_row = kb * V2_BK + ty + i * V2_BM;
            Bs[ty + i * V2_BM][tx] =
                (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < V2_BK; k++) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}

// ============================================================
// v3: Register Blocking TM=TN=4
//     每线程负责 4×4 个输出元素，tile 64×64，BLOCK_K=16
//     线程块大小 16×16（= 64/4 × 64/4）
// ============================================================
#define V3_BM 64
#define V3_BN 64
#define V3_BK 16
#define V3_TM  4
#define V3_TN  4

__global__ void gemm_v3_reg_blocking(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N, int K)
{
    __shared__ float As[V3_BM][V3_BK];
    __shared__ float Bs[V3_BK][V3_BN];

    int thread_row = threadIdx.y * V3_TM;
    int thread_col = threadIdx.x * V3_TN;
    int block_row  = blockIdx.y  * V3_BM;
    int block_col  = blockIdx.x  * V3_BN;

    float acc[V3_TM][V3_TN] = {};

    int num_threads = blockDim.x * blockDim.y;   // 16*16 = 256
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    for (int kb = 0; kb < (K + V3_BK - 1) / V3_BK; kb++) {
        // 协作加载 A tile [64 × 16]：256 线程各负责 64*16/256 = 4 个元素
        for (int i = tid; i < V3_BM * V3_BK; i += num_threads) {
            int r = i / V3_BK, c = i % V3_BK;
            int g_row = block_row + r;
            int g_col = kb * V3_BK + c;
            As[r][c] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : 0.0f;
        }

        // 协作加载 B tile [16 × 64]：256 线程各负责 16*64/256 = 4 个元素
        for (int i = tid; i < V3_BK * V3_BN; i += num_threads) {
            int r = i / V3_BN, c = i % V3_BN;
            int g_row = kb * V3_BK + r;
            int g_col = block_col + c;
            Bs[r][c] = (g_row < K && g_col < N) ? B[g_row * N + g_col] : 0.0f;
        }

        __syncthreads();

        // Register blocking 内层：每 k 步做 TM×TN 的外积累加
        #pragma unroll
        for (int k = 0; k < V3_BK; k++) {
            float a_frag[V3_TM], b_frag[V3_TN];
            #pragma unroll
            for (int tm = 0; tm < V3_TM; tm++)
                a_frag[tm] = As[thread_row + tm][k];
            #pragma unroll
            for (int tn = 0; tn < V3_TN; tn++)
                b_frag[tn] = Bs[k][thread_col + tn];
            #pragma unroll
            for (int tm = 0; tm < V3_TM; tm++)
                #pragma unroll
                for (int tn = 0; tn < V3_TN; tn++)
                    acc[tm][tn] += a_frag[tm] * b_frag[tn];
        }

        __syncthreads();
    }

    // 写回 C
    #pragma unroll
    for (int tm = 0; tm < V3_TM; tm++) {
        #pragma unroll
        for (int tn = 0; tn < V3_TN; tn++) {
            int g_row = block_row + thread_row + tm;
            int g_col = block_col + thread_col + tn;
            if (g_row < M && g_col < N)
                C[g_row * N + g_col] = acc[tm][tn];
        }
    }
}

// ============================================================
// v4: 组合版 — 128×128 tile, BLOCK_K=32, TM=TN=8, LDS padding
//     线程块大小 16×16（= 128/8 × 128/8）
// ============================================================
#define V4_BM 128
#define V4_BN 128
#define V4_BK  32
#define V4_TM   8
#define V4_TN   8

__global__ void gemm_v4_combined(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N, int K)
{
    // +1 消除 bank conflict：LDS 有 32 个 bank，padding 1 列让行间偏移错开
    __shared__ float As[V4_BM][V4_BK + 1];
    __shared__ float Bs[V4_BK][V4_BN + 1];

    int thread_row = threadIdx.y * V4_TM;
    int thread_col = threadIdx.x * V4_TN;
    int block_row  = blockIdx.y  * V4_BM;
    int block_col  = blockIdx.x  * V4_BN;

    float acc[V4_TM][V4_TN] = {};

    int num_threads = blockDim.x * blockDim.y;   // 256
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    for (int kb = 0; kb < (K + V4_BK - 1) / V4_BK; kb++) {
        // 加载 A tile [128 × 32]：每线程 128*32/256 = 16 个元素
        for (int i = tid; i < V4_BM * V4_BK; i += num_threads) {
            int r = i / V4_BK, c = i % V4_BK;
            int g_row = block_row + r;
            int g_col = kb * V4_BK + c;
            As[r][c] = (g_row < M && g_col < K) ? A[g_row * K + g_col] : 0.0f;
        }

        // 加载 B tile [32 × 128]：每线程 32*128/256 = 16 个元素
        for (int i = tid; i < V4_BK * V4_BN; i += num_threads) {
            int r = i / V4_BN, c = i % V4_BN;
            int g_row = kb * V4_BK + r;
            int g_col = block_col + c;
            Bs[r][c] = (g_row < K && g_col < N) ? B[g_row * N + g_col] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < V4_BK; k++) {
            float a_frag[V4_TM], b_frag[V4_TN];
            #pragma unroll
            for (int tm = 0; tm < V4_TM; tm++)
                a_frag[tm] = As[thread_row + tm][k];
            #pragma unroll
            for (int tn = 0; tn < V4_TN; tn++)
                b_frag[tn] = Bs[k][thread_col + tn];
            #pragma unroll
            for (int tm = 0; tm < V4_TM; tm++)
                #pragma unroll
                for (int tn = 0; tn < V4_TN; tn++)
                    acc[tm][tn] += a_frag[tm] * b_frag[tn];
        }

        __syncthreads();
    }

    #pragma unroll
    for (int tm = 0; tm < V4_TM; tm++) {
        #pragma unroll
        for (int tn = 0; tn < V4_TN; tn++) {
            int g_row = block_row + thread_row + tm;
            int g_col = block_col + thread_col + tn;
            if (g_row < M && g_col < N)
                C[g_row * N + g_col] = acc[tm][tn];
        }
    }
}

// ============================================================
// 主函数：正确性验证（CPU reference）
// ============================================================
static void cpu_gemm(const float* A, const float* B, float* C,
                     int M, int N, int K)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0.f;
            for (int k = 0; k < K; k++)
                s += A[i * K + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}

static float max_abs_err(const float* ref, const float* got, int n)
{
    float err = 0.f;
    for (int i = 0; i < n; i++)
        err = fmaxf(err, fabsf(ref[i] - got[i]));
    return err;
}

int main(int argc, char** argv)
{
    int M = 512, N = 512, K = 512;
    if (argc >= 4) {
        M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]);
    }
    printf("验证矩阵大小：M=%d, N=%d, K=%d\n", M, N, K);

    size_t bytes_A = (size_t)M * K * sizeof(float);
    size_t bytes_B = (size_t)K * N * sizeof(float);
    size_t bytes_C = (size_t)M * N * sizeof(float);

    float *h_A = (float*)malloc(bytes_A);
    float *h_B = (float*)malloc(bytes_B);
    float *h_C_ref = (float*)malloc(bytes_C);
    float *h_C_gpu = (float*)malloc(bytes_C);

    srand(42);
    for (int i = 0; i < M * K; i++) h_A[i] = (float)rand() / RAND_MAX - 0.5f;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)rand() / RAND_MAX - 0.5f;

    // CPU reference（只对小规模做，避免太慢）
    if (M <= 512 && N <= 512 && K <= 512) {
        cpu_gemm(h_A, h_B, h_C_ref, M, N, K);
    }

    float *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, bytes_A));
    HIP_CHECK(hipMalloc(&d_B, bytes_B));
    HIP_CHECK(hipMalloc(&d_C, bytes_C));
    HIP_CHECK(hipMemcpy(d_A, h_A, bytes_A, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B, bytes_B, hipMemcpyHostToDevice));

    // 测试各版本
    const char* names[] = {"v0_naive", "v1_lds", "v2_bk32", "v3_reg4", "v4_combined"};

    auto run_kernel = [&](int ver) {
        HIP_CHECK(hipMemset(d_C, 0, bytes_C));
        switch (ver) {
        case 0: {
            dim3 block(16, 16);
            dim3 grid((N + 15) / 16, (M + 15) / 16);
            gemm_v0_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
            break;
        }
        case 1: {
            dim3 block(V1_BN, V1_BM);
            dim3 grid((N + V1_BN - 1) / V1_BN, (M + V1_BM - 1) / V1_BM);
            gemm_v1_lds<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
            break;
        }
        case 2: {
            dim3 block(V2_BN, V2_BM);
            dim3 grid((N + V2_BN - 1) / V2_BN, (M + V2_BM - 1) / V2_BM);
            gemm_v2_larger_bk<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
            break;
        }
        case 3: {
            dim3 block(V3_BN / V3_TN, V3_BM / V3_TM);   // 16×16
            dim3 grid((N + V3_BN - 1) / V3_BN, (M + V3_BM - 1) / V3_BM);
            gemm_v3_reg_blocking<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
            break;
        }
        case 4: {
            dim3 block(V4_BN / V4_TN, V4_BM / V4_TM);   // 16×16
            dim3 grid((N + V4_BN - 1) / V4_BN, (M + V4_BM - 1) / V4_BM);
            gemm_v4_combined<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
            break;
        }
        }
        HIP_CHECK(hipDeviceSynchronize());
    };

    for (int ver = 0; ver <= 4; ver++) {
        run_kernel(ver);
        HIP_CHECK(hipMemcpy(h_C_gpu, d_C, bytes_C, hipMemcpyDeviceToHost));

        float err = -1.f;
        if (M <= 512 && N <= 512 && K <= 512)
            err = max_abs_err(h_C_ref, h_C_gpu, M * N);

        printf("%-16s  max_abs_err: %s\n",
               names[ver],
               err < 0 ? "skipped (too large)" : (err < 1e-3f ? "PASS" : "FAIL"));
        if (err >= 1e-3f)
            printf("  !! max_err = %.6f\n", err);
    }

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
    free(h_A); free(h_B); free(h_C_ref); free(h_C_gpu);
    return 0;
}
