// rocblas_baseline.cpp
// 调用 rocBLAS sgemm 作为性能上限对照
// 硬件目标：AI MAX 395 (gfx1151), ROCm 7.12.0
// 编译（在已激活 activate-rocm.sh 后）：
//   hipcc rocblas_baseline.cpp -lrocblas -O3 -o rocblas_baseline

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>

#define HIP_CHECK(call)                                                     \
    do {                                                                    \
        hipError_t err = (call);                                            \
        if (err != hipSuccess) {                                            \
            fprintf(stderr, "HIP error %s:%d: %s\n",                      \
                    __FILE__, __LINE__, hipGetErrorString(err));            \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

#define RB_CHECK(call)                                                      \
    do {                                                                    \
        rocblas_status s = (call);                                          \
        if (s != rocblas_status_success) {                                  \
            fprintf(stderr, "rocBLAS error %s:%d: status=%d\n",           \
                    __FILE__, __LINE__, (int)s);                            \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

// TFLOPS = 2*M*N*K / (time_ms * 1e-3) / 1e12
static double tflops(int M, int N, int K, double time_ms)
{
    return 2.0 * M * N * K / (time_ms * 1e-3) / 1e12;
}

int main(int argc, char** argv)
{
    // 默认形状扫描
    int shapes[][3] = {{512, 512, 512}, {1024, 1024, 1024},
                       {2048, 2048, 2048}, {4096, 4096, 4096}};
    int nshapes = 4;

    if (argc == 4) {
        // 单个形状模式：rocblas_baseline M N K
        shapes[0][0] = atoi(argv[1]);
        shapes[0][1] = atoi(argv[2]);
        shapes[0][2] = atoi(argv[3]);
        nshapes = 1;
    }

    int WARMUP = 5, REPEAT = 20;

    rocblas_handle handle;
    RB_CHECK(rocblas_create_handle(&handle));

    printf("%-30s  %8s  %10s\n", "shape", "time_ms", "TFLOPS");
    printf("%s\n", std::string(54, '-').c_str());

    for (int s = 0; s < nshapes; s++) {
        int M = shapes[s][0], N = shapes[s][1], K = shapes[s][2];
        size_t bytes_A = (size_t)M * K * sizeof(float);
        size_t bytes_B = (size_t)K * N * sizeof(float);
        size_t bytes_C = (size_t)M * N * sizeof(float);

        // 分配并初始化 host 数据
        std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N, 0.f);
        srand(42);
        for (auto& v : h_A) v = (float)rand() / RAND_MAX - 0.5f;
        for (auto& v : h_B) v = (float)rand() / RAND_MAX - 0.5f;

        float *d_A, *d_B, *d_C;
        HIP_CHECK(hipMalloc(&d_A, bytes_A));
        HIP_CHECK(hipMalloc(&d_B, bytes_B));
        HIP_CHECK(hipMalloc(&d_C, bytes_C));
        HIP_CHECK(hipMemcpy(d_A, h_A.data(), bytes_A, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_B, h_B.data(), bytes_B, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemset(d_C, 0, bytes_C));

        // rocBLAS sgemm：C = alpha*A*B + beta*C
        // 注意：rocBLAS 默认列主序，这里用 NN 模式并传入行主序矩阵
        // 等价技巧：C^T = B^T * A^T，交换 A/B 并翻转转置标志
        const float alpha = 1.0f, beta = 0.0f;

        // Warmup
        for (int i = 0; i < WARMUP; i++) {
            RB_CHECK(rocblas_sgemm(
                handle,
                rocblas_operation_none, rocblas_operation_none,
                N, M, K,
                &alpha,
                d_B, N,    // B^T 视角
                d_A, K,    // A^T 视角
                &beta,
                d_C, N));
        }
        HIP_CHECK(hipDeviceSynchronize());

        // 计时
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < REPEAT; i++) {
            RB_CHECK(rocblas_sgemm(
                handle,
                rocblas_operation_none, rocblas_operation_none,
                N, M, K,
                &alpha,
                d_B, N,
                d_A, K,
                &beta,
                d_C, N));
        }
        HIP_CHECK(hipDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double avg_ms     = elapsed_ms / REPEAT;
        double tf         = tflops(M, N, K, avg_ms);

        printf("M=%d N=%d K=%d  %8.3f  %10.3f\n", M, N, K, avg_ms, tf);

        HIP_CHECK(hipFree(d_A));
        HIP_CHECK(hipFree(d_B));
        HIP_CHECK(hipFree(d_C));
    }

    RB_CHECK(rocblas_destroy_handle(handle));
    return 0;
}
