/**
 * cpu_baseline.cpp
 *
 * part3-hip-kernels chapter2：向量加法 CPU 基线
 *
 * 用 -O2 -fopenmp 编译，多线程执行向量加法，作为 GPU 版本的性能参照。
 * 带宽按 3 次访存（读 a、读 b、写 c）计算。
 *
 * 编译：
 *   g++ -O2 -std=c++17 -fopenmp cpu_baseline.cpp -o cpu_baseline
 *
 * 用法：
 *   ./cpu_baseline [n] [repeat]
 *   ./cpu_baseline 16777216 10
 *   ./cpu_baseline 67108864 10
 *
 * 输出格式（每行一个 n）：
 *   cpu_n=<n>  best_ms=<t>  mean_ms=<t>  bw_GBs=<bw>  ok=<0|1>
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char** argv) {
    int n      = (argc > 1) ? std::atoi(argv[1]) : (1 << 24);  // 默认 16M
    int repeat = (argc > 2) ? std::atoi(argv[2]) : 10;

    if (n <= 0 || repeat <= 0) {
        std::fprintf(stderr, "Usage: %s [n] [repeat]\n", argv[0]);
        return 1;
    }

    std::vector<float> a(n, 1.0f), b(n, 2.0f), c(n, 0.0f);

    // warmup（让 OS 分页和缓存达到稳态）
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) c[i] = a[i] + b[i];

    double best_ms = 1e18;
    double sum_ms  = 0.0;

    for (int r = 0; r < repeat; ++r) {
        auto t0 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for
        for (int i = 0; i < n; ++i) c[i] = a[i] + b[i];
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        sum_ms += ms;
        if (ms < best_ms) best_ms = ms;
    }

    double mean_ms = sum_ms / repeat;

    // 正确性验证
    bool ok = true;
    for (int i = 0; i < n; ++i) {
        if (std::fabs(c[i] - 3.0f) > 1e-6f) {
            ok = false;
            break;
        }
    }

    // 带宽：3 次访存（读 a + 读 b + 写 c），取 best_ms
    double bw = (double)n * sizeof(float) * 3.0 / (best_ms / 1000.0) / 1e9;

#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
#else
    int nthreads = 1;
#endif

    std::printf(
        "cpu_n=%d  threads=%d  best_ms=%.4f  mean_ms=%.4f  bw_GBs=%.3f  ok=%d\n",
        n, nthreads, best_ms, mean_ms, bw, (int)ok
    );

    return ok ? 0 : 1;
}
