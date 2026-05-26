// code/part1-hardware-rocm/chapter4/bw_triad.hip
//
// STREAM Triad on the GPU: c[i] = a[i] + alpha * b[i]
// 没有任何复用——理论时间 ≈ 总搬运字节 / 有效带宽。
// 通过扫描 size 画 "L2 → MALL → DRAM" 的有效带宽阶梯。
//
// 编译：
//   hipcc -O3 bw_triad.hip -o bw_triad
// 运行（默认扫描）：
//   ./bw_triad
// 单尺寸：
//   ./bw_triad --bytes 134217728 --repeats 100
// 自定义尺寸列表（每数组 bytes，逗号分隔）：
//   ./bw_triad --sizes 1048576,4194304,16777216,33554432,134217728,268435456,1073741824
//
// 硬件上下文：AI MAX 395 (gfx1151) + ROCm 7.12.0 (rocm-sdk wheels) / hipcc 7.2.x
// 输出格式（一行一个 size，方便 grep）：
//   [bw_triad] bytes=<N> repeats=<R> time=<ms> BW=<GB/s>
//
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>
#include <string>
#include <sstream>

#define HIP_CHECK(expr) do {                                                      \
    hipError_t _err = (expr);                                                     \
    if (_err != hipSuccess) {                                                     \
        fprintf(stderr, "HIP error %s at %s:%d : %s\n", hipGetErrorString(_err),  \
                __FILE__, __LINE__, #expr);                                       \
        std::exit(1);                                                             \
    }                                                                             \
} while (0)

__global__ void triad(const float* __restrict__ a,
                      const float* __restrict__ b,
                      float* __restrict__ c,
                      float alpha, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + alpha * b[i];
}

__global__ void fill_kernel(float* p, float v, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

static void print_header() {
    char buf[64];
    std::time_t t = std::time(nullptr);
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S %z", std::localtime(&t));
    int dev = 0;
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDevice(&dev));
    HIP_CHECK(hipGetDeviceProperties(&prop, dev));
    printf("# bw_triad (STREAM Triad) — AI MAX 395 / gfx1151\n");
    printf("# timestamp: %s\n", buf);
    printf("# device:    %s (gcnArch=%s, %d CUs)\n",
           prop.name, prop.gcnArchName, prop.multiProcessorCount);
    printf("# (run rocm-smi separately for power/freq/mem state)\n");
    fflush(stdout);
}

static double bench_one(size_t bytes, int repeats, int warmup) {
    size_t n = bytes / sizeof(float);
    float *a = nullptr, *b = nullptr, *c = nullptr;
    HIP_CHECK(hipMalloc(&a, bytes));
    HIP_CHECK(hipMalloc(&b, bytes));
    HIP_CHECK(hipMalloc(&c, bytes));

    int block = 256;
    size_t grid_sz = (n + block - 1) / block;
    // gfx1151 grid x 上限远超 2^31，但保险起见 cap 到 int max.
    int grid = (grid_sz > (size_t)INT32_MAX) ? INT32_MAX : (int)grid_sz;

    fill_kernel<<<grid, block>>>(a, 1.0f, n);
    fill_kernel<<<grid, block>>>(b, 2.0f, n);
    fill_kernel<<<grid, block>>>(c, 0.0f, n);
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t s, e;
    HIP_CHECK(hipEventCreate(&s));
    HIP_CHECK(hipEventCreate(&e));

    for (int i = 0; i < warmup; ++i)
        triad<<<grid, block>>>(a, b, c, 2.0f, n);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipEventRecord(s));
    for (int i = 0; i < repeats; ++i)
        triad<<<grid, block>>>(a, b, c, 2.0f, n);
    HIP_CHECK(hipEventRecord(e));
    HIP_CHECK(hipEventSynchronize(e));

    float ms = 0.f;
    HIP_CHECK(hipEventElapsedTime(&ms, s, e));

    // Triad: 2 reads + 1 write = 3 * bytes per iteration
    double total_bytes = 3.0 * (double)bytes * (double)repeats;
    double gbps = total_bytes / ((double)ms * 1e-3) / 1e9;
    double ms_per_iter = (double)ms / (double)repeats;

    printf("[bw_triad] bytes=%zu repeats=%d time=%.4f ms time_per_iter=%.4f ms BW=%.2f GB/s\n",
           bytes, repeats, (double)ms, ms_per_iter, gbps);
    fflush(stdout);

    HIP_CHECK(hipEventDestroy(s));
    HIP_CHECK(hipEventDestroy(e));
    HIP_CHECK(hipFree(a));
    HIP_CHECK(hipFree(b));
    HIP_CHECK(hipFree(c));
    return gbps;
}

int main(int argc, char** argv) {
    // 默认扫描：1 MB → 1 GiB（带 1/4/16/32/128/256 MB / 1 GiB 等关键拐点）
    std::vector<size_t> sizes = {
        1ull << 20,   // 1 MiB
        4ull << 20,   // 4 MiB
        16ull << 20,  // 16 MiB
        32ull << 20,  // 32 MiB (MALL boundary)
        128ull << 20, // 128 MiB
        256ull << 20, // 256 MiB
        1ull << 30,   // 1 GiB
    };
    int repeats = 50;
    int warmup = 5;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { fprintf(stderr, "missing arg for %s\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--bytes") {
            sizes.clear();
            sizes.push_back((size_t)std::strtoull(next(), nullptr, 10));
        } else if (a == "--repeats") {
            repeats = std::atoi(next());
        } else if (a == "--warmup") {
            warmup = std::atoi(next());
        } else if (a == "--sizes") {
            sizes.clear();
            std::string list = next();
            std::stringstream ss(list);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (!item.empty()) sizes.push_back((size_t)std::strtoull(item.c_str(), nullptr, 10));
            }
        } else if (a == "-h" || a == "--help") {
            printf("Usage: %s [--bytes N | --sizes N1,N2,...] [--repeats R] [--warmup W]\n", argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown arg: %s\n", a.c_str());
            return 2;
        }
    }

    print_header();
    if (repeats < 50) repeats = 50;
    if (warmup < 5) warmup = 5;
    printf("# config: repeats=%d warmup=%d sizes=%zu entries\n", repeats, warmup, sizes.size());
    fflush(stdout);

    for (size_t bytes : sizes) {
        bench_one(bytes, repeats, warmup);
    }
    return 0;
}
