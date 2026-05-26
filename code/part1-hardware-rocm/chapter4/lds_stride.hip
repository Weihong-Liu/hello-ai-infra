// code/part1-hardware-rocm/chapter4/lds_stride.hip
//
// LDS bank conflict micro-benchmark：构造 stride=1 / 2 / 4 / 8 / 16 / 32
// 的 LDS 访问，观察吞吐变化，验证 4.4 节的 N-way 冲突。
//
// 编译：
//   hipcc -O3 lds_stride.hip -o lds_stride
// 运行：
//   ./lds_stride                       # 默认扫 1/2/4/8/16/32 + repeats=100
//   ./lds_stride --repeats 200 --iters 65536
//
// 硬件上下文：AI MAX 395 (gfx1151) + ROCm 7.12.0 (rocm-sdk wheels) / hipcc 7.2.x
// 输出格式（每个 stride 一行）：
//   [lds_stride] STRIDE=<S> repeats=<R> time=<ms_total> time_per_iter=<ms>
//
// 实现要点：
//   - LDS 大小 2048 floats = 8 KiB（fits in single CU 的 LDS）
//   - 每个 thread 在 n_iters 次循环里反复读 lds[(tid*STRIDE) & 2047]
//   - 输出 acc 写到 d_out 防止 dead-code-elim
//
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>

#define HIP_CHECK(expr) do {                                                      \
    hipError_t _err = (expr);                                                     \
    if (_err != hipSuccess) {                                                     \
        fprintf(stderr, "HIP error %s at %s:%d : %s\n", hipGetErrorString(_err),  \
                __FILE__, __LINE__, #expr);                                       \
        std::exit(1);                                                             \
    }                                                                             \
} while (0)

template <int STRIDE>
__global__ void lds_stride_kernel(const float* __restrict__ in,
                                  float* __restrict__ out,
                                  int n_iters) {
    __shared__ float lds[2048];
    int tid = threadIdx.x;
    // load LDS from global once
    for (int i = tid; i < 2048; i += blockDim.x)
        lds[i] = in[i];
    __syncthreads();

    float acc = 0.f;
    #pragma unroll 1
    for (int it = 0; it < n_iters; ++it) {
        int idx = (tid * STRIDE + it) & 2047;  // +it 避免编译器算成常量
        acc += lds[idx];
    }
    out[tid] = acc;  // sink 防止 DCE
}

template <int STRIDE>
static double run_one(const float* d_in, float* d_out, int block, int n_iters,
                      int repeats, int warmup) {
    hipEvent_t s, e;
    HIP_CHECK(hipEventCreate(&s));
    HIP_CHECK(hipEventCreate(&e));

    for (int i = 0; i < warmup; ++i)
        lds_stride_kernel<STRIDE><<<1, block>>>(d_in, d_out, n_iters);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipEventRecord(s));
    for (int i = 0; i < repeats; ++i)
        lds_stride_kernel<STRIDE><<<1, block>>>(d_in, d_out, n_iters);
    HIP_CHECK(hipEventRecord(e));
    HIP_CHECK(hipEventSynchronize(e));

    float ms = 0.f;
    HIP_CHECK(hipEventElapsedTime(&ms, s, e));

    double ms_total = (double)ms;
    double ms_per = ms_total / (double)repeats;
    printf("[lds_stride] STRIDE=%-2d repeats=%d iters=%d time=%.4f ms time_per_iter=%.4f ms\n",
           STRIDE, repeats, n_iters, ms_total, ms_per);
    fflush(stdout);

    HIP_CHECK(hipEventDestroy(s));
    HIP_CHECK(hipEventDestroy(e));
    return ms_total;
}

static void print_header(int block, int n_iters, int repeats, int warmup) {
    char buf[64];
    std::time_t t = std::time(nullptr);
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S %z", std::localtime(&t));
    int dev = 0;
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDevice(&dev));
    HIP_CHECK(hipGetDeviceProperties(&prop, dev));
    printf("# lds_stride (LDS bank conflict scan) — AI MAX 395 / gfx1151\n");
    printf("# timestamp: %s\n", buf);
    printf("# device:    %s (gcnArch=%s, %d CUs)\n",
           prop.name, prop.gcnArchName, prop.multiProcessorCount);
    printf("# config:    block=%d iters=%d repeats=%d warmup=%d\n",
           block, n_iters, repeats, warmup);
    fflush(stdout);
}

int main(int argc, char** argv) {
    int block = 256;
    int n_iters = 1 << 16;  // 65536
    int repeats = 100;
    int warmup = 5;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { fprintf(stderr, "missing arg for %s\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--block") block = std::atoi(next());
        else if (a == "--iters") n_iters = std::atoi(next());
        else if (a == "--repeats") repeats = std::atoi(next());
        else if (a == "--warmup") warmup = std::atoi(next());
        else if (a == "-h" || a == "--help") {
            printf("Usage: %s [--block B] [--iters I] [--repeats R] [--warmup W]\n", argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown arg: %s\n", a.c_str()); return 2;
        }
    }
    if (repeats < 50) repeats = 50;
    if (warmup < 5) warmup = 5;

    print_header(block, n_iters, repeats, warmup);

    float *d_in = nullptr, *d_out = nullptr;
    HIP_CHECK(hipMalloc(&d_in, 2048 * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out, block * sizeof(float)));
    HIP_CHECK(hipMemset(d_in, 1, 2048 * sizeof(float)));

    run_one<1>(d_in, d_out, block, n_iters, repeats, warmup);
    run_one<2>(d_in, d_out, block, n_iters, repeats, warmup);
    run_one<4>(d_in, d_out, block, n_iters, repeats, warmup);
    run_one<8>(d_in, d_out, block, n_iters, repeats, warmup);
    run_one<16>(d_in, d_out, block, n_iters, repeats, warmup);
    run_one<32>(d_in, d_out, block, n_iters, repeats, warmup);

    HIP_CHECK(hipFree(d_in));
    HIP_CHECK(hipFree(d_out));
    return 0;
}
