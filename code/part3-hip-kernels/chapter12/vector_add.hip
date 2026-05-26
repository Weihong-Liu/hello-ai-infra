/**
 * vector_add.hip
 *
 * part3-hip-kernels chapter2：从 Vector Add 理解 GPU 并行
 *
 * 三个 kernel 变体，由命令行 --kernel 选择：
 *   naive       - 一线程处理一个元素（基线）
 *   grid_stride - grid-stride loop，一线程可处理多个元素
 *   strided_bad - 故意写成步长访问（非合并访存反例）
 *
 * 编译示例（在激活 activate-rocm.sh 后）：
 *   hipcc vector_add.hip -O2 -o vector_add_bench
 *
 * 用法：
 *   ./vector_add_bench --kernel naive      --size 16777216 --block 256
 *   ./vector_add_bench --kernel grid_stride --size 16777216 --block 256
 *   ./vector_add_bench --kernel strided_bad --size 16777216 --block 256
 */

#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

// --------------------------------------------------------------------------
// 错误检查宏
// --------------------------------------------------------------------------
#define HIP_CHECK(call)                                                      \
  do {                                                                       \
    hipError_t _err = (call);                                                \
    if (_err != hipSuccess) {                                                \
      std::cerr << "[HIP ERROR] " << hipGetErrorString(_err)                 \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl;       \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

// --------------------------------------------------------------------------
// Kernel 1: naive
//   每个线程负责一个元素，索引 = blockIdx.x * blockDim.x + threadIdx.x
// --------------------------------------------------------------------------
__global__ void kernel_naive(const float* __restrict__ a,
                             const float* __restrict__ b,
                             float* __restrict__ c,
                             int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

// --------------------------------------------------------------------------
// Kernel 2: grid_stride loop
//   每个线程以 gridDim.x * blockDim.x 为步长遍历数组。
//   好处：grid 大小与数据量解耦；任何 n 都能跑，不需要额外的 ceiling 计算。
// --------------------------------------------------------------------------
__global__ void kernel_grid_stride(const float* __restrict__ a,
                                   const float* __restrict__ b,
                                   float* __restrict__ c,
                                   int n) {
  int stride = gridDim.x * blockDim.x;
  int idx    = blockIdx.x * blockDim.x + threadIdx.x;
  for (; idx < n; idx += stride) {
    c[idx] = a[idx] + b[idx];
  }
}

// --------------------------------------------------------------------------
// Kernel 3: strided_bad（故意写成非合并访存，作为反例）
//   线程 t 访问下标 t, t+blockDim.x, t+2*blockDim.x ...
//   即 warp 内相邻线程访问的地址相差 blockDim.x 个 float，
//   导致 cache line 利用率极低，是教科书级别的"访存反例"。
// --------------------------------------------------------------------------
__global__ void kernel_strided_bad(const float* __restrict__ a,
                                   const float* __restrict__ b,
                                   float* __restrict__ c,
                                   int n) {
  int t       = threadIdx.x;
  int blk     = blockIdx.x;
  int blk_dim = blockDim.x;
  int grid_dim = gridDim.x;

  // 以 threadIdx 为最慢变化维，blockIdx 为快变化维
  // 结果：warp 内的线程在全局内存里跨步 blk_dim 访问
  for (int i = blk; i < (n + blk_dim - 1) / blk_dim; i += grid_dim) {
    int idx = i * blk_dim + t;
    if (idx < n) {
      c[idx] = a[idx] + b[idx];
    }
  }
}

// --------------------------------------------------------------------------
// Host 端计时 + 校验
// --------------------------------------------------------------------------
struct BenchResult {
  double mean_ms;
  double min_ms;
  double bandwidth_gb_s;  // 按 3 次访存（读 a、读 b、写 c）计算
  bool   correct;
};

BenchResult run_benchmark(const std::string& kernel_name,
                          int n,
                          int block_size,
                          int warmup,
                          int repeat) {
  size_t bytes = (size_t)n * sizeof(float);

  // 在 CPU 上准备数据（a=1.0, b=2.0，期望 c=3.0）
  std::vector<float> h_a(n, 1.0f);
  std::vector<float> h_b(n, 2.0f);
  std::vector<float> h_c(n, 0.0f);

  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  HIP_CHECK(hipMalloc(&d_a, bytes));
  HIP_CHECK(hipMalloc(&d_b, bytes));
  HIP_CHECK(hipMalloc(&d_c, bytes));

  HIP_CHECK(hipMemcpy(d_a, h_a.data(), bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_b, h_b.data(), bytes, hipMemcpyHostToDevice));

  // 计算 grid size（naive 和 strided_bad 按 n/block_size 计算；
  // grid_stride 可以固定一个较小的 grid，这里为了对比统一用相同 grid）
  int grid_size = (n + block_size - 1) / block_size;

  // 用 HIP event 计时
  hipEvent_t ev_start, ev_stop;
  HIP_CHECK(hipEventCreate(&ev_start));
  HIP_CHECK(hipEventCreate(&ev_stop));

  // Warmup
  for (int i = 0; i < warmup; ++i) {
    if (kernel_name == "naive") {
      hipLaunchKernelGGL(kernel_naive, dim3(grid_size), dim3(block_size),
                         0, 0, d_a, d_b, d_c, n);
    } else if (kernel_name == "grid_stride") {
      hipLaunchKernelGGL(kernel_grid_stride, dim3(grid_size), dim3(block_size),
                         0, 0, d_a, d_b, d_c, n);
    } else {
      hipLaunchKernelGGL(kernel_strided_bad, dim3(grid_size), dim3(block_size),
                         0, 0, d_a, d_b, d_c, n);
    }
  }
  HIP_CHECK(hipDeviceSynchronize());

  // 正式计时
  std::vector<float> run_times_ms(repeat);
  for (int i = 0; i < repeat; ++i) {
    HIP_CHECK(hipEventRecord(ev_start));
    if (kernel_name == "naive") {
      hipLaunchKernelGGL(kernel_naive, dim3(grid_size), dim3(block_size),
                         0, 0, d_a, d_b, d_c, n);
    } else if (kernel_name == "grid_stride") {
      hipLaunchKernelGGL(kernel_grid_stride, dim3(grid_size), dim3(block_size),
                         0, 0, d_a, d_b, d_c, n);
    } else {
      hipLaunchKernelGGL(kernel_strided_bad, dim3(grid_size), dim3(block_size),
                         0, 0, d_a, d_b, d_c, n);
    }
    HIP_CHECK(hipEventRecord(ev_stop));
    HIP_CHECK(hipEventSynchronize(ev_stop));
    HIP_CHECK(hipEventElapsedTime(&run_times_ms[i], ev_start, ev_stop));
  }

  // 校验
  HIP_CHECK(hipMemcpy(h_c.data(), d_c, bytes, hipMemcpyDeviceToHost));
  float max_err = 0.0f;
  for (int i = 0; i < n; ++i) {
    max_err = std::max(max_err, std::abs(h_c[i] - 3.0f));
  }

  // 统计
  double sum = 0.0;
  double min_ms = run_times_ms[0];
  for (float t : run_times_ms) {
    sum += t;
    if (t < min_ms) min_ms = t;
  }
  double mean_ms = sum / repeat;
  double bw = (double)bytes * 3.0 / (min_ms / 1000.0) / 1e9;

  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_c));
  HIP_CHECK(hipEventDestroy(ev_start));
  HIP_CHECK(hipEventDestroy(ev_stop));

  return {mean_ms, min_ms, bw, max_err == 0.0f};
}

// --------------------------------------------------------------------------
// main
// --------------------------------------------------------------------------
int main(int argc, char** argv) {
  // 默认参数
  std::string kernel_name = "naive";
  int n          = 1 << 24;   // 16M floats
  int block_size = 256;
  int warmup     = 5;
  int repeat     = 30;

  // 简易 CLI 解析
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--kernel") == 0 && i + 1 < argc) {
      kernel_name = argv[++i];
    } else if (std::strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
      n = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--block") == 0 && i + 1 < argc) {
      block_size = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
      warmup = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) {
      repeat = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--help") == 0) {
      std::cout << "Usage: vector_add_bench [--kernel naive|grid_stride|strided_bad]"
                << " [--size N] [--block B] [--warmup W] [--repeat R]\n";
      return 0;
    }
  }

  // 校验 kernel 名
  if (kernel_name != "naive" && kernel_name != "grid_stride" &&
      kernel_name != "strided_bad") {
    std::cerr << "Unknown kernel: " << kernel_name << std::endl;
    return 1;
  }

  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));

  std::cout << "device_name: " << prop.name << std::endl;
  std::cout << "kernel: " << kernel_name << std::endl;
  std::cout << "vector_size: " << n << std::endl;
  std::cout << "block_size: " << block_size << std::endl;
  std::cout << "grid_size: " << (n + block_size - 1) / block_size << std::endl;
  std::cout << "warmup: " << warmup << std::endl;
  std::cout << "repeat: " << repeat << std::endl;

  BenchResult result = run_benchmark(kernel_name, n, block_size, warmup, repeat);

  std::cout << "mean_ms: " << result.mean_ms << std::endl;
  std::cout << "min_ms: " << result.min_ms << std::endl;
  std::cout << "bandwidth_gb_s: " << result.bandwidth_gb_s << std::endl;
  std::cout << "correct: " << (result.correct ? "true" : "false") << std::endl;
  std::cout << "status: " << (result.correct ? "PASS" : "FAIL") << std::endl;

  return result.correct ? 0 : 1;
}
