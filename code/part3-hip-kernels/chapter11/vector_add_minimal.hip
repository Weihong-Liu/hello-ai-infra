// vector_add_minimal.hip — 最小 HIP 向量加法程序
// 演示：host alloc/copy/free + 加法 kernel + HIP_CHECK 宏 + 结果验证
// 对应文档：docs/part3-hip-kernels/chapter11/index.md §11.3–§11.6

#include <hip/hip_runtime.h>

#include <cmath>
#include <iostream>
#include <vector>

// ────────────────────────────────────────────────────────────
// HIP_CHECK 宏：包裹每次 HIP API 调用，失败时打印位置并退出
// ────────────────────────────────────────────────────────────
#define HIP_CHECK(call)                                                      \
  do {                                                                       \
    hipError_t err = (call);                                                 \
    if (err != hipSuccess) {                                                 \
      std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__           \
                << ": " << hipGetErrorString(err) << std::endl;              \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

// ────────────────────────────────────────────────────────────
// kernel：向量加法，每个线程处理一个元素
// ────────────────────────────────────────────────────────────
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;  // 边界检查：多余线程直接返回
    c[idx] = a[idx] + b[idx];
}

int main() {
    const int n = 1 << 20;           // 1M 个元素
    const size_t bytes = n * sizeof(float);

    // ── Host 端分配并初始化 ──────────────────────────────────
    std::vector<float> h_a(n, 1.0f);
    std::vector<float> h_b(n, 2.0f);
    std::vector<float> h_c(n, 0.0f);

    // ── Device 端分配 ────────────────────────────────────────
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    HIP_CHECK(hipMalloc(&d_a, bytes));
    HIP_CHECK(hipMalloc(&d_b, bytes));
    HIP_CHECK(hipMalloc(&d_c, bytes));

    // ── 数据从 Host 传入 Device ──────────────────────────────
    HIP_CHECK(hipMemcpy(d_a, h_a.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b.data(), bytes, hipMemcpyHostToDevice));

    // ── 启动 kernel（异步）──────────────────────────────────
    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;  // 向上取整
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    HIP_CHECK(hipGetLastError());       // 检查 launch 时的配置错误
    HIP_CHECK(hipDeviceSynchronize());  // 等 GPU 跑完，捕获执行时的错误

    // ── 数据从 Device 拉回 Host ──────────────────────────────
    HIP_CHECK(hipMemcpy(h_c.data(), d_c, bytes, hipMemcpyDeviceToHost));

    // ── 验证结果：对比浮点最大绝对误差 ──────────────────────
    float max_error = 0.0f;
    for (int i = 0; i < n; ++i) {
        max_error = std::max(max_error, std::abs(h_c[i] - 3.0f));
    }

    // ── 打印摘要 ─────────────────────────────────────────────
    std::cout << "vector_size: " << n << std::endl;
    std::cout << "threads_per_block: " << threads << std::endl;
    std::cout << "blocks: " << blocks << std::endl;
    std::cout << "max_error: " << max_error << std::endl;
    std::cout << "status: " << (max_error == 0.0f ? "PASS" : "FAIL") << std::endl;

    // ── 释放 Device 内存 ─────────────────────────────────────
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));

    return max_error == 0.0f ? 0 : 1;
}
