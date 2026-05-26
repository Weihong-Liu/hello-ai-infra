// error_check.hip — 演示错误检查的必要性
// 对应文档：docs/part3-hip-kernels/chapter11/index.md §11.6
//
// 包含两类故意触发的错误演示：
//   1. 申请超出显存上限的内存（OOM 错误）
//   2. 使用超过 maxThreadsPerBlock 的 block size（launch 配置错误）
//
// 带 HIP_CHECK 的版本会打印清晰的错误信息并退出；
// 不带检查的版本会静默出错（通过命令行参数 --no-check 触发）。

#include <hip/hip_runtime.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

// ────────────────────────────────────────────────────────────
// HIP_CHECK 宏（与正文贴出的版本完全一致）
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

// 哑 kernel：什么都不做，用于触发 launch 配置错误演示
__global__ void dummy_kernel(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    (void)idx;
    (void)data;
}

// ────────────────────────────────────────────────────────────
// 演示 1：申请超出显存上限的内存（带检查版本）
// ────────────────────────────────────────────────────────────
void demo_oom_with_check() {
    std::cout << "\n=== 演示 1: OOM（带 HIP_CHECK）===" << std::endl;
    float* d_ptr = nullptr;
    // 申请 1 TB 内存，必然超出显存上限
    size_t huge = (size_t)1024 * 1024 * 1024 * 1024;
    std::cout << "尝试分配 1 TB 显存..." << std::endl;
    HIP_CHECK(hipMalloc(&d_ptr, huge));
    // 走到这里意味着没有触发错误（不应该发生）
    hipFree(d_ptr);
}

// ────────────────────────────────────────────────────────────
// 演示 2：超出 maxThreadsPerBlock 的 block size（带检查版本）
// ────────────────────────────────────────────────────────────
void demo_bad_launch_with_check() {
    std::cout << "\n=== 演示 2: 非法 launch 配置（带 HIP_CHECK）===" << std::endl;

    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    int max_threads = prop.maxThreadsPerBlock;
    std::cout << "maxThreadsPerBlock = " << max_threads << std::endl;

    // 故意超出上限
    int bad_threads = max_threads * 2;
    std::cout << "尝试以 blockDim=" << bad_threads << " 启动 kernel..." << std::endl;

    float* d_ptr = nullptr;
    HIP_CHECK(hipMalloc(&d_ptr, sizeof(float)));

    dummy_kernel<<<1, bad_threads>>>(d_ptr);
    HIP_CHECK(hipGetLastError());  // 此处应捕获 launch 配置错误
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(d_ptr));
}

int main(int argc, char* argv[]) {
    bool no_check = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--no-check") no_check = true;
    }

    std::cout << "error_check 演示程序" << std::endl;
    std::cout << "模式: " << (no_check ? "不带检查（静默出错）" : "带 HIP_CHECK（打印错误后退出）") << std::endl;

    if (no_check) {
        // ── 不带检查的版本：静默失败演示 ──────────────────────
        std::cout << "\n=== 不带检查版本：OOM 后继续跑 ===" << std::endl;
        float* d_bad = nullptr;
        size_t huge = (size_t)1024 * 1024 * 1024 * 1024;
        // 返回值被忽略：分配失败 d_bad 仍为 nullptr
        hipMalloc(&d_bad, huge);
        std::cout << "hipMalloc 返回，d_bad = " << (void*)d_bad << "（nullptr 表示分配失败）" << std::endl;
        std::cout << "程序继续执行但 d_bad 无效，后续操作将产生未定义行为" << std::endl;
        // 不调用 hipFree(nullptr) 以避免 crash，直接退出
        return 0;
    }

    // 带检查的版本：遇到第一个错误就会打印并 exit(1)
    // 先跑演示 1（OOM），出错后程序退出，演示 2 不会执行
    demo_oom_with_check();

    // 如果 OOM 没触发（理论上不可能），继续演示 2
    demo_bad_launch_with_check();

    return 0;
}
