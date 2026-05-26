// hello_hip.hip — 最小 HIP hello-from-device 程序
// 演示：设备属性查询 + device-side printf
// 对应文档：docs/part3-hip-kernels/chapter11/index.md §11.4

#include <hip/hip_runtime.h>
#include <cstdio>
#include <iostream>

#define HIP_CHECK(call)                                                      \
  do {                                                                       \
    hipError_t err = (call);                                                 \
    if (err != hipSuccess) {                                                 \
      std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__           \
                << ": " << hipGetErrorString(err) << std::endl;              \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

// kernel：从 Device 端打印问候信息（只让第一个线程打印）
__global__ void hello_kernel(int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        // 只让第一个线程打印，避免几千行重复输出
        printf("[device] hello from GPU! total threads = %d\n", n);
    }
}

int main() {
    // 查询设备属性
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));  // 0 是设备编号
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Global memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Warp size: " << prop.warpSize << std::endl;

    // 启动 kernel：1 个 block，64 个线程（对齐 wavefront 大小）
    const int threads = 64;
    const int blocks  = 1;
    hello_kernel<<<blocks, threads>>>(blocks * threads);
    HIP_CHECK(hipGetLastError());       // 检查 launch 时的配置错误
    HIP_CHECK(hipDeviceSynchronize());  // 等待 kernel 完成，捕获执行时的错误

    std::cout << "status: PASS" << std::endl;
    return 0;
}
