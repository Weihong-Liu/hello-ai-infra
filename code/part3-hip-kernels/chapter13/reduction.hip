// reduction.hip
// Part 3 Chapter 3: Reduction 优化
// 五个渐进优化版本 v0~v4，通过 --version 参数选择。
//
// 编译：
//   hipcc reduction.hip -O2 -o reduction
//
// 运行示例：
//   ./reduction --version 0 --n 16777216 --block 256 --warmup 5 --repeat 20
//   ./reduction --version 4 --n 268435456 --block 256 --warmup 3 --repeat 10

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>

// ─── HIP 错误检查宏 ───────────────────────────────────────────────────────────

#define HIP_CHECK(expr)                                                        \
    do {                                                                       \
        hipError_t _e = (expr);                                                \
        if (_e != hipSuccess) {                                                \
            fprintf(stderr, "HIP error at %s:%d: %s\n",                       \
                    __FILE__, __LINE__, hipGetErrorString(_e));                \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// ─── v0: 全局 atomicAdd（最朴素实现）────────────────────────────────────────

__global__ void reduce_v0_atomic(const float* __restrict__ input,
                                  float* output,
                                  int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(output, input[idx]);
    }
}

// ─── v1: LDS 归约树（每个 block 内用 shared memory 归约，最后 atomicAdd）────

__global__ void reduce_v1_lds(const float* __restrict__ input,
                               float* output,
                               int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 1. 加载元素到 LDS（越界补 0）
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    // 2. 归约树：stride 从 blockDim/2 开始每轮减半
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // 3. 线程 0 把 block 局部和写回全局内存
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

// ─── v2: Wavefront shuffle 优化（最后几轮用 __shfl_down 代替 LDS）────────────

// wavefront 内归约（无需 __syncthreads()，wavefront 内天然同步）
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset);
    }
    return val;
}

__global__ void reduce_v2_wavefront(const float* __restrict__ input,
                                     float* output,
                                     int n) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val = (idx < n) ? input[idx] : 0.0f;

    // 阶段 1：LDS 归约到 warpSize 大小
    sdata[tid] = val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride >= warpSize; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // 阶段 2：每个 wavefront 的前 warpSize 个线程用 shuffle 完成最后几轮
    if (tid < warpSize) {
        float warp_val = (blockDim.x >= warpSize) ? sdata[tid] : val;
        warp_val = warp_reduce_sum(warp_val);
        if (tid == 0) {
            atomicAdd(output, warp_val);
        }
    }
}

// ─── v3: 多元素展开 + 向量化加载（float4）─────────────────────────────────

__global__ void reduce_v3_unrolled(const float* __restrict__ input,
                                    float* output,
                                    int n) {
    extern __shared__ float sdata[];

    int tid      = threadIdx.x;
    int gid      = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_g = gridDim.x * blockDim.x;  // 全局步长

    // 每个线程在寄存器里做多元素局部累加
    float local_sum = 0.0f;

    // 尽量以 float4 加载（每次处理 4 个 float）
    // 假设 n 是 4 的倍数；不整除时退回标量处理
    int n4 = n / 4;
    int gid4 = blockIdx.x * blockDim.x + threadIdx.x;
    int stride4 = gridDim.x * blockDim.x;
    const float4* input4 = reinterpret_cast<const float4*>(input);

    // 向量化加载段
    for (int i = gid4; i < n4; i += stride4) {
        float4 v = input4[i];
        local_sum += v.x + v.y + v.z + v.w;
    }

    // 处理尾部（n 不是 4 整数倍时）
    int tail_start = n4 * 4;
    for (int i = tail_start + gid; i < n; i += stride_g) {
        local_sum += input[i];
    }

    // LDS + 归约树
    sdata[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s >= warpSize; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // 最后用 wavefront shuffle 完成
    if (tid < warpSize) {
        float warp_val = (blockDim.x >= warpSize) ? sdata[tid] : local_sum;
        warp_val = warp_reduce_sum(warp_val);
        if (tid == 0) {
            atomicAdd(output, warp_val);
        }
    }
}

// ─── v4: 两阶段 Reduction（避免大规模 atomicAdd 争用）──────────────────────

// 阶段 1：每个 block 归约到 partial_sums[blockIdx.x]
__global__ void reduce_v4_phase1(const float* __restrict__ input,
                                  float* partial_sums,
                                  int n) {
    extern __shared__ float sdata[];

    int tid      = threadIdx.x;
    int gid      = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_g = gridDim.x * blockDim.x;

    float local_sum = 0.0f;
    for (int i = gid; i < n; i += stride_g) {
        local_sum += input[i];
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s >= warpSize; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid < warpSize) {
        float warp_val = (blockDim.x >= warpSize) ? sdata[tid] : local_sum;
        warp_val = warp_reduce_sum(warp_val);
        if (tid == 0) {
            partial_sums[blockIdx.x] = warp_val;
        }
    }
}

// 阶段 2：单个 block 合并所有 partial_sums（不需要 atomicAdd）
__global__ void reduce_v4_phase2(const float* __restrict__ partial_sums,
                                  float* output,
                                  int num_partials) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    sdata[tid] = (tid < num_partials) ? partial_sums[tid] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s >= warpSize; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid < warpSize) {
        float warp_val = sdata[tid];
        warp_val = warp_reduce_sum(warp_val);
        if (tid == 0) {
            output[0] = warp_val;
        }
    }
}

// ─── CPU 参考实现（验证正确性）────────────────────────────────────────────

float cpu_reduce(const float* data, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += data[i];
    return static_cast<float>(sum);
}

// ─── Host launcher ─────────────────────────────────────────────────────────

struct RunResult {
    double ms;
    float  result;
};

RunResult run_version(int version, const float* h_input, int n,
                      int block_size, int warmup, int repeat) {
    float *d_input    = nullptr;
    float *d_output   = nullptr;
    float *d_partial  = nullptr;

    HIP_CHECK(hipMalloc(&d_input,  n * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_output, sizeof(float)));
    HIP_CHECK(hipMemcpy(d_input, h_input, n * sizeof(float),
                        hipMemcpyHostToDevice));

    int grid = (n + block_size - 1) / block_size;
    int shared_bytes = block_size * sizeof(float);

    // 限制 v4 阶段 1 的 grid 大小，保证阶段 2 单 block 能处理
    int grid_v4 = std::min(grid, 1024);

    auto launch_once = [&]() {
        HIP_CHECK(hipMemset(d_output, 0, sizeof(float)));

        switch (version) {
            case 0:
                reduce_v0_atomic<<<grid, block_size>>>(d_input, d_output, n);
                break;
            case 1:
                reduce_v1_lds<<<grid, block_size, shared_bytes>>>(
                    d_input, d_output, n);
                break;
            case 2:
                reduce_v2_wavefront<<<grid, block_size, shared_bytes>>>(
                    d_input, d_output, n);
                break;
            case 3:
                reduce_v3_unrolled<<<grid, block_size, shared_bytes>>>(
                    d_input, d_output, n);
                break;
            case 4:
                if (!d_partial) {
                    HIP_CHECK(hipMalloc(&d_partial,
                                        grid_v4 * sizeof(float)));
                }
                reduce_v4_phase1<<<grid_v4, block_size, shared_bytes>>>(
                    d_input, d_partial, n);
                reduce_v4_phase2<<<1, block_size, shared_bytes>>>(
                    d_partial, d_output, grid_v4);
                break;
            default:
                fprintf(stderr, "Unknown version: %d\n", version);
                exit(1);
        }
    };

    // warmup
    for (int i = 0; i < warmup; i++) launch_once();
    HIP_CHECK(hipDeviceSynchronize());

    // 计时
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < repeat; i++) launch_once();
    HIP_CHECK(hipDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count()
                / repeat;

    float h_result = 0.0f;
    HIP_CHECK(hipMemcpy(&h_result, d_output, sizeof(float),
                        hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_output));
    if (d_partial) HIP_CHECK(hipFree(d_partial));

    return {ms, h_result};
}

// ─── main ──────────────────────────────────────────────────────────────────

static void print_usage(const char* prog) {
    printf("Usage: %s [options]\n"
           "  --version <0-4>  Kernel version (default: all)\n"
           "  --n <int>        Number of elements (default: 16777216 = 16M)\n"
           "  --block <int>    Block size (default: 256)\n"
           "  --warmup <int>   Warmup iterations (default: 5)\n"
           "  --repeat <int>   Timing iterations (default: 20)\n"
           "  --all            Run all versions 0~4 and compare\n",
           prog);
}

int main(int argc, char** argv) {
    int version    = -1;  // -1 = all
    int n          = 16 * 1024 * 1024;  // 16M
    int block_size = 256;
    int warmup     = 5;
    int repeat     = 20;
    bool run_all   = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--version") == 0 && i + 1 < argc) {
            version = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--block") == 0 && i + 1 < argc) {
            block_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) {
            repeat = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--all") == 0) {
            run_all = true;
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (run_all) version = -1;

    // 初始化输入数据
    float* h_input = new float[n];
    for (int i = 0; i < n; i++) {
        h_input[i] = 1.0f;  // 全 1，期望结果 = n
    }

    float expected = cpu_reduce(h_input, n);
    printf("Expected sum (CPU): %.2f\n", expected);
    printf("n=%d  block_size=%d  warmup=%d  repeat=%d\n\n",
           n, block_size, warmup, repeat);

    double bytes = (double)n * sizeof(float);

    int v_start = (version < 0) ? 0 : version;
    int v_end   = (version < 0) ? 4 : version;

    for (int v = v_start; v <= v_end; v++) {
        RunResult r = run_version(v, h_input, n, block_size, warmup, repeat);
        double bw   = bytes / (r.ms / 1000.0) / 1e9;
        float  err  = fabsf(r.result - expected) / fabsf(expected + 1e-10f);

        printf("v%d  time=%.4f ms  bandwidth=%.3f GB/s  result=%.2f  "
               "rel_err=%.2e\n",
               v, r.ms, bw, r.result, err);
    }

    delete[] h_input;
    return 0;
}
