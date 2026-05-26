/**
 * softmax.hip — 四个 Softmax kernel 渐进版本
 *
 * v0: naive 三趟（pass 1: max, pass 2: exp+sum 不减 max, pass 3: div）
 * v1: 数值稳定三趟（pass 1: max, pass 2: exp+sum 减 max, pass 3: div）
 * v2: 单 kernel 三 pass 合并 + LDS 归约，减少全局访存趟数
 * v3: 单 kernel + float4 向量化加载（每 thread 处理 4 元素，要求 S % 4 == 0）
 *
 * 编译：
 *   hipcc softmax.hip -O2 -o build/softmax_bin
 *
 * 用法：
 *   ./build/softmax_bin [B] [S] [kernel_version] [--warmup W] [--repeat R] [--timing]
 *     B              batch size（行数），默认 8
 *     S              序列长度（列数），默认 2048
 *     kernel_version 0~3，默认全部跑（-1）
 *     --warmup W     warmup 次数，默认 3
 *     --repeat R     计时 repeat 次数，默认 10
 *     --timing       输出计时信息（ms 和 GB/s）
 *
 * 产出：logs/ 目录下的二进制文件，供 verify_vs_torch.py 读取
 */

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <functional>

// -----------------------------------------------------------------------
// 宏：错误检查
// -----------------------------------------------------------------------
#define HIP_CHECK(call)                                                       \
  do {                                                                        \
    hipError_t _e = (call);                                                   \
    if (_e != hipSuccess) {                                                   \
      fprintf(stderr, "[HIP] %s:%d  %s\n", __FILE__, __LINE__,               \
              hipGetErrorString(_e));                                          \
      exit(1);                                                                \
    }                                                                         \
  } while (0)

// -----------------------------------------------------------------------
// 辅助：LDS block 内归约（取最大值 或 求和）
// -----------------------------------------------------------------------
static constexpr int BLOCK = 256;  // block size (threads per block)

// block 内归约：求最大值
__device__ float block_reduce_max(float val, float* lds) {
  int tid = threadIdx.x;
  lds[tid] = val;
  __syncthreads();
  for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      lds[tid] = fmaxf(lds[tid], lds[tid + stride]);
    }
    __syncthreads();
  }
  return lds[0];
}

// block 内归约：求和
__device__ float block_reduce_sum(float val, float* lds) {
  int tid = threadIdx.x;
  lds[tid] = val;
  __syncthreads();
  for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      lds[tid] += lds[tid + stride];
    }
    __syncthreads();
  }
  return lds[0];
}

// -----------------------------------------------------------------------
// v0: Naive Softmax — 三趟，无数值稳定处理
// 注意：v0 故意不减 max，用于演示大值输入的数值问题
// -----------------------------------------------------------------------
__global__ void softmax_v0_pass1_max(const float* x, float* row_max, int S) {
  int row = blockIdx.x;
  const float* row_ptr = x + row * S;
  float mx = -1e30f;
  for (int i = threadIdx.x; i < S; i += blockDim.x) {
    mx = fmaxf(mx, row_ptr[i]);
  }
  __shared__ float lds[BLOCK];
  row_max[row] = block_reduce_max(mx, lds);
}

__global__ void softmax_v0_pass2_expsum(const float* x, float* exp_out,
                                         float* row_sum, int S) {
  int row = blockIdx.x;
  const float* row_ptr = x + row * S;
  float* exp_ptr = exp_out + row * S;
  float s = 0.f;
  for (int i = threadIdx.x; i < S; i += blockDim.x) {
    float e = expf(row_ptr[i]);  // 无减 max，大值会溢出为 Inf
    exp_ptr[i] = e;
    s += e;
  }
  __shared__ float lds[BLOCK];
  row_sum[row] = block_reduce_sum(s, lds);
}

__global__ void softmax_v0_pass3_div(float* exp_out, const float* row_sum,
                                      float* out, int S) {
  int row = blockIdx.x;
  float* exp_ptr = exp_out + row * S;
  float* out_ptr = out + row * S;
  float inv_sum = 1.f / row_sum[row];
  for (int i = threadIdx.x; i < S; i += blockDim.x) {
    out_ptr[i] = exp_ptr[i] * inv_sum;
  }
}

// v0 host 封装
void run_softmax_v0(const float* d_x, float* d_out,
                    float* d_exp_buf, float* d_row_max, float* d_row_sum,
                    int B, int S) {
  dim3 grid(B), block(BLOCK);
  softmax_v0_pass1_max<<<grid, block>>>(d_x, d_row_max, S);
  softmax_v0_pass2_expsum<<<grid, block>>>(d_x, d_exp_buf, d_row_sum, S);
  softmax_v0_pass3_div<<<grid, block>>>(d_exp_buf, d_row_sum, d_out, S);
  HIP_CHECK(hipDeviceSynchronize());
}

// -----------------------------------------------------------------------
// v1: 数值稳定 Softmax — 三趟，先减最大值
// -----------------------------------------------------------------------
__global__ void softmax_v1_pass1_max(const float* x, float* row_max, int S) {
  int row = blockIdx.x;
  const float* row_ptr = x + row * S;
  float mx = -1e30f;
  for (int i = threadIdx.x; i < S; i += blockDim.x) {
    mx = fmaxf(mx, row_ptr[i]);
  }
  __shared__ float lds[BLOCK];
  row_max[row] = block_reduce_max(mx, lds);
}

__global__ void softmax_v1_pass2_expsum(const float* x, const float* row_max,
                                         float* exp_out, float* row_sum,
                                         int S) {
  int row = blockIdx.x;
  const float* row_ptr = x + row * S;
  float* exp_ptr = exp_out + row * S;
  float mx = row_max[row];
  float s = 0.f;
  for (int i = threadIdx.x; i < S; i += blockDim.x) {
    float e = expf(row_ptr[i] - mx);  // 减去最大值，避免溢出
    exp_ptr[i] = e;
    s += e;
  }
  __shared__ float lds[BLOCK];
  row_sum[row] = block_reduce_sum(s, lds);
}

__global__ void softmax_v1_pass3_div(float* exp_out, const float* row_sum,
                                      float* out, int S) {
  int row = blockIdx.x;
  float* exp_ptr = exp_out + row * S;
  float* out_ptr = out + row * S;
  float inv_sum = 1.f / row_sum[row];
  for (int i = threadIdx.x; i < S; i += blockDim.x) {
    out_ptr[i] = exp_ptr[i] * inv_sum;
  }
}

void run_softmax_v1(const float* d_x, float* d_out,
                    float* d_exp_buf, float* d_row_max, float* d_row_sum,
                    int B, int S) {
  dim3 grid(B), block(BLOCK);
  softmax_v1_pass1_max<<<grid, block>>>(d_x, d_row_max, S);
  softmax_v1_pass2_expsum<<<grid, block>>>(d_x, d_row_max, d_exp_buf, d_row_sum, S);
  softmax_v1_pass3_div<<<grid, block>>>(d_exp_buf, d_row_sum, d_out, S);
  HIP_CHECK(hipDeviceSynchronize());
}

// -----------------------------------------------------------------------
// v2: 单 kernel 三归约合并 — 使用 LDS，减少全局访存趟数
// 流程（单 block 负责一行）：
//   pass 1: 线程私有遍历求 max，LDS 归约树得行 max（存 lds[BLOCK-1]）
//   pass 2: 计算 exp(x-max)，暂存到 out_ptr，LDS 归约树得行 sum
//   pass 3: 读 out_ptr（exp 值），原地乘以 inv_sum 写回
// 说明：
//   全局内存读：x 1 遍 + out（读 exp 值）1 遍
//   全局内存写：out 1 遍（最终结果）
//   无独立中间缓冲区（exp_out/row_max/row_sum）
// -----------------------------------------------------------------------
__global__ void softmax_v2_kernel(const float* __restrict__ x,
                                   float* __restrict__ out, int S) {
  __shared__ float lds[BLOCK];
  int row = blockIdx.x;
  const float* row_ptr = x + row * S;
  float* out_ptr = out + row * S;

  // --- pass 1: 求 max ---
  float mx = -1e30f;
  for (int i = threadIdx.x; i < S; i += BLOCK) {
    mx = fmaxf(mx, row_ptr[i]);
  }
  mx = block_reduce_max(mx, lds);
  // block_reduce_max 结束后 lds[0]=全行 max，但 lds 马上会被 pass2 复用
  // 把 max 存到 lds[BLOCK-1]（尾槽），pass2 开始前广播给所有线程
  if (threadIdx.x == 0) lds[BLOCK - 1] = mx;
  __syncthreads();
  mx = lds[BLOCK - 1];

  // --- pass 2: 计算 exp(x-max)，暂存到 out，累加 sum ---
  float s = 0.f;
  for (int i = threadIdx.x; i < S; i += BLOCK) {
    float e = expf(row_ptr[i] - mx);
    out_ptr[i] = e;  // 暂存 exp 值（待 pass3 原地乘 inv_sum）
    s += e;
  }
  s = block_reduce_sum(s, lds);
  if (threadIdx.x == 0) lds[BLOCK - 1] = s;
  __syncthreads();
  s = lds[BLOCK - 1];

  // --- pass 3: 读 exp，乘以 inv_sum，写回 ---
  float inv_s = 1.f / s;
  for (int i = threadIdx.x; i < S; i += BLOCK) {
    out_ptr[i] *= inv_s;
  }
}

void run_softmax_v2(const float* d_x, float* d_out, int B, int S) {
  dim3 grid(B), block(BLOCK);
  softmax_v2_kernel<<<grid, block>>>(d_x, d_out, S);
  HIP_CHECK(hipDeviceSynchronize());
}

// -----------------------------------------------------------------------
// v3: 向量化加载 + 单 kernel — 使用 float4 读取，每线程处理 4 元素
// 要求：S 为 4 的倍数
// -----------------------------------------------------------------------
__global__ void softmax_v3_kernel(const float* __restrict__ x,
                                   float* __restrict__ out, int S) {
  __shared__ float lds[BLOCK];
  int row = blockIdx.x;
  const float4* row4 = reinterpret_cast<const float4*>(x + row * S);
  float4* out4 = reinterpret_cast<float4*>(out + row * S);
  int S4 = S / 4;  // 向量化后每行有 S/4 个 float4

  // --- pass 1: 向量化读取，求 max ---
  float mx = -1e30f;
  for (int i = threadIdx.x; i < S4; i += BLOCK) {
    float4 v = row4[i];
    mx = fmaxf(mx, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
  }
  mx = block_reduce_max(mx, lds);
  if (threadIdx.x == 0) lds[BLOCK - 1] = mx;
  __syncthreads();
  mx = lds[BLOCK - 1];

  // --- pass 2: 向量化读取，计算 exp(x-max)，暂存到 out，累加 sum ---
  float s = 0.f;
  for (int i = threadIdx.x; i < S4; i += BLOCK) {
    float4 v = row4[i];
    float4 e;
    e.x = expf(v.x - mx);
    e.y = expf(v.y - mx);
    e.z = expf(v.z - mx);
    e.w = expf(v.w - mx);
    out4[i] = e;  // 向量化写出 exp
    s += e.x + e.y + e.z + e.w;
  }
  s = block_reduce_sum(s, lds);
  if (threadIdx.x == 0) lds[BLOCK - 1] = s;
  __syncthreads();
  s = lds[BLOCK - 1];

  // --- pass 3: 向量化读出 exp，除以 sum，向量化写回 ---
  float inv_s = 1.f / s;
  for (int i = threadIdx.x; i < S4; i += BLOCK) {
    float4 e = out4[i];
    e.x *= inv_s;
    e.y *= inv_s;
    e.z *= inv_s;
    e.w *= inv_s;
    out4[i] = e;
  }
}

void run_softmax_v3(const float* d_x, float* d_out, int B, int S) {
  if (S % 4 != 0) {
    fprintf(stderr, "[v3] S must be divisible by 4 (S=%d)，跳过 v3\n", S);
    return;
  }
  dim3 grid(B), block(BLOCK);
  softmax_v3_kernel<<<grid, block>>>(d_x, d_out, S);
  HIP_CHECK(hipDeviceSynchronize());
}

// -----------------------------------------------------------------------
// 计时辅助
// -----------------------------------------------------------------------
double measure_ms(std::function<void()> fn, int warmup, int repeat) {
  for (int i = 0; i < warmup; i++) fn();
  HIP_CHECK(hipDeviceSynchronize());

  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < repeat; i++) fn();
  HIP_CHECK(hipDeviceSynchronize());
  auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / repeat;
}

// -----------------------------------------------------------------------
// CPU 参考实现（数值稳定）
// -----------------------------------------------------------------------
void cpu_softmax(const float* x, float* out, int B, int S) {
  for (int b = 0; b < B; b++) {
    const float* row = x + b * S;
    float* o = out + b * S;
    float mx = row[0];
    for (int i = 1; i < S; i++) mx = fmaxf(mx, row[i]);
    float s = 0.f;
    for (int i = 0; i < S; i++) {
      o[i] = expf(row[i] - mx);
      s += o[i];
    }
    for (int i = 0; i < S; i++) o[i] /= s;
  }
}

float max_abs_error(const float* a, const float* b, int n) {
  float err = 0.f;
  for (int i = 0; i < n; i++) {
    err = fmaxf(err, fabsf(a[i] - b[i]));
  }
  return err;
}

// 写二进制结果到文件（供 verify_vs_torch.py 读取）
void write_bin(const char* path, const float* data, int n) {
  FILE* f = fopen(path, "wb");
  if (!f) { perror(path); return; }
  fwrite(data, sizeof(float), n, f);
  fclose(f);
}

// -----------------------------------------------------------------------
// 主函数
// -----------------------------------------------------------------------
int main(int argc, char* argv[]) {
  int B = 8;
  int S = 2048;
  int version = -1;  // -1 = 全跑
  int warmup = 3;
  int repeat = 10;
  bool do_timing = false;

  // 参数解析：按位置参数顺序填 B / S / version；--xxx 是 flag
  int positional_seen = 0;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
      warmup = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) {
      repeat = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--timing") == 0) {
      do_timing = true;
    } else if (argv[i][0] != '-') {
      switch (positional_seen) {
        case 0: B = atoi(argv[i]); break;
        case 1: S = atoi(argv[i]); break;
        case 2: version = atoi(argv[i]); break;
        default: break;
      }
      positional_seen++;
    }
  }

  printf("softmax_bin: B=%d S=%d\n", B, S);

  int N = B * S;
  size_t bytes = N * sizeof(float);

  // 初始化输入（固定种子 42，便于复现）
  std::vector<float> h_x(N), h_ref(N), h_out(N);
  srand(42);
  for (int i = 0; i < N; i++) {
    h_x[i] = ((float)rand() / RAND_MAX) * 4.f - 2.f;  // [-2, 2]
  }

  // CPU 参考
  cpu_softmax(h_x.data(), h_ref.data(), B, S);

  // 分配 GPU 内存
  float *d_x, *d_out, *d_exp_buf, *d_row_max, *d_row_sum;
  HIP_CHECK(hipMalloc(&d_x, bytes));
  HIP_CHECK(hipMalloc(&d_out, bytes));
  HIP_CHECK(hipMalloc(&d_exp_buf, bytes));
  HIP_CHECK(hipMalloc(&d_row_max, B * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_row_sum, B * sizeof(float)));

  HIP_CHECK(hipMemcpy(d_x, h_x.data(), bytes, hipMemcpyHostToDevice));

  // 确保 logs/ 目录存在（在 softmax_bin 的 CWD 下）
  // 注意：run_all.sh 在 chapter4/ 下执行，logs/ 由 run_all.sh 提前创建
  write_bin("logs/input.bin", h_x.data(), N);
  write_bin("logs/reference.bin", h_ref.data(), N);

  struct KernelDesc {
    int ver;
    const char* name;
    const char* bin_path;
  } kernels[] = {
    {0, "v0_naive",      "logs/output_v0.bin"},
    {1, "v1_stable",     "logs/output_v1.bin"},
    {2, "v2_merged_lds", "logs/output_v2.bin"},
    {3, "v3_vectorized", "logs/output_v3.bin"},
  };

  // 有效数据量（理论最优：读一次 x + 写一次 out）
  double data_gb = (double)bytes * 2.0 / 1e9;

  for (auto& k : kernels) {
    if (version >= 0 && version != k.ver) continue;

    HIP_CHECK(hipMemset(d_out, 0, bytes));

    // 计时（若 --timing 开启）
    double ms = 0.0;
    if (do_timing) {
      auto fn = [&]() {
        switch (k.ver) {
          case 0: run_softmax_v0(d_x, d_out, d_exp_buf, d_row_max, d_row_sum, B, S); break;
          case 1: run_softmax_v1(d_x, d_out, d_exp_buf, d_row_max, d_row_sum, B, S); break;
          case 2: run_softmax_v2(d_x, d_out, B, S); break;
          case 3: run_softmax_v3(d_x, d_out, B, S); break;
        }
      };
      ms = measure_ms(fn, warmup, repeat);
    } else {
      // 非计时模式：只跑一次
      HIP_CHECK(hipMemset(d_out, 0, bytes));
      switch (k.ver) {
        case 0: run_softmax_v0(d_x, d_out, d_exp_buf, d_row_max, d_row_sum, B, S); break;
        case 1: run_softmax_v1(d_x, d_out, d_exp_buf, d_row_max, d_row_sum, B, S); break;
        case 2: run_softmax_v2(d_x, d_out, B, S); break;
        case 3: run_softmax_v3(d_x, d_out, B, S); break;
      }
    }

    HIP_CHECK(hipMemcpy(h_out.data(), d_out, bytes, hipMemcpyDeviceToHost));
    write_bin(k.bin_path, h_out.data(), N);

    float err = max_abs_error(h_ref.data(), h_out.data(), N);
    if (do_timing) {
      double bw = data_gb / (ms / 1000.0);
      printf("  [%s] max_abs_error=%.2e  time=%.3f ms  bw=%.2f GB/s  %s\n",
             k.name, err, ms, bw, err < 1e-5f ? "PASS" : "WARN(large error)");
    } else {
      printf("  [%s] max_abs_error vs cpu_ref = %.2e  %s\n",
             k.name, err, err < 1e-5f ? "PASS" : "WARN(large error)");
    }
  }

  hipFree(d_x);
  hipFree(d_out);
  hipFree(d_exp_buf);
  hipFree(d_row_max);
  hipFree(d_row_sum);

  printf("Done. Binary outputs written to logs/\n");
  return 0;
}
