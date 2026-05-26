// layernorm.hip
// 教学版 LayerNorm：v0 三步分离 → v1 融合 → v2 融合 + float4 向量化
// 硬件目标：AI MAX 395 (gfx1151), ROCm 7.12.0
//
// 编译：
//   hipcc layernorm.hip -O2 -o layernorm
//
// 用法：
//   ./layernorm --version 0|1|2 --B 32 --S 128 --H 768
//   ./layernorm --version 2 --B 32 --S 512 --H 4096 --warmup 5 --repeat 20
//   ./layernorm --all   # 跑所有版本，默认形状

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <cassert>

// ─── HIP 错误检查宏 ──────────────────────────────────────────────────────────

#define HIP_CHECK(call)                                                         \
    do {                                                                        \
        hipError_t _e = (call);                                                 \
        if (_e != hipSuccess) {                                                 \
            fprintf(stderr, "[HIP ERROR] %s:%d  %s\n",                        \
                    __FILE__, __LINE__, hipGetErrorString(_e));                 \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// ─── 常量 ────────────────────────────────────────────────────────────────────

static constexpr float EPS = 1e-5f;

// ============================================================================
// v0：三步分离 — compute_mean / compute_var / normalize 各一个 kernel
// ============================================================================

// v0 阶段 1：对每行求均值
__global__ void ln_v0_mean(
    const float* __restrict__ input,   // [N_rows, H]
    float*       __restrict__ mean_out,// [N_rows]
    int H)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float sdata[];

    float local_sum = 0.0f;
    for (int i = tid; i < H; i += blockDim.x)
        local_sum += input[row * H + i];

    sdata[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0) mean_out[row] = sdata[0] / (float)H;
}

// v0 阶段 2：对每行求方差（需要已有 mean_out）
__global__ void ln_v0_var(
    const float* __restrict__ input,
    const float* __restrict__ mean_in, // [N_rows]
    float*       __restrict__ var_out, // [N_rows]
    int H)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float mu = mean_in[row];

    extern __shared__ float sdata[];

    float local_sq = 0.0f;
    for (int i = tid; i < H; i += blockDim.x) {
        float diff = input[row * H + i] - mu;
        local_sq += diff * diff;
    }

    sdata[tid] = local_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0) var_out[row] = sdata[0] / (float)H;
}

// v0 阶段 3：归一化 + γ/β
__global__ void ln_v0_norm(
    const float* __restrict__ input,
    const float* __restrict__ mean_in,
    const float* __restrict__ var_in,
    const float* __restrict__ weight,  // γ [H]
    const float* __restrict__ bias,    // β [H]
    float*       __restrict__ output,  // [N_rows, H]
    int H,
    float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float mu      = mean_in[row];
    float inv_std = rsqrtf(var_in[row] + eps);

    for (int i = tid; i < H; i += blockDim.x) {
        output[row * H + i] =
            weight[i] * (input[row * H + i] - mu) * inv_std + bias[i];
    }
}

// ============================================================================
// v1：融合 kernel — 单次 launch，LDS 内完成 mean + var + normalize
// ============================================================================

__global__ void layernorm_v1_fused(
    const float* __restrict__ input,   // [N_rows, H]
    const float* __restrict__ weight,  // [H]
    const float* __restrict__ bias,    // [H]
    float*       __restrict__ output,  // [N_rows, H]
    int H,
    float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    extern __shared__ float sdata[];

    const float* x = input  + row * H;
    float*       y = output + row * H;

    // 阶段 1：加载 + 局部求和 → mean
    float local_sum = 0.0f;
    for (int i = tid; i < H; i += blockDim.x)
        local_sum += x[i];

    sdata[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    float mean = sdata[0] / (float)H;
    __syncthreads();

    // 阶段 2：局部求 (x - mean)^2 之和 → var
    float local_sq = 0.0f;
    for (int i = tid; i < H; i += blockDim.x) {
        float diff = x[i] - mean;
        local_sq += diff * diff;
    }

    sdata[tid] = local_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    float var     = sdata[0] / (float)H;
    float inv_std = rsqrtf(var + eps);

    // 阶段 3：归一化 + γ/β → 写出
    for (int i = tid; i < H; i += blockDim.x)
        y[i] = weight[i] * (x[i] - mean) * inv_std + bias[i];
}

// ============================================================================
// v2：融合 + float4 向量化 — 要求 H 是 4 的倍数
// ============================================================================

__global__ void layernorm_v2_float4(
    const float* __restrict__ input,   // [N_rows, H]，H 须是 4 的倍数
    const float* __restrict__ weight,  // [H]
    const float* __restrict__ bias,    // [H]
    float*       __restrict__ output,  // [N_rows, H]
    int H,
    float eps)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int H4  = H / 4;

    extern __shared__ float sdata[];

    const float4* x4 = reinterpret_cast<const float4*>(input  + row * H);
    float4*       y4 = reinterpret_cast<float4*>(output + row * H);
    const float4* w4 = reinterpret_cast<const float4*>(weight);
    const float4* b4 = reinterpret_cast<const float4*>(bias);

    // 阶段 1：float4 加载 + 局部求和 → mean
    float local_sum = 0.0f;
    for (int i = tid; i < H4; i += blockDim.x) {
        float4 v = x4[i];
        local_sum += v.x + v.y + v.z + v.w;
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    float mean = sdata[0] / (float)H;
    __syncthreads();

    // 阶段 2：float4 加载 + 局部求 var
    float local_sq = 0.0f;
    for (int i = tid; i < H4; i += blockDim.x) {
        float4 v = x4[i];
        float d0 = v.x - mean, d1 = v.y - mean;
        float d2 = v.z - mean, d3 = v.w - mean;
        local_sq += d0*d0 + d1*d1 + d2*d2 + d3*d3;
    }

    sdata[tid] = local_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    float var     = sdata[0] / (float)H;
    float inv_std = rsqrtf(var + eps);

    // 阶段 3：float4 归一化写出
    for (int i = tid; i < H4; i += blockDim.x) {
        float4 xi = x4[i];
        float4 wi = w4[i];
        float4 bi = b4[i];
        float4 yi;
        yi.x = wi.x * (xi.x - mean) * inv_std + bi.x;
        yi.y = wi.y * (xi.y - mean) * inv_std + bi.y;
        yi.z = wi.z * (xi.z - mean) * inv_std + bi.z;
        yi.w = wi.w * (xi.w - mean) * inv_std + bi.w;
        y4[i] = yi;
    }
}

// ============================================================================
// CPU 参考实现（用于正确性对比）
// ============================================================================

void layernorm_cpu_ref(
    const float* input,
    const float* weight,
    const float* bias,
    float*       output,
    int N_rows, int H,
    float eps)
{
    for (int row = 0; row < N_rows; row++) {
        const float* x = input  + row * H;
        float*       y = output + row * H;

        double sum = 0.0;
        for (int i = 0; i < H; i++) sum += x[i];
        double mean = sum / H;

        double sq_sum = 0.0;
        for (int i = 0; i < H; i++) {
            double d = x[i] - mean;
            sq_sum += d * d;
        }
        double var     = sq_sum / H;
        double inv_std = 1.0 / sqrt(var + eps);

        for (int i = 0; i < H; i++)
            y[i] = (float)(weight[i] * ((double)x[i] - mean) * inv_std + bias[i]);
    }
}

// ============================================================================
// Host launcher 与计时工具
// ============================================================================

struct RunResult {
    double time_ms;
    double bandwidth_gbs;   // 有效带宽（读 input + 写 output，weight/bias 忽略）
};

// 返回相对误差（max |gpu - cpu| / (max |cpu| + 1e-8)）
double check_rel_err(const float* gpu_out, const float* cpu_ref, int N) {
    double max_diff = 0.0, max_val = 0.0;
    for (int i = 0; i < N; i++) {
        max_diff = std::max(max_diff, (double)std::abs(gpu_out[i] - cpu_ref[i]));
        max_val  = std::max(max_val,  (double)std::abs(cpu_ref[i]));
    }
    return max_diff / (max_val + 1e-8);
}

RunResult run_version(
    int version,
    const float* h_input,
    const float* h_weight,
    const float* h_bias,
    float* h_output,
    int N_rows, int H,
    int warmup, int repeat)
{
    long long total = (long long)N_rows * H;
    size_t row_bytes  = (size_t)N_rows * H * sizeof(float);
    size_t param_bytes = (size_t)H * sizeof(float);

    float *d_input = nullptr, *d_weight = nullptr, *d_bias = nullptr;
    float *d_output = nullptr;
    float *d_mean = nullptr, *d_var = nullptr;  // 仅 v0 使用

    HIP_CHECK(hipMalloc(&d_input,  row_bytes));
    HIP_CHECK(hipMalloc(&d_weight, param_bytes));
    HIP_CHECK(hipMalloc(&d_bias,   param_bytes));
    HIP_CHECK(hipMalloc(&d_output, row_bytes));

    HIP_CHECK(hipMemcpy(d_input,  h_input,  row_bytes,  hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_weight, h_weight, param_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_bias,   h_bias,   param_bytes, hipMemcpyHostToDevice));

    // blockDim：不超过 min(H, 1024)，向下取 2 的幂
    int block = 1;
    while (block * 2 <= H && block * 2 <= 1024) block *= 2;
    int shared_bytes = block * sizeof(float);

    if (version == 0) {
        HIP_CHECK(hipMalloc(&d_mean, N_rows * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_var,  N_rows * sizeof(float)));
    }

    auto launch_once = [&]() {
        if (version == 0) {
            ln_v0_mean<<<N_rows, block, shared_bytes>>>(d_input, d_mean, H);
            ln_v0_var <<<N_rows, block, shared_bytes>>>(d_input, d_mean, d_var, H);
            ln_v0_norm<<<N_rows, block>>>(
                d_input, d_mean, d_var, d_weight, d_bias, d_output, H, EPS);
        } else if (version == 1) {
            layernorm_v1_fused<<<N_rows, block, shared_bytes>>>(
                d_input, d_weight, d_bias, d_output, H, EPS);
        } else if (version == 2) {
            assert(H % 4 == 0 && "v2 要求 H 是 4 的倍数");
            layernorm_v2_float4<<<N_rows, block, shared_bytes>>>(
                d_input, d_weight, d_bias, d_output, H, EPS);
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

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / repeat;

    // 有效带宽 = (读 input + 写 output) / 时间
    // 忽略 weight/bias（通常被 L2 缓存）
    double data_gb = 2.0 * total * sizeof(float) / 1e9;
    double bw_gbs  = data_gb / (ms / 1000.0);

    HIP_CHECK(hipMemcpy(h_output, d_output, row_bytes, hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(d_input));
    HIP_CHECK(hipFree(d_weight));
    HIP_CHECK(hipFree(d_bias));
    HIP_CHECK(hipFree(d_output));
    if (d_mean) HIP_CHECK(hipFree(d_mean));
    if (d_var)  HIP_CHECK(hipFree(d_var));

    return {ms, bw_gbs};
}

// ============================================================================
// main：命令行解析 + 主流程
// ============================================================================

static void usage(const char* prog) {
    printf("用法：%s [选项]\n", prog);
    printf("  --version N     选择 kernel 版本（0/1/2）\n");
    printf("  --all           测试所有版本\n");
    printf("  --B N           Batch size（默认 32）\n");
    printf("  --S N           Sequence length（默认 128）\n");
    printf("  --H N           Hidden size（默认 768）\n");
    printf("  --warmup N      预热次数（默认 3）\n");
    printf("  --repeat N      计时重复次数（默认 20）\n");
    printf("  --no-verify     跳过 CPU 对照（仅测速）\n");
    printf("  --help          显示此帮助\n");
}

int main(int argc, char** argv) {
    int version   = 1;
    bool run_all  = false;
    bool verify   = true;
    int B         = 32;
    int S         = 128;
    int H         = 768;
    int warmup    = 3;
    int repeat    = 20;

    for (int i = 1; i < argc; i++) {
        if      (strcmp(argv[i], "--version") == 0) version   = atoi(argv[++i]);
        else if (strcmp(argv[i], "--all")     == 0) run_all   = true;
        else if (strcmp(argv[i], "--B")       == 0) B         = atoi(argv[++i]);
        else if (strcmp(argv[i], "--S")       == 0) S         = atoi(argv[++i]);
        else if (strcmp(argv[i], "--H")       == 0) H         = atoi(argv[++i]);
        else if (strcmp(argv[i], "--warmup")  == 0) warmup    = atoi(argv[++i]);
        else if (strcmp(argv[i], "--repeat")  == 0) repeat    = atoi(argv[++i]);
        else if (strcmp(argv[i], "--no-verify") == 0) verify  = false;
        else if (strcmp(argv[i], "--help")    == 0) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "未知选项：%s\n", argv[i]); usage(argv[0]); return 1; }
    }

    int N_rows  = B * S;
    long long total = (long long)N_rows * H;

    printf("=== LayerNorm 教学 Benchmark ===\n");
    printf("形状：B=%d  S=%d  H=%d  N_rows=%d\n", B, S, H, N_rows);
    printf("warmup=%d  repeat=%d\n\n", warmup, repeat);

    // 初始化随机输入
    srand(42);
    float* h_input  = new float[total];
    float* h_weight = new float[H];
    float* h_bias   = new float[H];
    float* h_output = new float[total];
    float* h_ref    = new float[total];

    for (long long i = 0; i < total; i++)
        h_input[i]  = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    for (int i = 0; i < H; i++) {
        h_weight[i] = (float)rand() / RAND_MAX * 0.5f + 0.75f;  // γ 约在 0.75~1.25
        h_bias[i]   = (float)rand() / RAND_MAX * 0.2f - 0.1f;   // β 约在 -0.1~0.1
    }

    // CPU 参考
    if (verify) {
        layernorm_cpu_ref(h_input, h_weight, h_bias, h_ref, N_rows, H, EPS);
    }

    // 确定要跑哪些版本
    int versions[3] = {0, 1, 2};
    int n_versions  = run_all ? 3 : 1;
    if (!run_all) versions[0] = version;

    printf("%-6s  %-12s  %-10s  %-14s  %-10s\n",
           "版本", "kernel", "耗时(ms)", "带宽(GB/s)", "相对误差");
    printf("%s\n", std::string(60, '-').c_str());

    const char* names[] = {"v0 三步分离", "v1 融合", "v2 融合+float4"};

    for (int vi = 0; vi < n_versions; vi++) {
        int v = versions[vi];

        if (v == 2 && H % 4 != 0) {
            printf("v%-5d  %-12s  %-10s  %-14s  SKIP (H不是4的倍数)\n",
                   v, names[v], "--", "--");
            continue;
        }

        RunResult res = run_version(
            v, h_input, h_weight, h_bias, h_output,
            N_rows, H, warmup, repeat);

        double rel_err = -1.0;
        if (verify) rel_err = check_rel_err(h_output, h_ref, total);

        if (verify) {
            printf("v%-5d  %-12s  %10.4f  %14.3f  %10.2e\n",
                   v, names[v], res.time_ms, res.bandwidth_gbs, rel_err);
        } else {
            printf("v%-5d  %-12s  %10.4f  %14.3f  (跳过验证)\n",
                   v, names[v], res.time_ms, res.bandwidth_gbs);
        }
        fflush(stdout);
    }

    delete[] h_input;
    delete[] h_weight;
    delete[] h_bias;
    delete[] h_output;
    delete[] h_ref;

    return 0;
}
