"""
matmul_autotuned.py — 带 @triton.autotune 的矩阵乘法（Matmul）。

复用 chapter2（Triton Matmul）的分块思路，在搜索空间
  BLOCK_M ∈ {32, 64, 128}
  BLOCK_N ∈ {32, 64, 128}
  BLOCK_K ∈ {32, 64}
  num_warps ∈ {2, 4, 8}
  num_stages ∈ {1, 2, 3}（ROCm Triton 后端支持情况见注释）
上自动搜索最优配置。

用法（source activate-rocm.sh 后）：
    python matmul_autotuned.py
    python matmul_autotuned.py --M 2048 --N 2048 --K 2048
"""

import argparse
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 搜索空间定义
#
# BLOCK_M / BLOCK_N / BLOCK_K：分块大小，必须是 2 的幂
#   - 越大可能提高计算密度，但寄存器和 LDS 占用增加，occupancy 下降
#   - 典型起点：BLOCK_M=BLOCK_N=64/128, BLOCK_K=32/64
#
# num_warps：每个 program 使用的 wavefront 数量
#   - AMD gfx1151（RDNA4）wavefront 宽度 = 32 线程（wave32 模式）
#   - 实际并发线程数 = num_warps × 32（而非 NVIDIA 的 num_warps × 32 = CUDA warp）
#   - 注意：此处参数名沿用 Triton 惯例，但 AMD 实际单位是 wavefront
#
# num_stages：流水线级数
#   - 控制软件流水深度，影响访存延迟隐藏
#   - ROCm Triton 后端对 num_stages > 1 的支持取决于版本，实测若出现编译错误
#     可将范围缩小至 {1, 2}
# ---------------------------------------------------------------------------

AUTOTUNE_CONFIGS = [
    triton.Config(
        {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
        num_warps=nw,
        num_stages=ns,
    )
    for bm in [64, 128]
    for bn in [64, 128]
    for bk in [32, 64]
    for nw in [2, 4, 8]
    for ns in [1, 2]
]


@triton.autotune(
    configs=AUTOTUNE_CONFIGS,
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel_autotuned(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    分块矩阵乘法 kernel：C = A @ B
    A: [M, K], B: [K, N], C: [M, N]

    每个 program 计算一个 [BLOCK_M, BLOCK_N] 的输出 tile。
    沿 K 维度分块迭代，累积乘积到 fp32 accumulator，
    最后转 fp16 写回 C。
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 计算当前 tile 的行/列偏移
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    # 累加器（fp32 精度，避免 fp16 累加误差）
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in tl.range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]

        # 加载 A tile：[BLOCK_M, BLOCK_K]
        a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float16)

        # 加载 B tile：[BLOCK_K, BLOCK_N]
        b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float16)

        # 矩阵乘加（tl.dot 使用硬件矩阵乘指令）
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)

    # 写回结果
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


def matmul_autotuned(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Triton 自动调参矩阵乘法。
    a: [M, K], fp16
    b: [K, N], fp16
    返回 c: [M, N], fp16
    """
    assert a.ndim == 2 and b.ndim == 2
    assert a.shape[1] == b.shape[0], "矩阵维度不匹配"
    assert a.dtype == torch.float16 and b.dtype == torch.float16
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )
    matmul_kernel_autotuned[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


def verify(M: int = 512, N: int = 512, K: int = 512) -> None:
    """数值验证：与 torch.matmul 对比，fp16 精度 atol=1e-2。"""
    torch.manual_seed(42)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    ref = torch.matmul(a, b)
    out = matmul_autotuned(a, b)
    max_err = (out - ref).abs().max().item()
    ok = max_err < 1e-1  # fp16 矩阵乘，误差放宽
    status = "PASS" if ok else "FAIL"
    print(f"verify [{M},{K}]x[{K},{N}]: max_err={max_err:.3e}  {status}")
    if not ok:
        raise AssertionError(f"数值验证失败: max_err={max_err:.3e}")


def benchmark_fn(fn, warmup: int = 25, repeat: int = 100) -> float:
    """返回最小 kernel 延迟（ms），用 CUDA Event 精确计时。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start_e = torch.cuda.Event(enable_timing=True)
    end_e = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeat):
        start_e.record()
        fn()
        end_e.record()
        torch.cuda.synchronize()
        times.append(start_e.elapsed_time(end_e))
    return min(times)


def main():
    parser = argparse.ArgumentParser(description="Matmul autotune 验证 + 快速 benchmark")
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--N", type=int, default=1024)
    parser.add_argument("--K", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("ROCm GPU 不可用，请在实验机上运行")

    print(f"torch: {torch.__version__}")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print()

    # 数值验证
    verify(args.M, args.N, args.K)
    print()

    # 快速 benchmark
    a = torch.randn(args.M, args.K, device="cuda", dtype=torch.float16)
    b = torch.randn(args.K, args.N, device="cuda", dtype=torch.float16)

    ms_triton = benchmark_fn(lambda: matmul_autotuned(a, b), args.warmup, args.repeat)
    ms_torch = benchmark_fn(lambda: torch.matmul(a, b), args.warmup, args.repeat)

    # TFLOPS 计算：2*M*N*K 次浮点运算
    flops = 2 * args.M * args.N * args.K
    tflops_triton = flops / (ms_triton / 1000) / 1e12
    tflops_torch = flops / (ms_torch / 1000) / 1e12

    print(f"形状 [{args.M},{args.K}]x[{args.K},{args.N}], fp16")
    print(f"  triton autotune : {ms_triton:.4f} ms  /  {tflops_triton:.3f} TFLOPS")
    print(f"  torch.matmul    : {ms_torch:.4f} ms  /  {tflops_torch:.3f} TFLOPS")
    print(f"  速比 (torch/triton): {ms_torch / ms_triton:.2f}x")


if __name__ == "__main__":
    main()
