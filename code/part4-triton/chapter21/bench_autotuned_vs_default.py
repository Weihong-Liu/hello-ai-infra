"""
bench_autotuned_vs_default.py — 对比默认固定 config 与 autotune 后的性能。

对 Softmax 和 Matmul 各自做两组对比：
  - Softmax：固定 BLOCK_SIZE=1024 vs autotune（自动选择 BLOCK_SIZE + num_warps）
  - Matmul：固定 BLOCK_M=BLOCK_N=BLOCK_K=64 vs autotune

扫描形状：
  Softmax: B=8, S ∈ {512, 1024, 2048, 4096, 8192}
  Matmul:  {512², 1024², 2048²}, fp16

输出：
  logs/bench_softmax_vs_default.csv
  logs/bench_matmul_vs_default.csv
  logs/bench_summary.log       — 人类可读汇总

用法（source activate-rocm.sh 后）：
    python bench_autotuned_vs_default.py
    python bench_autotuned_vs_default.py --warmup 10 --repeat 50
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from softmax_autotuned import softmax_autotuned
from matmul_autotuned import matmul_autotuned

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 基线：固定参数的 Softmax kernel（不用 autotune）
# ---------------------------------------------------------------------------

@triton.jit
def softmax_kernel_fixed(
    x_ptr, y_ptr,
    B, S,
    stride_xb, stride_xs,
    stride_yb, stride_ys,
    BLOCK_SIZE: tl.constexpr,
):
    """固定 BLOCK_SIZE 的 Softmax，用于与 autotune 版本对比。"""
    pid = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE)
    x_ptrs = x_ptr + pid * stride_xb + cols * stride_xs
    mask = cols < S
    x = tl.load(x_ptrs, mask=mask, other=-float("inf"))
    x_max = tl.max(x, axis=0)
    x = x - x_max
    x_exp = tl.exp(x)
    x_sum = tl.sum(x_exp, axis=0)
    y = x_exp / x_sum
    y_ptrs = y_ptr + pid * stride_yb + cols * stride_ys
    tl.store(y_ptrs, y, mask=mask)


def softmax_fixed(x: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    """固定 BLOCK_SIZE 的 Softmax 包装函数。"""
    assert x.ndim == 2
    B, S = x.shape
    y = torch.empty_like(x)
    BLOCK_SIZE = max(block_size, triton.next_power_of_2(S))
    softmax_kernel_fixed[(B,)](
        x, y, B, S,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


# ---------------------------------------------------------------------------
# 基线：固定参数的 Matmul kernel（不用 autotune）
# ---------------------------------------------------------------------------

@triton.jit
def matmul_kernel_fixed(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """固定 BLOCK_M/N/K 的 Matmul，用于与 autotune 版本对比。"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in tl.range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        a = tl.load(A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0).to(tl.float16)
        b = tl.load(B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                    mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0).to(tl.float16)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)


def matmul_fixed(a: torch.Tensor, b: torch.Tensor,
                 bm: int = 64, bn: int = 64, bk: int = 32) -> torch.Tensor:
    """固定 block size 的 Matmul 包装函数。"""
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = (triton.cdiv(M, bm), triton.cdiv(N, bn))
    matmul_kernel_fixed[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
    )
    return c


# ---------------------------------------------------------------------------
# 计时工具
# ---------------------------------------------------------------------------

def benchmark_fn(fn, warmup: int, repeat: int) -> float:
    """返回最小延迟（ms）。"""
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


# ---------------------------------------------------------------------------
# Softmax benchmark
# ---------------------------------------------------------------------------

def bench_softmax(warmup: int, repeat: int):
    B = 8
    S_values = [512, 1024, 2048, 4096, 8192]

    print("=== Softmax: 固定 BLOCK_SIZE vs autotune ===")
    header = f"{'形状':12s} {'fixed(ms)':>12s} {'auto(ms)':>12s} {'GB/s_fixed':>12s} {'GB/s_auto':>12s} {'提速比':>8s}"
    print(header)
    print("-" * 80)

    rows = []
    for S in S_values:
        x = torch.randn(B, S, device="cuda", dtype=torch.float32)

        ms_fixed = benchmark_fn(lambda: softmax_fixed(x), warmup, repeat)
        ms_auto = benchmark_fn(lambda: softmax_autotuned(x), warmup, repeat)

        bw = lambda ms: 2 * B * S * 4 / (ms / 1000) / 1e9
        ratio = ms_fixed / ms_auto

        shape_str = f"[{B},{S}]"
        print(f"{shape_str:12s} {ms_fixed:12.4f} {ms_auto:12.4f} "
              f"{bw(ms_fixed):12.2f} {bw(ms_auto):12.2f} {ratio:8.2f}x")
        rows.append({
            "shape": shape_str, "B": B, "S": S,
            "fixed_ms": ms_fixed, "auto_ms": ms_auto,
            "fixed_gbs": bw(ms_fixed), "auto_gbs": bw(ms_auto),
            "speedup": ratio,
        })

    # 写 CSV
    csv_path = LOG_DIR / "bench_softmax_vs_default.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n结果写入: {csv_path}")
    return rows


# ---------------------------------------------------------------------------
# Matmul benchmark
# ---------------------------------------------------------------------------

def bench_matmul(warmup: int, repeat: int):
    shapes = [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]

    print("=== Matmul: 固定 BLOCK vs autotune ===")
    header = f"{'形状':16s} {'fixed(ms)':>12s} {'auto(ms)':>12s} {'TFLOPS_fixed':>14s} {'TFLOPS_auto':>14s} {'提速比':>8s}"
    print(header)
    print("-" * 90)

    rows = []
    for M, N, K in shapes:
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)

        ms_fixed = benchmark_fn(lambda: matmul_fixed(a, b), warmup, repeat)
        ms_auto = benchmark_fn(lambda: matmul_autotuned(a, b), warmup, repeat)

        flops = 2 * M * N * K
        tf = lambda ms: flops / (ms / 1000) / 1e12
        ratio = ms_fixed / ms_auto

        shape_str = f"[{M},{N},{K}]"
        print(f"{shape_str:16s} {ms_fixed:12.4f} {ms_auto:12.4f} "
              f"{tf(ms_fixed):14.3f} {tf(ms_auto):14.3f} {ratio:8.2f}x")
        rows.append({
            "shape": shape_str, "M": M, "N": N, "K": K,
            "fixed_ms": ms_fixed, "auto_ms": ms_auto,
            "fixed_tflops": tf(ms_fixed), "auto_tflops": tf(ms_auto),
            "speedup": ratio,
        })

    csv_path = LOG_DIR / "bench_matmul_vs_default.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n结果写入: {csv_path}")
    return rows


# ---------------------------------------------------------------------------
# 汇总日志
# ---------------------------------------------------------------------------

def write_summary(softmax_rows, matmul_rows):
    import torch
    summary_path = LOG_DIR / "bench_summary.log"
    with open(summary_path, "w") as f:
        f.write(f"device: {torch.cuda.get_device_name(0)}\n")
        f.write(f"torch: {torch.__version__}\n\n")

        f.write("=== Softmax: 固定 BLOCK_SIZE vs autotune ===\n")
        f.write(f"{'形状':12s} {'fixed_ms':>10s} {'auto_ms':>10s} {'提速比':>8s}\n")
        for r in softmax_rows:
            f.write(f"{r['shape']:12s} {r['fixed_ms']:10.4f} {r['auto_ms']:10.4f} {r['speedup']:8.2f}x\n")

        f.write("\n=== Matmul: 固定 BLOCK vs autotune ===\n")
        f.write(f"{'形状':16s} {'fixed_ms':>10s} {'auto_ms':>10s} {'提速比':>8s}\n")
        for r in matmul_rows:
            f.write(f"{r['shape']:16s} {r['fixed_ms']:10.4f} {r['auto_ms']:10.4f} {r['speedup']:8.2f}x\n")

    print(f"\n汇总日志写入: {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("ROCm GPU 不可用，请在实验机上运行")

    print(f"torch: {torch.__version__}")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print()

    sm_rows = bench_softmax(args.warmup, args.repeat)
    print()
    mm_rows = bench_matmul(args.warmup, args.repeat)
    print()
    write_summary(sm_rows, mm_rows)


if __name__ == "__main__":
    main()
