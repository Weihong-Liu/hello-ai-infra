"""
bench_matmul.py — 形状扫描 + torch.matmul vs Triton naive vs Triton grouped 对比
对应文档：docs/part4-triton/chapter18/index.md §18.6
"""

import argparse
import csv
import sys
import torch
from matmul_triton import matmul_naive, matmul_grouped


def benchmark_torch_matmul(M, N, K, dtype=torch.float32, warmup=25, repeat=100):
    """用 CUDA Event 精确计时 torch.matmul，返回最小耗时（ms）。"""
    A = torch.randn(M, K, device="cuda", dtype=dtype)
    B = torch.randn(K, N, device="cuda", dtype=dtype)

    # warmup：让 JIT / rocBLAS 内部的 heuristic 选定 kernel
    for _ in range(warmup):
        _ = torch.matmul(A, B)
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        torch.matmul(A, B)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # ms

    ms = min(times)
    tflops = 2 * M * N * K / (ms * 1e-3) / 1e12
    return ms, tflops


def benchmark_triton(fn, M, N, K, dtype=torch.float32, warmup=25, repeat=100):
    """计时给定 Triton 函数，返回最小耗时（ms）。"""
    A = torch.randn(M, K, device="cuda", dtype=dtype)
    B = torch.randn(K, N, device="cuda", dtype=dtype)
    if dtype == torch.float16:
        A = A.to(torch.float16)
        B = B.to(torch.float16)

    # warmup（触发 JIT 编译 + kernel 选择）
    for _ in range(warmup):
        _ = fn(A, B)
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        fn(A, B)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    ms = min(times)
    tflops = 2 * M * N * K / (ms * 1e-3) / 1e12
    return ms, tflops


def run_benchmarks(warmup=25, repeat=100):
    sizes = [512, 1024, 2048, 4096]
    dtypes = [("fp32", torch.float32), ("fp16", torch.float16)]

    # tile 组合扫描（M=N=K=2048, fp32）
    tile_shapes = [
        (32, 32, 32),
        (64, 64, 32),
        (128, 128, 32),
        (64, 64, 64),
        (128, 64, 32),
    ]

    print("=" * 80)
    print("Tile 大小扫描（M=N=K=2048, fp32）")
    print("=" * 80)
    print(f"{'BLOCK_M':>8} {'BLOCK_N':>8} {'BLOCK_K':>8} {'TFLOPS':>10} {'ms':>10}")
    print("-" * 50)
    M = N = K = 2048
    tile_results = []
    for bm, bn, bk in tile_shapes:
        try:
            A = torch.randn(M, K, device="cuda", dtype=torch.float32)
            B = torch.randn(K, N, device="cuda", dtype=torch.float32)
            from matmul_triton import matmul_naive_kernel
            import triton

            def run_tile(A, B, BM=bm, BN=bn, BK=bk):
                C = torch.empty((M, N), device=A.device, dtype=torch.float32)
                grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
                matmul_naive_kernel[grid](
                    A, B, C, M, N, K,
                    A.stride(0), A.stride(1),
                    B.stride(0), B.stride(1),
                    C.stride(0), C.stride(1),
                    BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
                )
                return C

            for _ in range(warmup):
                run_tile(A, B)
            torch.cuda.synchronize()
            times = []
            for _ in range(repeat):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                run_tile(A, B)
                e.record()
                torch.cuda.synchronize()
                times.append(s.elapsed_time(e))
            ms_val = min(times)
            tf = 2 * M * N * K / (ms_val * 1e-3) / 1e12
            print(f"{bm:>8} {bn:>8} {bk:>8} {tf:>10.3f} {ms_val:>10.3f}")
            tile_results.append((bm, bn, bk, tf, ms_val))
        except Exception as ex:
            print(f"{bm:>8} {bn:>8} {bk:>8}   ERROR: {ex}")

    print()

    # 主 benchmark：形状 × dtype × 实现
    all_results = []
    for dtype_name, dtype in dtypes:
        print("=" * 80)
        print(f"fp={dtype_name}  (warmup={warmup}, repeat={repeat}, 取 min)")
        print("=" * 80)
        print(f"{'M=N=K':>6}  {'rocBLAS(TFLOPS)':>16} {'naive(TFLOPS)':>14} {'grouped(TFLOPS)':>16}  "
              f"{'naive/roc':>10} {'grp/roc':>8}")
        print("-" * 80)
        for sz in sizes:
            M = N = K = sz
            torch_ms, torch_tf = benchmark_torch_matmul(M, N, K, dtype=dtype,
                                                        warmup=warmup, repeat=repeat)

            # Triton naive 和 grouped 只支持 fp32 直接运行，fp16 需要 torch 转换
            # 注意：matmul_naive/grouped 内部累加用 fp32，输入可以是 fp16
            A16 = torch.randn(M, K, device="cuda", dtype=dtype)
            B16 = torch.randn(K, N, device="cuda", dtype=dtype)

            try:
                naive_ms, naive_tf = benchmark_triton(matmul_naive, M, N, K,
                                                      dtype=dtype, warmup=warmup, repeat=repeat)
            except Exception as ex:
                print(f"  naive ERROR: {ex}")
                naive_ms, naive_tf = float("nan"), float("nan")

            try:
                grouped_ms, grouped_tf = benchmark_triton(matmul_grouped, M, N, K,
                                                          dtype=dtype, warmup=warmup, repeat=repeat)
            except Exception as ex:
                print(f"  grouped ERROR: {ex}")
                grouped_ms, grouped_tf = float("nan"), float("nan")

            naive_ratio   = naive_tf   / torch_tf if torch_tf > 0 else float("nan")
            grouped_ratio = grouped_tf / torch_tf if torch_tf > 0 else float("nan")

            print(f"{sz:>6}  {torch_tf:>16.3f} {naive_tf:>14.3f} {grouped_tf:>16.3f}  "
                  f"{naive_ratio:>10.3f} {grouped_ratio:>8.3f}")

            all_results.append({
                "dtype": dtype_name, "M": M, "N": N, "K": K,
                "rocblas_tflops": torch_tf, "rocblas_ms": torch_ms,
                "naive_tflops": naive_tf, "naive_ms": naive_ms,
                "grouped_tflops": grouped_tf, "grouped_ms": grouped_ms,
                "naive_vs_rocblas": naive_ratio, "grouped_vs_rocblas": grouped_ratio,
            })
        print()

    return all_results, tile_results


def save_csv(all_results, path="logs/bench_matmul.csv"):
    if not all_results:
        return
    keys = list(all_results[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"CSV 已写入 {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()

    import os
    os.makedirs("logs", exist_ok=True)

    all_results, tile_results = run_benchmarks(warmup=args.warmup, repeat=args.repeat)
    save_csv(all_results, "logs/bench_matmul.csv")
    print("Benchmark 完成 ✓")
