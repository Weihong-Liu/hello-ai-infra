"""
Benchmark：Triton Softmax v1 / v2 / PyTorch F.softmax 在不同形状下的延迟和带宽对比。

形状扫描：B ∈ {1, 8, 32} × S ∈ {512, 2048, 8192, 32768}
指标：延迟（ms）、有效带宽（GB/s）、vs torch 速比

可选：与 HIP Softmax 对比（若 code/part3-hip-kernels/chapter14/ 已编译）

输出写到：
  logs/bench_summary.log   — 汇总表
  logs/bench_raw.csv       — 原始数据（供后续分析）

用法：
  python bench_softmax.py
  python bench_softmax.py --warmup 10 --repeat 50
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from softmax_triton import softmax_v1, softmax_v1_tuned, softmax_v2

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# 本章形状矩阵
SHAPES = [
    (1, 512),
    (1, 2048),
    (1, 8192),
    (1, 32768),
    (8, 512),
    (8, 2048),
    (8, 8192),
    (8, 32768),
    (32, 512),
    (32, 2048),
    (32, 8192),
    (32, 32768),
]


def benchmark_fn(fn, warmup: int, repeat: int) -> float:
    """
    用 CUDA Event 精确测量 GPU kernel 执行时间，返回最小延迟（ms）。
    GPU kernel 是异步提交的，必须用 Event 而非 time.perf_counter()。
    """
    # Warmup：让 JIT / autotuner 完成编译
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


def compute_bandwidth(B: int, S: int, ms: float, passes: int = 2) -> float:
    """
    计算有效带宽（GB/s）。
    passes：1 表示 1 读 1 写（v1），2 表示 2 读 1 写（v2），
            PyTorch / v1 用 passes=2（1读1写 = 2×数据量）。
    实际公式：(passes × B × S × sizeof(fp32)) / (ms/1000) / 1e9
    """
    bytes_moved = passes * B * S * 4  # float32 = 4 bytes
    bw = bytes_moved / (ms / 1000.0) / 1e9
    return bw


def run_benchmark(warmup: int, repeat: int):
    if not torch.cuda.is_available():
        print("SKIP: ROCm GPU 不可用", flush=True)
        sys.exit(0)

    device_name = torch.cuda.get_device_name(0)
    print(f"torch: {torch.__version__}", flush=True)
    print(f"device: {device_name}", flush=True)
    print(f"warmup={warmup}, repeat={repeat}", flush=True)
    print("=" * 80, flush=True)

    # 表头
    header = f"{'Shape':16s} {'Version':12s} {'ms':>8s} {'GB/s':>8s} {'vs_torch':>10s}"
    print(header, flush=True)
    print("-" * 80, flush=True)

    rows = []  # for CSV output

    for B, S in SHAPES:
        x = torch.randn(B, S, device="cuda", dtype=torch.float32)
        shape_str = f"[{B},{S}]"

        # --- PyTorch baseline ---
        ms_torch = benchmark_fn(lambda: F.softmax(x, dim=-1), warmup, repeat)
        bw_torch = compute_bandwidth(B, S, ms_torch, passes=2)
        print(f"{shape_str:16s} {'torch':12s} {ms_torch:8.4f} {bw_torch:8.2f} {'1.00x':>10s}", flush=True)
        rows.append({
            "shape": shape_str, "B": B, "S": S,
            "version": "torch", "ms": ms_torch, "gb_s": bw_torch, "vs_torch": 1.0,
        })

        # --- Triton v1 ---
        try:
            ms_v1 = benchmark_fn(lambda: softmax_v1(x), warmup, repeat)
            bw_v1 = compute_bandwidth(B, S, ms_v1, passes=2)
            ratio_v1 = ms_torch / ms_v1
            print(f"{shape_str:16s} {'triton_v1':12s} {ms_v1:8.4f} {bw_v1:8.2f} {ratio_v1:>10.2f}x", flush=True)
            rows.append({
                "shape": shape_str, "B": B, "S": S,
                "version": "triton_v1", "ms": ms_v1, "gb_s": bw_v1, "vs_torch": ratio_v1,
            })
        except Exception as e:
            print(f"{shape_str:16s} {'triton_v1':12s} ERROR: {e}", flush=True)

        # --- Triton v1 autotune ---
        try:
            ms_v1t = benchmark_fn(lambda: softmax_v1_tuned(x), warmup, repeat)
            bw_v1t = compute_bandwidth(B, S, ms_v1t, passes=2)
            ratio_v1t = ms_torch / ms_v1t
            print(f"{shape_str:16s} {'triton_v1t':12s} {ms_v1t:8.4f} {bw_v1t:8.2f} {ratio_v1t:>10.2f}x", flush=True)
            rows.append({
                "shape": shape_str, "B": B, "S": S,
                "version": "triton_v1_tuned", "ms": ms_v1t, "gb_s": bw_v1t, "vs_torch": ratio_v1t,
            })
        except Exception as e:
            print(f"{shape_str:16s} {'triton_v1t':12s} ERROR: {e}", flush=True)

        # --- Triton v2（分块 fused，block_size=1024）---
        try:
            ms_v2 = benchmark_fn(lambda: softmax_v2(x, block_size=1024), warmup, repeat)
            # v2 是 2 读 1 写，分子是 3× 数据量
            bw_v2 = compute_bandwidth(B, S, ms_v2, passes=3)
            ratio_v2 = ms_torch / ms_v2
            print(f"{shape_str:16s} {'triton_v2':12s} {ms_v2:8.4f} {bw_v2:8.2f} {ratio_v2:>10.2f}x", flush=True)
            rows.append({
                "shape": shape_str, "B": B, "S": S,
                "version": "triton_v2", "ms": ms_v2, "gb_s": bw_v2, "vs_torch": ratio_v2,
            })
        except Exception as e:
            print(f"{shape_str:16s} {'triton_v2':12s} ERROR: {e}", flush=True)

        print("", flush=True)

    print("=" * 80, flush=True)
    print(f"device_name: {device_name}", flush=True)

    # 写 CSV
    csv_path = LOG_DIR / "bench_raw.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["shape", "B", "S", "version", "ms", "gb_s", "vs_torch"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"原始数据写入: {csv_path}", flush=True)

    # 写汇总日志
    summary_path = LOG_DIR / "bench_summary.log"
    with open(summary_path, "w") as f:
        f.write(f"torch: {torch.__version__}\n")
        f.write(f"device: {device_name}\n")
        f.write(f"warmup={warmup}, repeat={repeat}\n\n")
        f.write(header + "\n")
        f.write("-" * 80 + "\n")
        for row in rows:
            vs = f"{row['vs_torch']:.2f}x"
            f.write(f"{row['shape']:16s} {row['version']:12s} {row['ms']:8.4f} {row['gb_s']:8.2f} {vs:>10s}\n")
    print(f"汇总日志写入: {summary_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Triton Softmax benchmark")
    parser.add_argument("--warmup", type=int, default=25,
                        help="Warmup 次数（默认 25，让 JIT/autotune 完成编译）")
    parser.add_argument("--repeat", type=int, default=100,
                        help="正式计时重复次数（默认 100，取最小值）")
    args = parser.parse_args()

    run_benchmark(args.warmup, args.repeat)


if __name__ == "__main__":
    main()
