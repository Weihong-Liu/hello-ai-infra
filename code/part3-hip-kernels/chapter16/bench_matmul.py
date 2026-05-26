#!/usr/bin/env python3
"""
bench_matmul.py
形状扫描 + 版本对比 + TFLOPS 计算
输出同时写入 logs/bench_matmul.log

前提：
    - 已编译 ./matmul_bench（由 run_all.sh 负责）
    - 已激活 activate-rocm.sh（PyTorch ROCm 可用，用于 torch 对照计时）

用法：
    python bench_matmul.py
    python bench_matmul.py --shapes 512 1024 2048 4096
    python bench_matmul.py --warmup 5 --repeat 20
"""

import argparse
import os
import subprocess
import sys
import time

# 理论峰值：AI MAX 395 fp32，约 17.6 TFLOPS（文档值，仅供参考）
PEAK_TFLOPS_FP32 = 17.6

LOG_DIR = "logs"
LOG_PATH = os.path.join(LOG_DIR, "bench_matmul.log")


def tflops(M, N, K, time_ms):
    return 2.0 * M * N * K / (time_ms * 1e-3) / 1e12


def bench_hip_kernel(exe, version, M, N, K, warmup, repeat):
    """
    调用独立的 HIP benchmark 可执行文件，返回 avg_ms。
    可执行文件约定：matmul_bench <version> <M> <N> <K> <warmup> <repeat>
    输出最后一行格式：avg_ms=<float>
    """
    cmd = [exe, str(version), str(M), str(N), str(K), str(warmup), str(repeat)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            return None, result.stderr.strip()
        for line in result.stdout.splitlines()[::-1]:
            if line.startswith("avg_ms="):
                return float(line.split("=")[1]), None
        return None, f"未找到 avg_ms 行，输出：{result.stdout}"
    except FileNotFoundError:
        return None, f"{exe} 未找到"
    except subprocess.TimeoutExpired:
        return None, "超时"


def bench_rocblas(M, N, K, warmup, repeat):
    """调用 rocblas_baseline，返回 avg_ms"""
    cmd = ["./rocblas_baseline", str(M), str(N), str(K)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            return None, result.stderr.strip()
        # 格式："M=... N=... K=...  <time_ms>  <TFLOPS>"
        for line in result.stdout.splitlines():
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    avg_ms = float(parts[-2])
                    return avg_ms, None
                except ValueError:
                    continue
        return None, f"无法解析输出：{result.stdout}"
    except FileNotFoundError:
        return None, "./rocblas_baseline 未找到"
    except subprocess.TimeoutExpired:
        return None, "超时"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", nargs="+", type=int,
                        default=[512, 1024, 2048, 4096],
                        help="M=N=K 的取值列表（方阵）")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)

    versions = [
        (0, "v0_naive"),
        (1, "v1_lds"),
        (2, "v2_bk32"),
        (3, "v3_reg4"),
        (4, "v4_combined"),
    ]

    header = f"{'shape':>24}  {'version':>16}  {'time_ms':>9}  {'TFLOPS':>8}  {'peak%':>7}  {'vs_rocblas':>10}"
    sep    = "-" * len(header)

    lines = [header, sep]
    print(header)
    print(sep)

    for size in args.shapes:
        M = N = K = size
        # 先跑 rocBLAS 得到上限
        rb_ms, rb_err = bench_rocblas(M, N, K, args.warmup, args.repeat)
        if rb_ms is None:
            rb_tf = None
            rb_label = f"ERR({rb_err})"
        else:
            rb_tf = tflops(M, N, K, rb_ms)
            rb_label = f"{rb_tf:.3f}"

        for ver_id, ver_name in versions:
            avg_ms, err = bench_hip_kernel(
                "./matmul_bench", ver_id, M, N, K, args.warmup, args.repeat)
            if avg_ms is None:
                row = f"M=N=K={size:5d}  {ver_name:>16}  {'ERR':>9}  {'N/A':>8}  {'N/A':>7}  {'N/A':>10}"
            else:
                tf  = tflops(M, N, K, avg_ms)
                pct = tf / PEAK_TFLOPS_FP32 * 100.0
                vs  = (tf / rb_tf * 100.0) if rb_tf else float("nan")
                row = (f"M=N=K={size:5d}  {ver_name:>16}  {avg_ms:9.3f}  "
                       f"{tf:8.3f}  {pct:6.1f}%  {vs:9.1f}%")
            print(row)
            lines.append(row)

        # rocBLAS 行
        if rb_ms is not None:
            rb_row = (f"M=N=K={size:5d}  {'rocblas':>16}  {rb_ms:9.3f}  "
                      f"{rb_tf:8.3f}  {'---':>7}  {'100.0%':>10}")
        else:
            rb_row = (f"M=N=K={size:5d}  {'rocblas':>16}  {'---':>9}  "
                      f"{rb_label:>8}  {'---':>7}  {'100.0%':>10}")
        print(rb_row)
        lines.append(rb_row)
        print(sep)
        lines.append(sep)

    with open(LOG_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n结果已写入 {LOG_PATH}")


if __name__ == "__main__":
    main()
