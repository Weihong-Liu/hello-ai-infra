#!/usr/bin/env python3
"""bench_reduction.py
Part 3 Chapter 3: Reduction 优化 — 跨 version × size 扫描 benchmark

用法（已 source activate-rocm.sh 环境后，在 chapter3/ 目录执行）：
    python bench_reduction.py

或通过 run_all.sh 调用。

输出：
  - 终端打印表格
  - logs/bench_summary.csv
"""

import subprocess
import csv
import os
import sys
import time
from pathlib import Path

# ─── 配置 ──────────────────────────────────────────────────────────────────

BINARY      = "./reduction"               # 编译后的可执行文件路径
VERSIONS    = [0, 1, 2, 3, 4]            # 要测试的 kernel 版本
SIZES_M     = [4, 16, 64, 256]           # 输入大小（单位 M 个元素）
BLOCK_SIZE  = 256
WARMUP      = 5
REPEAT      = 20
LOG_DIR     = Path("logs")

# AI MAX 395 理论峰值内存带宽（占位，实测后填入）
# 单位：GB/s；可通过 rocm-bandwidth-test 或 stream benchmark 测量
THEORETICAL_BW_GBS = None  # 🚧 待 job:p3c3-reduction 填充

VERSION_LABELS = {
    0: "v0 atomicAdd",
    1: "v1 LDS tree",
    2: "v2 wavefront",
    3: "v3 unrolled",
    4: "v4 two-pass",
}

# ─── 辅助函数 ──────────────────────────────────────────────────────────────

def n_from_m(m: int) -> int:
    """M 个元素 → 实际元素数（1M = 1024*1024）"""
    return m * 1024 * 1024


def run_single(version: int, n: int) -> dict:
    """运行一个 (version, n) 组合，解析输出并返回指标字典。"""
    cmd = [
        BINARY,
        "--version", str(version),
        "--n",       str(n),
        "--block",   str(BLOCK_SIZE),
        "--warmup",  str(WARMUP),
        "--repeat",  str(REPEAT),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
    except FileNotFoundError:
        print(f"[ERROR] Binary not found: {BINARY}. Run 'bash run_all.sh' first.")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        return {"version": version, "n_m": n // (1024 * 1024),
                "time_ms": None, "bw_gbs": None, "error": "TIMEOUT"}

    if result.returncode != 0:
        print(f"[WARN] version={version} n={n} exit={result.returncode}")
        print(result.stderr[:500])
        return {"version": version, "n_m": n // (1024 * 1024),
                "time_ms": None, "bw_gbs": None, "error": "FAILED"}

    # 解析最后一行输出（格式见 reduction.hip main 函数）
    # 示例：v1  time=0.4231 ms  bandwidth=159.123 GB/s  result=16777216.00  rel_err=0.00e+00
    time_ms  = None
    bw_gbs   = None
    rel_err  = None

    for line in result.stdout.splitlines():
        line = line.strip()
        if not line.startswith(f"v{version}"):
            continue
        parts = line.split()
        for p in parts:
            if p.startswith("time="):
                try:
                    time_ms = float(p.split("=")[1].rstrip("ms "))
                except ValueError:
                    pass
            elif p.startswith("bandwidth="):
                try:
                    bw_gbs = float(p.split("=")[1].rstrip("GB/s "))
                except ValueError:
                    pass
            elif p.startswith("rel_err="):
                try:
                    rel_err = float(p.split("=")[1])
                except ValueError:
                    pass

    return {
        "version":  version,
        "label":    VERSION_LABELS.get(version, f"v{version}"),
        "n_m":      n // (1024 * 1024),
        "time_ms":  time_ms,
        "bw_gbs":   bw_gbs,
        "rel_err":  rel_err,
        "error":    None,
    }


def print_table(rows: list, sizes_m: list):
    """按 version 为行、size 为列，打印带宽对比表。"""
    # 汇总：bw_table[version][size_m] = bw_gbs
    bw_table: dict = {}
    for r in rows:
        v = r["version"]
        s = r["n_m"]
        if v not in bw_table:
            bw_table[v] = {}
        bw_table[v][s] = r.get("bw_gbs")

    col_w = 14
    header = f"{'版本':<18}" + "".join(f"{'N=' + str(s) + 'M':>{col_w}}" for s in sizes_m)
    if THEORETICAL_BW_GBS:
        header += f"  (理论峰值: {THEORETICAL_BW_GBS:.1f} GB/s)"
    print("\n" + "=" * (18 + col_w * len(sizes_m)))
    print(f"  带宽对比表 (GB/s) — AI MAX 395, block={BLOCK_SIZE}")
    print("=" * (18 + col_w * len(sizes_m)))
    print(header)
    print("-" * (18 + col_w * len(sizes_m)))

    for v in VERSIONS:
        label = VERSION_LABELS.get(v, f"v{v}")
        row = f"{label:<18}"
        for s in sizes_m:
            bw = bw_table.get(v, {}).get(s)
            if bw is not None:
                row += f"{bw:>{col_w}.3f}"
            else:
                row += f"{'N/A':>{col_w}}"
        print(row)

    print("=" * (18 + col_w * len(sizes_m)) + "\n")


def save_csv(rows: list, path: Path):
    """把所有结果保存为 CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["version", "label", "n_m", "time_ms", "bw_gbs",
                  "rel_err", "error"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] 结果已保存至 {path}")


# ─── main ──────────────────────────────────────────────────────────────────

def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(BINARY):
        print(f"[ERROR] {BINARY} 不存在，请先编译：hipcc reduction.hip -O2 -o reduction")
        sys.exit(1)

    all_rows = []
    total = len(VERSIONS) * len(SIZES_M)
    done  = 0

    print(f"[INFO] 开始 benchmark：{len(VERSIONS)} 个版本 × {len(SIZES_M)} 个大小 = {total} 组")
    print(f"[INFO] block={BLOCK_SIZE}, warmup={WARMUP}, repeat={REPEAT}\n")

    for size_m in SIZES_M:
        n = n_from_m(size_m)
        for v in VERSIONS:
            done += 1
            print(f"[{done:>2}/{total}] version={v}  N={size_m}M ({n:,}) ... ", end="", flush=True)
            t_start = time.monotonic()
            row = run_single(v, n)
            elapsed = time.monotonic() - t_start

            if row.get("error"):
                print(f"FAIL ({row['error']})")
            else:
                bw = row.get("bw_gbs", 0) or 0
                ms = row.get("time_ms", 0) or 0
                pct = f"({bw / THEORETICAL_BW_GBS * 100:.1f}% peak)" \
                      if THEORETICAL_BW_GBS and bw else ""
                print(f"{ms:.4f} ms  {bw:.3f} GB/s {pct}  [{elapsed:.1f}s total]")

            all_rows.append(row)

    # 打印汇总表
    print_table(all_rows, SIZES_M)

    # 保存 CSV
    save_csv(all_rows, LOG_DIR / "bench_summary.csv")


if __name__ == "__main__":
    main()
