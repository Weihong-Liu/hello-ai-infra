#!/usr/bin/env python3
"""bench_vector_add.py
Part 3 Chapter 2: Vector Add — 跨 kernel × size × block_size 扫描 benchmark

用法（已 source activate-rocm.sh 环境后，在 chapter2/ 目录执行）：
    python3 bench_vector_add.py

或通过 run_all.sh 调用。

输出：
  - 终端打印表格
  - logs/bench_summary.csv     — 全量结果（每行一组配置）
  - logs/bench_blocksize.csv   — block size 扫描结果（固定 N=16M）
  - logs/bench_size.csv        — 输入规模扫描结果（固定 block=256）
"""

import subprocess
import csv
import os
import sys
import time
from pathlib import Path

# ─── 配置 ──────────────────────────────────────────────────────────────────

BINARY      = "./build/vector_add_bench"
KERNELS     = ["naive", "grid_stride", "strided_bad"]
SIZES_M     = [1, 4, 16, 64]           # 输入规模（单位：M 个元素）
BLOCK_SIZES = [64, 128, 256, 512, 1024]  # block size 扫描范围
DEFAULT_N_M = 16                         # block size 扫描时固定的 N（16M）
DEFAULT_BLK = 256                        # size 扫描时固定的 block size
WARMUP      = 5
REPEAT      = 30
LOG_DIR     = Path("logs")

KERNEL_LABELS = {
    "naive":       "naive（一线程一元素）",
    "grid_stride": "grid_stride（循环展开）",
    "strided_bad": "strided_bad（非合并反例）",
}

# ─── 辅助函数 ──────────────────────────────────────────────────────────────


def n_from_m(m: int) -> int:
    return m * 1024 * 1024


def run_single(kernel: str, n: int, block: int) -> dict:
    """运行一组 (kernel, n, block) 配置，解析输出返回指标字典。"""
    cmd = [
        BINARY,
        "--kernel", kernel,
        "--size",   str(n),
        "--block",  str(block),
        "--warmup", str(WARMUP),
        "--repeat", str(REPEAT),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
    except FileNotFoundError:
        print(
            f"[ERROR] Binary not found: {BINARY}\n"
            "        Run 'bash run_all.sh' or 'hipcc vector_add.hip -O2 -o build/vector_add_bench' first."
        )
        sys.exit(1)
    except subprocess.TimeoutExpired:
        return {
            "kernel": kernel, "n_m": n // (1024 * 1024), "block": block,
            "mean_ms": None, "min_ms": None, "bw_gbs": None, "correct": None,
            "error": "TIMEOUT",
        }

    if result.returncode not in (0, 1):  # returncode=1 表示正确性失败，也记录
        print(f"[WARN] kernel={kernel} n={n} block={block} exit={result.returncode}")
        if result.stderr:
            print(result.stderr[:500])

    # 解析 key: value 格式的输出（见 vector_add.hip main 函数）
    kv: dict = {}
    for line in result.stdout.splitlines():
        if ": " in line or ":" in line:
            parts = line.strip().split(":", 1)
            if len(parts) == 2:
                kv[parts[0].strip()] = parts[1].strip()

    def safe_float(key):
        try:
            return float(kv[key])
        except (KeyError, ValueError):
            return None

    return {
        "kernel":   kernel,
        "n_m":      n // (1024 * 1024),
        "block":    block,
        "mean_ms":  safe_float("mean_ms"),
        "min_ms":   safe_float("min_ms"),
        "bw_gbs":   safe_float("bandwidth_gb_s"),
        "correct":  kv.get("correct", "unknown"),
        "error":    None if result.returncode in (0, 1) else f"EXIT={result.returncode}",
    }


def print_bw_table(rows: list, row_key: str, col_key: str, col_vals: list,
                   row_label_fn=None, title: str = "带宽对比 (GB/s)"):
    """通用表格打印：row_key × col_key → bw_gbs"""
    bw_table: dict = {}
    for r in rows:
        rk = r[row_key]
        ck = r[col_key]
        if rk not in bw_table:
            bw_table[rk] = {}
        bw_table[rk][ck] = r.get("bw_gbs")

    col_w = 14
    row_keys_ordered = list(dict.fromkeys(r[row_key] for r in rows))

    header = f"{'':22}" + "".join(
        f"{(str(col_key) + '=' + str(v)):>{col_w}}" for v in col_vals
    )
    sep = "=" * (22 + col_w * len(col_vals))
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(header)
    print("-" * (22 + col_w * len(col_vals)))

    for rk in row_keys_ordered:
        label = row_label_fn(rk) if row_label_fn else str(rk)
        row = f"{label:<22}"
        for cv in col_vals:
            bw = bw_table.get(rk, {}).get(cv)
            row += f"{bw:>{col_w}.3f}" if bw is not None else f"{'N/A':>{col_w}}"
        print(row)

    print(sep + "\n")


def save_csv(rows: list, path: Path, fieldnames: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] 结果已保存至 {path}")


# ─── main ──────────────────────────────────────────────────────────────────


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(BINARY):
        print(
            f"[ERROR] {BINARY} 不存在。\n"
            "        请先运行：hipcc vector_add.hip -O2 -o build/vector_add_bench"
        )
        sys.exit(1)

    all_rows: list = []

    # ── 扫描 1：block size 扫描（N=16M，三个 kernel × 5 种 block size）────────
    blocksize_rows: list = []
    n_fixed = n_from_m(DEFAULT_N_M)
    total = len(KERNELS) * len(BLOCK_SIZES)
    done = 0

    print(f"[INFO] 扫描 1/2：block size 扫描（N={DEFAULT_N_M}M，block ∈ {BLOCK_SIZES}）")
    for kernel in KERNELS:
        for blk in BLOCK_SIZES:
            done += 1
            label = KERNEL_LABELS.get(kernel, kernel)
            print(f"  [{done:>2}/{total}] {label}, block={blk} ... ", end="", flush=True)
            t0 = time.monotonic()
            row = run_single(kernel, n_fixed, blk)
            elapsed = time.monotonic() - t0
            if row.get("error"):
                print(f"FAIL ({row['error']})")
            else:
                bw = row.get("bw_gbs") or 0
                ms = row.get("min_ms") or 0
                print(f"{ms:.4f} ms  {bw:.3f} GB/s  [{elapsed:.1f}s]")
            blocksize_rows.append(row)
            all_rows.append(row)

    print_bw_table(
        blocksize_rows, "kernel", "block", BLOCK_SIZES,
        row_label_fn=lambda k: KERNEL_LABELS.get(k, k)[:22],
        title=f"block size 扫描 — N={DEFAULT_N_M}M，带宽 (GB/s) @ AI MAX 395"
    )
    save_csv(blocksize_rows, LOG_DIR / "bench_blocksize.csv",
             ["kernel", "n_m", "block", "mean_ms", "min_ms", "bw_gbs", "correct", "error"])

    # ── 扫描 2：输入规模扫描（block=256，三个 kernel × 4 种 N）──────────────
    size_rows: list = []
    total = len(KERNELS) * len(SIZES_M)
    done = 0

    print(f"\n[INFO] 扫描 2/2：输入规模扫描（block={DEFAULT_BLK}，N ∈ {SIZES_M}M）")
    for kernel in KERNELS:
        for m in SIZES_M:
            done += 1
            n = n_from_m(m)
            label = KERNEL_LABELS.get(kernel, kernel)
            print(f"  [{done:>2}/{total}] {label}, N={m}M ... ", end="", flush=True)
            t0 = time.monotonic()
            row = run_single(kernel, n, DEFAULT_BLK)
            elapsed = time.monotonic() - t0
            if row.get("error"):
                print(f"FAIL ({row['error']})")
            else:
                bw = row.get("bw_gbs") or 0
                ms = row.get("min_ms") or 0
                print(f"{ms:.4f} ms  {bw:.3f} GB/s  [{elapsed:.1f}s]")
            size_rows.append(row)
            # 避免重复 N=16M block=256（已在 blocksize_rows 里了）
            if not (m == DEFAULT_N_M and DEFAULT_BLK == DEFAULT_BLK):
                all_rows.append(row)
            else:
                all_rows.append(row)

    print_bw_table(
        size_rows, "kernel", "n_m", SIZES_M,
        row_label_fn=lambda k: KERNEL_LABELS.get(k, k)[:22],
        title=f"输入规模扫描 — block={DEFAULT_BLK}，带宽 (GB/s) @ AI MAX 395"
    )
    save_csv(size_rows, LOG_DIR / "bench_size.csv",
             ["kernel", "n_m", "block", "mean_ms", "min_ms", "bw_gbs", "correct", "error"])

    # ── 汇总 CSV ────────────────────────────────────────────────────────────
    save_csv(all_rows, LOG_DIR / "bench_summary.csv",
             ["kernel", "n_m", "block", "mean_ms", "min_ms", "bw_gbs", "correct", "error"])


if __name__ == "__main__":
    main()
