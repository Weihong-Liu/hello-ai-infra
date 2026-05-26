#!/usr/bin/env python3
"""
bench_layernorm.py
Part 3 Chapter 5: LayerNorm 优化 — 版本 × 形状扫描 benchmark

形状矩阵：
  H in {768, 1024, 4096}（BERT-base / BERT-large / LLaMA-7B）
  B×S in {32×128=4096, 32×512=16384}

输出：
  - 终端表格（ms + GB/s）
  - logs/bench_layernorm.csv

用法（已激活 activate-rocm.sh，在 chapter5/ 目录执行）：
    python bench_layernorm.py
    python bench_layernorm.py --warmup 5 --repeat 30
    python bench_layernorm.py --H 768 1024 4096 --block 256 512 1024
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

BINARY   = "./layernorm"
LOG_DIR  = Path("logs")
LOG_CSV  = LOG_DIR / "bench_layernorm.csv"

VERSION_LABELS = {
    0: "v0 三步分离",
    1: "v1 融合",
    2: "v2 float4",
}

# 形状配置：(B, S, H) 三元组
DEFAULT_SHAPES = [
    (32, 128,  768),   # BERT-base
    (32, 128, 1024),   # BERT-large
    (32, 512, 4096),   # LLaMA-7B
]


def bench_one(version: int, B: int, S: int, H: int,
              warmup: int, repeat: int) -> dict:
    """
    调用 ./layernorm --no-verify 计时，解析输出行。
    输出格式（来自 layernorm.hip main）：
      v<N>  <name>  <time_ms>  <bw_gbs>  (跳过验证)
    """
    cmd = [
        BINARY,
        "--version", str(version),
        "--B", str(B),
        "--S", str(S),
        "--H", str(H),
        "--warmup", str(warmup),
        "--repeat", str(repeat),
        "--no-verify",
    ]
    base = {"version": version, "label": VERSION_LABELS.get(version, f"v{version}"),
            "B": B, "S": S, "H": H,
            "time_ms": None, "bw_gbs": None, "error": None}

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except FileNotFoundError:
        base["error"] = f"{BINARY} 未找到"
        return base
    except subprocess.TimeoutExpired:
        base["error"] = "TIMEOUT"
        return base

    if result.returncode != 0:
        base["error"] = f"exit={result.returncode}"
        return base

    # 解析输出：找到 v<version> 开头的行
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line.startswith(f"v{version}"):
            continue
        parts = line.split()
        # 格式：v<N>  <label_col1>  <label_col2>  <time_ms>  <bw_gbs>  <...>
        # 由于 label 可能含空格，采用从右往左解析数字的方式
        nums = []
        for p in parts:
            try:
                nums.append(float(p))
            except ValueError:
                pass
        if len(nums) >= 2:
            base["time_ms"] = nums[0]
            base["bw_gbs"]  = nums[1]
        break

    return base


def print_table(rows: list, H_list: list, shapes: list):
    """按版本为行、(B,S,H) 形状为列打印带宽表格。"""
    # 索引：key = (version, B, S, H)
    table = {}
    for r in rows:
        key = (r["version"], r["B"], r["S"], r["H"])
        table[key] = r

    col_w = 18
    shape_labels = [f"B{b}S{s}H{h}" for (b, s, h) in shapes]
    header = f"{'版本':<14}" + "".join(f"{lbl:>{col_w}}" for lbl in shape_labels)
    sep    = "-" * len(header)

    print("\n" + "=" * len(header))
    print(f"  LayerNorm 带宽对比表 (GB/s) — AI MAX 395 + ROCm 7.12.0")
    print("=" * len(header))
    print(header)
    print(sep)

    for v in sorted(VERSION_LABELS.keys()):
        label = VERSION_LABELS[v]
        row_str = f"{label:<14}"
        for (b, s, h) in shapes:
            r = table.get((v, b, s, h))
            if r is None:
                row_str += f"{'N/A':>{col_w}}"
            elif r.get("error"):
                row_str += f"{'ERR':>{col_w}}"
            elif r.get("bw_gbs") is not None:
                row_str += f"{r['bw_gbs']:>{col_w}.2f}"
            else:
                row_str += f"{'N/A':>{col_w}}"
        print(row_str)

    print("=" * len(header))
    print("单位：GB/s\n")


def save_csv(rows: list):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["version", "label", "B", "S", "H", "time_ms", "bw_gbs", "error"]
    with open(LOG_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] 结果已保存至 {LOG_CSV}")


def main():
    parser = argparse.ArgumentParser(description="LayerNorm 版本 × 形状 Benchmark")
    parser.add_argument("--H",       nargs="+", type=int, default=[768, 1024, 4096])
    parser.add_argument("--warmup",  type=int, default=3)
    parser.add_argument("--repeat",  type=int, default=20)
    parser.add_argument("--versions",nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(BINARY):
        print(f"[ERROR] {BINARY} 不存在，请先编译：hipcc layernorm.hip -O2 -o layernorm")
        sys.exit(1)

    # 构建形状列表
    shapes = []
    for h in args.H:
        if h in (768, 1024):
            shapes.append((32, 128, h))
        else:
            shapes.append((32, 512, h))

    total = len(args.versions) * len(shapes)
    all_rows = []
    done = 0

    print(f"[INFO] 开始 benchmark：{len(args.versions)} 版本 × {len(shapes)} 形状 = {total} 组")
    print(f"[INFO] warmup={args.warmup}, repeat={args.repeat}\n")

    for (B, S, H) in shapes:
        for v in args.versions:
            done += 1
            label = VERSION_LABELS.get(v, f"v{v}")
            print(f"[{done:>2}/{total}] v{v} ({label})  B={B} S={S} H={H} ... ",
                  end="", flush=True)

            if v == 2 and H % 4 != 0:
                print("SKIP (H 不是 4 的倍数)")
                all_rows.append({
                    "version": v, "label": label,
                    "B": B, "S": S, "H": H,
                    "time_ms": None, "bw_gbs": None, "error": "H%4!=0"
                })
                continue

            t_wall = time.monotonic()
            r = bench_one(v, B, S, H, args.warmup, args.repeat)
            elapsed = time.monotonic() - t_wall

            if r.get("error"):
                print(f"FAIL ({r['error']})")
            elif r.get("bw_gbs") is not None:
                print(f"{r['time_ms']:.4f} ms  {r['bw_gbs']:.2f} GB/s  [{elapsed:.1f}s]")
            else:
                print(f"时间={r.get('time_ms')}  带宽={r.get('bw_gbs')}  [{elapsed:.1f}s]")

            all_rows.append(r)

    H_list = [h for (_, _, h) in shapes]
    print_table(all_rows, H_list, shapes)
    save_csv(all_rows)


if __name__ == "__main__":
    main()
