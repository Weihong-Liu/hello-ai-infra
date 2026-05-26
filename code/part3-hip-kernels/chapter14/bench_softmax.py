#!/usr/bin/env python3
"""bench_softmax.py — 不同 (B, S) 下各 softmax 版本的吞吐扫描。

调用 ./build/softmax_bin <B> <S> <ver> --timing --warmup 3 --repeat 10，
解析 stdout 中的 time 和 bw 字段；若解析失败则退化为 wall-clock 计时。

结果写到：
  logs/bench_stdout.log    — 全量终端输出
  logs/bench_summary.csv   — 每 (B, S, ver) 一行，含 min_ms / mean_ms / bw_GBps / max_abs_error
"""
from __future__ import annotations

import csv
import math
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
BIN = HERE / "build" / "softmax_bin"
LOG = HERE / "logs"
LOG.mkdir(exist_ok=True)

# 测试形状矩阵
SHAPES = [
    (1,   512),
    (8,   512),
    (8,  1024),
    (8,  2048),
    (8,  4096),
    (16, 2048),
    (16, 4096),
    (32, 2048),
    (32, 4096),
    (32, 8192),
]
VERSIONS = [0, 1, 2, 3]
WARMUP = 3
REPEAT = 10


def run_one(b: int, s: int, ver: int) -> dict:
    """运行单次实验，返回统计字典。"""
    cmd = [
        str(BIN), str(b), str(s), str(ver),
        "--warmup", str(WARMUP),
        "--repeat", str(REPEAT),
        "--timing",
    ]

    times_ms: list[float] = []
    bw_gbs: list[float] = []
    max_abs_error: float | None = None

    # 1 次 warmup run（不计时）
    subprocess.run(cmd, capture_output=True, check=False)

    # REPEAT 次计时 run
    for _ in range(3):  # 外层重复 3 次取最佳，减少操作系统噪声
        t0 = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True, text=True, check=False)
        t1 = time.perf_counter()
        wall_ms = (t1 - t0) * 1000.0

        # 解析 softmax_bin 输出的 time= 和 bw= 字段
        for line in r.stdout.splitlines():
            if f"v{ver}" not in line and str(ver) not in line:
                continue
            m_time = re.search(r"time=([\d.]+)\s*ms", line)
            m_bw   = re.search(r"bw=([\d.]+)\s*GB/s", line)
            m_err  = re.search(r"max_abs_error=([\d.eE+-]+)", line)
            if m_time:
                times_ms.append(float(m_time.group(1)))
            if m_bw:
                bw_gbs.append(float(m_bw.group(1)))
            if m_err and max_abs_error is None:
                max_abs_error = float(m_err.group(1))

        if not times_ms:
            # 解析失败：退化到 wall-clock
            times_ms.append(wall_ms)

    return {
        "B": b,
        "S": s,
        "ver": ver,
        "min_ms":  round(min(times_ms), 4),
        "mean_ms": round(statistics.mean(times_ms), 4),
        "med_ms":  round(statistics.median(times_ms), 4),
        "bw_GBps": round(statistics.mean(bw_gbs), 3) if bw_gbs else float("nan"),
        "max_abs_error": max_abs_error if max_abs_error is not None else float("nan"),
    }


def fmt_bw(v: float) -> str:
    return f"{v:8.2f}" if math.isfinite(v) else "     nan"


def main() -> int:
    if not BIN.exists():
        print(f"[ERROR] {BIN} 不存在，请先执行 run_all.sh 编译")
        return 1

    rows: list[dict] = []
    header = (f"{'B':>4} {'S':>6} {'ver':>3} {'min_ms':>9} {'mean_ms':>9} "
              f"{'bw_GBps':>9} {'max_abs_err':>13}")
    sep = "-" * len(header)
    print(header)
    print(sep)

    for b, s in SHAPES:
        for v in VERSIONS:
            # v3 要求 S % 4 == 0；如不满足则跳过
            if v == 3 and s % 4 != 0:
                continue
            try:
                r = run_one(b, s, v)
            except Exception as exc:
                print(f"  [WARN] B={b} S={s} ver={v} 异常：{exc}", file=sys.stderr)
                continue

            rows.append(r)
            err = r["max_abs_error"]
            err_str = f"{err:.3e}" if math.isfinite(err) else "    nan"
            print(f"{r['B']:>4} {r['S']:>6} {r['ver']:>3} "
                  f"{r['min_ms']:>9.3f} {r['mean_ms']:>9.3f} "
                  f"{fmt_bw(r['bw_GBps'])} {err_str:>13}")

    if not rows:
        print("[ERROR] 未收集到任何结果")
        return 1

    csv_path = LOG / "bench_summary.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(sep)
    print(f"\n结果已写入 {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
