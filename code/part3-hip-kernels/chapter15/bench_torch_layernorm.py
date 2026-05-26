"""bench_torch_layernorm.py — torch.nn.LayerNorm baseline at the same shapes
used by the HIP v0/v1/v2 benchmarks. Reports per-shape kernel-only ms and the
effective bandwidth (3 × N × 4 bytes / time) computed exactly the same way as
bench_layernorm.py so the numbers are directly comparable.

Usage (after `source ../activate-rocm.sh`):
    python bench_torch_layernorm.py
    python bench_torch_layernorm.py --warmup 25 --repeat 100
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def bench_one(B: int, S: int, H: int, warmup: int, repeat: int) -> dict:
    x = torch.randn(B, S, H, device="cuda", dtype=torch.float32)
    ln = torch.nn.LayerNorm(H, eps=1e-5).cuda()
    for _ in range(warmup):
        _ = ln(x)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeat):
        start.record()
        _ = ln(x)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    t_ms = min(times)
    n_bytes = B * S * H * 4 * 3.0
    bw = n_bytes / (t_ms / 1000.0) / 1e9
    return {"B": B, "S": S, "H": H, "time_ms": t_ms, "bw_gbs": bw}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=50)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("ROCm GPU 不可用")

    print(f"torch: {torch.__version__}  device: {torch.cuda.get_device_name(0)}")
    shapes = [(32, 128, 768), (32, 128, 1024), (32, 512, 4096)]

    rows = []
    for B, S, H in shapes:
        r = bench_one(B, S, H, args.warmup, args.repeat)
        rows.append(r)
        print(f"  torch.nn.LayerNorm  B={B} S={S} H={H}  time_ms={r['time_ms']:.4f}  bw={r['bw_gbs']:.2f} GB/s")

    csv_path = LOG_DIR / "bench_torch_layernorm.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["B", "S", "H", "time_ms", "bw_gbs"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n写入: {csv_path}")


if __name__ == "__main__":
    main()
