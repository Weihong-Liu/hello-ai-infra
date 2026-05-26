"""Triton vector-copy bandwidth benchmark for chapter 7.5 Roofline plotting.

Sweeps a few sizes (footprint) so the reader can see where DRAM/MALL/L2
transitions sit on AI MAX 395. For each n it measures effective bandwidth.

Usage:
    python bench_triton_copy.py --output-json logs/triton_copy.json
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
from pathlib import Path

import torch
import triton
import triton.language as tl


@triton.jit
def copy_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=mask), mask=mask)


def bench_one(n, repeats=200, warmup=20, block=1024):
    x = torch.empty(n, dtype=torch.float32, device="cuda")
    y = torch.empty_like(x)
    grid = ((n + block - 1) // block,)

    for _ in range(warmup):
        copy_kernel[grid](x, y, n, BLOCK=block)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for i in range(repeats):
        starts[i].record()
        copy_kernel[grid](x, y, n, BLOCK=block)
        ends[i].record()
    torch.cuda.synchronize()
    times_ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]

    bytes_per_iter = 2 * n * 4  # read + write fp32
    median_s = statistics.median(times_ms) * 1e-3
    eff_bw = bytes_per_iter / median_s
    return {
        "n": n,
        "footprint_mib": (2 * n * 4) / (1 << 20),
        "median_ms": statistics.median(times_ms),
        "min_ms": min(times_ms),
        "mean_ms": statistics.mean(times_ms),
        "eff_bw_GBs": eff_bw / 1e9,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--output-json", type=Path)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("ROCm/CUDA torch backend unavailable")

    sizes = [
        1 << 16,   # 256 KiB pair (L2-resident)
        1 << 18,   # 1 MiB
        1 << 20,   # 4 MiB
        1 << 22,   # 16 MiB pair (around MALL)
        1 << 24,   # 64 MiB pair (DRAM territory)
        1 << 26,   # 256 MiB pair (DRAM, well above MALL)
    ]
    results = [bench_one(n, args.repeats, args.warmup) for n in sizes]

    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": getattr(__import__("triton"), "__version__", "?"),
        "hip_version": getattr(torch.version, "hip", None),
        "device_name": torch.cuda.get_device_name(0),
    }
    out = {"environment": env, "results": results}
    print(json.dumps(out, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
