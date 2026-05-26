"""PyTorch element-wise op benchmark for chapter 7.5 / Roofline FLOPS plotting.

Usage example:
    python bench_torch_op.py --shape 4096,4096 --dtype fp32 --repeats 200

Hardware: AI MAX 395 + ROCm 7.12.0 (gfx1151).

For FLOPS estimation we use a fused multiply-add expression `y = a * x + b`
(addcmul-like) which performs 2 FLOP per fp element of the output tensor.
We also include a `torch.softmax` measurement (memory-bound oriented) for the
report. The script prints mean / median / min / p95 / std for each config
and writes a JSON summary to the path given by --output-json.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    idx = (len(ordered) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    w = idx - lo
    return ordered[lo] * (1 - w) + ordered[hi] * w


def summarize(values):
    if not values:
        values = [0.0]
    return {
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "p95_ms": percentile(values, 0.95),
        "std_ms": statistics.pstdev(values),
    }


def bench_addcmul(shape, dtype, repeats=200, warmup=20):
    """y = a*x + b: 2 FLOP per fp element, 3 reads + 1 write per element."""
    a = torch.randn(*shape, dtype=dtype, device="cuda")
    x = torch.randn(*shape, dtype=dtype, device="cuda")
    b = torch.randn(*shape, dtype=dtype, device="cuda")
    y = torch.empty_like(a)

    for _ in range(warmup):
        torch.add(b, a * x, out=y)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for i in range(repeats):
        starts[i].record()
        torch.add(b, a * x, out=y)
        ends[i].record()
    torch.cuda.synchronize()

    times_ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    n = a.numel()
    elem_size = a.element_size()
    flops_per_iter = 2 * n  # 1 mul + 1 add per element
    bytes_per_iter = 4 * n * elem_size  # 3 reads + 1 write
    return times_ms, flops_per_iter, bytes_per_iter


def bench_softmax(shape, dtype, repeats=200, warmup=20):
    x = torch.randn(*shape, dtype=dtype, device="cuda")
    for _ in range(warmup):
        torch.softmax(x, dim=-1)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for i in range(repeats):
        starts[i].record()
        torch.softmax(x, dim=-1)
        ends[i].record()
    torch.cuda.synchronize()
    times_ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return times_ms


def bench_matmul(shape, dtype, repeats=50, warmup=10):
    """Square-ish matmul: 2*M*N*K FLOP for matmul-FMA accounting."""
    M, N = shape
    K = N
    a = torch.randn(M, K, dtype=dtype, device="cuda")
    b = torch.randn(K, N, dtype=dtype, device="cuda")
    for _ in range(warmup):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    for i in range(repeats):
        starts[i].record()
        torch.matmul(a, b)
        ends[i].record()
    torch.cuda.synchronize()
    times_ms = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    flops_per_iter = 2 * M * N * K
    elem_size = a.element_size()
    bytes_per_iter = elem_size * (M * K + K * N + M * N)  # one read each + one write
    return times_ms, flops_per_iter, bytes_per_iter


DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shape", default="4096,4096")
    p.add_argument("--dtype", default="fp32", choices=list(DTYPE_MAP.keys()))
    p.add_argument("--repeats", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--op", default="addcmul",
                   choices=["addcmul", "softmax", "matmul"])
    p.add_argument("--output-json", type=Path)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("ROCm/CUDA torch backend unavailable")

    shape = tuple(int(x) for x in args.shape.split(","))
    dtype = DTYPE_MAP[args.dtype]

    if args.op == "addcmul":
        times, flops_per_iter, bytes_per_iter = bench_addcmul(
            shape, dtype, args.repeats, args.warmup)
    elif args.op == "softmax":
        times = bench_softmax(shape, dtype, args.repeats, args.warmup)
        flops_per_iter = None
        # softmax: ~5 ops per element (max, sub, exp, sum, div) = 5N; reads N, writes N
        # but we keep BW reporting only, FLOPS estimate is approximate
        bytes_per_iter = 2 * shape[0] * shape[1] * dtype.itemsize if hasattr(dtype, "itemsize") else None
    elif args.op == "matmul":
        times, flops_per_iter, bytes_per_iter = bench_matmul(
            shape, dtype, args.repeats, args.warmup)

    stats = summarize(times)

    median_s = stats["median_ms"] * 1e-3
    achieved_flops = (flops_per_iter / median_s) if flops_per_iter else None
    achieved_bw = (bytes_per_iter / median_s) if bytes_per_iter else None
    arith_intensity = (flops_per_iter / bytes_per_iter) if (flops_per_iter and bytes_per_iter) else None

    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "hip_version": getattr(torch.version, "hip", None),
        "device_name": torch.cuda.get_device_name(0),
    }
    out = {
        "environment": env,
        "config": {
            "op": args.op,
            "shape": list(shape),
            "dtype": args.dtype,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "stats_ms": stats,
        "flops_per_iter": flops_per_iter,
        "bytes_per_iter": bytes_per_iter,
        "achieved_flops": achieved_flops,
        "achieved_bw_bytes_per_s": achieved_bw,
        "arithmetic_intensity_flop_per_byte": arith_intensity,
    }

    print(json.dumps(out, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
