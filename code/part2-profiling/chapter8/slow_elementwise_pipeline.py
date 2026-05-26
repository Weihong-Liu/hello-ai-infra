import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark a slow elementwise pipeline on ROCm GPU.")
    parser.add_argument("--size", type=int, default=1 << 24)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--mode", choices=["baseline", "keep_gpu"], default="baseline")
    parser.add_argument("--profile", choices=["none", "torch"], default="none")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--trace-file", type=Path)
    return parser.parse_args()


def resolve_dtype(name):
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def percentile(values, q):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = (len(ordered) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(values):
    if not values:
        values = [0.0]
    return {
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "p95_ms": percentile(values, 0.95),
    }


def event_elapsed_ms(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end)


def elementwise_pipeline(x):
    y = x * 1.0001
    y = torch.sin(y)
    y = y + 0.25
    y = torch.tanh(y)
    y = y * y
    y = torch.sqrt(y + 1.0)
    y = torch.relu(y - 0.1)
    return y


def make_cpu_input(size, dtype):
    base = torch.linspace(0.0, 1.0, size, dtype=torch.float32)
    return base.to(dtype=dtype)


def run_baseline(cpu_input, warmup, repeat):
    for _ in range(warmup):
        x = cpu_input.to("cuda")
        y = elementwise_pipeline(x)
        _ = y.to("cpu")
        torch.cuda.synchronize()

    h2d_times = []
    gpu_times = []
    d2h_times = []
    total_times = []
    checksum = None

    for _ in range(repeat):
        total_start = time.perf_counter()
        x, h2d_ms = event_elapsed_ms(lambda: cpu_input.to("cuda"))
        y, gpu_ms = event_elapsed_ms(lambda: elementwise_pipeline(x))
        cpu_output, d2h_ms = event_elapsed_ms(lambda: y.to("cpu"))
        torch.cuda.synchronize()
        total_end = time.perf_counter()

        h2d_times.append(h2d_ms)
        gpu_times.append(gpu_ms)
        d2h_times.append(d2h_ms)
        total_times.append((total_end - total_start) * 1000)
        checksum = float(cpu_output.float().sum().item())

    return {
        "h2d_ms": h2d_times,
        "gpu_pipeline_ms": gpu_times,
        "d2h_ms": d2h_times,
        "total_ms": total_times,
        "checksum": checksum,
    }


def run_keep_gpu(cpu_input, warmup, repeat):
    x = cpu_input.to("cuda")
    torch.cuda.synchronize()

    for _ in range(warmup):
        y = elementwise_pipeline(x)
        torch.cuda.synchronize()

    gpu_times = []
    total_times = []
    y = None

    for _ in range(repeat):
        total_start = time.perf_counter()
        y, gpu_ms = event_elapsed_ms(lambda: elementwise_pipeline(x))
        torch.cuda.synchronize()
        total_end = time.perf_counter()

        gpu_times.append(gpu_ms)
        total_times.append((total_end - total_start) * 1000)

    cpu_output = y.to("cpu")
    torch.cuda.synchronize()
    checksum = float(cpu_output.float().sum().item())

    return {
        "h2d_ms": [0.0] * repeat,
        "gpu_pipeline_ms": gpu_times,
        "d2h_ms": [0.0] * repeat,
        "total_ms": total_times,
        "checksum": checksum,
    }


def build_summary(args, raw_times):
    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "hip_version": getattr(torch.version, "hip", None),
    }
    config = {
        "size": args.size,
        "dtype": args.dtype,
        "mode": args.mode,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "profile": args.profile,
    }
    return {
        "environment": env,
        "config": config,
        "summary": {
            "h2d": summarize(raw_times["h2d_ms"]),
            "gpu_pipeline": summarize(raw_times["gpu_pipeline_ms"]),
            "d2h": summarize(raw_times["d2h_ms"]),
            "total": summarize(raw_times["total_ms"]),
        },
        "checksum": raw_times["checksum"],
        "status": "PASS",
    }


def print_summary(result):
    env = result["environment"]
    config = result["config"]
    print(f"python: {env['python']}")
    print(f"torch: {env['torch']}")
    print(f"hip_version: {env['hip_version']}")
    print(f"cuda_available: {env['cuda_available']}")
    if env["device_name"]:
        print(f"device_name: {env['device_name']}")
    print(f"size: {config['size']}")
    print(f"dtype: {config['dtype']}")
    print(f"mode: {config['mode']}")
    print(f"warmup: {config['warmup']}")
    print(f"repeat: {config['repeat']}")

    for name, stats in result["summary"].items():
        print(f"{name}_mean_ms: {stats['mean_ms']:.6f}")
        print(f"{name}_median_ms: {stats['median_ms']:.6f}")
        print(f"{name}_min_ms: {stats['min_ms']:.6f}")
        print(f"{name}_p95_ms: {stats['p95_ms']:.6f}")

    print(f"checksum: {result['checksum']:.6f}")
    print(f"status: {result['status']}")


def run_once(args, cpu_input):
    if args.mode == "baseline":
        return run_baseline(cpu_input, args.warmup, args.repeat)
    return run_keep_gpu(cpu_input, args.warmup, args.repeat)


def run_with_torch_profiler(args, cpu_input):
    if args.trace_file is None:
        raise SystemExit("--trace-file is required when --profile torch")

    args.trace_file.parent.mkdir(parents=True, exist_ok=True)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        raw_times = run_once(args, cpu_input)

    prof.export_chrome_trace(str(args.trace_file))
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    return raw_times


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("PyTorch ROCm backend is not available")

    dtype = resolve_dtype(args.dtype)
    cpu_input = make_cpu_input(args.size, dtype)

    if args.profile == "torch":
        raw_times = run_with_torch_profiler(args, cpu_input)
    else:
        raw_times = run_once(args, cpu_input)

    result = build_summary(args, raw_times)
    print_summary(result)

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
