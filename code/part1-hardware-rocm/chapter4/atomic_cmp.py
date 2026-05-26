#!/usr/bin/env python
# code/part1-hardware-rocm/chapter4/atomic_cmp.py
#
# §4.7.4 atomic 收尾代价对比 micro-benchmark。
#
# 对照三种"每线程加 1 到某个全局位置"的写法，看 atomic 频率/争用对吞吐
# 的影响：
#   (a) atomic_full     —— 所有 thread 都 atomicAdd 到同一个地址（最坏情况，
#       wave 里 64 个 lane 全部串行写一个 counter）；
#   (b) atomic_bucketed —— 把 thread 哈希到 1024 个桶，每个桶上做 atomicAdd
#       （轻冲突；模拟"先 reduce 再 atomic 提交"的现实写法）；
#   (c) plain_store     —— 普通非原子写入到 thread 自己的 slot（无冲突 baseline，
#       表征"理论上限"，每 thread 一次 store 不需要 RMW 也不需要 serialization）。
#
# 三种 kernel 的工作量都是"每个 thread 写一次"：n_threads 一致，差别只在
# atomic 频率与争用模式。所以 ops/s 的差距直接等于 atomic 的相对代价。
#
# 用法：
#   cd code/part1-hardware-rocm
#   source ./activate-rocm.sh
#   python chapter5/atomic_cmp.py
# 可选参数：
#   --sizes 65536,262144,1048576,4194304    # 扫多个 n_threads
#   --repeats 100  --warmup 10  --buckets 1024
#
# 硬件上下文：AI MAX 395 (gfx1151) + ROCm 7.12.0（torch 2.9.1+rocm7.12.0,
# triton 3.5.1+rocm7.12.0），与 §4.7.1–§4.7.3 一致。
#
# 输出：先打印 timestamp + rocm-smi 头，然后一行一种 mode 一行一个 size，
# CSV-friendly：mode, n_threads, time_ms, ops_per_s

import argparse
import datetime
import os
import shutil
import subprocess
import sys

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------


@triton.jit
def atomic_full_kernel(out_ptr, n, BLOCK: tl.constexpr):
    """每 thread atomicAdd(out[0], 1) —— 全部冲突到同一地址。"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    one = tl.where(mask, 1, 0)
    # atomic_add 接受 ptr + value 向量；同一地址重复 → 硬件序列化。
    tl.atomic_add(out_ptr + tl.zeros_like(offs), one, mask=mask)


@triton.jit
def atomic_bucketed_kernel(out_ptr, n, BUCKETS: tl.constexpr, BLOCK: tl.constexpr):
    """每 thread atomicAdd(out[tid % BUCKETS], 1) —— 分散到 BUCKETS 个桶。"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    bucket = offs % BUCKETS
    one = tl.where(mask, 1, 0)
    tl.atomic_add(out_ptr + bucket, one, mask=mask)


@triton.jit
def plain_store_kernel(out_ptr, n, BLOCK: tl.constexpr):
    """每 thread out[tid] = 1 —— 普通 store，无原子、无冲突。"""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.full([BLOCK], 1, tl.int32), mask=mask)


# ---------------------------------------------------------------------------
# Bench harness
# ---------------------------------------------------------------------------


def _bench(launch, n_threads: int, repeats: int, warmup: int) -> float:
    """通用 hipEvent 计时；launch 是 callable，无参；返回 total ms over `repeats`."""
    torch.cuda.synchronize()
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(repeats):
        launch()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e)


def run_atomic_full(n_threads: int, repeats: int, warmup: int, block: int):
    out = torch.zeros(1, dtype=torch.int32, device="cuda")
    grid = ((n_threads + block - 1) // block,)

    def launch():
        atomic_full_kernel[grid](out, n_threads, BLOCK=block)

    ms = _bench(launch, n_threads, repeats, warmup)
    return ms


def run_atomic_bucketed(n_threads: int, repeats: int, warmup: int, block: int, buckets: int):
    out = torch.zeros(buckets, dtype=torch.int32, device="cuda")
    grid = ((n_threads + block - 1) // block,)

    def launch():
        atomic_bucketed_kernel[grid](out, n_threads, BUCKETS=buckets, BLOCK=block)

    ms = _bench(launch, n_threads, repeats, warmup)
    return ms


def run_plain_store(n_threads: int, repeats: int, warmup: int, block: int):
    out = torch.zeros(n_threads, dtype=torch.int32, device="cuda")
    grid = ((n_threads + block - 1) // block,)

    def launch():
        plain_store_kernel[grid](out, n_threads, BLOCK=block)

    ms = _bench(launch, n_threads, repeats, warmup)
    return ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def print_header(args):
    ts = datetime.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    print(f"# atomic_cmp (atomic frequency / contention micro-bench) — AI MAX 395 / gfx1151")
    print(f"# timestamp: {ts}")
    print(f"# host:      {os.uname().nodename}")
    print(f"# torch:     {torch.__version__}  triton: {triton.__version__}")
    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        print(f"# device:    {prop.name} (gcnArchName={getattr(prop, 'gcnArchName', '?')}, {prop.multi_processor_count} CUs)")
    print(
        f"# config:    sizes={args.sizes} repeats={args.repeats} warmup={args.warmup}"
        f" block={args.block} buckets={args.buckets}"
    )
    print("# --- rocm-smi snapshot ---")
    rocm_smi = shutil.which("rocm-smi")
    if rocm_smi:
        try:
            out = subprocess.run(
                [rocm_smi, "--showuse", "--showmeminfo", "vram"],
                capture_output=True, text=True, timeout=10,
            )
            for line in (out.stdout + out.stderr).splitlines():
                print(f"#   {line}")
        except Exception as exc:
            print(f"#   rocm-smi failed: {exc}")
    else:
        print("#   rocm-smi: not in PATH")
    print("# --- output ---")
    print("mode,n_threads,time_ms_total,time_ms_per_iter,ops_per_s")
    sys.stdout.flush()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", default="65536,262144,1048576,4194304",
                   help="comma-separated n_threads to scan")
    p.add_argument("--repeats", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--block", type=int, default=1024)
    p.add_argument("--buckets", type=int, default=1024)
    args = p.parse_args()
    if args.repeats < 50:
        args.repeats = 50
    if args.warmup < 5:
        args.warmup = 5

    if not torch.cuda.is_available():
        print("ERROR: torch.cuda is not available", file=sys.stderr)
        sys.exit(1)

    print_header(args)

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    repeats = args.repeats
    warmup = args.warmup
    block = args.block
    buckets = args.buckets

    def emit(mode, n, ms_total):
        ms_per = ms_total / repeats
        # 每次 launch 完成 n 次"逻辑写入"——三种模式都是 n_threads 个写
        ops_total = float(n) * repeats
        ops_per_s = ops_total / (ms_total * 1e-3) if ms_total > 0 else float("nan")
        print(f"{mode},{n},{ms_total:.4f},{ms_per:.6f},{ops_per_s:.3e}")
        sys.stdout.flush()

    for n in sizes:
        ms = run_plain_store(n, repeats, warmup, block)
        emit("plain_store", n, ms)
        ms = run_atomic_bucketed(n, repeats, warmup, block, buckets)
        emit(f"atomic_bucketed_{buckets}", n, ms)
        ms = run_atomic_full(n, repeats, warmup, block)
        emit("atomic_full", n, ms)


if __name__ == "__main__":
    main()
