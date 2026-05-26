"""
bench_attention.py — Attention 性能 benchmark

扫描：
  - S ∈ {512, 1024, 2048, 4096}
  - D ∈ {64, 128}
  - causal ∈ {False, True}

固定：B=4, H=8, fp16

指标：中位数耗时（ms）、显存峰值（MB）

输出：控制台 + logs/bench_attention_D{D}_causal{causal}.csv
"""
import argparse
import csv
import os
import time

import torch
import torch.nn.functional as F

from attention_v1_naive import attention_v1
from attention_v2_blocked import attention_v2

LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')


def benchmark_fn(fn, warmup=20, rep=100):
    """运行 fn() warmup 次热身 + rep 次计时，返回中位数毫秒。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(rep):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def run_bench(
    B: int = 4,
    H: int = 8,
    D: int = 64,
    causal: bool = False,
    seq_list=None,
    out_csv: str = None,
):
    if seq_list is None:
        seq_list = [512, 1024, 2048, 4096]

    device = 'cuda'
    rows   = []

    print(f"\n=== Attention Benchmark: B={B} H={H} D={D} causal={causal} dtype=fp16 ===")
    print(f"Hardware: AI MAX 395 + ROCm 7.12.0")
    print(f"{'S':>6}  {'v1(ms)':>10}  {'v2(ms)':>10}  {'sdpa(ms)':>10}  "
          f"{'mem_v1(MB)':>12}  {'mem_v2(MB)':>12}")
    print("-" * 75)

    for S in seq_list:
        q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

        # v1
        torch.cuda.reset_peak_memory_stats()
        t_v1   = benchmark_fn(lambda: attention_v1(q, k, v))
        mem_v1 = torch.cuda.max_memory_allocated() / 1e6

        # v2
        torch.cuda.reset_peak_memory_stats()
        t_v2   = benchmark_fn(lambda: attention_v2(q, k, v, causal=causal))
        mem_v2 = torch.cuda.max_memory_allocated() / 1e6

        # torch SDPA
        t_sdpa = benchmark_fn(
            lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        )

        print(f"{S:>6}  {t_v1:>10.3f}  {t_v2:>10.3f}  {t_sdpa:>10.3f}  "
              f"{mem_v1:>12.1f}  {mem_v2:>12.1f}")

        rows.append(dict(
            B=B, H=H, S=S, D=D, causal=causal,
            v1_ms=t_v1, v2_ms=t_v2, sdpa_ms=t_sdpa,
            mem_v1_mb=mem_v1, mem_v2_mb=mem_v2,
        ))

    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n结果写入 {out_csv}")

    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Triton Attention benchmark")
    parser.add_argument('--B', type=int, default=4)
    parser.add_argument('--H', type=int, default=8)
    parser.add_argument('--D', type=int, default=64, choices=[64, 128])
    parser.add_argument('--causal', action='store_true')
    parser.add_argument(
        '--seq', type=int, nargs='+', default=[512, 1024, 2048, 4096],
        help='序列长度列表，例如 --seq 512 1024 2048 4096'
    )
    args = parser.parse_args()

    out_csv = os.path.join(
        LOG_DIR,
        f'bench_attention_D{args.D}_causal{args.causal}.csv'
    )
    run_bench(
        B=args.B, H=args.H, D=args.D, causal=args.causal,
        seq_list=args.seq, out_csv=out_csv,
    )
