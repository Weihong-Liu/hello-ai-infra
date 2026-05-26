"""
code/part1-hardware-rocm/chapter4/bw_footprint.py

Triton vector_copy footprint scan: 测量不同 footprint 下的有效带宽，
观察 "L0/L1 → L2 → MALL → DRAM" 的阶梯。

用法：
    python bw_footprint.py
    python bw_footprint.py --sizes 4096,65536,1048576,16777216,134217728
    python bw_footprint.py --repeats 200 --warmup 30

硬件上下文：AI MAX 395 (gfx1151) + ROCm 7.12.0 (rocm-sdk wheels)。
输出格式（一行一个 size，便于 grep）：
    [bw_footprint] n=<elems> bytes=<B> time=<ms_total> time_per_iter=<ms> eff_bw=<GB/s>
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import subprocess
import sys

import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception as exc:  # pragma: no cover
    HAS_TRITON = False
    _TRITON_ERR = exc


if HAS_TRITON:
    @triton.jit
    def copy_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        v = tl.load(x_ptr + offs, mask=mask)
        tl.store(y_ptr + offs, v, mask=mask)


def _print_header() -> None:
    now = _dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    print(f"# bw_footprint (Triton vector_copy) — AI MAX 395 / gfx1151")
    print(f"# timestamp: {now}")
    print(f"# python:    {sys.version.split()[0]}")
    print(f"# torch:     {torch.__version__}")
    if HAS_TRITON:
        print(f"# triton:    {triton.__version__}")
    else:  # pragma: no cover
        print(f"# triton:    NOT AVAILABLE ({_TRITON_ERR!r})")
    if torch.cuda.is_available():
        try:
            print(f"# device:    {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
    # rocm-smi snapshot to stdout (best effort)
    try:
        out = subprocess.run(
            ["rocm-smi", "--showuse", "--showmeminfo", "vram"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        print("# rocm-smi:")
        for line in out.stdout.splitlines():
            print(f"#   {line}")
    except Exception as e:
        print(f"# rocm-smi: unavailable ({e})")
    sys.stdout.flush()


def bench(n_elems: int, repeats: int, warmup: int, block: int = 1024) -> tuple[float, float]:
    if not HAS_TRITON:
        raise RuntimeError("Triton not available; cannot run footprint scan.")
    device = "cuda"
    x = torch.empty(n_elems, dtype=torch.float32, device=device)
    y = torch.empty_like(x)
    # 用 random fill 防止编译器把 copy 折叠
    x.uniform_(-1.0, 1.0)
    grid = ((n_elems + block - 1) // block,)

    for _ in range(warmup):
        copy_kernel[grid](x, y, n_elems, BLOCK=block)
    torch.cuda.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(repeats):
        copy_kernel[grid](x, y, n_elems, BLOCK=block)
    e.record()
    torch.cuda.synchronize()
    ms = s.elapsed_time(e)

    bytes_total = 2 * n_elems * 4 * repeats  # 1 read + 1 write fp32
    gbps = bytes_total / (ms * 1e-3) / 1e9
    return ms, gbps


def main() -> int:
    p = argparse.ArgumentParser()
    # 覆盖 L0/L1(几 KB) → L2(几 MB) → MALL(32 MB) → DRAM(几百 MB) 的尺寸点
    default_sizes = ",".join(str(n) for n in [
        4_096,        # 16 KiB  (L0/L1 territory)
        16_384,       # 64 KiB
        65_536,       # 256 KiB
        262_144,      # 1 MiB
        1_048_576,    # 4 MiB   (L2)
        4_194_304,    # 16 MiB
        8_388_608,    # 32 MiB  (≈ MALL boundary)
        16_777_216,   # 64 MiB
        67_108_864,   # 256 MiB
        134_217_728,  # 512 MiB
        268_435_456,  # 1 GiB
    ])
    p.add_argument("--sizes", default=default_sizes,
                   help="逗号分隔的元素数列表（fp32, 每元素 4 bytes）")
    p.add_argument("--repeats", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--block", type=int, default=1024)
    args = p.parse_args()

    if args.repeats < 100:
        args.repeats = 100
    if args.warmup < 5:
        args.warmup = 5

    _print_header()
    print(f"# config: repeats={args.repeats} warmup={args.warmup} block={args.block}")
    sys.stdout.flush()

    if not torch.cuda.is_available():
        print("ERROR: torch.cuda.is_available() is False — ROCm/HIP not visible to PyTorch.",
              file=sys.stderr)
        return 1
    if not HAS_TRITON:
        print(f"ERROR: triton import failed: {_TRITON_ERR!r}", file=sys.stderr)
        return 2

    for sz_str in args.sizes.split(","):
        sz_str = sz_str.strip()
        if not sz_str:
            continue
        n = int(sz_str)
        bytes_per_array = n * 4
        ms, gbps = bench(n, repeats=args.repeats, warmup=args.warmup, block=args.block)
        ms_per_iter = ms / args.repeats
        print(f"[bw_footprint] n={n:>11} bytes={bytes_per_array:>11} "
              f"time={ms:9.3f} ms time_per_iter={ms_per_iter:8.4f} ms "
              f"eff_bw={gbps:7.2f} GB/s")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
