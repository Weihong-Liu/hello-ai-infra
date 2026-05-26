#!/usr/bin/env python3
"""verify_vs_torch.py — 用 torch.nn.functional.softmax 对齐 v0/v1/v2/v3 输出。

工作流：
  1. 读 logs/input.bin（由 softmax_bin 写出，fp32，形状 [B, S]）
  2. 用 torch.softmax(x, dim=-1) 计算参考值（fp32）
  3. 与 logs/output_v0.bin ... output_v3.bin 对比，报告 max_diff / mean_diff / rel_err
  4. 打印每行求和是否为 1.0（完整性检查）

说明：
  - B/S 可通过命令行参数传入，默认与 softmax_bin 默认值一致（B=8, S=2048）
  - v0 故意不减 max，数值误差较大，标记为 WARN(expected) 而非 FAIL
  - v1/v2/v3 期望 max_diff < 1e-5
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

import numpy as np

LOG = Path(__file__).parent / "logs"


def load_bin(name: str, B: int, S: int) -> np.ndarray | None:
    p = LOG / name
    if not p.exists():
        return None
    expected = B * S
    data = np.fromfile(p, dtype=np.float32)
    if data.size != expected:
        print(f"  [WARN] {name}: 期望 {expected} 个元素，实际 {data.size}，形状可能不匹配")
    return data.reshape(B, S)


def main() -> int:
    parser = argparse.ArgumentParser(description="verify softmax outputs vs torch.softmax")
    parser.add_argument("--B", type=int, default=8, help="batch size")
    parser.add_argument("--S", type=int, default=2048, help="sequence length")
    args = parser.parse_args()

    B, S = args.B, args.S

    x = load_bin("input.bin", B, S)
    if x is None:
        print(f"[ERROR] logs/input.bin 不存在，请先跑 ./build/softmax_bin {B} {S}")
        return 1

    # 尝试使用 torch 计算参考值
    try:
        import torch
        import torch.nn.functional as F  # noqa: N812

        ref_t = F.softmax(torch.from_numpy(x).to(torch.float32), dim=-1)
        ref = ref_t.cpu().numpy()
        print(f"reference: torch.softmax(fp32)  shape={ref.shape}")
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] torch 不可用：{e}")
        print("退化为 numpy fp64 参考...")
        x64 = x.astype(np.float64)
        m = x64.max(axis=-1, keepdims=True)
        e64 = np.exp(x64 - m)
        ref = (e64 / e64.sum(axis=-1, keepdims=True)).astype(np.float32)
        print(f"reference: numpy fp64  shape={ref.shape}")

    row_sums = ref.sum(axis=-1)
    print(f"row_sum (ref)  min={row_sums.min():.6f}  max={row_sums.max():.6f}")
    print()

    # 各 kernel 版本对比
    versions = [
        ("v0", "output_v0.bin", True),   # (版本名, 文件名, 是否宽松判定)
        ("v1", "output_v1.bin", False),
        ("v2", "output_v2.bin", False),
        ("v3", "output_v3.bin", False),
    ]

    any_fail = False
    print(f"{'版本':12} {'max_diff':>12} {'mean_diff':>12} {'rel_err':>10} {'行和偏差':>10} {'状态':>30}")
    print("-" * 90)

    for ver_name, bin_name, is_lenient in versions:
        out = load_bin(bin_name, B, S)
        if out is None:
            print(f"  [{ver_name}] {bin_name} 不存在，跳过")
            continue

        diff = np.abs(out - ref)
        max_diff  = float(diff.max())
        mean_diff = float(diff.mean())
        ref_max   = float(np.abs(ref).max())
        rel_err   = max_diff / (ref_max + 1e-30)

        # 行和检查（应接近 1.0）
        row_sum_dev = float(np.abs(out.sum(axis=-1) - 1.0).max())

        ok = max_diff < 1e-5
        if is_lenient:
            tag = "WARN(expected: v0 不数值稳定)" if not ok else "PASS"
        else:
            tag = "PASS" if ok else "FAIL"
            if not ok:
                any_fail = True

        print(f"  {ver_name:<10} {max_diff:>12.3e} {mean_diff:>12.3e} "
              f"{rel_err:>10.3e} {row_sum_dev:>10.3e} {tag:>30}")

    print()
    if any_fail:
        print("[FAIL] 一个或多个 kernel 版本未通过数值验证（threshold=1e-5）")
        return 1
    print("[PASS] 所有验证通过（v0 的 WARN 属于预期行为）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
