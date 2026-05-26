"""
验证 Triton Softmax 实现与 torch.nn.functional.softmax 的数值对齐。

测试覆盖：
  - 所有版本（v1 / v1_tuned / v2）
  - 本章使用的所有形状：B ∈ {1, 8, 32} × S ∈ {512, 2048, 8192, 32768}
  - 大数值输入（测试数值稳定性）
  - fp32 精度

输出写到 logs/verify_vs_torch.log，失败时返回非零退出码。
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from softmax_triton import softmax_v1, softmax_v1_tuned, softmax_v2

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# 本章实验形状矩阵
SHAPES = [
    (1, 512),
    (1, 2048),
    (1, 8192),
    (1, 32768),
    (8, 512),
    (8, 2048),
    (8, 8192),
    (8, 32768),
    (32, 512),
    (32, 2048),
    (32, 8192),
    (32, 32768),
]

# 数值稳定性测试用的极端输入
EXTREME_SHAPES = [(8, 2048), (8, 8192)]

ATOL = 1e-5
RTOL = 1e-5


def verify_shape(B: int, S: int, version_name: str, fn, x: torch.Tensor):
    """
    对单个形状和版本做数值验证。
    返回 (ok: bool, max_diff: float)。
    """
    y_ref = F.softmax(x, dim=-1)
    y = fn(x)
    max_diff = (y - y_ref).abs().max().item()
    ok = max_diff < ATOL
    return ok, max_diff


def run_all_verifications():
    if not torch.cuda.is_available():
        print("SKIP: ROCm GPU 不可用", flush=True)
        sys.exit(0)

    print(f"torch: {torch.__version__}", flush=True)
    print(f"device: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"atol={ATOL}, rtol={RTOL}", flush=True)
    print("=" * 60, flush=True)

    versions = [
        ("v1", softmax_v1),
        ("v1_tuned", softmax_v1_tuned),
        ("v2", softmax_v2),
    ]

    all_passed = True
    results = []

    # --- 标准形状验证 ---
    print("[ 标准形状验证 ]", flush=True)
    torch.manual_seed(42)
    for B, S in SHAPES:
        x = torch.randn(B, S, device="cuda", dtype=torch.float32)
        for vname, fn in versions:
            ok, max_diff = verify_shape(B, S, vname, fn, x)
            status = "PASS" if ok else "FAIL"
            line = f"  [{B:3d}, {S:5d}] {vname:10s}: max_diff={max_diff:.2e}  {status}"
            print(line, flush=True)
            results.append(line)
            if not ok:
                all_passed = False

    print("", flush=True)

    # --- 数值稳定性验证（大数值输入） ---
    print("[ 数值稳定性验证（大数值输入） ]", flush=True)
    torch.manual_seed(99)
    for B, S in EXTREME_SHAPES:
        # 输入值域 [-500, 500]，会触发 naive softmax 溢出
        x_large = torch.rand(B, S, device="cuda", dtype=torch.float32) * 1000 - 500
        for vname, fn in versions:
            ok, max_diff = verify_shape(B, S, vname, fn, x_large)
            status = "PASS" if ok else "FAIL"
            line = f"  [{B:3d}, {S:5d}] {vname:10s} large_input: max_diff={max_diff:.2e}  {status}"
            print(line, flush=True)
            results.append(line)
            if not ok:
                all_passed = False

    print("", flush=True)
    print("=" * 60, flush=True)
    overall = "ALL PASS" if all_passed else "SOME FAIL"
    print(f"overall: {overall}", flush=True)

    # 写日志
    log_path = LOG_DIR / "verify_vs_torch.log"
    with open(log_path, "w") as f:
        f.write(f"torch: {torch.__version__}\n")
        f.write(f"device: {torch.cuda.get_device_name(0)}\n")
        f.write(f"atol={ATOL}, rtol={RTOL}\n")
        f.write("\n".join(results))
        f.write(f"\noverall: {overall}\n")
    print(f"日志写入: {log_path}", flush=True)

    return all_passed


if __name__ == "__main__":
    passed = run_all_verifications()
    sys.exit(0 if passed else 1)
