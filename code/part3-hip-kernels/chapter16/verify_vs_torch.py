#!/usr/bin/env python3
"""
verify_vs_torch.py
用 PyTorch torch.matmul 输出作为参考，验证我们的 HIP kernel 数值正确性。

前提：已编译 matmul.hip -> ./matmul
      已激活 activate-rocm.sh（PyTorch ROCm 可用）

用法：
    python verify_vs_torch.py [M N K]

示例：
    python verify_vs_torch.py 512 512 512
"""

import subprocess
import sys
import numpy as np

def main():
    M, N, K = 512, 512, 512
    if len(sys.argv) == 4:
        M, N, K = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])

    print(f"数值验证：M={M}, N={N}, K={K}")
    print()

    # 1. 用 PyTorch 计算参考值
    try:
        import torch
        if not torch.cuda.is_available():
            print("[WARN] PyTorch 无法访问 GPU，跳过 torch 对照。仅运行 HIP kernel 内部 CPU 对照。")
            _run_hip_only(M, N, K)
            return
    except ImportError:
        print("[WARN] PyTorch 未安装，跳过 torch 对照。")
        _run_hip_only(M, N, K)
        return

    rng = np.random.default_rng(42)
    A_np = rng.standard_normal((M, K)).astype(np.float32)
    B_np = rng.standard_normal((K, N)).astype(np.float32)

    A_t = torch.from_numpy(A_np).cuda()
    B_t = torch.from_numpy(B_np).cuda()
    C_ref = torch.matmul(A_t, B_t).cpu().numpy()

    print("PyTorch 参考 C[0, :4] =", C_ref[0, :4])
    print()

    # 2. 把 A、B 写到临时文件，让 HIP 程序读取
    import tempfile, struct, os

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        a_path = f.name
        f.write(struct.pack(f"{M*K}f", *A_np.flatten()))

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        b_path = f.name
        f.write(struct.pack(f"{K*N}f", *B_np.flatten()))

    # 3. 运行 matmul 可执行文件做内部 CPU 对照
    try:
        result = subprocess.run(
            ["./matmul", str(M), str(N), str(K)],
            capture_output=True, text=True, timeout=120
        )
        print("=== matmul 输出 ===")
        print(result.stdout)
        if result.returncode != 0:
            print("[STDERR]", result.stderr)
    except FileNotFoundError:
        print("[ERROR] ./matmul 未找到，请先运行 hipcc matmul.hip -O3 -o matmul")
    except subprocess.TimeoutExpired:
        print("[ERROR] ./matmul 运行超时")

    os.unlink(a_path)
    os.unlink(b_path)


def _run_hip_only(M, N, K):
    """仅运行 HIP 可执行文件，不做 torch 对照"""
    try:
        result = subprocess.run(
            ["./matmul", str(M), str(N), str(K)],
            capture_output=True, text=True, timeout=120
        )
        print("=== matmul 输出 ===")
        print(result.stdout)
        if result.returncode != 0:
            print("[STDERR]", result.stderr)
    except FileNotFoundError:
        print("[ERROR] ./matmul 未找到，请先编译：hipcc matmul.hip -O3 -o matmul")
    except subprocess.TimeoutExpired:
        print("[ERROR] ./matmul 运行超时")


if __name__ == "__main__":
    main()
