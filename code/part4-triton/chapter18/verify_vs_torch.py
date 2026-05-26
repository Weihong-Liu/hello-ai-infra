"""
verify_vs_torch.py — 验证 Triton Matmul 与 torch.matmul 的数值对齐
对应文档：docs/part4-triton/chapter18/index.md §18.5.3
"""

import torch
from matmul_triton import matmul_naive, matmul_grouped


def verify_naive():
    print("=== 验证 matmul_naive ===")
    test_shapes = [
        (512, 512, 512),
        (1000, 999, 998),   # 三维都不整除
        (1024, 1024, 1024),
        (2000, 1500, 800),  # 非对称
    ]
    for M, N, K in test_shapes:
        A = torch.randn(M, K, device="cuda", dtype=torch.float32)
        B = torch.randn(K, N, device="cuda", dtype=torch.float32)
        ref = torch.matmul(A, B)
        out = matmul_naive(A, B)
        assert torch.allclose(ref, out, atol=1e-3, rtol=1e-3), \
            f"形状 ({M},{N},{K}) 验证失败，max_err={( ref - out).abs().max()}"
        print(f"  ({M:5d}, {N:5d}, {K:5d})  max_err={( ref - out).abs().max():.2e}  PASS")
    print("所有形状验证通过 ✓")


def verify_grouped():
    print("\n=== 验证 matmul_grouped ===")
    test_shapes = [
        (512, 512, 512),
        (1000, 999, 998),
        (1024, 1024, 1024),
        (2000, 1500, 800),
    ]
    for M, N, K in test_shapes:
        A = torch.randn(M, K, device="cuda", dtype=torch.float32)
        B = torch.randn(K, N, device="cuda", dtype=torch.float32)
        ref = torch.matmul(A, B)
        out = matmul_grouped(A, B)
        assert torch.allclose(ref, out, atol=1e-3, rtol=1e-3), \
            f"形状 ({M},{N},{K}) grouped 验证失败，max_err={( ref - out).abs().max()}"
        print(f"  ({M:5d}, {N:5d}, {K:5d})  max_err={( ref - out).abs().max():.2e}  PASS")
    print("所有形状验证通过 ✓")


if __name__ == "__main__":
    verify_naive()
    verify_grouped()
    print("\n全部验证完成 ✓")
