"""
verify_vs_torch.py — 验证 v1/v2 与 PyTorch SDPA 的数值一致性

测试矩阵：
  - causal=False / True
  - S ∈ {128, 256, 512, 1024}
  - D ∈ {64, 128}

Pass 条件：max abs error < 1e-1（fp16 精度下的合理门槛）
"""
import sys
import torch
import torch.nn.functional as F

from attention_v1_naive import attention_v1
from attention_v2_blocked import attention_v2


def verify(fn, q, k, v, causal, fn_name, tol=0.1):
    if fn_name == 'v1':
        out = fn(q, k, v)
    else:
        out = fn(q, k, v, causal=causal)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)

    err_max  = (out.float() - ref.float()).abs().max().item()
    err_mean = (out.float() - ref.float()).abs().mean().item()
    status   = "PASS" if err_max < tol else "FAIL"
    return status, err_max, err_mean


def run_all():
    device = 'cuda'
    results = []

    configs = [
        (1, 4, S, D, causal)
        for S in [128, 256, 512, 1024]
        for D in [64, 128]
        for causal in [False, True]
    ]

    all_pass = True
    print(f"{'impl':>4}  {'S':>6}  {'D':>4}  {'causal':>6}  {'max_err':>10}  {'mean_err':>10}  {'status':>6}")
    print("-" * 70)

    for B, H, S, D, causal in configs:
        q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

        for fn, fn_name in [(attention_v1, 'v1'), (attention_v2, 'v2')]:
            status, err_max, err_mean = verify(fn, q, k, v, causal, fn_name)
            print(f"{fn_name:>4}  {S:>6}  {D:>4}  {str(causal):>6}  "
                  f"{err_max:>10.6f}  {err_mean:>10.6f}  {status:>6}")
            if status != "PASS":
                all_pass = False
            results.append(dict(
                impl=fn_name, S=S, D=D, causal=causal,
                err_max=err_max, err_mean=err_mean, status=status,
            ))

    print("-" * 70)
    if all_pass:
        print("所有验证 PASS")
    else:
        print("部分验证 FAIL，请检查上方详情")
        sys.exit(1)

    return results


if __name__ == '__main__':
    run_all()
