"""
attention_v1_naive.py — 教学版 Triton Attention（v1：不物化 QK^T，但两遍遍历 K）

特点：
  - 每个 program 处理 output 的一行（第 row_id 行）
  - 第一遍遍历 K 找全局 max（数值稳定）
  - 第二遍遍历 K+V 做加权求和
  - 代码逻辑最直接，便于理解三阶段数据流
  - 缺点：K 读两遍、j 维度无并行

适用形状：
  q, k, v: [B, H, S, D], fp16, contiguous
  D 必须是 2 的幂次且 <= 128
"""
import torch
import triton
import triton.language as tl


@triton.jit
def attention_v1_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    B, H, S, D,
    scale,
    BLOCK_D: tl.constexpr,
):
    pid      = tl.program_id(0)
    batch_id = pid // (H * S)
    rem      = pid  % (H * S)
    head_id  = rem  // S
    row_id   = rem  % S

    d_offs = tl.arange(0, BLOCK_D)

    # 加载 q_i: [BLOCK_D]
    q_ptr = Q + batch_id * stride_qb + head_id * stride_qh + row_id * stride_qs
    q     = tl.load(q_ptr + d_offs).to(tl.float32)  # [BLOCK_D]

    # 第一遍：找所有 score 的全局 max（数值稳定基准）
    m = tl.full([1], float('-inf'), dtype=tl.float32)
    for j in range(S):
        k_ptr = K + batch_id * stride_kb + head_id * stride_kh + j * stride_ks
        k     = tl.load(k_ptr + d_offs).to(tl.float32)
        s     = tl.sum(q * k) * scale  # scalar
        m     = tl.maximum(m, s)

    # 第二遍：exp(s - m) 加权 V，归一化
    denom = tl.zeros([1], dtype=tl.float32)
    acc   = tl.zeros([BLOCK_D], dtype=tl.float32)
    for j in range(S):
        k_ptr = K + batch_id * stride_kb + head_id * stride_kh + j * stride_ks
        k     = tl.load(k_ptr + d_offs).to(tl.float32)
        s     = tl.sum(q * k) * scale
        w     = tl.exp(s - m)
        denom = denom + w

        v_ptr = V + batch_id * stride_vb + head_id * stride_vh + j * stride_vs
        v     = tl.load(v_ptr + d_offs).to(tl.float32)
        acc   = acc + w * v

    acc   = acc / denom  # 归一化

    # 写回输出
    o_ptr = Out + batch_id * stride_ob + head_id * stride_oh + row_id * stride_os
    tl.store(o_ptr + d_offs, acc.to(tl.float16))


def attention_v1(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    教学版 Attention，不物化 QK^T 中间矩阵，但顺序两遍遍历 K。

    Args:
        q: [B, H, S, D], fp16, contiguous
        k: [B, H, S, D], fp16, contiguous
        v: [B, H, S, D], fp16, contiguous
    Returns:
        out: [B, H, S, D], fp16
    """
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous(), \
        "q, k, v 必须是 contiguous tensor"
    assert q.dtype == torch.float16, "仅支持 fp16"
    B, H, S, D = q.shape
    assert D in (32, 64, 128), f"head dim D={D} 必须是 32/64/128"

    out   = torch.empty_like(q)
    scale = float(D ** -0.5)
    grid  = (B * H * S,)

    attention_v1_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, S, D, scale,
        BLOCK_D=D,
    )
    return out


if __name__ == '__main__':
    import torch.nn.functional as F

    device = 'cuda'
    B, H, S, D = 1, 4, 256, 64
    q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
    k = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
    v = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

    out_v1  = attention_v1(q, k, v)
    out_ref = F.scaled_dot_product_attention(q, k, v)

    err = (out_v1.float() - out_ref.float()).abs().max().item()
    print(f"v1 vs torch SDPA max abs error: {err:.6f}")
    assert err < 0.1, f"误差过大：{err}"
    print("attention_v1 smoke test PASS")
