"""
attention_v2_blocked.py — 分块 Triton Attention（v2：online softmax，仅 forward）

特点：
  - 每个 program 处理 output 的一行（第 row_id 行）
  - 分块遍历 K/V（每次 BLOCK_S 行），只遍历一遍
  - Online softmax：维护 running max、running sum、running acc
    纠正因子 corr = exp(m_old - m_new) 确保数值稳定
  - 支持 causal mask（j > i 位置 score 设为 -inf）
  - 通过 @triton.autotune 搜索最优 BLOCK_S

显存分析：
  - 不物化 S×S 的中间矩阵，显存 O(B·H·S·D)
  - 与物化版相比：B=4, H=16, S=4096, D=64 时节省约 2 GB

适用形状：
  q, k, v: [B, H, S, D], fp16, contiguous
  D 必须是 2 的幂次且 <= 128
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': bs}, num_warps=nw, num_stages=ns)
        for bs in (32, 64, 128, 256)
        for nw in (2, 4, 8)
        for ns in (1, 2, 3)
    ],
    key=['S', 'D'],
)
@triton.jit
def attention_v2_kernel(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    B, H, S, D,
    scale,
    causal,
    BLOCK_S: tl.constexpr,
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
    q     = tl.load(q_ptr + d_offs).to(tl.float32)

    # Online softmax 状态
    m_i = tl.full([1], float('-inf'), dtype=tl.float32)  # running max
    l_i = tl.zeros([1], dtype=tl.float32)                # running sum(exp(...))
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)          # running weighted acc

    # 分块遍历 K/V（只遍历一遍）
    for block_start in range(0, S, BLOCK_S):
        block_offs = block_start + tl.arange(0, BLOCK_S)
        mask       = block_offs < S

        # 加载 K block: [BLOCK_S, BLOCK_D]
        k_ptr = K + batch_id * stride_kb + head_id * stride_kh
        k_blk = tl.load(
            k_ptr + block_offs[:, None] * stride_ks + d_offs[None, :],
            mask=mask[:, None], other=0.0
        ).to(tl.float32)

        # score: q_i · k_j / sqrt(D)，结果 [BLOCK_S]
        s = tl.sum(q[None, :] * k_blk, axis=1) * scale

        # 对越界位置的 score 设为 -inf（避免 padding 影响）
        s = tl.where(mask, s, float('-inf'))

        # Causal mask：j > i 的位置只能被"未来"看到，当前 query 不应关注
        if causal:
            s = tl.where(block_offs > row_id, float('-inf'), s)

        # Online softmax 更新
        m_blk = tl.max(s, keep_dims=True)
        m_new = tl.maximum(m_i, m_blk)

        # 纠正因子：把历史累积量从 m_i 基准调整到 m_new 基准
        corr  = tl.exp(m_i - m_new)
        p     = tl.exp(s - m_new)                           # [BLOCK_S]
        l_i   = l_i * corr + tl.sum(p, keep_dims=True)
        acc   = acc * corr

        # 加载 V block: [BLOCK_S, BLOCK_D]，加权累加
        v_ptr = V + batch_id * stride_vb + head_id * stride_vh
        v_blk = tl.load(
            v_ptr + block_offs[:, None] * stride_vs + d_offs[None, :],
            mask=mask[:, None], other=0.0
        ).to(tl.float32)
        # p[:, None] * v_blk: [BLOCK_S, BLOCK_D] → sum over BLOCK_S → [BLOCK_D]
        acc  += tl.sum(p[:, None] * v_blk, axis=0)
        m_i   = m_new

    # 归一化并写回
    out   = acc / l_i
    o_ptr = Out + batch_id * stride_ob + head_id * stride_oh + row_id * stride_os
    tl.store(o_ptr + d_offs, out.to(tl.float16))


def attention_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """
    分块 Attention，online softmax，仅 forward。
    不物化 [S, S] 中间矩阵，显存 O(B·H·S·D)。

    Args:
        q: [B, H, S, D], fp16, contiguous
        k: [B, H, S, D], fp16, contiguous
        v: [B, H, S, D], fp16, contiguous
        causal: 是否启用因果掩码（j > i 位置设为 -inf）
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

    attention_v2_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, S, D, scale,
        int(causal),
        BLOCK_D=D,
    )
    return out


if __name__ == '__main__':
    import torch.nn.functional as F

    device = 'cuda'
    B, H, S, D = 1, 4, 256, 64

    for causal in [False, True]:
        q = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H, S, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H, S, D, device=device, dtype=torch.float16)

        out_v2  = attention_v2(q, k, v, causal=causal)
        out_ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)

        err = (out_v2.float() - out_ref.float()).abs().max().item()
        print(f"v2 causal={causal} vs torch SDPA max abs error: {err:.6f}")
        assert err < 0.1, f"误差过大 (causal={causal}): {err}"
        print(f"attention_v2 causal={causal} smoke test PASS")
