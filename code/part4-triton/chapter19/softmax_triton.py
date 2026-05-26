"""
Triton Softmax 实现：v1（行宽 <= BLOCK_SIZE）和 v2（大行分块 fused）。

用法（激活 ROCm 环境后）：
    python softmax_triton.py
    python softmax_triton.py --version v2 --B 8 --S 32768
"""

import argparse

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# v1：一行一个 program，BLOCK_SIZE >= S（必须是 2 的幂）
# ---------------------------------------------------------------------------

@triton.jit
def _softmax_kernel_v1(
    x_ptr, y_ptr,
    B, S,
    stride_xb, stride_xs,
    stride_yb, stride_ys,
    BLOCK_SIZE: tl.constexpr,
):
    """
    数值稳定 Softmax v1。
    每个 program 处理矩阵的一行。BLOCK_SIZE 须 >= S 且为 2 的幂。
    越界位置用 -inf 填充，确保 tl.max 不受影响。
    """
    pid = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE)
    x_ptrs = x_ptr + pid * stride_xb + cols * stride_xs
    mask = cols < S

    # 加载整行，越界填 -inf（对 tl.max 安全）
    x = tl.load(x_ptrs, mask=mask, other=-float("inf"))

    # 数值稳定：减去行最大值
    x_max = tl.max(x, axis=0)
    x = x - x_max

    # 指数、归一化
    x_exp = tl.exp(x)
    x_sum = tl.sum(x_exp, axis=0)
    y = x_exp / x_sum

    # 写回（越界位置不写）
    y_ptrs = y_ptr + pid * stride_yb + cols * stride_ys
    tl.store(y_ptrs, y, mask=mask)


def softmax_v1(x: torch.Tensor) -> torch.Tensor:
    """
    Triton Softmax v1：行级，单 pass，BLOCK_SIZE = next_power_of_2(S)。
    适合 S 较小（<= ~8192）的场景；S 过大时 BLOCK_SIZE 过大导致 occupancy 下降。
    """
    assert x.ndim == 2, "输入必须是 2D 张量 [B, S]"
    B, S = x.shape
    BLOCK_SIZE = triton.next_power_of_2(S)
    y = torch.empty_like(x)
    # grid = (B,)：每行启动一个 program
    _softmax_kernel_v1[(B,)](
        x, y,
        B, S,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y


# ---------------------------------------------------------------------------
# v1 autotune 版本：对不同 S 自动选 BLOCK_SIZE
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
        triton.Config({"BLOCK_SIZE": 2048}),
        triton.Config({"BLOCK_SIZE": 4096}),
        triton.Config({"BLOCK_SIZE": 8192}),
    ],
    key=["S"],
)
@triton.jit
def _softmax_kernel_v1_tuned(
    x_ptr, y_ptr,
    B, S,
    stride_xb, stride_xs,
    stride_yb, stride_ys,
    BLOCK_SIZE: tl.constexpr,
):
    """v1 autotune 版本，Triton 自动为不同 S 选择最优 BLOCK_SIZE。"""
    pid = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE)
    x_ptrs = x_ptr + pid * stride_xb + cols * stride_xs
    mask = cols < S

    x = tl.load(x_ptrs, mask=mask, other=-float("inf"))
    x_max = tl.max(x, axis=0)
    x = x - x_max
    x_exp = tl.exp(x)
    x_sum = tl.sum(x_exp, axis=0)
    y = x_exp / x_sum

    y_ptrs = y_ptr + pid * stride_yb + cols * stride_ys
    tl.store(y_ptrs, y, mask=mask)


def softmax_v1_tuned(x: torch.Tensor) -> torch.Tensor:
    """Triton Softmax v1 autotune 版本。"""
    assert x.ndim == 2
    B, S = x.shape
    y = torch.empty_like(x)
    _softmax_kernel_v1_tuned[(B,)](
        x, y,
        B, S,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
    )
    return y


# ---------------------------------------------------------------------------
# v2：大行分块 fused，BLOCK_SIZE 独立于 S，双 pass online reduction
# ---------------------------------------------------------------------------

@triton.jit
def _softmax_kernel_v2(
    x_ptr, y_ptr,
    B, S,
    stride_xb, stride_xs,
    stride_yb, stride_ys,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton Softmax v2：双 pass online reduction。
    BLOCK_SIZE 可独立于 S 选取（典型值 1024/2048），寄存器占用固定。
    代价：每行数据被读两次（Pass 1 求 max/sum，Pass 2 写结果）。
    适合 S 很大（>= 8192）或 v1 因 BLOCK_SIZE 过大导致 occupancy 过低时使用。
    """
    pid = tl.program_id(axis=0)
    row_start_x = pid * stride_xb
    row_start_y = pid * stride_yb

    # --- Pass 1：Online reduction，求行最大值和归一化分母 ---
    # 初始化：row_max = -inf，row_sum = 0
    row_max = tl.full((1,), float("-inf"), dtype=tl.float32)
    row_sum = tl.zeros((1,), dtype=tl.float32)

    for block_start in tl.range(0, S, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < S
        x_block = tl.load(
            x_ptr + row_start_x + cols * stride_xs,
            mask=mask,
            other=float("-inf"),
        )
        block_max = tl.max(x_block, axis=0)

        # Online max 更新：把已累积的 sum 按新 max 修正
        new_max = tl.maximum(row_max, block_max)
        row_sum = row_sum * tl.exp(row_max - new_max) + \
                  tl.sum(tl.exp(x_block - new_max), axis=0)
        row_max = new_max

    # --- Pass 2：写回归一化结果 ---
    for block_start in tl.range(0, S, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < S
        x_block = tl.load(
            x_ptr + row_start_x + cols * stride_xs,
            mask=mask,
            other=float("-inf"),
        )
        y_block = tl.exp(x_block - row_max) / row_sum
        tl.store(
            y_ptr + row_start_y + cols * stride_ys,
            y_block,
            mask=mask,
        )


def softmax_v2(x: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    """
    Triton Softmax v2：分块 fused，支持任意大 S。
    block_size 须为 2 的幂，典型值 1024 或 2048。
    """
    assert x.ndim == 2
    assert (block_size & (block_size - 1)) == 0, "block_size 必须是 2 的幂"
    B, S = x.shape
    y = torch.empty_like(x)
    _softmax_kernel_v2[(B,)](
        x, y,
        B, S,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_SIZE=block_size,
    )
    return y


# ---------------------------------------------------------------------------
# 快速验证
# ---------------------------------------------------------------------------

def verify(B: int, S: int, version: str = "v1") -> None:
    """对指定形状和版本做数值验证（与 torch.nn.functional.softmax 对齐）。"""
    import torch.nn.functional as F

    torch.manual_seed(42)
    x = torch.randn(B, S, device="cuda", dtype=torch.float32)
    y_ref = F.softmax(x, dim=-1)

    if version == "v1":
        y = softmax_v1(x)
        label = "v1"
    elif version == "v1_tuned":
        y = softmax_v1_tuned(x)
        label = "v1_tuned"
    elif version == "v2":
        y = softmax_v2(x)
        label = "v2"
    else:
        raise ValueError(f"未知版本: {version}")

    max_diff = (y - y_ref).abs().max().item()
    ok = max_diff < 1e-5
    status = "PASS" if ok else "FAIL"
    print(f"verify [{B},{S}] {label}: max_diff={max_diff:.2e} {status}")
    if not ok:
        raise AssertionError(f"数值验证失败：max_diff={max_diff:.2e}")


def main():
    parser = argparse.ArgumentParser(description="Triton Softmax 快速验证")
    parser.add_argument("--version", choices=["v1", "v1_tuned", "v2", "all"],
                        default="all")
    parser.add_argument("--B", type=int, default=8)
    parser.add_argument("--S", type=int, default=2048)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("ROCm GPU 不可用，请在实验机上运行")

    print(f"torch: {torch.__version__}")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"形状: [{args.B}, {args.S}]")
    print()

    if args.version == "all":
        for v in ["v1", "v1_tuned", "v2"]:
            verify(args.B, args.S, v)
    else:
        verify(args.B, args.S, args.version)


if __name__ == "__main__":
    main()
