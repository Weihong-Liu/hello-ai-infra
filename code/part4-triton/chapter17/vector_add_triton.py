"""
vector_add_triton.py
Triton 向量加法完整实现，与 torch + 数值对齐验证。

用法（已激活 venv 后）：
    python vector_add_triton.py

预期输出（成功时）：
    [1M 元素] max_diff=0.00e+00  PASS: Triton vector_add 与 torch + 数值对齐
    [非整除] max_diff=0.00e+00  PASS: 非整除形状 n=1500 对齐
"""

import sys

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel：每个 program 处理长度为 BLOCK_SIZE 的连续区间
# ---------------------------------------------------------------------------
@triton.jit
def vector_add_kernel(
    x_ptr,                     # 输入向量 x 的指针（torch.Tensor → 指针自动转换）
    y_ptr,                     # 输入向量 y 的指针
    out_ptr,                   # 输出向量的指针
    n_elements,                # 向量长度
    BLOCK_SIZE: tl.constexpr,  # 每个 program 处理的元素数，编译时常量（2 的幂）
):
    # 1. 获取当前 program 的 ID（等价于 HIP 里的 blockIdx.x）
    pid = tl.program_id(axis=0)

    # 2. 计算当前 program 负责的元素偏移量向量 [pid*BLOCK_SIZE, ..., (pid+1)*BLOCK_SIZE-1]
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # 3. 边界 mask：超出向量长度的位置不做读写
    mask = offsets < n_elements

    # 4. 带 mask 的向量化加载（越界位置默认填 0）
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # 5. element-wise 向量加法（Triton 自动处理广播）
    out = x + y

    # 6. 带 mask 的向量化存储（mask=False 的位置不写，不破坏边界外的内存）
    tl.store(out_ptr + offsets, out, mask=mask)


# ---------------------------------------------------------------------------
# Python 调用侧：分配输出 tensor，计算 grid，启动 kernel
# ---------------------------------------------------------------------------
def vector_add(x: torch.Tensor, y: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    """
    Triton 向量加法。

    Args:
        x: 输入向量，必须在 CUDA 设备上（AMD GPU via ROCm）
        y: 输入向量，形状与 x 相同
        block_size: 每个 Triton program 处理的元素数，必须是 2 的幂

    Returns:
        out: x + y 的结果张量，与 x 同 dtype 和 device
    """
    assert x.shape == y.shape, f"形状不匹配：x={x.shape}, y={y.shape}"
    assert x.is_cuda and y.is_cuda, "输入必须在 GPU（CUDA/ROCm）上"
    assert x.is_contiguous() and y.is_contiguous(), "输入必须是连续（contiguous）张量"

    out = torch.empty_like(x)
    n = x.numel()

    # triton.cdiv(n, BLOCK_SIZE) = ceil(n / BLOCK_SIZE)，确保最后一段元素不遗漏
    grid = (triton.cdiv(n, block_size),)

    # 方括号 [grid] 传 launch grid，圆括号传 kernel 参数
    # BLOCK_SIZE 以关键字参数传入，Triton 会为每个不同值单独 JIT 编译
    vector_add_kernel[grid](
        x, y, out,
        n_elements=n,
        BLOCK_SIZE=block_size,
    )
    return out


# ---------------------------------------------------------------------------
# 数值验证
# ---------------------------------------------------------------------------
def verify(n: int, dtype=torch.float32, block_size: int = 1024, label: str = "") -> bool:
    x = torch.rand(n, device="cuda", dtype=dtype)
    y = torch.rand(n, device="cuda", dtype=dtype)

    out_triton = vector_add(x, y, block_size=block_size)
    torch.cuda.synchronize()

    out_ref = x + y
    max_diff = (out_triton - out_ref).abs().max().item()

    tag = f"[{label}]" if label else f"[n={n}]"
    if max_diff < 1e-5:
        print(f"{tag} max_diff={max_diff:.2e}  PASS: Triton vector_add 与 torch + 数值对齐")
        return True
    else:
        print(f"{tag} max_diff={max_diff:.2e}  FAIL: 数值对齐失败", file=sys.stderr)
        return False


def main() -> int:
    print(f"triton: {__import__('triton').__version__}")
    print(f"torch:  {torch.__version__}")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print()

    all_pass = True

    # 1M 元素（BLOCK_SIZE 的整数倍）
    all_pass &= verify(1 << 20, label="1M 元素, fp32, BLOCK=1024")

    # 非整除形状，验证 mask 边界处理正确
    all_pass &= verify(1500,    label="n=1500, fp32, 非整除")
    all_pass &= verify(2049,    label="n=2049, fp32, 非整除")
    all_pass &= verify(1,       label="n=1, fp32, 最小情形")

    # 不同 block_size
    all_pass &= verify(1 << 16, block_size=512,  label="64K 元素, BLOCK=512")
    all_pass &= verify(1 << 16, block_size=2048, label="64K 元素, BLOCK=2048")

    # fp16 精度（atol 宽松到 1e-3）
    x = torch.rand(1 << 20, device="cuda", dtype=torch.float16)
    y = torch.rand(1 << 20, device="cuda", dtype=torch.float16)
    out_t = vector_add(x, y)
    out_r = x + y
    max_diff_fp16 = (out_t.float() - out_r.float()).abs().max().item()
    fp16_ok = max_diff_fp16 < 1e-2
    print(f"[1M 元素, fp16] max_diff={max_diff_fp16:.2e}  {'PASS' if fp16_ok else 'FAIL'}")
    all_pass &= fp16_ok

    print()
    print("=== 全部验证通过 ===" if all_pass else "=== 有验证失败 ===")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
