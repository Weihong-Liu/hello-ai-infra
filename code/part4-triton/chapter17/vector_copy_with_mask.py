"""
vector_copy_with_mask.py
演示 Triton mask 机制的最简例子：向量条件拷贝。

包含三个示例：
  1. 基础拷贝（验证全量 mask 行为）
  2. 偶数索引拷贝（组合 mask：范围 & 条件）
  3. 阈值拷贝（数据驱动的 mask，演示 other= 的作用）

用法：
    python vector_copy_with_mask.py
"""

import sys

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1：基础拷贝（等价于 memcpy，演示最简 mask 用法）
# ---------------------------------------------------------------------------
@triton.jit
def copy_kernel(
    src_ptr, dst_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """每个 program 拷贝 BLOCK_SIZE 个元素，用 mask 处理边界。"""
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    val = tl.load(src_ptr + offsets, mask=mask)
    tl.store(dst_ptr + offsets, val, mask=mask)


def triton_copy(src: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    dst = torch.empty_like(src)
    n = src.numel()
    grid = (triton.cdiv(n, block_size),)
    copy_kernel[grid](src, dst, n_elements=n, BLOCK_SIZE=block_size)
    return dst


# ---------------------------------------------------------------------------
# Kernel 2：只拷贝偶数索引的元素，奇数位置填 0
# ---------------------------------------------------------------------------
@triton.jit
def copy_even_indices_kernel(
    src_ptr, dst_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    mask 示例：组合两个条件。
      - in_bounds：偏移量不超过向量长度
      - is_even：偏移量是偶数索引

    tl.load 对 mask=False 的位置返回 other=0.0。
    tl.store 对 mask=False 的位置不做任何写操作（保持原始内存）。
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # 条件 1：在向量范围内
    in_bounds = offsets < n_elements
    # 条件 2：偶数索引
    is_even   = (offsets % 2) == 0
    # 组合 mask：两个条件同时满足才读/写
    load_mask = in_bounds & is_even

    # 越界或奇数位置：tl.load 返回 other=0.0（不读实际内存）
    val = tl.load(src_ptr + offsets, mask=load_mask, other=0.0)

    # 写出时用 in_bounds 而非 load_mask：
    #   - 偶数位置写入读取到的值
    #   - 奇数位置写入 0（load 时 other=0.0 已经把 val 置为 0）
    #   - 越界位置不写（mask=False）
    tl.store(dst_ptr + offsets, val, mask=in_bounds)


def triton_copy_even(src: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    dst = torch.zeros_like(src)  # 初始化为 0，奇数位置保持 0
    n = src.numel()
    grid = (triton.cdiv(n, block_size),)
    copy_even_indices_kernel[grid](src, dst, n_elements=n, BLOCK_SIZE=block_size)
    return dst


# ---------------------------------------------------------------------------
# Kernel 3：阈值拷贝（数据驱动的 mask）
# ---------------------------------------------------------------------------
@triton.jit
def copy_above_threshold_kernel(
    src_ptr, dst_ptr,
    n_elements,
    threshold,           # 阈值，标量（通过 kernel 参数传入）
    fill_value,          # 低于阈值位置的填充值
    BLOCK_SIZE: tl.constexpr,
):
    """
    把 src 中大于 threshold 的元素拷贝到 dst，
    低于 threshold 的位置填充 fill_value。

    演示：mask 由数据内容（而非索引位置）决定。
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    in_bounds = offsets < n_elements

    # 先无条件加载（越界位置 other=threshold，确保不会被误判为"大于阈值"）
    val = tl.load(src_ptr + offsets, mask=in_bounds, other=threshold)

    # 数据驱动的 mask：val > threshold
    above = val > threshold

    # 低于阈值的位置用 fill_value 替换
    # tl.where(condition, x, y) ← Triton 的三元选择，逐元素操作
    out = tl.where(above, val, fill_value)

    tl.store(dst_ptr + offsets, out, mask=in_bounds)


def triton_copy_threshold(
    src: torch.Tensor,
    threshold: float = 0.5,
    fill_value: float = 0.0,
    block_size: int = 1024,
) -> torch.Tensor:
    dst = torch.empty_like(src)
    n = src.numel()
    grid = (triton.cdiv(n, block_size),)
    copy_above_threshold_kernel[grid](
        src, dst,
        n_elements=n,
        threshold=threshold,
        fill_value=fill_value,
        BLOCK_SIZE=block_size,
    )
    return dst


# ---------------------------------------------------------------------------
# 验证与演示
# ---------------------------------------------------------------------------
def main() -> int:
    print(f"triton: {__import__('triton').__version__}")
    print(f"torch:  {torch.__version__}")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print()

    all_pass = True

    # -------------------------------------------------------------------
    # 示例 1：基础拷贝（整除 & 非整除）
    # -------------------------------------------------------------------
    print("=== 示例 1：基础拷贝（copy_kernel）===")
    for n, label in [(1 << 20, "1M 整除"), (1500, "非整除 1500"), (1, "最小 n=1")]:
        src = torch.rand(n, device="cuda")
        dst = triton_copy(src)
        torch.cuda.synchronize()
        diff = (dst - src).abs().max().item()
        ok = diff < 1e-6
        print(f"  n={n:>8} ({label}): max_diff={diff:.2e}  {'PASS' if ok else 'FAIL'}")
        all_pass &= ok
    print()

    # -------------------------------------------------------------------
    # 示例 2：偶数索引拷贝（组合 mask）
    # -------------------------------------------------------------------
    print("=== 示例 2：偶数索引拷贝（copy_even_indices_kernel）===")
    n = 16
    src = torch.arange(n, device="cuda", dtype=torch.float32)
    dst = triton_copy_even(src, block_size=16)
    torch.cuda.synchronize()

    # 参考实现（CPU）
    ref = torch.zeros(n, device="cuda", dtype=torch.float32)
    ref[0::2] = src[0::2]

    diff = (dst - ref).abs().max().item()
    ok = diff < 1e-6
    print(f"  src = {src.tolist()}")
    print(f"  dst = {dst.tolist()}")
    print(f"  ref = {ref.tolist()}")
    print(f"  max_diff={diff:.2e}  {'PASS' if ok else 'FAIL'}")
    all_pass &= ok
    print()

    # -------------------------------------------------------------------
    # 示例 3：阈值拷贝（tl.where 用法）
    # -------------------------------------------------------------------
    print("=== 示例 3：阈值拷贝（copy_above_threshold_kernel）===")
    torch.manual_seed(42)
    n = 1 << 18
    src = torch.rand(n, device="cuda")
    threshold = 0.5

    dst = triton_copy_threshold(src, threshold=threshold, fill_value=-1.0)
    torch.cuda.synchronize()

    # 参考实现
    ref = torch.where(src > threshold, src, torch.full_like(src, -1.0))
    diff = (dst - ref).abs().max().item()
    ok = diff < 1e-6

    above_count = (src > threshold).sum().item()
    print(f"  n={n}, threshold={threshold}")
    print(f"  高于阈值元素数 = {above_count} / {n}  ({100*above_count/n:.1f}%)")
    print(f"  max_diff={diff:.2e}  {'PASS' if ok else 'FAIL'}")
    all_pass &= ok
    print()

    # -------------------------------------------------------------------
    # mask=None 的危险性演示（仅用注释说明，不实际执行越界操作）
    # -------------------------------------------------------------------
    print("=== 注意：mask=None 的风险 ===")
    print("  如果 n 不是 BLOCK_SIZE 的整数倍且不传 mask，tl.load 会读越界地址，")
    print("  tl.store 会写越界地址，可能引发 segfault 或数据损坏。")
    print("  本脚本中所有 kernel 均传入 mask，规避此问题。")
    print()

    print("=== 全部验证通过 ===" if all_pass else "=== 有验证失败 ===")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
