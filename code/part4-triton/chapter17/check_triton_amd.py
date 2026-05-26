"""
check_triton_amd.py
验证 AMD ROCm 环境下 Triton 是否可用。

用法（已激活 venv 后）：
    python check_triton_amd.py

预期输出（成功时）：
    triton_version: 3.5.1+rocm7.12.0
    torch_version:  2.9.1+rocm7.12.0
    device_count:   1
    device_0:       Radeon 8060S Graphics  (gfx1151)
    vector_add_check: PASS  max_diff=0.0
"""

import sys

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# 最小 vector add kernel，验证 Triton backend 是否能编译并执行
# ---------------------------------------------------------------------------
@triton.jit
def _vec_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def triton_vec_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.shape == y.shape
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _vec_add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


def main() -> int:
    print(f"triton_version: {triton.__version__}")
    print(f"torch_version:  {torch.__version__}")

    if not torch.cuda.is_available():
        print("ERROR: torch.cuda.is_available() = False — GPU 不可见，请检查 ROCm 环境", file=sys.stderr)
        return 1

    device_count = torch.cuda.device_count()
    print(f"device_count:   {device_count}")
    for i in range(device_count):
        name = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        print(f"device_{i}:       {name}  (gcn_arch={getattr(props, 'gcn_arch_name', 'N/A')})")

    # 做一次实际的 Triton kernel 调用，验证编译 + 执行全链路
    N = 1 << 20  # 1M 元素
    x = torch.rand(N, device="cuda", dtype=torch.float32)
    y = torch.rand(N, device="cuda", dtype=torch.float32)

    out_triton = triton_vec_add(x, y)
    torch.cuda.synchronize()

    out_ref = x + y
    max_diff = (out_triton - out_ref).abs().max().item()
    status = "PASS" if max_diff < 1e-5 else "FAIL"
    print(f"vector_add_check: {status}  max_diff={max_diff}")

    if status != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
