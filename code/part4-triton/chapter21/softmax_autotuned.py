"""
softmax_autotuned.py — 带 @triton.autotune 的行级 Softmax。

复用 chapter3（Triton Softmax）的 v1 kernel 思路，
用 @triton.autotune 对 BLOCK_SIZE 和 num_warps 做自动搜索。

搜索空间：
  BLOCK_SIZE ∈ {512, 1024, 2048, 4096, 8192}
  num_warps  ∈ {2, 4, 8}
  key        = ["S"]  —— 对不同行宽分别缓存最优配置

用法（source activate-rocm.sh 后）：
    python softmax_autotuned.py
    python softmax_autotuned.py --B 8 --S 32768
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 搜索空间定义
#
# BLOCK_SIZE：每个 program 一次性处理的列数
#   - 必须是 2 的幂，且满足 BLOCK_SIZE >= S（v1 单 pass 约束）
#   - 对于 S=512/1024：小 BLOCK_SIZE（512/1024）足以
#   - 对于 S=8192/32768：BLOCK_SIZE 必须 >= S，寄存器压力随之升高
#   - autotune 中超出 S 的 BLOCK_SIZE 会通过 mask 处理越界，不影响正确性
#
# num_warps：每个 program 使用的 wavefront 数量
#   - AMD gfx1151（RDNA4）以 wave32 运行（每 wavefront 32 线程）
#   - num_warps=4 时共 128 个并发线程
#   - 通常 BLOCK_SIZE 越大，需要更多 wavefront 摊薄延迟
#
# key=["S"]：行宽不同时，autotune 分别为每个 S 缓存最优 config
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512},  num_warps=2),
        triton.Config({"BLOCK_SIZE": 512},  num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8),
    ],
    key=["S"],
)
@triton.jit
def softmax_kernel_autotuned(
    x_ptr, y_ptr,
    B, S,
    stride_xb, stride_xs,
    stride_yb, stride_ys,
    BLOCK_SIZE: tl.constexpr,
):
    """
    数值稳定 Softmax（v1 单 pass），由 autotune 自动选择 BLOCK_SIZE 与 num_warps。

    每个 program 处理一行（pid 对应行号）。
    BLOCK_SIZE >= S 时，mask 处理越界位置（填 -inf 对 tl.max 安全，填 0 对 tl.sum 安全）。
    """
    pid = tl.program_id(axis=0)
    cols = tl.arange(0, BLOCK_SIZE)
    x_ptrs = x_ptr + pid * stride_xb + cols * stride_xs
    mask = cols < S

    # 加载整行，越界位置填 -inf
    x = tl.load(x_ptrs, mask=mask, other=-float("inf"))

    # 数值稳定：减去行最大值（减后最大指数输入 = 0，不溢出）
    x_max = tl.max(x, axis=0)
    x = x - x_max

    # 指数 + 归一化
    x_exp = tl.exp(x)
    x_sum = tl.sum(x_exp, axis=0)
    y = x_exp / x_sum

    # 写回（越界位置不写）
    y_ptrs = y_ptr + pid * stride_yb + cols * stride_ys
    tl.store(y_ptrs, y, mask=mask)


def softmax_autotuned(x: torch.Tensor) -> torch.Tensor:
    """
    Triton 自动调参 Softmax，对每种行宽 S 自动选择最优 BLOCK_SIZE 与 num_warps。

    x: [B, S], float32
    返回 y: [B, S], float32
    """
    assert x.ndim == 2, "输入必须是 2D 张量 [B, S]"
    B, S = x.shape
    y = torch.empty_like(x)
    # grid = (B,)：每行一个 program
    softmax_kernel_autotuned[(B,)](
        x, y,
        B, S,
        x.stride(0), x.stride(1),
        y.stride(0), y.stride(1),
    )
    return y


def verify(B: int, S: int) -> None:
    """数值验证：与 F.softmax 对比，atol=1e-5。"""
    torch.manual_seed(42)
    x = torch.randn(B, S, device="cuda", dtype=torch.float32)
    ref = F.softmax(x, dim=-1)
    out = softmax_autotuned(x)
    max_err = (out - ref).abs().max().item()
    ok = max_err < 1e-5
    status = "PASS" if ok else "FAIL"
    print(f"verify [{B},{S}]: max_err={max_err:.2e}  {status}")
    if not ok:
        raise AssertionError(f"数值验证失败: [{B},{S}] max_err={max_err:.2e}")


def benchmark_fn(fn, warmup: int = 25, repeat: int = 100) -> float:
    """返回最小 kernel 延迟（ms）。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start_e = torch.cuda.Event(enable_timing=True)
    end_e = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeat):
        start_e.record()
        fn()
        end_e.record()
        torch.cuda.synchronize()
        times.append(start_e.elapsed_time(end_e))
    return min(times)


def main():
    parser = argparse.ArgumentParser(description="Softmax autotune 验证 + 快速 benchmark")
    parser.add_argument("--B", type=int, default=8)
    parser.add_argument("--S", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("ROCm GPU 不可用，请在实验机上运行")

    print(f"torch: {torch.__version__}")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print()

    # 数值验证
    for S in [512, 2048, 8192]:
        verify(args.B, S)
    print()

    # 快速 benchmark
    x = torch.randn(args.B, args.S, device="cuda", dtype=torch.float32)
    ms_triton = benchmark_fn(lambda: softmax_autotuned(x), args.warmup, args.repeat)
    ms_torch = benchmark_fn(lambda: F.softmax(x, dim=-1), args.warmup, args.repeat)

    bw_triton = 2 * args.B * args.S * 4 / (ms_triton / 1000) / 1e9
    bw_torch = 2 * args.B * args.S * 4 / (ms_torch / 1000) / 1e9

    print(f"形状 [{args.B},{args.S}], fp32")
    print(f"  triton autotune : {ms_triton:.4f} ms  /  {bw_triton:.2f} GB/s")
    print(f"  torch F.softmax : {ms_torch:.4f} ms  /  {bw_torch:.2f} GB/s")
    print(f"  速比 (torch/triton): {ms_torch / ms_triton:.2f}x")


if __name__ == "__main__":
    main()
