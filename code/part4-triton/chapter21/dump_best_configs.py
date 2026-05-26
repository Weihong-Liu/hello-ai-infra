"""
dump_best_configs.py — 运行 autotune，将每种 (kernel, key) 下的最优 config 写入 JSON。

autotune 第一次调用时对每个 key 值（如 S=512, S=2048 等）跑完所有候选 config，
之后缓存在 kernel 对象的内部字典里。本脚本：
  1. 对 Softmax 和 Matmul 的代表性 key 值触发 autotune；
  2. 从 kernel.best_config 读出胜出的 config；
  3. 把结果写到 logs/best_configs.json。

用法（source activate-rocm.sh 后）：
    python dump_best_configs.py

注意：
  - autotune 首次运行可能耗时数分钟（每个 config 都要 JIT 编译 + benchmark）。
  - 结果依赖硬件和 ROCm / Triton 版本，换机器或升级环境后应重新跑一遍。
  - best_config 字典的 key 是 autotune 的 key 参数在调用时的值组成的元组。
"""

import json
from pathlib import Path

import torch

from softmax_autotuned import softmax_kernel_autotuned, softmax_autotuned
from matmul_autotuned import matmul_kernel_autotuned, matmul_autotuned

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Softmax 的代表性行宽（与正文 benchmark 保持一致）
SOFTMAX_S_VALUES = [512, 1024, 2048, 4096, 8192]
SOFTMAX_B = 8

# Matmul 的代表性形状
MATMUL_SHAPES = [
    (512,  512,  512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 2048, 2048),
]


def trigger_softmax_autotune():
    """对各个 S 值触发 Softmax autotune，返回 best config 字典。"""
    print("=== Softmax autotune ===")
    results = {}
    for S in SOFTMAX_S_VALUES:
        x = torch.randn(SOFTMAX_B, S, device="cuda", dtype=torch.float32)
        # 第一次调用触发 autotune；后续调用使用缓存
        _ = softmax_autotuned(x)
        torch.cuda.synchronize()

        # Autotuner.cache 是 key 元组 → Config 的映射；
        # key=["S"] 但实际 cache key 会被 Triton 拼上张量 dtype，
        # 形如 (S, 'torch.float32', 'torch.float32')。先按首元素匹配。
        cache = softmax_kernel_autotuned.cache
        cfg = next((v for k, v in cache.items() if k and k[0] == S), None)
        if cfg is not None:
            entry = {
                "BLOCK_SIZE": cfg.kwargs["BLOCK_SIZE"],
                "num_warps": cfg.num_warps,
                "num_stages": cfg.num_stages,
            }
        else:
            entry = {"note": "best_config 未找到，可能 autotune 尚未完成"}

        results[str(S)] = entry
        print(f"  S={S:6d}: {entry}")

    return results


def trigger_matmul_autotune():
    """对各个 (M,N,K) 触发 Matmul autotune，返回 best config 字典。"""
    print("=== Matmul autotune ===")
    results = {}
    for M, N, K in MATMUL_SHAPES:
        a = torch.randn(M, K, device="cuda", dtype=torch.float16)
        b = torch.randn(K, N, device="cuda", dtype=torch.float16)
        _ = matmul_autotuned(a, b)
        torch.cuda.synchronize()

        # key=["M", "N", "K"] 但实际 cache key 会拼上张量 dtype。按前三个元素匹配。
        cache = matmul_kernel_autotuned.cache
        cfg = next((v for k, v in cache.items() if len(k) >= 3 and k[:3] == (M, N, K)), None)
        if cfg is not None:
            entry = {
                "BLOCK_M": cfg.kwargs["BLOCK_M"],
                "BLOCK_N": cfg.kwargs["BLOCK_N"],
                "BLOCK_K": cfg.kwargs["BLOCK_K"],
                "num_warps": cfg.num_warps,
                "num_stages": cfg.num_stages,
            }
        else:
            entry = {"note": "best_config 未找到，可能 autotune 尚未完成"}

        shape_key = f"{M}x{N}x{K}"
        results[shape_key] = entry
        print(f"  [{M},{N},{K}]: {entry}")

    return results


def main():
    if not torch.cuda.is_available():
        raise SystemExit("ROCm GPU 不可用，请在实验机上运行")

    device_name = torch.cuda.get_device_name(0)
    print(f"torch: {torch.__version__}")
    print(f"device: {device_name}")
    print()

    import triton
    triton_version = triton.__version__

    softmax_configs = trigger_softmax_autotune()
    print()
    matmul_configs = trigger_matmul_autotune()

    output = {
        "hardware": device_name,
        "triton_version": triton_version,
        "torch_version": torch.__version__,
        "softmax_best_configs": softmax_configs,
        "matmul_best_configs": matmul_configs,
    }

    out_path = LOG_DIR / "best_configs.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print()
    print(f"最优配置已写入: {out_path}")


if __name__ == "__main__":
    main()
