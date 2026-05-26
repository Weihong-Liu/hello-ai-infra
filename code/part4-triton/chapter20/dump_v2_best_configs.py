"""dump_v2_best_configs.py — trigger v2 attention autotune across all
(S, D, causal) configurations used in bench_attention.py and dump the best
config per (S, D) tuple to logs/v2_best_configs.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from attention_v2_blocked import attention_v2, attention_v2_kernel

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

B, H = 4, 8
SHAPES = [(S, D, causal) for D in (64, 128) for S in (512, 1024, 2048, 4096) for causal in (False, True)]


def main():
    if not torch.cuda.is_available():
        raise SystemExit("ROCm GPU 不可用")

    print(f"torch: {torch.__version__}  device: {torch.cuda.get_device_name(0)}")

    for S, D, causal in SHAPES:
        q = torch.randn(B, H, S, D, device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        _ = attention_v2(q, k, v, causal=causal)
        torch.cuda.synchronize()
        print(f"  warmed S={S} D={D} causal={causal}")

    cache = attention_v2_kernel.cache
    out = {}
    for k, cfg in cache.items():
        # autotune key=['S', 'D'] but Triton appends tensor dtypes
        S = k[0] if len(k) >= 1 else None
        D = k[1] if len(k) >= 2 else None
        label = f"S={S}_D={D}"
        out[label] = {
            "key_tuple": [str(x) for x in k],
            "BLOCK_S": cfg.kwargs.get("BLOCK_S"),
            "num_warps": cfg.num_warps,
            "num_stages": cfg.num_stages,
        }
        print(f"  best  S={S} D={D}: BLOCK_S={cfg.kwargs.get('BLOCK_S')} num_warps={cfg.num_warps} num_stages={cfg.num_stages}")

    out_path = LOG_DIR / "v2_best_configs.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n写入: {out_path}")


if __name__ == "__main__":
    main()
