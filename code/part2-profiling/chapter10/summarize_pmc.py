"""Summarize per-kernel PMC counter values from rocprofv3 single-counter runs.

Reads each chapter4/profiles/pmc_<NAME>/pmc_<NAME>_counter_collection.csv
and prints aggregate stats (mean / median / total / per-kernel-type mean) for
the named counter. Output is a JSON written to chapter4/logs/pmc_summary.json.
"""
from __future__ import annotations

import csv
import json
import re
import statistics
import sys
from pathlib import Path


COUNTERS = [
    "FETCH_SIZE",
    "WRITE_SIZE",
    "L2CacheHit",
    "MemUnitBusy",
    "OccupancyPercent",
    "Wavefronts",
    "VALUInsts",
    "GPUBusy",
]


def short_name(kernel_name: str) -> str:
    """Strip C++ template noise to a short label."""
    m = re.search(r"(?:CUDAFunctorOnSelf_(?:add|sub|mul)|"
                  r"AUnaryFunctor.*?(MulFunctor|AddFunctor|SubFunctor)|"
                  r"BinaryFunctor.*?(MulFunctor|AddFunctor|SubFunctor)|"
                  r"(sin_kernel_cuda|cos_kernel_cuda|tanh_kernel_cuda|"
                  r"sqrt_kernel_cuda|relu_kernel_cuda|launch_clamp_scalar))",
                  kernel_name)
    if m:
        # take first non-empty group
        for g in m.groups():
            if g:
                return g
    # fallback: chop generics
    return kernel_name[:60]


def summarize_counter(name: str, base: Path) -> dict:
    csv_path = base / f"pmc_{name}" / f"pmc_{name}_counter_collection.csv"
    if not csv_path.exists():
        return {"counter": name, "error": f"missing {csv_path}"}
    rows = list(csv.DictReader(open(csv_path)))
    values = [float(r["Counter_Value"]) for r in rows]
    by_kernel: dict[str, list[float]] = {}
    for r in rows:
        k = short_name(r["Kernel_Name"])
        by_kernel.setdefault(k, []).append(float(r["Counter_Value"]))
    return {
        "counter": name,
        "n_kernel_dispatches": len(values),
        "total": sum(values),
        "mean": statistics.mean(values) if values else 0,
        "median": statistics.median(values) if values else 0,
        "min": min(values) if values else 0,
        "max": max(values) if values else 0,
        "per_kernel_mean": {k: statistics.mean(v) for k, v in by_kernel.items()},
        "per_kernel_count": {k: len(v) for k, v in by_kernel.items()},
    }


def main():
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out = {c: summarize_counter(c, base) for c in COUNTERS}
    print(json.dumps(out, indent=2))
    out_path = base.parent / "logs" / "pmc_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
