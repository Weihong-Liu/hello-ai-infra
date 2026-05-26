#!/usr/bin/env bash
# run_all.sh — 顺序运行 chapter4 的全部实验
# 调用前提：已在 code/part4-triton/ 目录下 source ./activate-rocm.sh
# 产出：logs/ 下的 .csv 和 .log 文件

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

echo "=== [1/3] 数值验证：verify_vs_torch.py ==="
python verify_vs_torch.py 2>&1 | tee "${LOG_DIR}/verify_vs_torch.log"
echo ""

echo "=== [2/3] Benchmark D=64, causal=False ==="
python bench_attention.py --D 64 2>&1 | tee "${LOG_DIR}/bench_attention_D64_causalFalse.log"
echo ""

echo "=== [2/3] Benchmark D=64, causal=True ==="
python bench_attention.py --D 64 --causal 2>&1 | tee "${LOG_DIR}/bench_attention_D64_causalTrue.log"
echo ""

echo "=== [2/3] Benchmark D=128, causal=False ==="
python bench_attention.py --D 128 2>&1 | tee "${LOG_DIR}/bench_attention_D128_causalFalse.log"
echo ""

echo "=== [2/3] Benchmark D=128, causal=True ==="
python bench_attention.py --D 128 --causal 2>&1 | tee "${LOG_DIR}/bench_attention_D128_causalTrue.log"
echo ""

echo "=== [3/3] 全部完成 ==="
echo "日志写入 ${LOG_DIR}/"
ls -lh "${LOG_DIR}/"
