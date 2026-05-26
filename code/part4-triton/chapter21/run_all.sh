#!/usr/bin/env bash
# run_all.sh — 第21章 Triton 自动调参 实验入口
#
# 前提：已在 code/part4-triton/ 目录下执行
#   uv sync && source ./activate-rocm.sh
#
# 顺序步骤：
#   1. 数值验证（Softmax + Matmul）
#   2. dump_best_configs：触发 autotune，导出各形状最优 config -> JSON
#   3. bench_autotuned_vs_default：固定 config vs autotune 性能对比
#
# 所有日志写到 chapter5/logs/
# 失败时返回非零退出码。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# 打印环境信息
echo "=== [0/3] 环境信息 ===" | tee "${LOG_DIR}/run_all.log"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${LOG_DIR}/run_all.log"
echo "Python: $(python --version 2>&1)" | tee -a "${LOG_DIR}/run_all.log"
echo "Triton: $(python -c 'import triton; print(triton.__version__)' 2>&1)" \
    | tee -a "${LOG_DIR}/run_all.log"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>&1)" \
    | tee -a "${LOG_DIR}/run_all.log"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))' 2>&1)" \
    | tee -a "${LOG_DIR}/run_all.log"
echo "" | tee -a "${LOG_DIR}/run_all.log"

# --- 步骤 1：数值验证 ---
echo "=== [1/3] 数值验证 ===" | tee -a "${LOG_DIR}/run_all.log"
python softmax_autotuned.py --B 8 --S 2048 2>&1 | tee "${LOG_DIR}/verify_softmax.log"
python matmul_autotuned.py  --M 1024 --N 1024 --K 1024 2>&1 | tee "${LOG_DIR}/verify_matmul.log"
echo "数值验证完成" | tee -a "${LOG_DIR}/run_all.log"
echo ""

# --- 步骤 2：dump best configs ---
echo "=== [2/3] dump_best_configs（触发 autotune，导出最优 config）===" | tee -a "${LOG_DIR}/run_all.log"
# autotune 首次编译可能需要数分钟
python dump_best_configs.py 2>&1 | tee "${LOG_DIR}/dump_best_configs.log"
echo "dump 完成" | tee -a "${LOG_DIR}/run_all.log"
echo ""

# --- 步骤 3：固定 vs autotune 性能对比 ---
echo "=== [3/3] bench_autotuned_vs_default（warmup=25, repeat=100）===" | tee -a "${LOG_DIR}/run_all.log"
python bench_autotuned_vs_default.py --warmup 25 --repeat 100 2>&1 | tee "${LOG_DIR}/bench_vs_default.log"
echo "Benchmark 完成" | tee -a "${LOG_DIR}/run_all.log"
echo ""

echo "=== 全部步骤完成 ===" | tee -a "${LOG_DIR}/run_all.log"
echo "日志目录: ${LOG_DIR}" | tee -a "${LOG_DIR}/run_all.log"
ls "${LOG_DIR}/" | tee -a "${LOG_DIR}/run_all.log"
