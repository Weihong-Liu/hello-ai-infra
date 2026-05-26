#!/usr/bin/env bash
# run_all.sh — Triton Matmul 章节实验入口
#
# 假设调用前已经执行：
#   cd code/part4-triton
#   source ./activate-rocm.sh
#
# 所有输出写到 chapter2/logs/
# 失败时返回非零退出码。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "=== Triton Matmul 实验 ===" | tee "$LOG_DIR/run_all.log"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_DIR/run_all.log"
echo "Python: $(python --version 2>&1)" | tee -a "$LOG_DIR/run_all.log"
echo "Triton: $(python -c 'import triton; print(triton.__version__)' 2>&1)" \
    | tee -a "$LOG_DIR/run_all.log"
echo "" | tee -a "$LOG_DIR/run_all.log"

# --- 步骤 1：数值验证 ---
echo "[ 步骤 1 ] 数值验证 vs torch.matmul ..." | tee -a "$LOG_DIR/run_all.log"
python verify_vs_torch.py 2>&1 | tee "$LOG_DIR/verify_vs_torch.log"
echo "数值验证完成" | tee -a "$LOG_DIR/run_all.log"
echo "" | tee -a "$LOG_DIR/run_all.log"

# --- 步骤 2：Benchmark ---
echo "[ 步骤 2 ] Benchmark（warmup=25, repeat=100）..." | tee -a "$LOG_DIR/run_all.log"
python bench_matmul.py --warmup 25 --repeat 100 2>&1 | tee "$LOG_DIR/bench_matmul.log"
echo "Benchmark 完成" | tee -a "$LOG_DIR/run_all.log"
echo "" | tee -a "$LOG_DIR/run_all.log"

echo "=== 全部步骤完成 ===" | tee -a "$LOG_DIR/run_all.log"
echo "日志目录: $LOG_DIR" | tee -a "$LOG_DIR/run_all.log"
ls "$LOG_DIR/" | tee -a "$LOG_DIR/run_all.log"
