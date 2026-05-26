#!/usr/bin/env bash
# run_all.sh — Triton 编程模型章节实验入口
#
# 假设调用前已经执行：
#   cd code/part4-triton
#   source ./activate-rocm.sh
#
# 所有输出写到 chapter1/logs/
# 失败时返回非零退出码。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "=== Triton 编程模型实验 ===" | tee "$LOG_DIR/run_all.log"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_DIR/run_all.log"
echo "Python: $(python --version 2>&1)" | tee -a "$LOG_DIR/run_all.log"
echo "Triton: $(python -c 'import triton; print(triton.__version__)' 2>&1)" \
    | tee -a "$LOG_DIR/run_all.log"
echo "Torch:  $(python -c 'import torch; print(torch.__version__)' 2>&1)" \
    | tee -a "$LOG_DIR/run_all.log"
echo "" | tee -a "$LOG_DIR/run_all.log"

# --- 步骤 1：环境验证（check_triton_amd.py）---
echo "[ 步骤 1 ] 验证 Triton AMD 环境..." | tee -a "$LOG_DIR/run_all.log"
python check_triton_amd.py 2>&1 | tee "$LOG_DIR/check_triton_amd.log"
echo "环境验证完成" | tee -a "$LOG_DIR/run_all.log"
echo "" | tee -a "$LOG_DIR/run_all.log"

# --- 步骤 2：向量加法验证（vector_add_triton.py）---
echo "[ 步骤 2 ] 运行向量加法数值验证..." | tee -a "$LOG_DIR/run_all.log"
python vector_add_triton.py 2>&1 | tee "$LOG_DIR/vector_add_triton.log"
echo "向量加法验证完成" | tee -a "$LOG_DIR/run_all.log"
echo "" | tee -a "$LOG_DIR/run_all.log"

# --- 步骤 3：mask 演示（vector_copy_with_mask.py）---
echo "[ 步骤 3 ] 运行向量 mask 拷贝演示..." | tee -a "$LOG_DIR/run_all.log"
python vector_copy_with_mask.py 2>&1 | tee "$LOG_DIR/vector_copy_with_mask.log"
echo "mask 演示完成" | tee -a "$LOG_DIR/run_all.log"
echo "" | tee -a "$LOG_DIR/run_all.log"

echo "=== 全部步骤完成 ===" | tee -a "$LOG_DIR/run_all.log"
echo "日志目录: $LOG_DIR" | tee -a "$LOG_DIR/run_all.log"
ls "$LOG_DIR/" | tee -a "$LOG_DIR/run_all.log"
