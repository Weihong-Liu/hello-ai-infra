#!/usr/bin/env bash
# run.sh —— part3 chapter4 softmax 编译 + 验证 + benchmark
#
# 前置：已 source activate-rocm.sh，且 part3-hip-kernels venv 已 uv sync
# 用法：bash run.sh
#
# 产出：
#   build/softmax_bin
#   logs/compile.log
#   logs/verify.log              —— v0/v1/v2/v3 vs CPU 参考的 max_abs_error
#   logs/torch_compare.log       —— v1/v2/v3 vs torch.softmax 的 max_diff
#   logs/bench.log               —— B/S 扫描，per-version 计时
#   logs/bench_summary.csv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs"
BUILD_DIR="$SCRIPT_DIR/build"
mkdir -p "$LOG_DIR" "$BUILD_DIR"

command -v hipcc >/dev/null || { echo "[ERROR] hipcc 未在 PATH" >&2; exit 1; }

# -------- 1. 编译 --------
echo "=== [1/4] 编译 softmax.hip ==="
hipcc softmax.hip -O2 -o "$BUILD_DIR/softmax_bin" \
    2>&1 | tee "$LOG_DIR/compile.log"

# softmax.hip 主程序里写 logs/*.bin（基于 CWD），所以保持 CWD=chapter4
# -------- 2. 跑各版本 vs CPU 参考 --------
echo ""
echo "=== [2/4] 正确性验证（v0/v1/v2/v3 vs CPU ref，B=8 S=2048）==="
"$BUILD_DIR/softmax_bin" 8 2048 2>&1 | tee "$LOG_DIR/verify.log"

# -------- 3. 与 PyTorch 对齐 --------
echo ""
echo "=== [3/4] 与 PyTorch torch.softmax 对齐 ==="
python3 verify_vs_torch.py 2>&1 | tee "$LOG_DIR/torch_compare.log" || \
    echo "[WARN] torch 对齐失败（可能 torch 不可用），保留日志继续。" \
        | tee -a "$LOG_DIR/torch_compare.log"

# -------- 4. 不同 hidden size 吞吐扫描 --------
echo ""
echo "=== [4/4] B/S 扫描 ==="
BENCH_LOG="$LOG_DIR/bench.log"
: > "$BENCH_LOG"
python3 bench_softmax.py 2>&1 | tee "$BENCH_LOG"

echo ""
echo "=== 完成 ==="
ls -lh "$LOG_DIR"
