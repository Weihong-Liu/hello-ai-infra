#!/usr/bin/env bash
# run_all.sh — part3 chapter4 Softmax 优化
# 一键编译 → 验证正确性 → 与 PyTorch 对齐 → benchmark → rocprof 采集
#
# 前置：
#   - 已 source ../activate-rocm.sh（hipcc、python3、rocprof 在 PATH 中）
#   - 在 code/part3-hip-kernels/chapter14/ 目录下执行
#
# 产出（logs/ 目录）：
#   logs/compile.log           — hipcc 编译日志
#   logs/verify.log            — v0/v1/v2/v3 vs CPU 参考的 max_abs_error
#   logs/verify_torch.log      — v1/v2/v3 vs torch.softmax 的 max_diff
#   logs/bench_stdout.log      — bench_softmax.py 全量终端输出
#   logs/bench_summary.csv     — 多形状 × 多版本性能数据（CSV）
#   logs/rocprof_v1.log        — v1 三趟稳定版的 rocprof 输出
#   logs/rocprof_v3.log        — v3 float4 向量化版的 rocprof 输出
#   logs/rocprof_v1_stats.csv  — rocprof --stats 输出（v1）
#   logs/rocprof_v3_stats.csv  — rocprof --stats 输出（v3）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs"
BUILD_DIR="$SCRIPT_DIR/build"
mkdir -p "$LOG_DIR" "$BUILD_DIR"

# ─── 工具检查 ────────────────────────────────────────────────────────────────
check_cmd() {
    if ! command -v "$1" &>/dev/null; then
        echo "[ERROR] 命令未找到：$1。请先 source activate-rocm.sh。" >&2
        exit 1
    fi
}
check_cmd hipcc
check_cmd python3

# ─── 步骤 1：编译 ─────────────────────────────────────────────────────────────
echo "=== [1/5] 编译 softmax.hip ==="
hipcc softmax.hip -O2 -std=c++17 -o "$BUILD_DIR/softmax_bin" \
    2>&1 | tee "$LOG_DIR/compile.log"

if [[ ! -f "$BUILD_DIR/softmax_bin" ]]; then
    echo "[ERROR] 编译失败，请查看 logs/compile.log" >&2
    exit 1
fi
echo "[OK] 编译成功 → $BUILD_DIR/softmax_bin"

# ─── 步骤 2：正确性验证（vs CPU 参考）──────────────────────────────────────
echo ""
echo "=== [2/5] 正确性验证（v0/v1/v2/v3 vs CPU 参考，B=8 S=2048）==="
VERIFY_LOG="$LOG_DIR/verify.log"
"$BUILD_DIR/softmax_bin" 8 2048 2>&1 | tee "$VERIFY_LOG"
echo "[OK] 正确性验证完成 → $VERIFY_LOG"

# ─── 步骤 3：与 PyTorch torch.softmax 对齐 ──────────────────────────────────
echo ""
echo "=== [3/5] 与 PyTorch torch.softmax 对齐（B=8 S=2048）==="
TORCH_LOG="$LOG_DIR/verify_torch.log"
python3 verify_vs_torch.py --B 8 --S 2048 2>&1 | tee "$TORCH_LOG" || {
    echo "[WARN] torch 对齐检测到问题，保留日志继续执行。" \
        | tee -a "$TORCH_LOG"
}
echo "[OK] PyTorch 对齐完成 → $TORCH_LOG"

# ─── 步骤 4：多形状 × 多版本 Benchmark ──────────────────────────────────────
echo ""
echo "=== [4/5] 多形状 Benchmark（v0/v1/v2/v3 × 10 种形状）==="
BENCH_LOG="$LOG_DIR/bench_stdout.log"
python3 bench_softmax.py 2>&1 | tee "$BENCH_LOG"

if [[ -f "$LOG_DIR/bench_summary.csv" ]]; then
    echo "[OK] Benchmark 完成 → $LOG_DIR/bench_summary.csv"
else
    echo "[WARN] bench_summary.csv 未生成，可能 benchmark 途中出错。" >&2
fi

# ─── 步骤 5：rocprof 计数器采集（v1 vs v3，B=32 S=4096）──────────────────────
echo ""
echo "=== [5/5] rocprof 采集（v1 三趟稳定 vs v3 float4，B=32 S=4096）==="

BENCH_B=32
BENCH_S=4096

if command -v rocprof &>/dev/null; then
    # v1：三趟稳定版
    echo "  [rocprof] 采集 v1 (B=$BENCH_B S=$BENCH_S)..."
    rocprof --stats \
        -o "$LOG_DIR/rocprof_v1_stats.csv" \
        "$BUILD_DIR/softmax_bin" "$BENCH_B" "$BENCH_S" 1 --warmup 0 --repeat 3 \
        >"$LOG_DIR/rocprof_v1.log" 2>&1 || true
    echo "  [OK] v1 rocprof → $LOG_DIR/rocprof_v1.log"

    # v3：float4 向量化版
    echo "  [rocprof] 采集 v3 (B=$BENCH_B S=$BENCH_S)..."
    rocprof --stats \
        -o "$LOG_DIR/rocprof_v3_stats.csv" \
        "$BUILD_DIR/softmax_bin" "$BENCH_B" "$BENCH_S" 3 --warmup 0 --repeat 3 \
        >"$LOG_DIR/rocprof_v3.log" 2>&1 || true
    echo "  [OK] v3 rocprof → $LOG_DIR/rocprof_v3.log"
else
    echo "  [WARN] rocprof 未在 PATH 中，跳过 profiling 采集。"
    echo "  请确认已 source activate-rocm.sh 且 ROCm 安装正确。"
fi

# ─── 完成摘要 ────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  run_all.sh 完成"
echo "========================================"
echo "产出文件："
for f in \
    "$LOG_DIR/compile.log" \
    "$LOG_DIR/verify.log" \
    "$LOG_DIR/verify_torch.log" \
    "$LOG_DIR/bench_stdout.log" \
    "$LOG_DIR/bench_summary.csv" \
    "$LOG_DIR/rocprof_v1.log" \
    "$LOG_DIR/rocprof_v3.log" \
    "$LOG_DIR/rocprof_v1_stats.csv" \
    "$LOG_DIR/rocprof_v3_stats.csv"; do
    if [[ -f "$f" ]]; then
        echo "  [x] $f"
    else
        echo "  [ ] $f (未生成)"
    fi
done
