#!/usr/bin/env bash
# run_all.sh
# Part 3 Chapter 3: Reduction 优化 — 一键编译 + benchmark + rocprof 采集
#
# 前置：已 source activate-rocm.sh，在 chapter3/ 目录执行
#
# 用法：
#   bash run_all.sh
#
# 产出（logs/ 目录）：
#   logs/compile.log          — hipcc 编译日志
#   logs/bench_summary.csv    — 全量 benchmark 结果（by bench_reduction.py）
#   logs/bench_stdout.log     — bench 全量终端输出
#   logs/rocprof_v1.log       — v1 LDS tree 的 rocprof 计数器采集
#   logs/rocprof_v3.log       — v3 unrolled 的 rocprof 计数器采集
#   logs/rocprof_v1_stats.csv — rocprof --stats 输出（v1）
#   logs/rocprof_v3_stats.csv — rocprof --stats 输出（v3）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# ─── 工具检查 ────────────────────────────────────────────────────────────────

check_cmd() {
    if ! command -v "$1" &>/dev/null; then
        echo "[ERROR] 命令未找到：$1。请先 source activate-rocm.sh。" >&2
        exit 1
    fi
}

check_cmd hipcc
check_cmd python3

# ─── 步骤 1：编译 ─────────────────────────────────────────────────────────

echo "=== [1/4] 编译 reduction.hip ==="
hipcc reduction.hip -O2 -o reduction \
    2>&1 | tee "$LOG_DIR/compile.log"

if [[ ! -f ./reduction ]]; then
    echo "[ERROR] 编译失败，请查看 logs/compile.log" >&2
    exit 1
fi
echo "[OK] 编译成功 → ./reduction"

# ─── 步骤 2：正确性验证 ─────────────────────────────────────────────────────

echo ""
echo "=== [2/4] 正确性验证（N=1M，全版本）==="
VERIFY_LOG="$LOG_DIR/verify.log"
./reduction --all --n 1048576 --warmup 2 --repeat 5 2>&1 | tee "$VERIFY_LOG"
echo "[OK] 验证完成 → $VERIFY_LOG"

# ─── 步骤 3：全量 Benchmark ─────────────────────────────────────────────────

echo ""
echo "=== [3/4] 全量 Benchmark (4M / 16M / 64M / 256M) ==="
BENCH_LOG="$LOG_DIR/bench_stdout.log"
python3 bench_reduction.py 2>&1 | tee "$BENCH_LOG"

if [[ ! -f "$LOG_DIR/bench_summary.csv" ]]; then
    echo "[WARN] bench_summary.csv 未生成，可能 benchmark 途中出错。" >&2
fi
echo "[OK] Benchmark 完成 → $LOG_DIR/bench_summary.csv"

# ─── 步骤 4：rocprof 计数器采集（v1 vs v3，N=64M）──────────────────────────

echo ""
echo "=== [4/4] rocprof 采集（v1 LDS tree vs v3 unrolled，N=64M）==="

ROCPROF_N=67108864  # 64M

if command -v rocprof &>/dev/null; then
    # v1：LDS 归约树
    echo "  [rocprof] 采集 v1 ..."
    rocprof --stats \
        -o "$LOG_DIR/rocprof_v1_stats.csv" \
        ./reduction --version 1 --n "$ROCPROF_N" --warmup 0 --repeat 3 \
        >"$LOG_DIR/rocprof_v1.log" 2>&1 || true
    echo "  [OK] v1 rocprof → $LOG_DIR/rocprof_v1.log"

    # v3：多元素展开 + 向量化
    echo "  [rocprof] 采集 v3 ..."
    rocprof --stats \
        -o "$LOG_DIR/rocprof_v3_stats.csv" \
        ./reduction --version 3 --n "$ROCPROF_N" --warmup 0 --repeat 3 \
        >"$LOG_DIR/rocprof_v3.log" 2>&1 || true
    echo "  [OK] v3 rocprof → $LOG_DIR/rocprof_v3.log"
else
    echo "  [WARN] rocprof 未在 PATH 中，跳过 profiling 采集。"
    echo "  请确认 source activate-rocm.sh 后 rocprof 可用。"
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
