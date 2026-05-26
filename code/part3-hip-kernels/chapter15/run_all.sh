#!/usr/bin/env bash
# run_all.sh — part3-hip-kernels chapter5: LayerNorm 优化
# 前置：已 source ../activate-rocm.sh（在 code/part3-hip-kernels/ 目录下执行）
# 本脚本在 chapter5/ 目录下自包含执行，把所有日志写到 ./logs/
#
# 产出（logs/ 目录）：
#   logs/compile.log              — hipcc 编译日志
#   logs/verify.log               — 数值验证（v0/v1/v2 vs CPU 参考）
#   logs/verify_torch.log         — verify_vs_torch.py 输出
#   logs/bench_stdout.log         — bench_layernorm.py 全量终端输出
#   logs/bench_layernorm.csv      — 版本 × 形状 benchmark 结果（CSV）
#   logs/rocprof_v0.log           — v0 三步分离 rocprof 输出
#   logs/rocprof_v1.log           — v1 融合 rocprof 输出
#   logs/rocprof_v2.log           — v2 float4 rocprof 输出
#   logs/rocprof_v0_stats.csv     — rocprof --stats（v0）
#   logs/rocprof_v1_stats.csv     — rocprof --stats（v1）
#   logs/rocprof_v2_stats.csv     — rocprof --stats（v2）

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

echo "========================================"
echo "  Part 3 Chapter 5: LayerNorm 优化"
echo "========================================"
echo ""

# ─── 步骤 1：编译 ─────────────────────────────────────────────────────────

echo "=== [1/5] 编译 layernorm.hip ==="
hipcc layernorm.hip -O2 -o layernorm \
    2>&1 | tee "$LOG_DIR/compile.log"

if [[ ! -f ./layernorm ]]; then
    echo "[ERROR] 编译失败，请查看 logs/compile.log" >&2
    exit 1
fi
echo "[OK] 编译成功 → ./layernorm"
echo ""

# ─── 步骤 2：数值验证（v0/v1/v2，默认形状 B=4 S=32 H=768）────────────────

echo "=== [2/5] 数值验证（--all，B=4 S=32 H=768）==="
VERIFY_LOG="$LOG_DIR/verify.log"
./layernorm --all --B 4 --S 32 --H 768 --warmup 1 --repeat 3 \
    2>&1 | tee "$VERIFY_LOG"
echo "[OK] 验证完成 → $VERIFY_LOG"
echo ""

# 额外：H=1024 和 H=4096 各验一遍
for H_SIZE in 1024 4096; do
    echo "  验证 H=${H_SIZE} ..."
    ./layernorm --all --B 4 --S 32 --H "$H_SIZE" --warmup 1 --repeat 3 \
        2>&1 >> "$VERIFY_LOG"
done
echo ""

# ─── 步骤 3：verify_vs_torch.py ──────────────────────────────────────────

echo "=== [3/5] verify_vs_torch.py（PyTorch ROCm 对照）==="
TORCH_LOG="$LOG_DIR/verify_torch.log"
python3 verify_vs_torch.py --B 4 --S 32 --H 768 \
    2>&1 | tee "$TORCH_LOG" || \
    echo "[WARN] verify_vs_torch.py 返回非 0，请检查 $TORCH_LOG"
echo "[OK] torch 验证 → $TORCH_LOG"
echo ""

# ─── 步骤 4：全量 Benchmark ──────────────────────────────────────────────

echo "=== [4/5] 全量 Benchmark (H=768/1024/4096, v0/v1/v2) ==="
BENCH_LOG="$LOG_DIR/bench_stdout.log"
python3 bench_layernorm.py \
    --H 768 1024 4096 \
    --warmup 3 \
    --repeat 20 \
    2>&1 | tee "$BENCH_LOG"

if [[ ! -f "$LOG_DIR/bench_layernorm.csv" ]]; then
    echo "[WARN] bench_layernorm.csv 未生成，可能 benchmark 途中出错。" >&2
fi
echo "[OK] Benchmark 完成 → $LOG_DIR/bench_layernorm.csv"
echo ""

# ─── 步骤 5：rocprof 计数器采集（v0/v1/v2，H=4096，B=32 S=128）────────────

echo "=== [5/5] rocprof 采集（v0/v1/v2，B=32 S=128 H=4096）==="

ROCPROF_B=32
ROCPROF_S=128
ROCPROF_H=4096

if command -v rocprof &>/dev/null; then
    for VER in 0 1 2; do
        echo "  [rocprof] 采集 v${VER} ..."
        rocprof --stats \
            -o "$LOG_DIR/rocprof_v${VER}_stats.csv" \
            ./layernorm \
                --version "$VER" \
                --B "$ROCPROF_B" --S "$ROCPROF_S" --H "$ROCPROF_H" \
                --warmup 0 --repeat 3 --no-verify \
            >"$LOG_DIR/rocprof_v${VER}.log" 2>&1 || true
        echo "  [OK] v${VER} rocprof → $LOG_DIR/rocprof_v${VER}.log"
    done
else
    echo "  [WARN] rocprof 未在 PATH 中，跳过 profiling 采集。"
    echo "  请确认 source activate-rocm.sh 后 rocprof 可用。"
fi
echo ""

# ─── 完成摘要 ────────────────────────────────────────────────────────────────

echo "========================================"
echo "  run_all.sh 完成"
echo "========================================"
echo "产出文件："
EXPECTED_FILES=(
    "$LOG_DIR/compile.log"
    "$LOG_DIR/verify.log"
    "$LOG_DIR/verify_torch.log"
    "$LOG_DIR/bench_stdout.log"
    "$LOG_DIR/bench_layernorm.csv"
    "$LOG_DIR/rocprof_v0.log"
    "$LOG_DIR/rocprof_v1.log"
    "$LOG_DIR/rocprof_v2.log"
    "$LOG_DIR/rocprof_v0_stats.csv"
    "$LOG_DIR/rocprof_v1_stats.csv"
    "$LOG_DIR/rocprof_v2_stats.csv"
)
for f in "${EXPECTED_FILES[@]}"; do
    if [[ -f "$f" ]]; then
        echo "  [x] $f"
    else
        echo "  [ ] $f (未生成)"
    fi
done
