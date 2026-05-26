#!/usr/bin/env bash
# run_all.sh — part3-hip-kernels chapter6 实验主入口
# 执行前提：已 source ../activate-rocm.sh（在 code/part3-hip-kernels/ 下）
# 本脚本把所有日志写到 ./logs/（相对 chapter6 目录）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

echo "=== [1/4] 编译 HIP kernels ==="
hipcc matmul.hip      -O3 -o matmul      2>&1 | tee "$LOG_DIR/compile_matmul.log"
hipcc matmul_bench.hip -O3 -o matmul_bench 2>&1 | tee "$LOG_DIR/compile_bench.log"

echo "=== [2/4] 编译 rocBLAS baseline ==="
hipcc rocblas_baseline.cpp -lrocblas -O3 -o rocblas_baseline \
    2>&1 | tee "$LOG_DIR/compile_rocblas.log" || {
    echo "[WARN] rocBLAS 编译失败，跳过 rocBLAS 对照步骤"
    SKIP_ROCBLAS=1
}
SKIP_ROCBLAS=${SKIP_ROCBLAS:-0}

echo ""
echo "=== [3/4] 正确性验证 ==="
./matmul 512 512 512 2>&1 | tee "$LOG_DIR/verify_correctness.log"
echo "verify_vs_torch.py（需要 PyTorch ROCm）："
python verify_vs_torch.py 512 512 512 2>&1 | tee -a "$LOG_DIR/verify_correctness.log" || \
    echo "[WARN] verify_vs_torch.py 失败，继续..."

echo ""
echo "=== [4/4] 性能基准测试 ==="

# rocBLAS 单独形状扫描
if [[ "$SKIP_ROCBLAS" == "0" ]]; then
    echo "--- rocBLAS 形状扫描 ---"
    for SIZE in 512 1024 2048 4096; do
        ./rocblas_baseline $SIZE $SIZE $SIZE 2>&1
    done 2>&1 | tee "$LOG_DIR/bench_rocblas.log"
fi

# HIP kernel 版本逐一形状扫描
echo "--- HIP kernel 版本对比 ---"
for VER in 0 1 2 3 4; do
    VER_NAMES=("v0_naive" "v1_lds" "v2_bk32" "v3_reg4" "v4_combined")
    VER_NAME="${VER_NAMES[$VER]}"
    echo "版本 $VER ($VER_NAME)："
    for SIZE in 512 1024 2048 4096; do
        # v0 在大矩阵很慢，超过 2048 跳过
        if [[ "$VER" == "0" && "$SIZE" -gt 2048 ]]; then
            echo "  M=N=K=$SIZE  avg_ms=SKIP（v0 过慢）"
            continue
        fi
        ./matmul_bench $VER $SIZE $SIZE $SIZE 5 20 2>&1
    done
done 2>&1 | tee "$LOG_DIR/bench_kernels.log"

# Python 综合报告（如果 bench_matmul.py 能运行）
echo ""
echo "--- 综合 Python 报告 ---"
python bench_matmul.py --shapes 512 1024 2048 4096 --warmup 5 --repeat 20 \
    2>&1 | tee "$LOG_DIR/bench_matmul.log" || \
    echo "[WARN] bench_matmul.py 失败，见上方分项日志"

echo ""
echo "=== 所有实验完成，日志在 $LOG_DIR/ ==="
ls -lh "$LOG_DIR/"
