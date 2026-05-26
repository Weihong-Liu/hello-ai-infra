#!/usr/bin/env bash
# run.sh —— part3 chapter2 vector add 编译 + benchmark sweep
#
# 前置：已 source activate-rocm.sh
# 用法：bash run.sh
#
# 步骤：
#   1) 编译 vector_add.hip
#   2) 跑 CPU baseline（vector_add_cpu.cpp）
#   3) 三个 kernel × 三个 block size × 一个 size 的 benchmark sweep
#   4) 三个 kernel × 几个 size 的扫描，记录 GB/s
#
# 产出：
#   build/vector_add_bench
#   build/vector_add_cpu
#   logs/compile.log
#   logs/cpu_baseline.log
#   logs/sweep_block.log
#   logs/sweep_size.log
#   logs/summary.csv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/logs"
BUILD_DIR="$SCRIPT_DIR/build"
mkdir -p "$LOG_DIR" "$BUILD_DIR"

command -v hipcc >/dev/null || { echo "[ERROR] hipcc 未在 PATH" >&2; exit 1; }

# -------- 编译 --------
echo "=== [1/4] 编译 vector_add ==="
hipcc vector_add.hip -O2 -o "$BUILD_DIR/vector_add_bench" \
    2>&1 | tee "$LOG_DIR/compile.log"

# CPU baseline 直接 inline 写一个小 C++ 程序，避免额外文件
echo "=== [2/4] 编译 + 跑 CPU baseline ==="
cat > /tmp/_vec_add_cpu.cpp <<'EOF'
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
int main(int argc, char** argv) {
  int n = (argc > 1) ? atoi(argv[1]) : (1 << 24);
  int repeat = (argc > 2) ? atoi(argv[2]) : 5;
  std::vector<float> a(n, 1.0f), b(n, 2.0f), c(n, 0.0f);
  // warmup
  for (int i = 0; i < n; ++i) c[i] = a[i] + b[i];
  double best = 1e18;
  for (int r = 0; r < repeat; ++r) {
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n; ++i) c[i] = a[i] + b[i];
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (ms < best) best = ms;
  }
  // sanity
  bool ok = (c[0] == 3.0f && c[n - 1] == 3.0f);
  double bw = (double)n * sizeof(float) * 3.0 / (best / 1000.0) / 1e9;
  printf("cpu_n=%d  best_ms=%.4f  bw_GBs=%.2f  ok=%d\n", n, best, bw, (int)ok);
  return ok ? 0 : 1;
}
EOF
g++ -O2 -std=c++17 /tmp/_vec_add_cpu.cpp -o "$BUILD_DIR/vector_add_cpu"
"$BUILD_DIR/vector_add_cpu" 16777216 5 2>&1 | tee "$LOG_DIR/cpu_baseline.log"
"$BUILD_DIR/vector_add_cpu" 67108864 5 2>&1 | tee -a "$LOG_DIR/cpu_baseline.log"

# -------- block size sweep --------
echo ""
echo "=== [3/4] block size sweep（n=16M）==="
SWEEP_LOG="$LOG_DIR/sweep_block.log"
: > "$SWEEP_LOG"
for kernel in naive grid_stride strided_bad; do
  for blk in 64 128 256 512 1024; do
    echo "--- kernel=$kernel block=$blk ---" | tee -a "$SWEEP_LOG"
    "$BUILD_DIR/vector_add_bench" \
        --kernel "$kernel" --size 16777216 --block "$blk" \
        --warmup 5 --repeat 30 \
        2>&1 | tee -a "$SWEEP_LOG"
  done
done

# -------- size sweep --------
echo ""
echo "=== [4/4] size sweep（block=256）==="
SIZE_LOG="$LOG_DIR/sweep_size.log"
: > "$SIZE_LOG"
for kernel in naive grid_stride strided_bad; do
  for n in 1048576 4194304 16777216 67108864; do
    echo "--- kernel=$kernel size=$n ---" | tee -a "$SIZE_LOG"
    "$BUILD_DIR/vector_add_bench" \
        --kernel "$kernel" --size "$n" --block 256 \
        --warmup 5 --repeat 30 \
        2>&1 | tee -a "$SIZE_LOG"
  done
done

# -------- 汇总 CSV --------
echo ""
echo "=== 生成 summary.csv ==="
SUM="$LOG_DIR/summary.csv"
{
  echo "kernel,size,block,mean_ms,min_ms,bandwidth_gb_s"
  python3 - "$SWEEP_LOG" "$SIZE_LOG" <<'PY'
import re, sys
patterns = {
    'kernel': re.compile(r'^kernel:\s*(\S+)'),
    'size':   re.compile(r'^vector_size:\s*(\d+)'),
    'block':  re.compile(r'^block_size:\s*(\d+)'),
    'mean':   re.compile(r'^mean_ms:\s*(\S+)'),
    'min':    re.compile(r'^min_ms:\s*(\S+)'),
    'bw':     re.compile(r'^bandwidth_gb_s:\s*(\S+)'),
}
for path in sys.argv[1:]:
    cur = {}
    with open(path) as f:
        for line in f:
            for k, p in patterns.items():
                m = p.match(line.strip())
                if m:
                    cur[k] = m.group(1)
            if 'bw' in cur and 'kernel' in cur and 'size' in cur and 'block' in cur:
                print(f"{cur['kernel']},{cur['size']},{cur['block']},"
                      f"{cur.get('mean','')},{cur.get('min','')},{cur['bw']}")
                cur = {}
PY
} > "$SUM"
echo "summary.csv:"
cat "$SUM"

echo ""
echo "=== 完成 ==="
ls -lh "$LOG_DIR"
