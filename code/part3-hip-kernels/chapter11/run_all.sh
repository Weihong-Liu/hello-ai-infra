#!/usr/bin/env bash
# run_all.sh — 编译并运行 chapter1 全部 HIP 程序，日志写入 ./logs/
# 前提：已 source ../activate-rocm.sh（ROCm 环境已激活）
# 用法：bash run_all.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p build logs

# ────────────────────────────────────────────────────────────
# 辅助函数
# ────────────────────────────────────────────────────────────
log_step() { echo "==> $*"; }

compile() {
    local src="$1"
    local out="$2"
    log_step "编译 $src → build/$out"
    hipcc "$src" -O2 -o "build/$out" 2>&1 | tee "logs/${out}_compile.log"
    echo "compile_status: PASS"
}

run_binary() {
    local bin="$1"
    local log="$2"
    shift 2
    log_step "运行 build/$bin $*"
    "./build/$bin" "$@" 2>&1 | tee "logs/${log}.log"
    echo "run_status: PASS"
}

# ────────────────────────────────────────────────────────────
# 1. hello_hip：最小 hello-from-device，打印设备属性 + device printf
# ────────────────────────────────────────────────────────────
compile hello_hip.hip hello_hip
run_binary hello_hip hello_hip

# ────────────────────────────────────────────────────────────
# 2. vector_add_minimal：host alloc/copy/free + 加法 kernel + 验证
# ────────────────────────────────────────────────────────────
compile vector_add_minimal.hip vector_add_minimal
run_binary vector_add_minimal vector_add_minimal

# ────────────────────────────────────────────────────────────
# 3. error_check（带 HIP_CHECK 版本）：演示错误信息输出
#    预期：程序以 exit(1) 结束并打印 HIP error 信息，这是正常行为
# ────────────────────────────────────────────────────────────
compile error_check.hip error_check
log_step "运行 build/error_check（预期打印 HIP error 后 exit 1）"
set +e
./build/error_check 2>&1 | tee logs/error_check.log
ERROR_EXIT=$?
set -e
if [ "$ERROR_EXIT" -ne 0 ]; then
    echo "error_check exited with code $ERROR_EXIT (预期 non-zero，演示成功)"
else
    echo "error_check exited with code 0 (意外：预期触发错误)" >&2
    exit 1
fi

# ────────────────────────────────────────────────────────────
# 4. error_check --no-check 版本：演示静默出错
#    此版本正常退出（exit 0），不触发 set -e
# ────────────────────────────────────────────────────────────
log_step "运行 build/error_check --no-check（演示静默出错）"
run_binary error_check error_check_no_check --no-check

# ────────────────────────────────────────────────────────────
# 汇总
# ────────────────────────────────────────────────────────────
echo ""
echo "=============================="
echo "chapter1 smoke tests 全部完成"
echo "日志目录: $SCRIPT_DIR/logs/"
ls logs/
echo "=============================="
