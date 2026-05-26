#!/usr/bin/env bash
# code/part1-hardware-rocm/chapter4/run_all.sh
#
# 一键编译 + 跑三个 micro-benchmark，stdout 同时 tee 到 logs/。
# 在 AMD-AIMAX395 上运行（远程实验机），假设当前 cwd = code/part1-hardware-rocm，
# 并已经 source ./activate-rocm.sh。
#
# 用法：
#   cd /home/modelscope/lwh/hello-ai-infra/code/part1-hardware-rocm
#   source ./activate-rocm.sh
#   bash chapter4/run_all.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${HERE}/logs"
mkdir -p "${LOG_DIR}"

stamp() { date '+%Y-%m-%d %H:%M:%S %z'; }

run_with_log() {
    local name="$1"; shift
    local log="${LOG_DIR}/${name}.log"
    {
        echo "# === ${name} ==="
        echo "# host:  $(hostname)"
        echo "# pwd:   $(pwd)"
        echo "# start: $(stamp)"
        echo "# cmd:   $*"
        echo "# --- rocm-smi snapshot ---"
        if command -v rocm-smi >/dev/null 2>&1; then
            rocm-smi --showuse --showmeminfo vram 2>&1 | sed 's/^/#   /' || true
        else
            echo "#   rocm-smi: not in PATH"
        fi
        echo "# --- hipcc/hip ---"
        hipcc --version 2>&1 | head -3 | sed 's/^/#   /' || true
        echo "# --- output ---"
    } | tee "${log}"

    set +e
    "$@" 2>&1 | tee -a "${log}"
    local rc=${PIPESTATUS[0]}
    set -e

    {
        echo "# --- end ---"
        echo "# end:   $(stamp)"
        echo "# rc:    ${rc}"
    } | tee -a "${log}"

    if [[ ${rc} -ne 0 ]]; then
        echo "[run_all] ${name} failed with rc=${rc}, log=${log}" >&2
        exit "${rc}"
    fi
}

# 1. compile HIP benches
echo "[run_all] compiling bw_triad ..."
hipcc -O3 "${HERE}/bw_triad.hip" -o "${HERE}/bw_triad"

echo "[run_all] compiling lds_stride ..."
hipcc -O3 "${HERE}/lds_stride.hip" -o "${HERE}/lds_stride"

# 2. run benches
run_with_log bw_triad     "${HERE}/bw_triad"
run_with_log lds_stride   "${HERE}/lds_stride"
run_with_log bw_footprint python "${HERE}/bw_footprint.py"
run_with_log atomic_cmp   python "${HERE}/atomic_cmp.py"

echo "[run_all] all done; logs in ${LOG_DIR}"
