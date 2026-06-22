#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "请使用 'source $0' 而不是直接执行" >&2
    exit 1
fi

ROCM_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROCM_VENV="${ROCM_PROJECT_DIR}/.venv"

if [[ ! -d "${ROCM_VENV}" ]]; then
    echo "[activate-rocm] .venv 不存在：${ROCM_VENV}" >&2
    echo "[activate-rocm] 请先在本篇目录下执行 'uv sync'" >&2
    return 1
fi

source "${ROCM_VENV}/bin/activate"

ROCM_SDK_ROOT="$(find "${ROCM_VENV}/lib" -type d -name _rocm_sdk_devel -print -quit 2>/dev/null)"
if [[ -z "${ROCM_SDK_ROOT}" ]] && command -v rocm-sdk >/dev/null 2>&1; then
    rocm-sdk init >/dev/null 2>&1 || true
    ROCM_SDK_ROOT="$(find "${ROCM_VENV}/lib" -type d -name _rocm_sdk_devel -print -quit 2>/dev/null)"
fi
if [[ -z "${ROCM_SDK_ROOT}" ]]; then
    ROCM_SDK_ROOT="$(find "${ROCM_VENV}/lib" -type d -name _rocm_sdk_core -print -quit 2>/dev/null)"
fi
if [[ -z "${ROCM_SDK_ROOT}" ]]; then
    ROCM_SDK_ROOT="${ROCM_VENV}"
fi

export ROCM_PROJECT_DIR
export ROCM_VENV
export ROCM_PATH="${ROCM_SDK_ROOT}"
export HIP_PATH="${ROCM_SDK_ROOT}"

if [[ -d "${ROCM_VENV}/bin" ]]; then
    export PATH="${ROCM_VENV}/bin:${PATH}"
fi
if [[ -d "${ROCM_PATH}/bin" ]]; then
    export PATH="${ROCM_PATH}/bin:${PATH}"
fi
if [[ -d "${ROCM_PATH}/lib" ]]; then
    export LD_LIBRARY_PATH="${ROCM_PATH}/lib:${LD_LIBRARY_PATH:-}"
fi
if [[ -d "${ROCM_PATH}/lib64" ]]; then
    export LD_LIBRARY_PATH="${ROCM_PATH}/lib64:${LD_LIBRARY_PATH:-}"
fi

echo "Activated ROCm uv environment:"
echo "  PROJECT:    ${ROCM_PROJECT_DIR}"
echo "  VENV:       ${ROCM_VENV}"
echo "  ROCM_PATH:  ${ROCM_PATH}"
echo "  Python:     $(command -v python)"
python --version
