#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# ROCm / TheRock uv Project + Virtual Environment Installer
#
# 功能：
#   - 自动检测系统环境
#   - 自动检测 AMD GPU 和 GPU 架构
#   - 自动安装 / 检测 uv
#   - 自动安装 / 检测 fzf
#   - TUI 选择 GPU 架构
#   - TUI 选择 ROCm 版本
#   - TUI 选择安装模式
#   - TUI 选择 uv 项目目录 / 虚拟环境目录
#   - 使用 uv init 初始化项目
#   - 使用 uv venv 创建 .venv
#   - 从 AMD ROCm wheel 源安装 ROCm Python 包
#
# 注意：
#   这个脚本只负责 Python 虚拟环境和 ROCm wheel 包。
#   不负责安装 amdgpu-dkms / 内核驱动。
#
# 用法：
#   ./rocm-uv-env.sh
#   ./rocm-uv-env.sh --arch gfx94X-dcgpu
#   ./rocm-uv-env.sh --version 7.9.0 --arch gfx94X-dcgpu
#   ./rocm-uv-env.sh --project-dir ~/rocm-uv-projects/rocm-7.9-gfx94X
#   ./rocm-uv-env.sh --venv ~/rocm-venvs/rocm-7.9-gfx94X
#

set -euo pipefail

#######################################
# Basic Config
#######################################

SCRIPT_VERSION="1.2.0"

ROCM_WHL_BASE="https://repo.amd.com/rocm/whl"

GPU_ARCH=""
ROCM_VERSION=""
PYTHON_BIN="python3.11"

PROJECT_DIR=""
VENV_PATH=""
VENV_DIR_NAME=".venv"

NON_INTERACTIVE=false
FORCE_RECREATE=false
INSTALL_FZF=true
USE_UV_INIT=true

INSTALL_MODE="full"
# full: rocm[libraries,devel]，失败后 fallback 到 rocm-sdk-core/devel/libraries
# minimal: rocm-core rocm-smi-lib

# 区域 / PyPI 镜像
REGION=""           # cn | global；空表示走交互或自动判断
PYPI_MIRROR=""      # 显式指定时绕过镜像菜单

#######################################
# Colors
#######################################

RED=$'\033[0;31m'
GREEN=$'\033[0;32m'
YELLOW=$'\033[1;33m'
BLUE=$'\033[0;34m'
CYAN=$'\033[0;36m'
MAGENTA=$'\033[0;35m'
BOLD=$'\033[1m'
NC=$'\033[0m'

#######################################
# Utils
#######################################

log() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
    exit 1
}

has_cmd() {
    command -v "$1" >/dev/null 2>&1
}

setup_user_path() {
    export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:/usr/local/bin:${PATH}"

    if [[ -n "${SUDO_USER:-}" ]] && [[ "$SUDO_USER" != "root" ]]; then
        local sudo_home
        sudo_home="$(eval echo "~${SUDO_USER}" 2>/dev/null || true)"
        if [[ -n "$sudo_home" && -d "$sudo_home" ]]; then
            export PATH="${sudo_home}/.local/bin:${sudo_home}/.cargo/bin:${PATH}"
        fi
    fi
}

http_get() {
    local url="$1"

    if has_cmd curl; then
        curl -fsSL --connect-timeout 15 "$url" 2>/dev/null || true
    elif has_cmd wget; then
        wget -qO- --timeout=15 "$url" 2>/dev/null || true
    else
        true
    fi
}

http_status() {
    local url="$1"

    if has_cmd curl; then
        curl -s -o /dev/null -w "%{http_code}" --connect-timeout 8 "$url" 2>/dev/null || echo "000"
    elif has_cmd wget; then
        wget -q --spider --timeout=8 "$url" 2>/dev/null && echo "200" || echo "000"
    else
        echo "000"
    fi
}

draw_box() {
    local text="$1"
    local width=$(( ${#text} + 4 ))
    local border
    border=$(printf '═%.0s' $(seq 1 "$width"))

    echo -e "${BLUE}╔${border}╗${NC}"
    echo -e "${BLUE}║${NC}  ${BOLD}${text}${NC}  ${BLUE}║${NC}"
    echo -e "${BLUE}╚${border}╝${NC}"
}

print_header() {
    clear || true
    echo ""
    echo -e "${CYAN}  ██████╗  ██████╗  ██████╗███╗   ███╗     ██╗   ██╗██╗   ██╗${NC}"
    echo -e "${CYAN}  ██╔══██╗██╔═══██╗██╔════╝████╗ ████║     ██║   ██║██║   ██║${NC}"
    echo -e "${CYAN}  ██████╔╝██║   ██║██║     ██╔████╔██║     ██║   ██║██║   ██║${NC}"
    echo -e "${CYAN}  ██╔══██╗██║   ██║██║     ██║╚██╔╝██║     ██║   ██║╚██╗ ██╔╝${NC}"
    echo -e "${CYAN}  ██║  ██║╚██████╔╝╚██████╗██║ ╚═╝ ██║     ╚██████╔╝ ╚████╔╝ ${NC}"
    echo -e "${CYAN}  ╚═╝  ╚═╝ ╚═════╝  ╚═════╝╚═╝     ╚═╝      ╚═════╝   ╚═══╝  ${NC}"
    echo ""
    echo -e "  ${BOLD}ROCm / TheRock uv Project Installer v${SCRIPT_VERSION}${NC}"
    echo -e "  ${MAGENTA}Create ROCm Python environments with uv init + uv venv${NC}"
    echo ""
    echo -e "  ─────────────────────────────────────────────────────────────"
    echo ""
}

confirm() {
    local prompt="$1"
    local default="${2:-N}"

    if [[ "$NON_INTERACTIVE" == "true" ]]; then
        return 0
    fi

    local answer
    read -r -p "$prompt" answer

    if [[ -z "$answer" ]]; then
        answer="$default"
    fi

    [[ "$answer" =~ ^[Yy]$ ]]
}

expand_path() {
    local path="$1"

    if [[ "$path" == "~" ]]; then
        echo "$HOME"
    elif [[ "$path" == "~/"* ]]; then
        echo "${HOME}/${path#~/}"
    else
        echo "$path"
    fi
}

sanitize_name() {
    echo "$1" \
        | tr '[:upper:]' '[:lower:]' \
        | tr '.' '-' \
        | tr '_' '-' \
        | tr '/' '-' \
        | tr ':' '-' \
        | sed -E 's/[^a-z0-9-]+/-/g; s/^-+//; s/-+$//'
}

#######################################
# Help / Arguments
#######################################

show_help() {
    cat <<EOF
ROCm / TheRock uv Project Installer v${SCRIPT_VERSION}

Usage:
  $0 [options]

Options:
  --version VERSION        ROCm package version, e.g. 7.9.0
  --arch ARCH             GPU arch: gfx94X-dcgpu, gfx950-dcgpu, gfx1151
  --gpu-arch ARCH         Same as --arch
  --python PYTHON         Python binary, default: python3.11

  --project-dir PATH      uv project directory, venv will be PATH/.venv
  --venv PATH             Explicit virtual environment path

  --minimal               Install minimal packages: rocm-core rocm-smi-lib
  --full                  Install full package set, default
  --region cn|global      Region; affects PyPI mirror selection
  --pypi-mirror URL       Use this PyPI mirror as default index (overrides menu)
  --force                 Recreate venv if exists
  --non-interactive       Do not show menus (uses --region default = global)
  --no-fzf                Do not auto-install/use fzf
  --no-uv-init            Only create venv, do not run uv init
  --help, -h              Show help

Examples:
  $0
  $0 --arch gfx94X-dcgpu
  $0 --version 7.9.0 --arch gfx94X-dcgpu
  $0 --project-dir ~/rocm-uv-projects/rocm-7.9.0-gfx94X-dcgpu
  $0 --venv ~/rocm-venvs/rocm-7.9.0-gfx94X-dcgpu
  $0 --minimal --arch gfx1151

  # Non-interactive 全自动（推荐 CI / 脚本调用）
  $0 --non-interactive --version 7.12.0 --arch gfx1151 \\
     --region cn --pypi-mirror https://pypi.tuna.tsinghua.edu.cn/simple \\
     --project-dir /path/to/repo

GPU Architecture:
  gfx94X-dcgpu    MI325X, MI300X, MI300A
  gfx950-dcgpu    MI355X, MI350X
  gfx1151         Ryzen AI APU, Strix, Hawk

EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --version)
                ROCM_VERSION="${2:-}"
                shift 2
                ;;
            --arch|--gpu-arch)
                GPU_ARCH="${2:-}"
                shift 2
                ;;
            --python)
                PYTHON_BIN="${2:-}"
                shift 2
                ;;
            --project-dir)
                PROJECT_DIR="${2:-}"
                shift 2
                ;;
            --venv)
                VENV_PATH="${2:-}"
                shift 2
                ;;
            --minimal)
                INSTALL_MODE="minimal"
                shift
                ;;
            --full)
                INSTALL_MODE="full"
                shift
                ;;
            --region)
                REGION="${2:-}"
                shift 2
                ;;
            --pypi-mirror)
                PYPI_MIRROR="${2:-}"
                shift 2
                ;;
            --force)
                FORCE_RECREATE=true
                shift
                ;;
            --non-interactive)
                NON_INTERACTIVE=true
                shift
                ;;
            --no-fzf)
                INSTALL_FZF=false
                shift
                ;;
            --no-uv-init)
                USE_UV_INIT=false
                shift
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                error "Unknown option: $1"
                ;;
        esac
    done
}

#######################################
# System Detection
#######################################

detect_os() {
    OS_ID="unknown"
    OS_VERSION="unknown"
    OS_NAME="unknown"

    if [[ -f /etc/os-release ]]; then
        # shellcheck source=/dev/null
        . /etc/os-release
        OS_ID="${ID:-unknown}"
        OS_VERSION="${VERSION_ID:-unknown}"
        OS_NAME="${PRETTY_NAME:-unknown}"
    fi

    ARCH="$(uname -m)"

    echo -e "  ${GREEN}✓${NC} OS:           ${OS_NAME}"
    echo -e "  ${GREEN}✓${NC} Kernel:       $(uname -r)"
    echo -e "  ${GREEN}✓${NC} Architecture: ${ARCH}"

    if [[ "$ARCH" != "x86_64" ]]; then
        warn "ROCm wheel 环境通常面向 x86_64；当前架构是 ${ARCH}"
    fi
}

detect_pkg_manager() {
    PKG_MGR=""

    if has_cmd apt-get; then
        PKG_MGR="apt"
    elif has_cmd dnf; then
        PKG_MGR="dnf"
    elif has_cmd yum; then
        PKG_MGR="yum"
    else
        PKG_MGR="unknown"
    fi

    echo -e "  ${GREEN}✓${NC} Package Mgr:  ${PKG_MGR}"
}

install_basic_tools() {
    local need_packages=()

    has_cmd curl || need_packages+=("curl")
    has_cmd wget || need_packages+=("wget")
    has_cmd lspci || need_packages+=("pciutils")

    if [[ ${#need_packages[@]} -eq 0 ]]; then
        return 0
    fi

    warn "Missing tools: ${need_packages[*]}"

    if [[ "$EUID" -ne 0 ]]; then
        if ! has_cmd sudo; then
            warn "没有 sudo，无法自动安装依赖：${need_packages[*]}"
            return 0
        fi
    fi

    case "$PKG_MGR" in
        apt)
            sudo apt-get update
            sudo apt-get install -y "${need_packages[@]}" || true
            ;;
        dnf|yum)
            sudo "$PKG_MGR" install -y "${need_packages[@]}" || true
            ;;
        *)
            warn "未知包管理器，请手动安装：${need_packages[*]}"
            ;;
    esac
}

detect_gpu() {
    echo ""
    draw_box "Detecting AMD GPU"
    echo ""

    GPU_LIST=""

    if has_cmd lspci; then
        GPU_LIST="$(lspci -nn | grep -iE 'VGA|Display|3D' | grep -iE 'AMD|ATI|Advanced Micro Devices' || true)"
    fi

    if [[ -z "$GPU_LIST" ]]; then
        warn "未检测到 AMD GPU"
        if has_cmd lspci; then
            echo ""
            echo "Detected display devices:"
            lspci -nn | grep -iE 'VGA|Display|3D' || true
            echo ""
        fi
    else
        while IFS= read -r line; do
            echo -e "  ${GREEN}✓${NC} ${line}"
        done <<< "$GPU_LIST"
    fi
}

detect_gpu_architecture() {
    local detected=""

    if has_cmd rocminfo; then
        local info
        info="$(rocminfo 2>/dev/null || true)"

        if echo "$info" | grep -qE 'gfx950'; then
            detected="gfx950-dcgpu"
        elif echo "$info" | grep -qE 'gfx94[0-9]'; then
            detected="gfx94X-dcgpu"
        elif echo "$info" | grep -qE 'gfx1151'; then
            detected="gfx1151"
        fi
    fi

    if [[ -z "$detected" ]] && has_cmd lspci; then
        local gpu_info
        gpu_info="$(lspci -nn | grep -iE 'VGA|Display|3D' | grep -iE 'AMD|ATI|Advanced Micro Devices' || true)"

        if echo "$gpu_info" | grep -qiE 'MI355|MI350|gfx950'; then
            detected="gfx950-dcgpu"
        elif echo "$gpu_info" | grep -qiE 'MI325|MI300|gfx94|Aldebaran|CDNA3'; then
            detected="gfx94X-dcgpu"
        elif echo "$gpu_info" | grep -qiE 'gfx1151|Strix|Hawk|Ryzen AI'; then
            detected="gfx1151"
        fi
    fi

    echo "$detected"
}

validate_gpu_arch() {
    local arch="$1"

    case "$arch" in
        gfx94X-dcgpu|gfx950-dcgpu|gfx1151)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

#######################################
# uv / fzf Installation
#######################################

ensure_uv() {
    echo ""
    draw_box "Checking uv"
    echo ""

    setup_user_path

    if has_cmd uv; then
        echo -e "  ${GREEN}✓${NC} uv found: $(uv --version)"
        echo -e "  ${GREEN}✓${NC} uv path:  $(command -v uv)"
        return 0
    fi

    log "uv not found, installing uv..."

    if ! has_cmd curl; then
        error "curl not found. Please install curl first."
    fi

    curl -LsSf https://astral.sh/uv/install.sh | sh

    setup_user_path

    if ! has_cmd uv; then
        error "uv installed but not found in PATH. Try: export PATH=\"\$HOME/.local/bin:\$PATH\""
    fi

    echo -e "  ${GREEN}✓${NC} uv installed: $(uv --version)"
    echo -e "  ${GREEN}✓${NC} uv path:      $(command -v uv)"
}

ensure_fzf() {
    if [[ "$INSTALL_FZF" != "true" ]]; then
        return 0
    fi

    if [[ "$NON_INTERACTIVE" == "true" ]]; then
        return 0
    fi

    if has_cmd fzf; then
        return 0
    fi

    echo ""
    draw_box "Checking fzf"
    echo ""

    log "fzf not found, trying to install fzf..."

    case "$PKG_MGR" in
        apt)
            sudo apt-get update
            sudo apt-get install -y fzf || true
            ;;
        dnf|yum)
            sudo "$PKG_MGR" install -y fzf || true
            ;;
        *)
            warn "未知包管理器，跳过 fzf 自动安装"
            ;;
    esac

    if has_cmd fzf; then
        echo -e "  ${GREEN}✓${NC} fzf installed"
    else
        warn "fzf 不可用，将使用普通 select 菜单"
    fi
}

#######################################
# ROCm Wheel Index
#######################################

arch_index_url() {
    local arch="$1"
    echo "${ROCM_WHL_BASE}/${arch}/"
}

check_arch_index() {
    local arch="$1"
    local url
    url="$(arch_index_url "$arch")"

    local status
    status="$(http_status "$url")"

    [[ "$status" == "200" ]]
}

fetch_versions_from_wheel_index() {
    local arch="$1"
    local base_url="${ROCM_WHL_BASE}/${arch}"

    local html=""
    local versions=""

    html="$(http_get "${base_url}/rocm/")"

    if [[ -n "$html" ]]; then
        versions="$(
            echo "$html" \
                | grep -oE 'rocm-[0-9]+\.[0-9]+(\.[0-9]+)?([a-zA-Z0-9._+-]*)?\.(whl|tar\.gz)' \
                | sed -E 's/^rocm-//; s/\.(whl|tar\.gz)$//' \
                | sort -Vr \
                | uniq
        )"
    fi

    if [[ -z "$versions" ]]; then
        html="$(http_get "${base_url}/rocm-sdk-core/")"

        if [[ -n "$html" ]]; then
            versions="$(
                echo "$html" \
                    | grep -oE 'rocm_sdk_core-[0-9]+\.[0-9]+(\.[0-9]+)?([a-zA-Z0-9._+-]*)?-[^"]+\.whl' \
                    | sed -E 's/^rocm_sdk_core-//; s/-[^-]+-[^-]+-[^-]+\.whl$//' \
                    | sort -Vr \
                    | uniq
            )"
        fi
    fi

    if [[ -z "$versions" ]]; then
        html="$(http_get "${base_url}/")"

        local dirs
        dirs="$(
            echo "$html" \
                | grep -oE 'href="[^"]+/"' \
                | sed -E 's/href="//; s|/"||' \
                | grep -E '^(rocm|rocm-sdk-core|rocm-sdk-devel|rocm-sdk-libraries|rocm-core|rocm-smi-lib)$' \
                || true
        )"

        local all_versions=""
        while IFS= read -r dir; do
            [[ -z "$dir" ]] && continue

            local sub_html
            sub_html="$(http_get "${base_url}/${dir}/")"

            if [[ -n "$sub_html" ]]; then
                all_versions="$(
                    {
                        echo "$all_versions"
                        echo "$sub_html" \
                            | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?([a-zA-Z0-9._+-]*)?' \
                            | grep -E '^[0-9]+\.[0-9]+'
                    } | sort -Vr | uniq
                )"
            fi
        done <<< "$dirs"

        versions="$all_versions"
    fi

    [[ -n "$versions" ]] && echo "$versions"
}

#######################################
# TUI Menus
#######################################

select_gpu_arch() {
    if [[ -n "$GPU_ARCH" ]]; then
        if validate_gpu_arch "$GPU_ARCH"; then
            return 0
        else
            error "Invalid GPU arch: ${GPU_ARCH}"
        fi
    fi

    local detected_arch
    detected_arch="$(detect_gpu_architecture || true)"

    if [[ "$NON_INTERACTIVE" == "true" ]]; then
        if [[ -n "$detected_arch" ]]; then
            GPU_ARCH="$detected_arch"
            log "Auto-detected GPU arch: ${GPU_ARCH}"
            return 0
        fi

        error "Non-interactive mode requires --arch. Example: --arch gfx94X-dcgpu"
    fi

    echo ""
    draw_box "Select GPU Architecture"
    echo ""

    local options=()

    if [[ -n "$detected_arch" ]]; then
        options+=("${detected_arch}    Auto-detected")
    fi

    if [[ "$detected_arch" != "gfx94X-dcgpu" ]]; then
        options+=("gfx94X-dcgpu    MI325X / MI300X / MI300A")
    fi

    if [[ "$detected_arch" != "gfx950-dcgpu" ]]; then
        options+=("gfx950-dcgpu    MI355X / MI350X")
    fi

    if [[ "$detected_arch" != "gfx1151" ]]; then
        options+=("gfx1151         Ryzen AI APU / Strix / Hawk")
    fi

    if has_cmd fzf; then
        local selection
        selection="$(printf '%s\n' "${options[@]}" | fzf \
            --layout=reverse \
            --border=rounded \
            --prompt="GPU Arch ❯ " \
            --header="Select target ROCm wheel architecture")"

        [[ -z "$selection" ]] && error "No GPU architecture selected"

        GPU_ARCH="$(echo "$selection" | awk '{print $1}')"
    else
        PS3=$'\n\033[0;36mYour choice: \033[0m'
        select opt in "${options[@]}"; do
            if [[ -n "$opt" ]]; then
                GPU_ARCH="$(echo "$opt" | awk '{print $1}')"
                break
            else
                echo "Invalid selection"
            fi
        done
    fi

    if ! validate_gpu_arch "$GPU_ARCH"; then
        error "Invalid GPU arch selected: ${GPU_ARCH}"
    fi

    log "Selected GPU arch: ${GPU_ARCH}"
}

select_rocm_version() {
    if [[ -n "$ROCM_VERSION" ]]; then
        return 0
    fi

    echo ""
    draw_box "Fetching ROCm Versions"
    echo ""

    local versions
    versions="$(fetch_versions_from_wheel_index "$GPU_ARCH" || true)"

    if [[ -z "$versions" ]]; then
        warn "无法从 ${ROCM_WHL_BASE}/${GPU_ARCH}/ 自动解析版本"
        warn "将使用手动输入版本"
        read -r -p "Enter ROCm version, e.g. 7.9.0: " ROCM_VERSION
        [[ -z "$ROCM_VERSION" ]] && error "No ROCm version entered"
        return 0
    fi

    echo -e "  ${GREEN}✓${NC} Found available ROCm versions:"
    echo "$versions" | sed 's/^/    - /'
    echo ""

    if [[ "$NON_INTERACTIVE" == "true" ]]; then
        ROCM_VERSION="$(echo "$versions" | head -n1)"
        log "Non-interactive: using latest ROCm version: ${ROCM_VERSION}"
        return 0
    fi

    if has_cmd fzf; then
        local selection
        selection="$(echo "$versions" | fzf \
            --layout=reverse \
            --border=rounded \
            --prompt="ROCm Version ❯ " \
            --header="Select ROCm package version for ${GPU_ARCH}")"

        [[ -z "$selection" ]] && error "No ROCm version selected"
        ROCM_VERSION="$selection"
    else
        local arr=()
        while IFS= read -r v; do
            [[ -n "$v" ]] && arr+=("$v")
        done <<< "$versions"

        PS3=$'\n\033[0;36mYour choice: \033[0m'
        select opt in "${arr[@]}"; do
            if [[ -n "$opt" ]]; then
                ROCM_VERSION="$opt"
                break
            else
                echo "Invalid selection"
            fi
        done
    fi

    log "Selected ROCm version: ${ROCM_VERSION}"
}

select_install_mode() {
    if [[ "$NON_INTERACTIVE" == "true" ]]; then
        return 0
    fi

    echo ""
    draw_box "Select Install Mode"
    echo ""

    local options=(
        "full       Install rocm[libraries,devel]    Recommended"
        "minimal    Install rocm-core rocm-smi-lib   Smaller"
    )

    if has_cmd fzf; then
        local selection
        selection="$(printf '%s\n' "${options[@]}" | fzf \
            --layout=reverse \
            --border=rounded \
            --prompt="Install Mode ❯ " \
            --header="Choose ROCm package set")"

        if [[ -n "$selection" ]]; then
            INSTALL_MODE="$(echo "$selection" | awk '{print $1}')"
        fi
    else
        PS3=$'\n\033[0;36mYour choice: \033[0m'
        select opt in "${options[@]}"; do
            if [[ -n "$opt" ]]; then
                INSTALL_MODE="$(echo "$opt" | awk '{print $1}')"
                break
            else
                echo "Invalid selection"
            fi
        done
    fi

    log "Selected install mode: ${INSTALL_MODE}"
}

#######################################
# Region / PyPI Mirror Selection
#######################################

# 国内 PyPI 镜像候选（顺序无所谓，会按测速排序）
CN_PYPI_MIRRORS=(
    "https://pypi.tuna.tsinghua.edu.cn/simple|Tsinghua TUNA"
    "https://mirrors.aliyun.com/pypi/simple/|Aliyun"
    "https://pypi.mirrors.ustc.edu.cn/simple/|USTC"
    "https://mirrors.cloud.tencent.com/pypi/simple/|Tencent Cloud"
    "https://mirrors.bfsu.edu.cn/pypi/web/simple|BFSU"
)

normalize_region() {
    case "$1" in
        cn|china|CN|China) echo "cn" ;;
        global|intl|international|GLOBAL) echo "global" ;;
        *) echo "" ;;
    esac
}

select_region() {
    if [[ -n "$REGION" ]]; then
        local norm
        norm="$(normalize_region "$REGION")"
        [[ -z "$norm" ]] && error "Invalid --region: ${REGION} (expected cn|global)"
        REGION="$norm"
        log "Using region: ${REGION}"
        return 0
    fi

    if [[ "$NON_INTERACTIVE" == "true" ]]; then
        REGION="global"
        log "Non-interactive: defaulting region to global"
        return 0
    fi

    echo ""
    draw_box "Select Region"
    echo ""

    local options=(
        "cn        中国大陆（使用国内 PyPI 镜像）"
        "global    海外 / 直连官方 PyPI"
    )

    if has_cmd fzf; then
        local selection
        selection="$(printf '%s\n' "${options[@]}" | fzf \
            --layout=reverse --border=rounded \
            --prompt="Region ❯ " \
            --header="Choose your region (affects PyPI mirror, NOT the AMD ROCm index)")"
        [[ -z "$selection" ]] && error "No region selected"
        REGION="$(echo "$selection" | awk '{print $1}')"
    else
        PS3=$'\n\033[0;36mYour choice: \033[0m'
        select opt in "${options[@]}"; do
            if [[ -n "$opt" ]]; then
                REGION="$(echo "$opt" | awk '{print $1}')"
                break
            else
                echo "Invalid selection"
            fi
        done
    fi

    log "Selected region: ${REGION}"
}

probe_mirror_latency() {
    # 输出格式："<latency> <url> <name>"，按 latency 升序
    local entry url name latency tmp
    tmp="$(mktemp)"

    for entry in "${CN_PYPI_MIRRORS[@]}"; do
        url="${entry%%|*}"
        name="${entry##*|}"
        (
            local t
            t="$(curl -o /dev/null -s -w "%{time_total}" \
                --connect-timeout 3 --max-time 5 "$url" 2>/dev/null || echo "999")"
            # 把 999/异常值规整成可排序数字
            [[ -z "$t" ]] && t="999"
            printf '%s %s %s\n' "$t" "$url" "$name"
        ) >> "$tmp" &
    done
    wait

    sort -n "$tmp"
    rm -f "$tmp"
}

select_pypi_mirror() {
    if [[ -n "$PYPI_MIRROR" ]]; then
        log "Using explicit PyPI mirror: ${PYPI_MIRROR}"
        return 0
    fi

    if [[ "$REGION" != "cn" ]]; then
        log "Region=global, no PyPI mirror configured (uses default PyPI)"
        return 0
    fi

    echo ""
    draw_box "Probing CN PyPI Mirrors"
    echo ""
    log "Testing latency to ${#CN_PYPI_MIRRORS[@]} mirrors (~5s)..."

    local probed
    probed="$(probe_mirror_latency)"

    echo ""
    echo -e "  ${CYAN}Latency results (sorted):${NC}"
    while IFS= read -r line; do
        local lat url name
        lat="$(echo "$line" | awk '{print $1}')"
        url="$(echo "$line" | awk '{print $2}')"
        name="$(echo "$line" | cut -d' ' -f3-)"
        printf "    %-7s  %-50s  %s\n" "${lat}s" "$url" "$name"
    done <<< "$probed"
    echo ""

    if [[ "$NON_INTERACTIVE" == "true" ]]; then
        PYPI_MIRROR="$(echo "$probed" | head -n1 | awk '{print $2}')"
        log "Non-interactive: auto-picked fastest mirror: ${PYPI_MIRROR}"
        return 0
    fi

    # 取最快的 3 个进菜单
    local top3
    top3="$(echo "$probed" | head -n3)"

    local options=()
    while IFS= read -r line; do
        local lat url name
        lat="$(echo "$line" | awk '{print $1}')"
        url="$(echo "$line" | awk '{print $2}')"
        name="$(echo "$line" | cut -d' ' -f3-)"
        options+=("${url}|[${lat}s] ${name}")
    done <<< "$top3"

    if has_cmd fzf; then
        local selection
        selection="$(
            printf '%s\n' "${options[@]}" \
                | awk -F'|' '{printf "%-55s  %s\n", $1, $2}' \
                | fzf --layout=reverse --border=rounded \
                    --prompt="PyPI Mirror ❯ " \
                    --header="Top 3 fastest CN mirrors"
        )"
        [[ -z "$selection" ]] && error "No mirror selected"
        PYPI_MIRROR="$(echo "$selection" | awk '{print $1}')"
    else
        echo "Top 3 fastest:"
        local i=1 opt
        for opt in "${options[@]}"; do
            echo "  $i) $(echo "$opt" | awk -F'|' '{printf "%-55s  %s", $1, $2}')"
            i=$((i+1))
        done
        echo ""
        read -r -p "Your choice [1-3]: " choice
        case "$choice" in
            1|"") PYPI_MIRROR="$(echo "${options[0]}" | awk -F'|' '{print $1}')" ;;
            2)    PYPI_MIRROR="$(echo "${options[1]}" | awk -F'|' '{print $1}')" ;;
            3)    PYPI_MIRROR="$(echo "${options[2]}" | awk -F'|' '{print $1}')" ;;
            *)    PYPI_MIRROR="$(echo "${options[0]}" | awk -F'|' '{print $1}')" ;;
        esac
    fi

    log "Selected PyPI mirror: ${PYPI_MIRROR}"
}

#######################################
# Python / Project / Venv Path
#######################################

ensure_python() {
    echo ""
    draw_box "Checking Python"
    echo ""

    if has_cmd "$PYTHON_BIN"; then
        echo -e "  ${GREEN}✓${NC} Python found: $($PYTHON_BIN --version)"
        return 0
    fi

    warn "${PYTHON_BIN} not found"

    if has_cmd python3.12; then
        PYTHON_BIN="python3.12"
    elif has_cmd python3.11; then
        PYTHON_BIN="python3.11"
    elif has_cmd python3.10; then
        PYTHON_BIN="python3.10"
    elif has_cmd python3; then
        PYTHON_BIN="python3"
    else
        error "No Python 3 found"
    fi

    echo -e "  ${GREEN}✓${NC} Fallback Python: $($PYTHON_BIN --version)"
}

default_project_dir() {
    local safe_ver
    safe_ver="$(echo "$ROCM_VERSION" | tr '/' '_' | tr ':' '_')"

    echo "${HOME}/rocm-uv-projects/rocm-${safe_ver}-${GPU_ARCH}"
}

default_venv_path() {
    echo "$(default_project_dir)/${VENV_DIR_NAME}"
}

select_project_and_venv_path() {
    if [[ -n "$PROJECT_DIR" ]]; then
        PROJECT_DIR="$(expand_path "$PROJECT_DIR")"
        VENV_PATH="${PROJECT_DIR}/${VENV_DIR_NAME}"
        log "Using project dir: ${PROJECT_DIR}"
        log "Using venv path: ${VENV_PATH}"
        return 0
    fi

    if [[ -n "$VENV_PATH" ]]; then
        VENV_PATH="$(expand_path "$VENV_PATH")"
        PROJECT_DIR="$(dirname "$VENV_PATH")"
        log "Using explicit venv path: ${VENV_PATH}"
        log "Inferred project dir: ${PROJECT_DIR}"
        return 0
    fi

    local default_dir
    default_dir="$(default_project_dir)"

    if [[ "$NON_INTERACTIVE" == "true" ]]; then
        PROJECT_DIR="$default_dir"
        VENV_PATH="${PROJECT_DIR}/${VENV_DIR_NAME}"
        log "Non-interactive: using project dir: ${PROJECT_DIR}"
        log "Non-interactive: using venv path: ${VENV_PATH}"
        return 0
    fi

    echo ""
    draw_box "Select uv Project / Virtual Environment Path"
    echo ""

    local home_dir="${HOME}/rocm-uv-projects/rocm-${ROCM_VERSION}-${GPU_ARCH}"
    local opt_dir="/opt/rocm-uv-projects/rocm-${ROCM_VERSION}-${GPU_ARCH}"

    local options=(
        "default    ${default_dir}"
        "home       ${home_dir}"
        "opt        ${opt_dir}"
        "custom     Enter custom project dir"
        "venv       Enter explicit venv path"
    )

    if has_cmd fzf; then
        local selection
        selection="$(printf '%s\n' "${options[@]}" | fzf \
            --layout=reverse \
            --border=rounded \
            --prompt="Project Dir ❯ " \
            --header="Choose uv project dir. Default venv will be .venv")"

        [[ -z "$selection" ]] && error "No project dir selected"

        local choice
        choice="$(echo "$selection" | awk '{print $1}')"

        case "$choice" in
            default)
                PROJECT_DIR="$default_dir"
                VENV_PATH="${PROJECT_DIR}/${VENV_DIR_NAME}"
                ;;
            home)
                PROJECT_DIR="$home_dir"
                VENV_PATH="${PROJECT_DIR}/${VENV_DIR_NAME}"
                ;;
            opt)
                PROJECT_DIR="$opt_dir"
                VENV_PATH="${PROJECT_DIR}/${VENV_DIR_NAME}"
                ;;
            custom)
                echo ""
                read -r -p "Enter custom uv project dir: " custom_dir
                [[ -z "$custom_dir" ]] && error "Custom project dir cannot be empty"
                PROJECT_DIR="$(expand_path "$custom_dir")"
                VENV_PATH="${PROJECT_DIR}/${VENV_DIR_NAME}"
                ;;
            venv)
                echo ""
                read -r -p "Enter explicit venv path: " custom_venv
                [[ -z "$custom_venv" ]] && error "Custom venv path cannot be empty"
                VENV_PATH="$(expand_path "$custom_venv")"
                PROJECT_DIR="$(dirname "$VENV_PATH")"
                ;;
            *)
                PROJECT_DIR="$default_dir"
                VENV_PATH="${PROJECT_DIR}/${VENV_DIR_NAME}"
                ;;
        esac
    else
        echo "1) Default project: ${default_dir}"
        echo "2) Home project:    ${home_dir}"
        echo "3) Opt project:     ${opt_dir}"
        echo "4) Custom project dir"
        echo "5) Explicit venv path"
        echo ""

        read -r -p "Your choice [1-5]: " choice

        case "$choice" in
            1|"")
                PROJECT_DIR="$default_dir"
                VENV_PATH="${PROJECT_DIR}/${VENV_DIR_NAME}"
                ;;
            2)
                PROJECT_DIR="$home_dir"
                VENV_PATH="${PROJECT_DIR}/${VENV_DIR_NAME}"
                ;;
            3)
                PROJECT_DIR="$opt_dir"
                VENV_PATH="${PROJECT_DIR}/${VENV_DIR_NAME}"
                ;;
            4)
                echo ""
                read -r -p "Enter custom uv project dir: " custom_dir
                [[ -z "$custom_dir" ]] && error "Custom project dir cannot be empty"
                PROJECT_DIR="$(expand_path "$custom_dir")"
                VENV_PATH="${PROJECT_DIR}/${VENV_DIR_NAME}"
                ;;
            5)
                echo ""
                read -r -p "Enter explicit venv path: " custom_venv
                [[ -z "$custom_venv" ]] && error "Custom venv path cannot be empty"
                VENV_PATH="$(expand_path "$custom_venv")"
                PROJECT_DIR="$(dirname "$VENV_PATH")"
                ;;
            *)
                warn "Invalid choice, using default project dir"
                PROJECT_DIR="$default_dir"
                VENV_PATH="${PROJECT_DIR}/${VENV_DIR_NAME}"
                ;;
        esac
    fi

    PROJECT_DIR="$(expand_path "$PROJECT_DIR")"
    VENV_PATH="$(expand_path "$VENV_PATH")"

    log "Selected project dir: ${PROJECT_DIR}"
    log "Selected venv path: ${VENV_PATH}"
}

ensure_project_permission() {
    PROJECT_DIR="$(expand_path "$PROJECT_DIR")"
    VENV_PATH="$(expand_path "$VENV_PATH")"

    if [[ "$PROJECT_DIR" == /opt/* ]] && [[ "$EUID" -ne 0 ]]; then
        warn "You selected a project dir under /opt, but current user is not root."
        warn "Project dir: ${PROJECT_DIR}"
        echo ""
        echo -e "Re-run with sudo, for example:"
        echo -e "  ${CYAN}sudo env \"PATH=\$PATH\" $0 --version ${ROCM_VERSION} --arch ${GPU_ARCH} --project-dir ${PROJECT_DIR}${NC}"
        echo ""

        if [[ "$NON_INTERACTIVE" == "true" ]]; then
            error "Cannot create project under /opt without root in non-interactive mode"
        fi

        read -r -p "Use home project dir instead? (Y/n): " use_home
        if [[ -z "$use_home" || "$use_home" =~ ^[Yy]$ ]]; then
            PROJECT_DIR="${HOME}/rocm-uv-projects/rocm-${ROCM_VERSION}-${GPU_ARCH}"
            VENV_PATH="${PROJECT_DIR}/${VENV_DIR_NAME}"
            log "Changed project dir to: ${PROJECT_DIR}"
            log "Changed venv path to: ${VENV_PATH}"
        else
            error "Please re-run with sudo or choose another path"
        fi
    fi

    local parent_dir
    parent_dir="$(dirname "$PROJECT_DIR")"

    mkdir -p "$parent_dir"

    if [[ ! -w "$parent_dir" ]]; then
        error "Parent directory is not writable: ${parent_dir}"
    fi
}

#######################################
# uv init / venv / install
#######################################

init_uv_project() {
    echo ""
    draw_box "Initializing uv Project"
    echo ""

    if [[ "$USE_UV_INIT" != "true" ]]; then
        warn "uv init skipped by --no-uv-init"
        mkdir -p "$PROJECT_DIR"
        return 0
    fi

    ensure_project_permission

    mkdir -p "$PROJECT_DIR"
    cd "$PROJECT_DIR"

    local package_name
    package_name="$(basename "$PROJECT_DIR")"
    package_name="$(sanitize_name "$package_name")"

    if [[ -f "pyproject.toml" ]]; then
        warn "pyproject.toml already exists, skipping uv init"
        "$PYTHON_BIN" - "pyproject.toml" "$package_name" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
package_name = sys.argv[2]
text = path.read_text()

pattern = re.compile(r'(?m)^(\[project\]\n(?:[^\[]*?\n)?name\s*=\s*)"[^"]+"')
new_text, count = pattern.subn(rf'\1"{package_name}"', text, count=1)

if count:
    path.write_text(new_text)
PY
    else
        log "Running uv init in ${PROJECT_DIR}"

        uv init \
            --name "$package_name" \
            --bare \
            --no-readme \
            --no-workspace \
            .
    fi

    if [[ ! -f ".python-version" ]]; then
        "$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' > .python-version
    fi

    echo -e "  ${GREEN}✓${NC} uv project dir: ${PROJECT_DIR}"
}

create_rocm_venv() {
    echo ""
    draw_box "Creating ROCm uv Environment"
    echo ""

    if [[ -z "$PROJECT_DIR" ]]; then
        PROJECT_DIR="$(default_project_dir)"
    fi

    if [[ -z "$VENV_PATH" ]]; then
        VENV_PATH="${PROJECT_DIR}/${VENV_DIR_NAME}"
    fi

    PROJECT_DIR="$(expand_path "$PROJECT_DIR")"
    VENV_PATH="$(expand_path "$VENV_PATH")"

    mkdir -p "$PROJECT_DIR"
    cd "$PROJECT_DIR"

    if [[ -d "$VENV_PATH" ]]; then
        if [[ "$FORCE_RECREATE" == "true" ]]; then
            warn "Removing existing venv: ${VENV_PATH}"
            rm -rf "$VENV_PATH"
        else
            warn "Venv already exists: ${VENV_PATH}"
            if confirm "Recreate it? (y/N): " "N"; then
                rm -rf "$VENV_PATH"
            else
                log "Using existing venv"
            fi
        fi
    fi

    if [[ ! -d "$VENV_PATH" ]]; then
        log "Creating venv: ${VENV_PATH}"
        uv venv "$VENV_PATH" --python "$PYTHON_BIN"
    fi

    echo -e "  ${GREEN}✓${NC} Venv path: ${VENV_PATH}"
}

#######################################
# Configure indexes in pyproject.toml
# (PyPI mirror as default + ROCm AMD index)
#######################################

configure_pyproject_indexes() {
    echo ""
    draw_box "Configuring Indexes in pyproject.toml"
    echo ""

    local pyproject="${PROJECT_DIR}/pyproject.toml"
    local whl_url
    whl_url="$(arch_index_url "$GPU_ARCH")"

    if [[ ! -f "$pyproject" ]]; then
        error "pyproject.toml not found at ${pyproject}"
    fi

    local venv_python="${VENV_PATH}/bin/python"
    if [[ ! -x "$venv_python" ]]; then
        error "Venv Python not found: ${venv_python}"
    fi

    "$venv_python" - "$pyproject" "$whl_url" "${PYPI_MIRROR:-}" "$GPU_ARCH" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
rocm_url = sys.argv[2]
pypi_mirror = sys.argv[3] or ""
gpu_arch = sys.argv[4]

text = path.read_text()
original = text

def has_index_named(name: str) -> bool:
    return f'name = "{name}"' in text

def append_block(text: str, block: str) -> str:
    if not text.endswith("\n"):
        text += "\n"
    return text + block

# 1) PyPI 镜像（如果用户选了），作为 default index 替代 PyPI
if pypi_mirror and not has_index_named("pypi-mirror"):
    block = (
        "\n[[tool.uv.index]]\n"
        'name = "pypi-mirror"\n'
        f'url = "{pypi_mirror}"\n'
        "default = true\n"
    )
    text = append_block(text, block)

# 2) AMD ROCm wheel index — 必须 explicit = true！
#    否则 amd 索引对未知包（如 requests）返回 403 会让 uv 整个解析失败。
#    explicit 后只对 [tool.uv.sources] 显式映射的包生效。
if not has_index_named("rocm-amd"):
    block = (
        "\n[[tool.uv.index]]\n"
        'name = "rocm-amd"\n'
        f'url = "{rocm_url}"\n'
        "explicit = true\n"
    )
    text = append_block(text, block)

# 3) [tool.uv.sources] —— 把所有可能用到的 rocm 包定向到 rocm-amd
#    注意：sources 仅作用于 direct dependencies，所以 add_rocm_packages
#    必须把这些包都列成 direct deps（不能只 add 一个 rocm 元包让它拉 transitive）。
ROCM_PKGS = [
    "rocm",
    "rocm-sdk-core",
    "rocm-sdk-devel",
    "rocm-sdk-libraries",
    f"rocm-sdk-libraries-{gpu_arch}",
    "rocm-core",
    "rocm-smi-lib",
]

if "[tool.uv.sources]" not in text:
    lines = ["", "[tool.uv.sources]"]
    for pkg in ROCM_PKGS:
        lines.append(f'{pkg} = {{ index = "rocm-amd" }}')
    text = append_block(text, "\n".join(lines) + "\n")
else:
    for pkg in ROCM_PKGS:
        if f'\n{pkg} = ' not in text:
            text = text.replace(
                "[tool.uv.sources]",
                f'[tool.uv.sources]\n{pkg} = {{ index = "rocm-amd" }}',
                1,
            )

if text != original:
    path.write_text(text)
    sys.stderr.write(f"[indexes] updated {path}\n")
else:
    sys.stderr.write(f"[indexes] no change to {path}\n")
PY

    echo -e "  ${GREEN}✓${NC} pyproject.toml indexes configured:"
    if [[ -n "${PYPI_MIRROR:-}" ]]; then
        echo -e "      pypi-mirror (default): ${PYPI_MIRROR}"
    else
        echo -e "      pypi (default):        official PyPI"
    fi
    echo -e "      rocm-amd (explicit):   ${whl_url}"
    echo -e "      [tool.uv.sources]:     all rocm-* packages mapped"
}

#######################################
# Add ROCm packages via `uv add`
# 这样依赖直接进 [project.dependencies]，
# 后续 uv sync 在任意机器上都能复现。
#######################################

add_rocm_packages() {
    echo ""
    draw_box "Adding ROCm Packages via uv add"
    echo ""

    cd "$PROJECT_DIR"

    log "Project dir:  ${PROJECT_DIR}"
    log "Venv path:    ${VENV_PATH}"
    log "ROCm version: ${ROCM_VERSION}"
    log "Install mode: ${INSTALL_MODE}"

    local venv_python="${VENV_PATH}/bin/python"
    if [[ ! -x "$venv_python" ]]; then
        error "Venv Python not found: ${venv_python}"
    fi

    # 让 uv 用项目里的 venv
    export UV_PROJECT_ENVIRONMENT="$VENV_PATH"

    # 关键：所有 rocm 包都要做成 direct dependency，
    # 这样上面 configure_pyproject_indexes 写的 [tool.uv.sources] 才生效。
    # 只 add 一个 rocm 元包的话，transitive 的 rocm-sdk-* 会绕开 sources，
    # 跑去 default index（PyPI 镜像）找不到然后失败。

    if [[ "$INSTALL_MODE" == "minimal" ]]; then
        log "uv add rocm-core / rocm-smi-lib (==${ROCM_VERSION})"

        if ! uv add "rocm-core==${ROCM_VERSION}" "rocm-smi-lib==${ROCM_VERSION}"; then
            warn "minimal 锁版本失败，尝试不锁版本"
            uv add rocm-core rocm-smi-lib
        fi
    else
        log "uv add rocm + rocm-sdk-{core,devel,libraries-${GPU_ARCH}} (==${ROCM_VERSION})"

        if ! uv add \
            "rocm==${ROCM_VERSION}" \
            "rocm-sdk-core==${ROCM_VERSION}" \
            "rocm-sdk-devel==${ROCM_VERSION}" \
            "rocm-sdk-libraries-${GPU_ARCH}==${ROCM_VERSION}"
        then
            warn "full 安装失败，回退到 minimal（rocm-core + rocm-smi-lib）"
            uv add rocm-core rocm-smi-lib
        fi
    fi

    echo -e "\n  ${GREEN}✓${NC} ROCm packages added (recorded in pyproject.toml + installed)"
}

write_activation_helper() {
    echo ""
    draw_box "Writing Activation Helper"
    echo ""

    local helper="${PROJECT_DIR}/activate-rocm.sh"

    # 这个 helper 必须能跨机器使用，所以路径全部相对脚本自身计算
    cat > "$helper" <<'EOF'
#!/usr/bin/env bash
# Activate ROCm uv project environment.
# 路径全部相对本脚本所在目录解析，clone 到任意机器都可用。

# 必须 source 而不是直接执行，否则 venv 激活不会保留到调用方 shell
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "请使用 'source $0' 而不是直接执行" >&2
    exit 1
fi

ROCM_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROCM_VENV="${ROCM_PROJECT_DIR}/.venv"

if [[ ! -d "${ROCM_VENV}" ]]; then
    echo "[activate-rocm] .venv 不存在：${ROCM_VENV}" >&2
    echo "[activate-rocm] 请先在项目目录下执行 'uv sync'" >&2
    return 1
fi

# shellcheck source=/dev/null
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
EOF

    chmod +x "$helper"

    echo -e "  ${GREEN}✓${NC} Helper (portable): ${helper}"
}

verify_env() {
    echo ""
    draw_box "Verifying Environment"
    echo ""

    local venv_python="${VENV_PATH}/bin/python"

    echo -e "  ${CYAN}Project dir:${NC}"
    echo "  ${PROJECT_DIR}"

    echo ""
    echo -e "  ${CYAN}Venv path:${NC}"
    echo "  ${VENV_PATH}"

    echo ""
    echo -e "  ${CYAN}Python:${NC}"
    "$venv_python" --version || true

    echo ""
    echo -e "  ${CYAN}Installed ROCm packages:${NC}"
    uv pip list --python "$venv_python" | grep -iE 'rocm|hip|amd' || true

    echo ""
    echo -e "  ${CYAN}Check ROCm package version:${NC}"
    "$venv_python" -m pip show rocm-core 2>/dev/null | grep -E 'Name|Version' || true

    echo ""
    echo -e "  ${CYAN}System ROCm commands:${NC}"

    if has_cmd rocminfo; then
        echo -n "  rocminfo: "
        if rocminfo >/dev/null 2>&1; then
            echo -e "${GREEN}PASS${NC}"
        else
            echo -e "${YELLOW}FOUND but failed${NC}"
        fi
    else
        echo -e "  rocminfo: ${YELLOW}not found${NC}"
    fi

    if has_cmd rocm-smi; then
        echo -n "  rocm-smi: "
        if rocm-smi >/dev/null 2>&1; then
            echo -e "${GREEN}PASS${NC}"
        else
            echo -e "${YELLOW}FOUND but failed${NC}"
        fi
    else
        echo -e "  rocm-smi: ${YELLOW}not found${NC}"
    fi

    if [[ -e /dev/kfd ]]; then
        echo -e "  /dev/kfd: ${GREEN}present${NC}"
    else
        echo -e "  /dev/kfd: ${YELLOW}missing${NC}"
    fi
}

print_summary() {
    echo ""
    draw_box "Done"
    echo ""

    echo -e "  ${GREEN}ROCm uv project environment created successfully.${NC}"
    echo ""
    echo -e "  ${BOLD}Environment:${NC}"
    echo -e "    Project dir:  ${CYAN}${PROJECT_DIR}${NC}"
    echo -e "    Venv path:    ${CYAN}${VENV_PATH}${NC}"
    echo -e "    GPU arch:     ${CYAN}${GPU_ARCH}${NC}"
    echo -e "    ROCm version: ${CYAN}${ROCM_VERSION}${NC}"
    echo -e "    Install mode: ${CYAN}${INSTALL_MODE}${NC}"
    echo ""
    echo -e "  ${BOLD}Activate:${NC}"
    echo -e "    ${CYAN}source ${VENV_PATH}/bin/activate${NC}"
    echo ""
    echo -e "  ${BOLD}Or use helper:${NC}"
    echo -e "    ${CYAN}source ${PROJECT_DIR}/activate-rocm.sh${NC}"
    echo ""
    echo -e "  ${BOLD}Check packages:${NC}"
    echo -e "    ${CYAN}uv pip list --python ${VENV_PATH}/bin/python | grep -i rocm${NC}"
    echo ""
    echo -e "  ${BOLD}Check ROCm version in venv:${NC}"
    echo -e "    ${CYAN}${VENV_PATH}/bin/python -m pip show rocm-core | grep Version${NC}"
    echo ""
}

#######################################
# Main
#######################################

main() {
    parse_args "$@"

    setup_user_path

    print_header

    draw_box "Detecting System"
    echo ""
    detect_os
    detect_pkg_manager
    install_basic_tools

    detect_gpu

    ensure_uv
    ensure_fzf
    ensure_python

    select_gpu_arch

    if ! check_arch_index "$GPU_ARCH"; then
        error "AMD wheel index not available for arch: ${GPU_ARCH}, URL: $(arch_index_url "$GPU_ARCH")"
    fi

    select_rocm_version
    select_install_mode
    select_region
    select_pypi_mirror
    select_project_and_venv_path

    init_uv_project
    create_rocm_venv
    configure_pyproject_indexes
    add_rocm_packages
    write_activation_helper
    verify_env
    print_summary
}

main "$@"
