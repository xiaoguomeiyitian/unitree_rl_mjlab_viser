#!/usr/bin/env bash
# ============================================================================
# install_env.sh — 一键安装 unitree_rl_mjlab_viser 运行环境
#
# 自动检测 NVIDIA GPU, 智能选择安装路径:
#
#   有 NVIDIA GPU (任何版本, 含笔记本 MX 系列之外的独立卡)
#     → 安装 PyTorch cu128 wheel + nvidia 运行时 + libegl1-mesa
#     → 运行时: MUJOCO_GL=egl (EGL 后端, 性能最佳)
#
#   无 NVIDIA GPU / 仅有 AMD 集成显卡 / WSL2 无 GPU 透传
#     → 安装 PyTorch cpu wheel + libosmesa6
#     → 运行时: CUDA_VISIBLE_DEVICES="" MUJOCO_GL=osmesa
#
# 用法:
#   ./scripts/install_env.sh                     # 自动检测 + 交互确认
#   ./scripts/install_env.sh --force-gpu         # 强制 GPU 模式 (无 GPU 则失败)
#   ./scripts/install_env.sh --force-cpu         # 强制 CPU 模式
#   ./scripts/install_env.sh --recreate          # 删除现有 venv 后重建
#   ./scripts/install_env.sh --mirror cn         # 国内镜像 (阿里云)
#   ./scripts/install_env.sh --python 3.12       # 指定 Python 版本
#   ./scripts/install_env.sh --no-apt            # 跳过 apt (假定系统依赖已装)
#   ./scripts/install_env.sh -h                  # 显示帮助
#
# 设计原则:
#   1. 复用 gr00t_mjlab_autodl/install_native.sh 的成熟函数
#   2. 不修改用户的 ~/.bashrc
#   3. venv 放在 $PROJECT_DIR/.venv (项目目录内, 已在 .gitignore)
#   4. Python 版本 3.10–3.13 (mjlab 1.2.0 要求)
#   5. 钉死 setuptools<82 (PyTorch 2.11.0+cu128 要求)
#
# 预期: GPU 路径 ~5–10 分钟 (PyTorch wheel 下载 ~1.5GB), CPU 路径 ~3–5 分钟
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RL_MJLAB_ROOT="$(cd "$PROJECT_ROOT/../unitree_rl_mjlab" && pwd 2>/dev/null || echo "")"

# ── 颜色 & 日志 ──────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
info()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
step()  { echo -e "${BLUE}[→]${NC} $1"; }
fail()  { echo -e "${RED}[✗]${NC} $1"; exit 1; }
banner(){ echo -e "\n${BOLD}${BLUE}$1${NC}"; }

# ── 默认参数 ─────────────────────────────────────────────────────────────
FORCE_MODE=""          # "" (auto) | "gpu" | "cpu"
MIRROR=""              # "" = official, "cn" = 阿里云
NO_APT=false
RECREATE_VENV=false
PYTHON_VERSION="3.10"

VENV_DIR="$PROJECT_ROOT/.venv"
VENV_BIN="$VENV_DIR/bin"

# ── 版本常量 ─────────────────────────────────────────────────────────────
# GPU 路径
PYTORCH_GPU_VERSION="2.11.0+cu128"
TORCHVISION_GPU_VERSION="0.26.0+cu128"
TORCHAUDIO_GPU_VERSION="2.11.0+cu128"
TRITON_VERSION="3.6.0"
CUDA_INDEX="https://download.pytorch.org/whl/cu128"
NVIDIA_CUDNN_VERSION="9.19.0.56"
NVIDIA_CUBLAS_VERSION="12.8.4.1"
NVIDIA_CUFFT_VERSION="11.3.3.83"
NVIDIA_CURAND_VERSION="10.3.9.90"
NVIDIA_CUSPARSE_VERSION="12.5.8.93"
NVIDIA_CUSOLVER_VERSION="11.7.3.90"
NVIDIA_NCCL_VERSION="2.28.9"
NVIDIA_NVSHMEM_VERSION="3.4.5"
NVIDIA_NVJITLINK_VERSION="12.8.93"
NVIDIA_CUDA_NVRTC_VERSION="12.8.93"
NVIDIA_NVTX_VERSION="12.8.90"
NVIDIA_CUPTI_VERSION="12.8.90"
NVIDIA_CUDA_RUNTIME_VERSION="12.8.90"

# CPU 路径
PYTORCH_CPU_VERSION="2.11.0+cpu"
TORCHVISION_CPU_VERSION="0.26.0+cpu"
TORCHAUDIO_CPU_VERSION="2.11.0+cpu"

# 通用
MUJOCO_RANGE=">=3.5.0,<3.6.0"
WARP_RANGE=">=1.12.0,<1.13.0"
MJLAB_VERSION="1.2.0"

# 全局状态
GPU_CC=""
GPU_NAME=""
DRIVER_VER=""
PYTHON_BIN=""
PYTHON_VER=""
SELECTED_MODE=""        # 最终选择的模式: "gpu" | "cpu"
PYTHON_VERSION="3.10"

# ── 用法说明 ────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
用法: $0 [选项]

选项:
  --force-gpu               强制 GPU 模式 (无 NVIDIA GPU 则报错退出)
  --force-cpu               强制 CPU 模式 (跳过 GPU 检测)
  --recreate                删除现有 venv 后重建
  --mirror <cn|official>    pip/apt 镜像源 (默认 official, 国内用户选 cn)
  --python <ver>            Python 版本偏好: 3.10 | 3.11 | 3.12 | 3.13 (默认 3.10)
  --no-apt                  跳过 apt (假定系统依赖已装好)
  -h, --help                显示本帮助

示例:
  $0                                # 自动检测 + 交互确认
  $0 --force-gpu                    # 服务器: 强制装 GPU 版本
  $0 --force-cpu                    # 笔记本没独显: 装 CPU 版本
  $0 --mirror cn --force-gpu        # 国内服务器一键安装
  $0 --recreate --force-gpu         # 重建环境
  $0 --no-apt                       # 跳过 apt (用户自己装过依赖)

环境变量 (高级):
  CUDA_VISIBLE_DEVICES      若已设置且非空, 默认走 GPU 模式
                            若为空, 强制 CPU 模式
EOF
}

# ── 参数解析 ────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force-gpu)         FORCE_MODE="gpu"; shift ;;
        --force-cpu)         FORCE_MODE="cpu"; shift ;;
        --recreate)          RECREATE_VENV=true; shift ;;
        --mirror)            MIRROR="$2"; shift 2 ;;
        --python)            PYTHON_VERSION="$2"; shift 2 ;;
        --no-apt)            NO_APT=true; shift ;;
        -h|--help)           usage; exit 0 ;;
        *) fail "未知参数: $1 (用 --help 查看用法)" ;;
    esac
done

# ═════════════════════════════════════════════════════════════════════════════
# 1. 环境检测
# ═════════════════════════════════════════════════════════════════════════════

detect_os() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS_ID="$ID"
        OS_VERSION="$VERSION_ID"
        info "OS: $PRETTY_NAME"
        case "$OS_ID" in
            ubuntu|debian) ;;
            *)
                warn "OS $OS_ID 未在测试列表 (Ubuntu 22.04 / 24.04 / Debian), 可能可以工作但不保证"
                ;;
        esac
    else
        warn "无法识别操作系统 (缺少 /etc/os-release), 将尝试继续"
        OS_ID="unknown"
    fi
}

detect_gpu() {
    # 在用户强制 CPU 模式时跳过检测
    if [ "$FORCE_MODE" = "cpu" ]; then
        info "用户指定 --force-cpu, 跳过 GPU 检测"
        return 1
    fi

    if ! command -v nvidia-smi &>/dev/null; then
        warn "未找到 nvidia-smi (NVIDIA 驱动未装?)"
        return 1
    fi

    if ! nvidia-smi &>/dev/null; then
        warn "nvidia-smi 调用失败 (驱动异常?)"
        return 1
    fi

    GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)

    # 集成显卡 (如 NVIDIA Optimus 的集成 GPU) 名字通常含 "NVS" 或 "Quadro"
    # 但对于 ml 训练, 即使是入门独显 (MX 系列之外) 也有价值
    # 这里只拒绝明显不是 GPU 的情况 (None / Unknown)
    if [ -z "$GPU_NAME" ] || [ "$GPU_NAME" = "Unknown" ]; then
        warn "nvidia-smi 返回的 GPU 名称无效"
        return 1
    fi

    info "GPU: $GPU_NAME (compute_cap=sm_$GPU_CC, driver $DRIVER_VER)"

    # CUDA 12.x 需要 Driver ≥ 525.60
    if [ -n "$DRIVER_VER" ]; then
        DRIVER_MAJOR="${DRIVER_VER%%.*}"
        if [ "$DRIVER_MAJOR" -lt 525 ] 2>/dev/null; then
            warn "Driver $DRIVER_VER 较旧 (建议 ≥ 525 支持 CUDA 12.x), PyTorch 可能无法使用 GPU"
        fi
    fi

    return 0
}

detect_python() {
    PYTHON_BIN=""

    # 按优先级查找 (venv 优先)
    local candidates=(
        "$VENV_BIN/python3"
        "/usr/bin/python${PYTHON_VERSION}"
        "/usr/bin/python3.12"
        "/usr/bin/python3.11"
        "/usr/bin/python3.10"
        "/usr/local/bin/python${PYTHON_VERSION}"
        "/usr/local/bin/python3.12"
        "/usr/local/bin/python3.11"
        "/usr/local/bin/python3.10"
        "$(command -v python3)"
        "$(command -v python)"
    )

    for c in "${candidates[@]}"; do
        if [ -x "$c" ]; then
            local ver
            ver=$("$c" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
            if [ -n "$ver" ]; then
                local major="${ver%%.*}"
                local minor="${ver##*.}"
                # mjlab 1.2.0: Python >= 3.10, < 3.14
                if [ "$major" = "3" ] && [ "$minor" -ge 10 ] && [ "$minor" -lt 14 ] 2>/dev/null; then
                    PYTHON_BIN="$c"
                    PYTHON_VER="$ver"
                    PYTHON_VERSION="$ver"
                    break
                fi
            fi
        fi
    done

    if [ -z "$PYTHON_BIN" ]; then
        fail "未找到 Python 3.10+ (mjlab 1.2.0 需要 >= 3.10, < 3.14). 安装: 'sudo apt install python3.10 python3.10-venv python3.10-dev'"
    fi

    info "Python: $PYTHON_BIN ($($PYTHON_BIN --version 2>&1))"
}

select_install_mode() {
    banner "[0/8] 选择安装模式"

    local detected="cpu"
    if [ -n "$GPU_NAME" ]; then
        detected="gpu"
    fi

    if [ -n "$FORCE_MODE" ]; then
        SELECTED_MODE="$FORCE_MODE"
        if [ "$SELECTED_MODE" = "gpu" ] && [ -z "$GPU_NAME" ]; then
            fail "--force-gpu 但未检测到 NVIDIA GPU (nvidia-smi 不可用或 GPU 数量为 0)"
        fi
        info "用户指定模式: $SELECTED_MODE"
        return
    fi

    # 询问用户
    echo ""
    echo -e "  ${BOLD}自动检测结果:${NC}"
    if [ "$detected" = "gpu" ]; then
        echo -e "    检测到 NVIDIA GPU: ${GREEN}$GPU_NAME${NC}"
        echo -e "    推荐: ${CYAN}GPU 模式${NC} (cu128 PyTorch + EGL 后端, 性能最佳)"
    else
        echo -e "    未检测到 NVIDIA GPU"
        echo -e "    推荐: ${CYAN}CPU 模式${NC} (cpu PyTorch + OSMesa 后端)"
    fi
    echo ""
    echo "  请选择安装模式:"
    echo -e "    ${CYAN}1${NC}) GPU 模式 — 有 NVIDIA 显卡, 训练和仿真都在 GPU 上跑"
    echo -e "    ${CYAN}2${NC}) CPU 模式 — 没有 NVIDIA 显卡, 物理仿真在 CPU 上跑 (慢但可用)"
    echo -e "    ${CYAN}3${NC}) 退出"
    echo ""

    local default_choice="1"
    [ "$detected" = "cpu" ] && default_choice="2"

    while true; do
        read -p "  请选择 [1-3, 默认 $default_choice]: " choice
        choice="${choice:-$default_choice}"
        case "$choice" in
            1) SELECTED_MODE="gpu"; break ;;
            2) SELECTED_MODE="cpu"; break ;;
            3) echo "已取消"; exit 0 ;;
            *) echo "  无效选择, 请输入 1, 2 或 3" ;;
        esac
    done

    info "已选择: $SELECTED_MODE 模式"
}

# ═════════════════════════════════════════════════════════════════════════════
# 2. 安装步骤
# ═════════════════════════════════════════════════════════════════════════════

install_system_deps() {
    if [ "$NO_APT" = true ]; then
        info "跳过 apt (--no-apt)"
        return 0
    fi

    if [ "$OS_ID" = "unknown" ]; then
        warn "无法识别 OS, 跳过 apt (请手动装 Python + venv + OpenGL 库)"
        return 0
    fi

    banner "[1/8] 安装系统依赖 (apt)"

    local pkgs=(
        # Python 与构建工具
        python3-dev python3-venv python3-pip
        build-essential cmake ninja-build pkg-config
        # Git 与 LFS
        git git-lfs curl wget ca-certificates
        # OpenGL 通用库
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1
        # 网络工具
        net-tools iputils-ping
        # ffmpeg (录制视频)
        ffmpeg
    )

    # 按模式追加 GL 后端
    if [ "$SELECTED_MODE" = "gpu" ]; then
        pkgs+=(libegl1-mesa libopengl0 libgles2-mesa)
    else
        pkgs+=(libosmesa6)
    fi

    # 选 python3.12-venv / dev (若系统已有该版本)
    case "$PYTHON_VERSION" in
        3.12)
            if apt-cache show python3.12-venv &>/dev/null 2>&1; then
                pkgs+=(python3.12 python3.12-dev python3.12-venv)
            fi
            ;;
        3.11)
            if apt-cache show python3.11-venv &>/dev/null 2>&1; then
                pkgs+=(python3.11 python3.11-dev python3.11-venv)
            fi
            ;;
        3.13)
            if apt-cache show python3.13-venv &>/dev/null 2>&1; then
                pkgs+=(python3.13 python3.13-dev python3.13-venv)
            fi
            ;;
        # 3.10 默认已装
    esac

    # 镜像源 (cn)
    if [ "$MIRROR" = "cn" ]; then
        info "使用阿里云 apt 镜像"
        if [ -w /etc/apt/sources.list ]; then
            sed -i.bak 's|http://archive.ubuntu.com|http://mirrors.aliyun.com|g; s|http://security.ubuntu.com|http://mirrors.aliyun.com|g' /etc/apt/sources.list
        else
            sudo sed -i.bak 's|http://archive.ubuntu.com|http://mirrors.aliyun.com|g; s|http://security.ubuntu.com|http://mirrors.aliyun.com|g' /etc/apt/sources.list
        fi
    fi

    info "更新 apt 索引..."
    if [ -w /var/lib/apt/lists ] && command -v sudo &>/dev/null; then
        sudo apt-get update
    elif [ "$(id -u)" = "0" ]; then
        apt-get update
    else
        warn "apt-get update 需要 sudo 权限"
        sudo apt-get update
    fi

    info "安装 ${#pkgs[@]} 个包..."
    if [ "$(id -u)" = "0" ]; then
        apt-get install -y --no-install-recommends "${pkgs[@]}"
    else
        sudo apt-get install -y --no-install-recommends "${pkgs[@]}"
    fi

    info "apt 安装完成"
}

create_venv() {
    banner "[2/8] 创建 Python 虚拟环境"

    if [ -d "$VENV_DIR" ] && [ "$RECREATE_VENV" = false ]; then
        info "检测到现有 venv: $VENV_DIR"
        if "$VENV_BIN/python" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
            info "现有 venv 可用 (Python $("$VENV_BIN/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')), 跳过重建"
            info "(用 --recreate 强制重建)"
            return 0
        fi
        warn "现有 venv 不可用 (Python 版本不满足), 将删除重建"
    fi

    if [ -d "$VENV_DIR" ]; then
        info "删除旧 venv..."
        rm -rf "$VENV_DIR"
    fi

    info "创建 venv: $VENV_DIR (Python: $PYTHON_BIN)"
    "$PYTHON_BIN" -m venv "$VENV_DIR"

    # 关键: 不要升级 setuptools 到 82+
    # PyTorch 2.11.0+cu128 requires setuptools<82
    info "升级 pip + wheel (保持 setuptools<82)..."
    "$VENV_BIN/pip" install --upgrade "pip>=23" "wheel" "setuptools<82"

    info "venv 创建完成"
}

setup_pip_mirror() {
    if [ "$MIRROR" = "cn" ]; then
        info "配置 pip 镜像源: 阿里云"
        "$VENV_BIN/pip" config set global.index-url "https://mirrors.aliyun.com/pypi/simple/"
        "$VENV_BIN/pip" config set global.trusted-host "mirrors.aliyun.com"
    else
        "$VENV_BIN/pip" config unset global.index-url 2>/dev/null || true
        info "使用 PyPI 官方源"
    fi
}

install_pytorch_gpu() {
    banner "[3/8] 安装 PyTorch + CUDA runtime ($PYTORCH_GPU_VERSION)"

    info "3a. PyTorch 主包 (~850MB)..."
    "$VENV_BIN/pip" install --no-cache-dir --no-deps \
        --index-url "$CUDA_INDEX" \
        "torch==$PYTORCH_GPU_VERSION" \
        "torchvision==$TORCHVISION_GPU_VERSION" \
        "torchaudio==$TORCHAUDIO_GPU_VERSION"

    info "3b. CUDA 数学库 (cudnn/cublas/cufft/curand/cusparse/cusolver)..."
    "$VENV_BIN/pip" install --no-cache-dir \
        --index-url "$CUDA_INDEX" \
        "nvidia-cudnn-cu12==$NVIDIA_CUDNN_VERSION" \
        "nvidia-cublas-cu12==$NVIDIA_CUBLAS_VERSION" \
        "nvidia-cufft-cu12==$NVIDIA_CUFFT_VERSION" \
        "nvidia-curand-cu12==$NVIDIA_CURAND_VERSION" \
        "nvidia-cusparse-cu12==$NVIDIA_CUSPARSE_VERSION" \
        "nvidia-cusolver-cu12==$NVIDIA_CUSOLVER_VERSION"

    info "3c. CUDA 通信 (nccl/nvjitlink/nvtx/...)..."
    "$VENV_BIN/pip" install --no-cache-dir \
        --index-url "$CUDA_INDEX" \
        "nvidia-nccl-cu12==$NVIDIA_NCCL_VERSION" \
        "nvidia-nvshmem-cu12==$NVIDIA_NVSHMEM_VERSION" \
        "nvidia-nvjitlink-cu12==$NVIDIA_NVJITLINK_VERSION" \
        "nvidia-cuda-nvrtc-cu12==$NVIDIA_CUDA_NVRTC_VERSION" \
        "nvidia-nvtx-cu12==$NVIDIA_NVTX_VERSION" \
        "nvidia-cuda-cupti-cu12==$NVIDIA_CUPTI_VERSION" \
        "nvidia-cuda-runtime-cu12==$NVIDIA_CUDA_RUNTIME_VERSION"

    info "3d. triton (~200MB)..."
    "$VENV_BIN/pip" install --no-cache-dir \
        --index-url "$CUDA_INDEX" \
        "triton==$TRITON_VERSION"

    # 验证 GPU
    info "验证 PyTorch + GPU..."
    "$VENV_BIN/python" -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  CUDA version: {torch.version.cuda}')
    print(f'  Device: {torch.cuda.get_device_name(0)}')
    x = torch.randn(100, 100, device='cuda')
    y = x @ x.T
    print(f'  GPU compute test: OK (sum={y.sum().item():.2f})')
" 2>&1 | tee /tmp/torch_check.log
    if grep -q "GPU compute test: OK" /tmp/torch_check.log; then
        info "PyTorch + GPU 验证成功 ✓"
    else
        warn "PyTorch + GPU 验证失败, 但不影响后续安装 (请检查 CUDA 驱动版本)"
    fi
}

install_pytorch_cpu() {
    banner "[3/8] 安装 PyTorch (CPU: $PYTORCH_CPU_VERSION)"

    info "PyTorch CPU wheel (~250MB, 比 GPU 版小 3x)..."
    "$VENV_BIN/pip" install --no-cache-dir --no-deps \
        --index-url "https://download.pytorch.org/whl/cpu" \
        "torch==$PYTORCH_CPU_VERSION" \
        "torchvision==$TORCHVISION_CPU_VERSION" \
        "torchaudio==$TORCHAUDIO_CPU_VERSION"

    info "验证 PyTorch (CPU)..."
    "$VENV_BIN/python" -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()} (期望 False)')
# CPU 计算测试
x = torch.randn(100, 100)
y = x @ x.T
print(f'  CPU compute test: OK (sum={y.sum().item():.2f})')
" 2>&1
    info "PyTorch CPU 安装验证完成 ✓"
}

install_mujoco_warp() {
    banner "[4/8] 安装 mujoco + warp-lang"

    info "mujoco $MUJOCO_RANGE..."
    "$VENV_BIN/pip" install --no-cache-dir "mujoco$MUJOCO_RANGE"

    info "warp-lang $WARP_RANGE..."
    "$VENV_BIN/pip" install --no-cache-dir "warp-lang$WARP_RANGE"

    info "验证 mujoco + warp..."
    "$VENV_BIN/python" -c "
import mujoco
import warp as wp
print(f'  mujoco: {mujoco.__version__}')
print(f'  warp: {wp.__version__}')
wp.init()
print(f'  warp.init(): OK')
devices = wp.get_devices()
print(f'  Available devices: {[str(d) for d in devices]}')
" || warn "mujoco/warp 验证有警告, 不一定致命"
}

install_mjlab() {
    banner "[5/8] 安装 mjlab (editable from ../unitree_rl_mjlab)"

    if [ -z "$RL_MJLAB_ROOT" ] || [ ! -d "$RL_MJLAB_ROOT" ]; then
        fail "未找到兄弟项目: ../unitree_rl_mjlab (期望与本项目同级目录存在 unitree_rl_mjlab/)"
    fi
    if [ ! -f "$RL_MJLAB_ROOT/setup.py" ]; then
        fail "$RL_MJLAB_ROOT/setup.py 不存在 (破损的 unitree_rl_mjlab 目录)"
    fi

    info "RL_MJLAB_ROOT: $RL_MJLAB_ROOT"

    # mjlab editable 安装 (含 90+ 依赖)
    info "pip install -e . (mjlab + 90+ 依赖, 约 1GB, 可能需要 5-10 分钟)..."
    (cd "$RL_MJLAB_ROOT" && "$VENV_BIN/pip" install --no-cache-dir -e .)

    info "验证 mjlab..."
    "$VENV_BIN/python" -c "
import mjlab
import mujoco_warp
print(f'  mjlab: {getattr(mjlab, \"__version__\", \"installed\")}')
print(f'  mujoco-warp: {getattr(mujoco_warp, \"__version__\", \"installed\")}')
" || warn "mjlab 验证有警告"
}

install_viser_stack() {
    banner "[6/8] 安装 Viser + Tyro"

    info "viser + tyro (从本项目 pyproject.toml)..."
    "$VENV_BIN/pip" install --no-cache-dir "viser>=0.2.0" "tyro>=0.9.0"

    # 然后装本项目自己 (editable)
    info "pip install -e . (本项目 unitree_rl_mjlab_viser)..."
    (cd "$PROJECT_ROOT" && "$VENV_BIN/pip" install --no-cache-dir -e .)

    info "Viser 栈安装完成"
}

# ═════════════════════════════════════════════════════════════════════════════
# 3. 健康检查
# ═════════════════════════════════════════════════════════════════════════════

health_check() {
    banner "[7/8] 健康检查"

    echo ""
    echo -e "${BOLD}=== 环境摘要 ===${NC}"
    echo "  OS:        ${OS_ID:-$PRETTY_NAME}"
    echo "  Python:    $PYTHON_BIN ($PYTHON_VER)"
    echo "  GPU:       ${GPU_NAME:-未检测到} (sm_${GPU_CC:-none})"
    echo "  模式:      $SELECTED_MODE"
    echo "  venv:      $VENV_DIR"
    echo "  镜像:      ${MIRROR:-official}"
    echo ""

    "$VENV_BIN/python" -c "
import sys
checks = []

# torch
try:
    import torch
    cuda_status = torch.cuda.is_available()
    checks.append(('torch', f'{torch.__version__} (CUDA={cuda_status})'))
except Exception as e:
    checks.append(('torch', f'FAIL: {e}'))

# mujoco
try:
    import mujoco
    checks.append(('mujoco', mujoco.__version__))
except Exception as e:
    checks.append(('mujoco', f'FAIL: {e}'))

# warp
try:
    import warp as wp
    wp.init()
    devices = [str(d) for d in wp.get_devices()]
    checks.append(('warp-lang', f'{wp.__version__} devices={devices}'))
except Exception as e:
    checks.append(('warp-lang', f'FAIL: {e}'))

# mujoco-warp
try:
    import mujoco_warp
    checks.append(('mujoco-warp', getattr(mujoco_warp, '__version__', 'installed')))
except Exception as e:
    checks.append(('mujoco-warp', f'FAIL: {e}'))

# mjlab
try:
    import mjlab
    checks.append(('mjlab', getattr(mjlab, '__version__', '1.2.0')))
except Exception as e:
    checks.append(('mjlab', f'FAIL: {e}'))

# unitree_rl_mjlab (editable)
try:
    import src.tasks
    checks.append(('unitree_rl_mjlab', 'editable OK'))
except Exception as e:
    checks.append(('unitree_rl_mjlab', f'FAIL: {e}'))

# viser + tyro
try:
    import viser, tyro
    checks.append(('viser', viser.__version__ if hasattr(viser, '__version__') else 'installed'))
    checks.append(('tyro', tyro.__version__ if hasattr(tyro, '__version__') else 'installed'))
except Exception as e:
    checks.append(('viser/tyro', f'FAIL: {e}'))

# unitree_viser (本项目)
try:
    import unitree_viser
    checks.append(('unitree_viser', unitree_viser.__version__))
except Exception as e:
    checks.append(('unitree_viser', f'FAIL: {e}'))

errors = 0
for name, ver in checks:
    status = '\033[0;32mOK  \033[0m' if not ver.startswith('FAIL') else '\033[0;31mFAIL\033[0m'
    print(f'  {status} {name:25s} {ver}')
    if ver.startswith('FAIL'):
        errors += 1
print()
if errors == 0:
    print('\033[0;32m健康检查全部通过 ✓\033[0m')
else:
    print(f'\033[0;31m发现 {errors} 个问题\033[0m')
    sys.exit(1)
"
}

# ═════════════════════════════════════════════════════════════════════════════
# 4. 运行时环境配置 (写文件, 供 start.sh 读取)
# ═════════════════════════════════════════════════════════════════════════════

write_runtime_env() {
    banner "[8/8] 配置运行时环境"

    local env_file="$VENV_DIR/runtime_env.sh"
    {
        echo "# Auto-generated by install_env.sh on $(date -Iseconds 2>/dev/null || date)"
        echo "# Source this file before running unitree-viser:"
        echo "#   source $env_file"
        echo ""
        echo "export UNITREE_VISER_MODE='$SELECTED_MODE'"
        if [ "$SELECTED_MODE" = "gpu" ]; then
            echo "export MUJOCO_GL='egl'"
            echo "unset CUDA_VISIBLE_DEVICES"
        else
            echo "export CUDA_VISIBLE_DEVICES=''"
            echo "export MUJOCO_GL='osmesa'"
        fi
        # nvidia 库路径 (仅 GPU)
        if [ "$SELECTED_MODE" = "gpu" ] && [ -d "$VENV_DIR/lib/python${PYTHON_VER}/site-packages/nvidia" ]; then
            local nvidia_libs
            nvidia_libs=$(find "$VENV_DIR/lib/python${PYTHON_VER}/site-packages/nvidia" -type d -name 'lib' 2>/dev/null | sort -u | tr '\n' ':')
            if [ -n "$nvidia_libs" ]; then
                echo "export LD_LIBRARY_PATH='${nvidia_libs}\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}'"
            fi
        fi
    } > "$env_file"

    info "运行时环境已写入: $env_file"
    echo ""
    cat "$env_file"
    echo ""
}

print_summary() {
    cat <<EOF

${BOLD}${GREEN}╔════════════════════════════════════════════════════════════════╗
║              安装完成 ✓                                       ║
╚════════════════════════════════════════════════════════════════╝${NC}

${BOLD}下一步:${NC}

  1. 激活虚拟环境:
     ${CYAN}source $VENV_DIR/bin/activate${NC}

  2. 运行时环境变量 (start.sh 会自动检测, 但手动运行时需要 source):
     ${CYAN}source $VENV_DIR/runtime_env.sh${NC}

  3. 验证安装:
     ${CYAN}./start.sh version${NC}      # 显示版本 + GPU/CPU 模式
     ${CYAN}./start.sh test${NC}         # 跑 smoke test
     ${CYAN}./start.sh list${NC}         # 列出可用任务

  4. 启动训练:
     ${CYAN}./start.sh train Unitree-G1-Flat --num-envs 64 --max-iterations 100${NC}

  5. 启动仿真:
     ${CYAN}./start.sh sim Unitree-G1-Flat${NC}

${BOLD}当前模式:${NC} $SELECTED_MODE
${BOLD}venv 路径:${NC}  $VENV_DIR
${BOLD}Python:${NC}     $PYTHON_BIN ($PYTHON_VER)

${BOLD}常见问题:${NC}

  Q: 想切换到 GPU/CPU 模式?
  A: 重新运行 $0 --force-gpu/cpu --recreate

  Q: 训练时报 "CUDA out of memory"?
  A: 减小 --num-envs (从 1024 降到 256 或 64)

  Q: CPU 模式下 mjlab 有 bug?
  A: mjlab 1.2.0 的 CPU 模式已知有部分测试跳过 (Likely bug on CPU MjWarp),
     但训练/仿真的核心流程可用. 详见上游 mjlab FAQ.

  Q: 想升级某个包?
  A: source $VENV_DIR/bin/activate && pip install -U <pkg>
EOF
}

# ═════════════════════════════════════════════════════════════════════════════
# 主流程
# ═════════════════════════════════════════════════════════════════════════════

main() {
    banner "═══ unitree_rl_mjlab_viser 一键安装 ═══"

    # 1. 检测
    step "检测主机环境..."
    detect_os
    detect_gpu || true    # GPU 缺失不致命
    detect_python

    # 2. 选择模式
    select_install_mode

    # 3. 安装计划确认
    echo ""
    echo -e "${BOLD}安装计划:${NC}"
    echo "  模式:    $SELECTED_MODE"
    echo "  Python:  $PYTHON_VER ($PYTHON_BIN)"
    echo "  GPU:     sm_${GPU_CC:-none}"
    echo "  apt:     $([ "$NO_APT" = true ] && echo "跳过" || echo "执行")"
    echo "  venv:    $([ "$RECREATE_VENV" = true ] && echo "重建" || echo "复用或新建")"
    echo "  镜像:    ${MIRROR:-official}"
    echo ""
    read -p "继续? [Y/n]: " confirm
    case "${confirm:-Y}" in
        [yY]|[yY][eE][sS]) ;;
        *) echo "已取消"; exit 0 ;;
    esac

    # 4. 安装
    install_system_deps
    create_venv
    setup_pip_mirror

    if [ "$SELECTED_MODE" = "gpu" ]; then
        install_pytorch_gpu
    else
        install_pytorch_cpu
    fi

    install_mujoco_warp
    install_mjlab
    install_viser_stack

    # 5. 健康检查 + 写运行时 env
    health_check
    write_runtime_env

    print_summary
}

main "$@"
