#!/usr/bin/env bash
# ============================================================================
# unitree_rl_mjlab_viser 启动入口脚本
#
# 交互式:   ./start.sh
# 非交互:   ./start.sh <mode> [args...]
#
# 模式:
#   train   训练 (含可选 Viser 浏览器可视化)
#   sim     仿真模式 (浏览器中运行已训练策略 / 零动作)
#   list    列出所有可用任务 ID
#   test    运行 smoke test
#   doctor  环境健康检查
#
# 兼容:
#   - 自动检测本地 .venv 或复用 gr00t_mjlab_autodl/.venv (已有 torch+cu13)
#   - 自动检测 ../unitree_rl_mjlab 兄弟项目
#   - 自动 PYTHONPATH 集成
#
# 用法示例:
#   ./start.sh                                       # 交互式
#   ./start.sh list                                  # 列出所有任务
#   ./start.sh train Unitree-G1-Flat --num-envs 64   # 非交互训练
#   ./start.sh sim Unitree-Go2-Flat                  # 仿真 (zero policy)
#   ./start.sh sim Unitree-G1-Flat --checkpoint model.pt
#   ./start.sh test                                  # 跑 smoke test
#   ./start.sh doctor                                # 环境健康检查
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
SRC_DIR="$PROJECT_DIR/src"
SIBLING_RL_DIR="$(cd "$PROJECT_DIR/../unitree_rl_mjlab" 2>/dev/null && pwd || echo "")"
SIBLING_GR00T_DIR="$(cd "$PROJECT_DIR/../gr00t_mjlab_autodl" 2>/dev/null && pwd || echo "")"

# ── 全局变量 ───────────────────────────────────────────────────────────────
MODE=""
TASK=""
VISER_PORT="20006"
VISER_FPS="30"
VISER_ENV_IDX="0"
ENABLE_CONTROL="false"
HEADLESS="false"

# train 专用
NUM_ENVS=""
MAX_ITERATIONS=""
SEED=""
SAVE_INTERVAL=""
MOTION_FILE=""
LOG_ROOT=""
RESUME="false"
RESUME_CHECKPOINT=""
TRAIN_DEVICE="cuda:0"   # 训练默认 GPU (若不可用则回退 cpu)
USE_WANDB="false"         # 训练默认禁用 wandb (使用 tensorboard)

# sim 专用
CHECKPOINT=""
POLICY="zero"          # zero | random
NUM_ENVS_SIM="1"
SIM_DEVICE="cpu"       # 仿真默认 CPU (不占 GPU 显存)
INJECT_COMMANDS="true"
COMMAND_NAME="twist"
MAX_STEPS=""

# sim 命令注入源 (DDS 遥控器集成)
COMMAND_SOURCE="gui"   # gui | dds | both
DDS_DOMAIN="0"
DDS_INTERFACE="lo"
ROBOT_KEY="go2_0"      # 决定订阅 rt/{robot_key}/wirelesscontroller
DDS_TIMEOUT="0.5"      # 秒; 超时后归零

# deps 专用
DEPS_INSTALL="false"
DEPS_FULL="false"      # 是否装重型依赖 (torch + mjlab + mujoco-warp)

# 颜色
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[1;36m'; BOLD='\033[1m'; NC='\033[0m'

log_banner() { echo -e "${BOLD}${BLUE}$1${NC}"; }
log_info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_cmd()   { echo -e "${CYAN}[CMD]${NC}   $1"; }

# ── venv 检测 ──────────────────────────────────────────────────────────────
# 优先级:
#   1. $PROJECT_DIR/.venv  (项目本地 venv)
#   2. $SIBLING_GR00T_DIR/.venv  (gr00t_mjlab_autodl, 含完整 torch+cu13)
#   3. 系统 Python
detect_python() {
    if [ -x "$PROJECT_DIR/.venv/bin/python" ]; then
        PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"
        PIP_BIN="$PROJECT_DIR/.venv/bin/pip"
		VENV_DIR="$PROJECT_DIR/.venv"
		VENV_LABEL="本地 venv (${PYTHON_BIN})"
	elif [ -n "$SIBLING_GR00T_DIR" ] && [ -x "$SIBLING_GR00T_DIR/.venv/bin/python" ]; then
		PYTHON_BIN="$SIBLING_GR00T_DIR/.venv/bin/python"
		PIP_BIN="$SIBLING_GR00T_DIR/.venv/bin/pip"
		VENV_DIR="$SIBLING_GR00T_DIR/.venv"
		VENV_LABEL="gr00t_mjlab_autodl venv (${PYTHON_BIN})"
	elif command -v python3.12 &>/dev/null; then
		PYTHON_BIN="$(command -v python3.12)"
		PIP_BIN="$(command -v pip3.12)"
		VENV_DIR=""
		VENV_LABEL="系统 Python 3.12"
	else
		PYTHON_BIN="$(command -v python3)"
		PIP_BIN="$(command -v pip3)"
		VENV_DIR=""
	fi
}

# ── 运行时环境自动检测 (GPU/CPU) ────────────────────────────────────
# 每次启动时调用 nvidia-smi 探测, 自动设置:
#   有 NVIDIA GPU:  MUJOCO_GL=egl
#   无 NVIDIA GPU + 有 libOSMesa:  CUDA_VISIBLE_DEVICES=""  MUJOCO_GL=osmesa
#   无 NVIDIA GPU + 无 libOSMesa:  CUDA_VISIBLE_DEVICES=""  MUJOCO_GL=egl (headless via swrast)
#
# 如果用户手动 export 了 CUDA_VISIBLE_DEVICES 或 MUJOCO_GL, 尊重用户选择.
# 输出用日志, 让用户清楚知道当前是哪种模式.
setup_runtime_env() {
    # 用户已显式设置 → 尊重之
    if [ -n "${MUJOCO_GL:-}" ]; then
        log_info "运行时: 用户已设置 MUJOCO_GL=$MUJOCO_GL, 尊重"
        return 0
    fi

    local detected_gpu=false
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        local gpu_name
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
        if [ -n "$gpu_name" ] && [ "$gpu_name" != "Unknown" ]; then
            detected_gpu=true
            log_info "运行时: 检测到 NVIDIA GPU ($gpu_name), 启用 EGL 后端 (MUJOCO_GL=egl)"
        fi
    fi

    if [ "$detected_gpu" = true ]; then
        export MUJOCO_GL="egl"
        # nvidia 库路径 (如 .venv/lib/.../nvidia/*/lib)
        if [ -n "${PYTHON_BIN:-}" ] && [ -x "$PYTHON_BIN" ]; then
            local pyver
            pyver=$("$PYTHON_BIN" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
            local site_nvidia="$VENV_DIR/lib/python${pyver}/site-packages/nvidia"
            if [ -d "$site_nvidia" ]; then
                local extra_ld
                extra_ld=$(find "$site_nvidia" -type d -name 'lib' 2>/dev/null | sort -u | tr '\n' ':')
                if [ -n "$extra_ld" ]; then
                    export LD_LIBRARY_PATH="${extra_ld}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
                    log_info "运行时: 已添加 nvidia 库到 LD_LIBRARY_PATH"
                fi
            fi
        fi
    else
        export CUDA_VISIBLE_DEVICES=""
        # CPU 模式: 优先 OSMesa (软件光栅化, 无显示);若没装则降级 EGL
        # (EGL 配合 swrast 也能 headless 工作, 但需要 libEGL + libGL)
        if ldconfig -p 2>/dev/null | grep -q "libOSMesa\.so" \
            || [ -f /usr/lib/x86_64-linux-gnu/libOSMesa.so ] \
            || [ -f /usr/lib/x86_64-linux-gnu/libOSMesa.so.6 ]; then
            export MUJOCO_GL="osmesa"
            log_info "运行时: 未检测到 NVIDIA GPU, 切换到 CPU+OSMesa 模式 (CUDA_VISIBLE_DEVICES=\"\", MUJOCO_GL=osmesa)"
        else
            export MUJOCO_GL="egl"
            log_warn "运行时: 未检测到 NVIDIA GPU, 且未安装 libOSMesa"
            log_info "        降级到 EGL 后端 (MUJOCO_GL=egl), 配合 swrast 软件渲染可 headless 跑"
        fi
    fi
}

# ── 检查关键依赖 ──────────────────────────────────────────────────────────
check_core_deps() {
    local missing=()
    "$PYTHON_BIN" -c "import viser" 2>/dev/null   || missing+=("viser")
    "$PYTHON_BIN" -c "import tyro" 2>/dev/null    || missing+=("tyro")
    "$PYTHON_BIN" -c "import mjlab" 2>/dev/null   || missing+=("mjlab")
    if [ ${#missing[@]} -gt 0 ]; then
        log_warn "缺少核心依赖: ${missing[*]}"
        return 1
    fi
    return 0
}

# ── 构建 PYTHONPATH ──────────────────────────────────────────────────────
# 关键: 把 unitree_rl_mjlab/src 放第一位, 让 `import src` 优先解析为兄弟项目的
# src 包 (有 __init__.py, 含 SRC_PATH 等), 而不是本项目空 src 目录
# (后者会被 Python 当作 namespace package, 导致 `from src import SRC_PATH` 失败)
build_pythonpath() {
    local pp=""
    [ -n "$SIBLING_RL_DIR" ] && pp="$SIBLING_RL_DIR/src"
    pp="$pp:$SRC_DIR"
    PYTHONPATH="$pp"
    export PYTHONPATH
    # 同步给 cli.py 的 sys.path 修复用
    [ -n "$SIBLING_RL_DIR" ] && export UNITREE_RL_MJLAB_SRC="$SIBLING_RL_DIR/src"
}

# ── 检查上游 task 注册表是否可导入 ─────────────────────────────────────
check_task_registry() {
    if [ -z "$SIBLING_RL_DIR" ]; then
        log_error "未找到 ../unitree_rl_mjlab 兄弟项目"
        log_info "请将本项目放在 unitree_rl_mjlab 同级目录:"
        log_info "  /home/kxy/work/unitree/unitree_rl_mjlab_viser/"
        log_info "  /home/kxy/work/unitree/unitree_rl_mjlab/"
        return 1
    fi
    if [ ! -d "$SIBLING_RL_DIR/src" ]; then
        log_error "$SIBLING_RL_DIR/src 不存在 (破损的 unitree_rl_mjlab 目录)"
        return 1
    fi
    return 0
}

# ── 询问/读取单个值 ──────────────────────────────────────────────────────
prompt_select() {
    local prompt="$1"; shift; local options=("$@")
    local depth="${PROMPT_DEPTH:-0}"
    if [ "$depth" -ge 3 ]; then
        log_warn "递归过深, 使用默认 (1)" >&2
        echo "0"; return
    fi
    PROMPT_DEPTH=$((depth+1))
    echo -e "${BOLD}${prompt}${NC}" >&2
    for i in "${!options[@]}"; do echo -e "  ${CYAN}$((i+1))${NC}) ${options[$i]}" >&2; done
    local choice
    if ! read -p "请选择 [1-${#options[@]}] (默认 1): " choice; then
        # EOF / 中断: 默认选 0
        echo "0"; PROMPT_DEPTH=0; return
    fi
    if [ -z "$choice" ]; then echo "0"; PROMPT_DEPTH=0; return; fi
    if [[ ! "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt "${#options[@]}" ]; then
        log_error "无效选择，请重新输入" >&2
        prompt_select "$prompt" "${options[@]}"
    else
        PROMPT_DEPTH=0
        echo "$((choice-1))"
    fi
}

prompt_input() {
    local p="$1" d="${2:-}" req="${3:-false}"
    while true; do
        if [ -n "$d" ]; then read -p "$p [$d]: " v; v="${v:-$d}"
        else read -p "$p: " v; fi
        [ "$req" = "true" ] && [ -z "$v" ] && { log_warn "此项必填"; continue; }
        echo "$v"; return
    done
}

# placeholder_for_fix

prompt_yn() {
    local p="$1" d="${2:-y}"
    local v
    if [ "$d" = "y" ]; then read -p "$p [Y/n]: " v; v="${v:-y}"
    else read -p "$p [y/N]: " v; v="${v:-n}"; fi
    case "${v,,}" in y|yes) echo "true" ;; *) echo "false" ;; esac
}

# ── 用法说明 ──────────────────────────────────────────────────────────────
show_usage() {
    cat <<EOF
用法:
  $0                              # 交互式启动
  $0 <mode> [args...]             # 非交互式启动

模式:
  train     训练 RL 策略 (可选 Viser 可视化)
  sim       仿真模式 (浏览器中控制虚拟机器人)
  list      列出所有可用任务
  test      运行 smoke test
  doctor    环境健康检查

通用参数:
  --task <id>              任务 ID (e.g. Unitree-G1-Flat)
  --viser-port <N>         Viser HTTP/WS 端口 (默认: 20006, 0 = 禁用)
  --viser-fps <F>          Viser 渲染 FPS (默认: 30)
  --headless               禁用 Viser, 仅后台运行

训练参数 (train):
  --num-envs <N>           并行环境数 (默认按 GPU 显存自动)
  --max-iterations <N>     最大迭代 (默认 10001)
  --seed <N>               随机种子
  --save-interval <N>      模型保存间隔
  --motion-file <path>     tracking 任务必填 (--motion-file dance.npz)
  --enable-control         启用 Viser 训练控制 (暂停/单步/速度)
  --resume                 从检查点恢复
  --checkpoint <path>      检查点路径 (同时启用 --resume)

仿真参数 (sim):
  --num-envs <N>           环境数 (默认 1)
  --checkpoint <path>      加载训练好的 .pt 策略
  --policy <zero|random>   无 checkpoint 时使用的策略
  --no-inject-commands     关闭命令注入 GUI
  --max-steps <N>          最大步数 (默认无限)
  --command-name <name>    命令 term 名称 (默认 twist)

DDS 命令注入 (unitree_remote_ctrl 集成):
  --command-source {gui,dds,both}    命令注入源 (默认 gui)
                                       gui  - Viser 浏览器滑块
                                       dds  - 订阅 rt/{robot_key}/wirelesscontroller
                                       both - 同时启用, dds 优先
  --dds-domain <N>         CycloneDDS 域 ID (默认 0)
  --dds-interface <name>   DDS 网络接口 (默认 lo, 跨机用 enp*)
  --robot-key <key>        DDS topic 后缀 (默认 go2_0)
  --dds-timeout <sec>      DDS 消息超时 (默认 0.5s, 超时归零)

示例:
  $0 list
  $0 doctor
  $0 train Unitree-G1-Flat --num-envs 64 --max-iterations 100
  $0 train Unitree-G1-Flat --viser-port 0 --num-envs 2048   # 无 Viser 训练
  $0 train Unitree-G1-Tracking-No-State-Estimation --motion-file dance1.npz
  $0 sim Unitree-G1-Flat                                    # 零动作
  $0 sim Unitree-G1-Flat --checkpoint logs/.../model_500.pt
  $0 sim Unitree-G1-Flat --policy random
  $0 test
EOF
}

# ══════════════════════════════════════════════════════════════════════════════
# 命令构建
# ══════════════════════════════════════════════════════════════════════════════
_opt()  { [ "${2:-}" != "" ] && CMD_ARGS="$CMD_ARGS --$1 $2" || true; }
_flag() { [ "${2:-}" = "true" ] && CMD_ARGS="$CMD_ARGS --$1" || true; }

build_cmd() {
    CMD_ARGS=""
    case "$MODE" in
        train)
            CMD_ARGS="--task $TASK"
            [ "$VISER_PORT" != "0" ] && { _opt viser-port "$VISER_PORT"; _opt viser-fps "$VISER_FPS"; _opt viser-env-idx "$VISER_ENV_IDX"; _flag enable-control "$ENABLE_CONTROL"; }
            _flag headless "$HEADLESS"
            _opt num-envs "$NUM_ENVS"
            _opt max-iterations "$MAX_ITERATIONS"
            _opt seed "$SEED"
            _opt save-interval "$SAVE_INTERVAL"
            _opt motion-file "$MOTION_FILE"
            _opt log-root "$LOG_ROOT"
            _flag resume "$RESUME"
            _opt checkpoint "$RESUME_CHECKPOINT"
            _opt device "$TRAIN_DEVICE"
            _flag use-wandb "$USE_WANDB"
            ;;
        sim)
            CMD_ARGS="--task $TASK"
            _opt viser-port "$VISER_PORT"
            _opt viser-env-idx "$VISER_ENV_IDX"
            _flag headless "$HEADLESS"
            _opt num-envs "$NUM_ENVS_SIM"
            _opt checkpoint "$CHECKPOINT"
            _opt policy "$POLICY"
            _opt device "$SIM_DEVICE"
            # tyro 的 inject_commands: bool=True, 配套 --no-inject-commands
            [ "${INJECT_COMMANDS:-true}" = "false" ] && CMD_ARGS="$CMD_ARGS --no-inject-commands"
            _opt command-name "$COMMAND_NAME"
            # DDS 命令注入源 (unitree_remote_ctrl 集成)
            _opt command-source "$COMMAND_SOURCE"
            _opt dds-domain "$DDS_DOMAIN"
            _opt dds-interface "$DDS_INTERFACE"
            _opt robot-key "$ROBOT_KEY"
            _opt dds-timeout "$DDS_TIMEOUT"
            _opt max-steps "$MAX_STEPS"
            ;;
        list)   CMD_ARGS="" ;;
        test)   CMD_ARGS="" ;;
        doctor) CMD_ARGS="" ;;
    esac
}

run_cmd() {
    build_cmd
    case "$MODE" in
        train)  log_info "启动训练: unitree-viser cli train-args $CMD_ARGS" ;;
        sim)    log_info "启动仿真: unitree-viser cli sim-args $CMD_ARGS" ;;
        list)   log_info "列出可用任务" ;;
        test)   log_info "运行 smoke test" ;;
        doctor) log_info "环境健康检查" ;;
    esac

    # 自动检测 GPU/CPU 并设置运行时环境变量 (每次启动都跑)
    setup_runtime_env

    log_info "Python: $VENV_LABEL"
    log_info "PYTHONPATH: $PYTHONPATH"
    echo ""

    case "$MODE" in
        train)
            exec "$PYTHON_BIN" -m unitree_viser.cli train-args $CMD_ARGS
            ;;
        sim)
            exec "$PYTHON_BIN" -m unitree_viser.cli sim-args $CMD_ARGS
            ;;
        list)
            # 使用统一的 scan_tasks (注册表 → 离线扫描 → 硬编码)
            scan_tasks
            sort_tasks_by_category
            if [ ${#TASK_ARRAY[@]} -eq 0 ]; then
                log_error "未发现任何任务"
            else
                echo ""
                log_banner "═══ 可用任务 (共 ${#TASK_ARRAY[@]} 个) ═══"
                local num=0
                local current_cat=""
                for t in "${TASK_ARRAY[@]}"; do
                    local cat=$(get_task_category "$t")
                    if [ "$cat" != "$current_cat" ]; then
                        current_cat="$cat"
                        echo ""
                        case "$cat" in
                            velocity) echo -e "  ${BOLD}${CYAN}▼ 速度控制 (velocity)${NC} — 基座速度命令 (vx/vy/wz)" ;;
                            tracking) echo -e "  ${BOLD}${CYAN}▼ 运动跟踪 (tracking)${NC} — 参考 motion 复现" ;;
                        esac
                    fi
                    num=$((num+1))
                    # 编号 + 任务ID (固定列宽) + 中文标签
                    # 注意: printf %-Ns 按字节算,中文会偏,这里用纯 ASCII ID 算宽度 OK
                    printf "  %2d. ${BOLD}%-50s${NC}  ${CYAN}%s${NC}\n" "$num" "$t" "$(get_task_label "$t")"
                done
                echo ""
                log_info "tracking 任务需提供 --motion-file (e.g. dance.npz)"
                log_info "选择任务: 跑 './start.sh' 走交互式菜单 (2级选择)"
            fi
            ;;
        test)
            exec "$PYTHON_BIN" tests/test_smoke.py
            ;;
        doctor)
            echo ""
            log_banner "═══ unitree_rl_mjlab_viser ═══"
            "$PYTHON_BIN" -c "import unitree_viser; print(f'  version: {unitree_viser.__version__}')" 2>/dev/null || echo "  version: N/A"
            echo "  Python:  $($PYTHON_BIN --version 2>&1)"
            echo "  venv:    $VENV_LABEL"
            echo "  sibling: $SIBLING_RL_DIR"
            [ -n "$SIBLING_GR00T_DIR" ] && echo "  gr00t:   $SIBLING_GR00T_DIR"
            echo ""
            log_banner "═══ 运行时环境 ═══"
            echo "  MUJOCO_GL:           ${MUJOCO_GL:-<未设置>}"
            echo "  CUDA_VISIBLE_DEVICES: '${CUDA_VISIBLE_DEVICES:-<未设置>}'"
            if command -v nvidia-smi &>/dev/null; then
                local gpu_info
                gpu_info=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)
                if [ -n "$gpu_info" ]; then
                    echo "  nvidia-smi:         $gpu_info"
                fi
            fi
            echo ""
            log_banner "═══ 关键依赖 ═══"
            for pkg in viser tyro mjlab torch mujoco warp rsl_rl_lib mujoco_warp; do
                case "$pkg" in
                    rsl_rl_lib) import_pkg="rsl_rl"; metadata_pkg="rsl-rl-lib" ;;
                    warp)       import_pkg="warp"; metadata_pkg="warp-lang" ;;
                    mjlab)      import_pkg="mjlab"; metadata_pkg="mjlab" ;;
                    *)          import_pkg="$pkg"; metadata_pkg="$pkg" ;;
                esac
                ver=$("$PYTHON_BIN" -c "
import importlib, importlib.metadata as md
try:
    print(md.version('$metadata_pkg'))
    raise SystemExit(0)
except Exception:
    pass
try:
    m = importlib.import_module('$import_pkg')
    print(getattr(m, '__version__', 'unknown'))
except Exception as e:
    print(f'✗ {type(e).__name__}: {e}')
" 2>/dev/null)
                printf "  %-15s %s\n" "$pkg:" "$ver"
            done
            echo ""
            log_info "环境健康检查完成"
            ;;
    esac
}

# ══════════════════════════════════════════════════════════════════════════════
# 交互式模式配置
# ══════════════════════════════════════════════════════════════════════════════
select_mode() {
    local idx=$(prompt_select "请选择启动模式:" \
        "train  — 训练 (含可选 Viser 浏览器可视化)" \
        "sim    — 仿真 (浏览器驱动虚拟机器人)" \
        "list   — 列出所有可用任务 ID" \
        "test   — 运行 smoke test" \
        "doctor — 环境健康检查")
    case $idx in
        0) MODE="train" ;;
        1) MODE="sim" ;;
        2) MODE="list" ;;
        3) MODE="test" ;;
        4) MODE="doctor" ;;
    esac
    log_info "已选择: $MODE"
}

# ── 任务扫描 ──────────────────────────────────────────────────────────────
# 优先级:
#   1. mjlab Python 注册表 (--list-tasks)    ← 完整但需要 warp+torch
#   2. 离线 grep task_id="..." from __init__.py  ← 离线/快速
#   3. 硬编码兜底列表
# 输出: 写入全局 TASK_ARRAY
scan_tasks() {
    TASK_ARRAY=()
    local tasks=""

    # 1. Python 注册表
    if tasks=$("$PYTHON_BIN" -c "
import sys
sys.path.insert(0, '$SIBLING_RL_DIR/src')
import mjlab.tasks  # noqa
import src.tasks  # noqa
from mjlab.tasks.registry import list_tasks
for t in sorted(list_tasks()): print(t)
" 2>/dev/null) && [ -n "$tasks" ]; then
        log_info "✓ 通过 Python 注册表加载 (${tasks//$'\n'/ | }) "
        while IFS= read -r t; do TASK_ARRAY+=("$t"); done <<< "$tasks"
        return
    fi

    # 2. 离线 grep __init__.py 中的 task_id
    if [ -d "$SIBLING_RL_DIR/src/tasks" ]; then
        tasks=$(grep -rhE '^\s*task_id="Unitree-[^"]+"' "$SIBLING_RL_DIR/src/tasks" 2>/dev/null \
                | sed -E 's/.*task_id="(Unitree-[^"]+)".*/\1/' \
                | sort -u)
        if [ -n "$tasks" ]; then
            log_info "✓ 离线扫描到 ${tasks//$'\n'/ | } "
            while IFS= read -r t; do TASK_ARRAY+=("$t"); done <<< "$tasks"
            return
        fi
    fi

    # 3. 硬编码兜底
    log_warn "使用内置任务列表 (兄弟项目不可访问)"
    TASK_ARRAY=(
        "Unitree-Go2-Flat"      "Unitree-Go2-Rough"
        "Unitree-G1-Flat"       "Unitree-G1-Rough"
        "Unitree-G1-23Dof-Flat" "Unitree-G1-23Dof-Rough"
        "Unitree-G1-Tracking"   "Unitree-G1-Tracking-No-State-Estimation"
        "Unitree-G1-23Dof-Tracking" "Unitree-G1-23Dof-Tracking-No-State-Estimation"
        "Unitree-H1_2-Flat"     "Unitree-H1_2-Rough"
        "Unitree-H2-Flat"       "Unitree-H2-Rough"
        "Unitree-A2-Flat"       "Unitree-A2-Rough"
        "Unitree-As2-Flat"      "Unitree-As2-Rough"
        "Unitree-R1-Flat"       "Unitree-R1-Rough"
    )
}

# ── 任务元数据 (中文标签 + 分类) ───────────────────────────────────────────
# 任务分类: velocity (速度控制) / tracking (运动跟踪)
get_task_category() {
    case "$1" in
        *-Tracking*) echo "tracking" ;;
        *)           echo "velocity" ;;
    esac
}

# 分类中文名
get_category_label() {
    case "$1" in
        velocity) echo "速度控制" ;;
        tracking) echo "运动跟踪" ;;
        *)        echo "$1" ;;
    esac
}

# 任务 ID → 中文标签 (例如 "Go2 四足 · 平坦地形")
get_task_label() {
    local id="$1"
    local rest="${id#Unitree-}"
    local terrain=""
    case "$rest" in
        *-Tracking-No-State-Estimation) terrain="运动跟踪(无状态估计)"; rest="${rest%-Tracking-No-State-Estimation}" ;;
        *-Tracking)                     terrain="运动跟踪";              rest="${rest%-Tracking}" ;;
        *-Flat)                          terrain="平坦地形";              rest="${rest%-Flat}" ;;
        *-Rough)                         terrain="崎岖地形";              rest="${rest%-Rough}" ;;
    esac
    local robot_cn
    case "$rest" in
        A2)        robot_cn="A2 四足" ;;
        As2)       robot_cn="As2 四足" ;;
        G1)        robot_cn="G1 人形(29Dof)" ;;
        G1-23Dof)  robot_cn="G1 人形(23Dof)" ;;
        Go2)       robot_cn="Go2 四足" ;;
        Go2W)      robot_cn="Go2W 轮足" ;;
        H1)        robot_cn="H1 人形" ;;
        H1_2)      robot_cn="H1-2 人形" ;;
        H2)        robot_cn="H2 人形" ;;
        R1)        robot_cn="R1 人形" ;;
        *)         robot_cn="$rest" ;;
    esac
    if [ -z "$terrain" ]; then
        echo "$robot_cn"
    else
        echo "$robot_cn · $terrain"
    fi
}

# 把 TASK_ARRAY 排序: velocity 在前, tracking 在后
sort_tasks_by_category() {
    local velocity_list=() tracking_list=()
    for t in "${TASK_ARRAY[@]}"; do
        case "$(get_task_category "$t")" in
            velocity) velocity_list+=("$t") ;;
            tracking) tracking_list+=("$t") ;;
        esac
    done
    TASK_ARRAY=("${velocity_list[@]}" "${tracking_list[@]}")
}

# 二级任务菜单: 先选类别, 再选具体任务 (附中文说明)
select_task() {
    log_info "正在加载任务列表..."
    scan_tasks
    if [ ${#TASK_ARRAY[@]} -eq 0 ]; then
        log_error "未发现任何任务"; exit 1
    fi
    sort_tasks_by_category
    log_info "共 ${#TASK_ARRAY[@]} 个任务 (速度 $(( $(printf '%s\n' "${TASK_ARRAY[@]}" | grep -vc -- '-Tracking') )) 个 + 跟踪 4 个)"

    # 验证 --task 指定
    if [ -n "$TASK" ]; then
        if printf '%s\n' "${TASK_ARRAY[@]}" | grep -qx "$TASK"; then
            log_info "使用指定任务: $TASK ($(get_task_label "$TASK"))"
            return
        else
            log_warn "任务 '$TASK' 不在列表, 将让重新选择"
            TASK=""
        fi
    fi

    # 一级: 选类别
    local num_velocity=0 num_tracking=0
    for t in "${TASK_ARRAY[@]}"; do
        case "$(get_task_category "$t")" in
            velocity) num_velocity=$((num_velocity+1)) ;;
            tracking) num_tracking=$((num_tracking+1)) ;;
        esac
    done

    local cat_opts=()
    [ "$num_velocity" -gt 0 ] && cat_opts+=("velocity — 速度控制 / 基座 vx·vy·wz 命令 (${num_velocity} 个)")
    [ "$num_tracking" -gt 0 ] && cat_opts+=("tracking — 运动跟踪 / 参考 motion 复现 (${num_tracking} 个)")
    cat_opts+=("all      — 列出全部 ${#TASK_ARRAY[@]} 个任务 (不分类别)")
    cat_opts+=("custom   — 手动输入任务 ID...")

    echo ""
    local cat_idx=$(prompt_select "请选择任务类别:" "${cat_opts[@]}")
    local total_cats=${#cat_opts[@]}
    local all_idx=$((total_cats - 2))
    local custom_idx=$((total_cats - 1))

    case $cat_idx in
        "$all_idx")
            _select_task_from_list "全部" "${TASK_ARRAY[@]}"
            ;;
        "$custom_idx")
            TASK=$(prompt_input "请输入任务 ID (e.g. Unitree-G1-Flat)" "" true)
            ;;
        *)
            local cat="${cat_opts[$cat_idx]%% *}"
            local cat_label=$(get_category_label "$cat")
            local filtered=()
            for t in "${TASK_ARRAY[@]}"; do
                [ "$(get_task_category "$t")" = "$cat" ] && filtered+=("$t")
            done
            _select_task_from_list "$cat_label" "${filtered[@]}"
            ;;
    esac
    log_info "已选择任务: $TASK ($(get_task_label "$TASK"))"
}

# 内部: 从指定列表里选一个 (附中文标签)
_select_task_from_list() {
    local title="$1"; shift
    local tasks=("$@")
    echo ""
    local opts=()
    for t in "${tasks[@]}"; do
        opts+=("$t  —  $(get_task_label "$t")")
    done
    local idx=$(prompt_select "请选择 $title 任务 (共 ${#tasks[@]} 个):" "${opts[@]}")
    TASK="${tasks[$idx]}"
}

# ── 扫描已有训练模型 (用于恢复训练) ──────────────────────────────────────
SCAN_MODELS=()
SCAN_MODEL_PATHS=()
scan_train_models() {
    SCAN_MODELS=(); SCAN_MODEL_PATHS=()
    local log_base="logs/viser"
    [ -d "$log_base" ] || return 0
    # 递归搜索所有 model_*.pt, 按 run_dir (父目录) 分组找最佳模型
    local best_model="" best_iter=-1 best_run_dir=""
    while IFS= read -r f; do
        local fname; fname=$(basename "$f")
        if [[ "$fname" =~ ^model_([0-9]+) ]]; then
            local iter="${BASH_REMATCH[1]}"
            local run_dir; run_dir=$(dirname "$f")
            if [ "$iter" -gt "$best_iter" ]; then
                best_iter="$iter"; best_model="$f"; best_run_dir="$run_dir"
            fi
        fi
    done < <(find "$log_base" -name "model_*.pt" 2>/dev/null | sort)
    if [ -n "$best_model" ]; then
        SCAN_MODELS+=("$(basename "$best_run_dir") — $(basename "$best_model") (iter $best_iter)")
        SCAN_MODEL_PATHS+=("$best_model")
    fi
}

config_train() {
    # ── 1. 选择机器人类型 ──
    echo ""; log_banner "── 机器人选择 ──"; echo ""
    local ROBOT_OPTS=("Go2 (四足)" "G1 (人形 29Dof)" "G1-23Dof (人形 23Dof)" "H1-2 (人形)" "H1 (人形)" "H2 (人形)" "A2 (四足)" "AS2 (四足)" "R1 (人形)")
    local ROBOT_VALS=("Go2" "G1" "G1-23Dof" "H1_2" "H1" "H2" "A2" "AS2" "R1")
    local ri=$(prompt_select "选择机器人:" "${ROBOT_OPTS[@]}")
    local ROBOT_TYPE="${ROBOT_VALS[$ri]}"
    log_info "机器人: $ROBOT_TYPE"

    # ── 2. 选择地形/任务类别 ──
    echo ""; log_banner "── 任务类别选择 ──"; echo ""
    local TERRAIN_OPTS=("velocity — 速度控制 (vx/vy/wz 命令)" "tracking — 运动跟踪 (参考 motion 复现)")
    local TERRAIN_VALS=("velocity" "tracking")
    local ti=$(prompt_select "选择任务类别:" "${TERRAIN_OPTS[@]}")
    local TERRAIN_TYPE="${TERRAIN_VALS[$ti]}"
    log_info "任务类别: $TERRAIN_TYPE"

    # ── 3. 根据机器人+地形自动推导任务 ID ──
    case "$TERRAIN_TYPE" in
        velocity)
            # velocity: 选地形
            echo ""
            local terrain_opts=("Flat — 平坦地形" "Rough — 崎岖地形")
            local terrain_vals=("Flat" "Rough")
            local tti=$(prompt_select "选择地形:" "${terrain_opts[@]}")
            TASK="Unitree-${ROBOT_TYPE}-${terrain_vals[$tti]}"
            ;;
        tracking)
            # tracking: 选是否带状态估计
            echo ""
            local track_opts=("Tracking — 标准模式" "Tracking-No-State-Estimation — 无状态估计")
            local track_vals=("Tracking" "Tracking-No-State-Estimation")
            local tti=$(prompt_select "选择跟踪模式:" "${track_opts[@]}")
            TASK="Unitree-${ROBOT_TYPE}-${track_vals[$tti]}"
            ;;
    esac
    log_info "任务: $TASK ($(get_task_label "$TASK"))"

    # ── 4. 恢复训练选项 ──
    echo ""
    RESUME="false"; RESUME_CHECKPOINT=""
    scan_train_models
    if [ ${#SCAN_MODELS[@]} -gt 0 ]; then
        local opts=("${SCAN_MODELS[@]}" "从头开始 (不加载)")
        local mi=$(prompt_select "发现已有训练记录:" "${opts[@]}")
        if [ "$mi" -lt "${#SCAN_MODELS[@]}" ]; then
            RESUME="true"
            RESUME_CHECKPOINT="${SCAN_MODEL_PATHS[$mi]}"
            log_info "恢复训练: $RESUME_CHECKPOINT"
        else
            log_info "从头开始"
        fi
    else
        log_info "无训练记录, 从头开始"
    fi

    # ── 5. 训练超参 (直接输入) ──
    echo ""; log_banner "── 训练超参配置 ──"; echo ""
    MAX_ITERATIONS=$(prompt_input "最大迭代次数" "10000")
    NUM_ENVS=$(prompt_input "并行环境数" "1024")
    log_info "环境数: $NUM_ENVS, 迭代: $MAX_ITERATIONS"

    # ── 6. Viser 选项 (默认启用, 只问端口号) ──
    echo ""
    VISER_PORT=$(prompt_input "Viser 端口" "20006")
    ENABLE_CONTROL="true"

    # ── 7. wandb 选项 ──
    echo ""
    if [ "$(prompt_yn "启用 wandb 日志记录?" "n")" = "true" ]; then
        USE_WANDB="true"
    else
        USE_WANDB="false"
    fi

    # ── 8. 设备选择 (默认 GPU, 无 GPU 则跳过) ──
    echo ""
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        local gpu_name
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
        log_info "检测到 NVIDIA GPU: $gpu_name"
        if [ "$(prompt_yn "训练使用 GPU (cuda:0)?" "y")" = "true" ]; then
            TRAIN_DEVICE="cuda:0"
        else
            TRAIN_DEVICE="cpu"
        fi
    else
        TRAIN_DEVICE="cpu"
        log_info "未检测到 NVIDIA GPU, 训练使用 CPU (跳过选择)"
    fi

    # ── 9. tracking 任务需要 motion file ──
    if [[ "$TASK" == *"-Tracking"* ]]; then
        echo ""
        log_warn "Tracking 任务需要 motion 文件 (.npz)"
        local default_motion="$SIBLING_RL_DIR/src/assets/motions/g1/dance1_subject2.npz"
        if [ -f "$default_motion" ]; then
            log_info "检测到默认: $default_motion"
            MOTION_FILE=$(prompt_input "Motion 文件" "$default_motion")
        else
            MOTION_FILE=$(prompt_input "Motion 文件 (.npz)" "" true)
        fi
    fi
}

config_sim() {
    select_task

    # checkpoint
    echo ""
    local ckpt_default=""
    if [ -d "logs" ]; then
        ckpt_default=$(find logs -name "model_*.pt" 2>/dev/null | sort -V | tail -1 || echo "")
    fi
    if [ -n "$ckpt_default" ]; then
        log_info "找到最新训练产物: $ckpt_default"
        if [ "$(prompt_yn "使用此模型?" "y")" = "true" ]; then
            CHECKPOINT="$ckpt_default"
        else
            CHECKPOINT=$(prompt_input "Checkpoint 路径 (留空用 zero policy)" "")
        fi
    else
        CHECKPOINT=$(prompt_input "Checkpoint 路径 (留空用 zero policy)" "")
    fi

    if [ -z "$CHECKPOINT" ]; then
        local idx=$(prompt_select "无 checkpoint, 使用什么策略?" "zero (零动作, 站立)" "random (随机动作)")
        case $idx in 0) POLICY="zero" ;; 1) POLICY="random" ;; esac
    fi

    # 设备选择 (默认 CPU, 不占 GPU 显存; 无 GPU 则跳过)
    echo ""
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        local gpu_name
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
        log_info "检测到 NVIDIA GPU: $gpu_name"
        if [ "$(prompt_yn "仿真使用 CPU (推荐, 不占 GPU 显存)?" "y")" != "true" ]; then
            SIM_DEVICE="cuda:0"
        else
            SIM_DEVICE="cpu"
        fi
    else
        SIM_DEVICE="cpu"
        log_info "未检测到 NVIDIA GPU, 仿真使用 CPU (跳过选择)"
    fi

    if [ "$(prompt_yn "启用 Viser 浏览器可视化?" "y")" = "true" ]; then
        VISER_PORT=$(prompt_input "Viser 端口" "20006")
        if [ "$(prompt_yn "启用命令注入 (vx/vy/wz 滑块)?" "y")" = "true" ]; then
            INJECT_COMMANDS="true"
        else
            INJECT_COMMANDS="false"
        fi
    else
        HEADLESS="true"
    fi

    MAX_STEPS=$(prompt_input "最大步数 (留空=无限)" "")
}

config_deps() {
    if [ "$(prompt_yn "检查依赖?" "y")" = "true" ]; then
        DEPS_INSTALL="false"
    fi
    if [ "$(prompt_yn "自动安装缺失依赖?" "n")" = "true" ]; then
        DEPS_INSTALL="true"
        if [ "$(prompt_yn "装完整重型依赖 (torch + mjlab + mjwarp, ~3GB)?" "n")" = "true" ]; then
            DEPS_FULL="true"
        fi
    fi
}

# ══════════════════════════════════════════════════════════════════════════════
# 配置确认
# ══════════════════════════════════════════════════════════════════════════════
confirm_and_start() {
    echo ""
    log_banner "══════════════════ 配置确认 ══════════════════"
    echo ""
    case "$MODE" in
        train)
            echo -e "  模式:       ${BOLD}train${NC}"
            echo -e "  任务:       ${BOLD}${TASK}${NC}"
            echo -e "  环境数:     ${BOLD}${NUM_ENVS:-自动}${NC}"
            echo -e "  迭代:       ${BOLD}${MAX_ITERATIONS:-自动}${NC}"
            echo -e "  Viser:      ${BOLD}$([ "$HEADLESS" = "true" ] && echo "禁用" || echo "http://localhost:$VISER_PORT")${NC}"
            if [ "$ENABLE_CONTROL" = "true" ]; then
                echo -e "  训练控制:   ${BOLD}启用 (暂停/单步/速度)${NC}"
            fi
            if [ -n "$MOTION_FILE" ]; then
                echo -e "  Motion:     ${BOLD}${MOTION_FILE}${NC}"
            fi
            if [ "$RESUME" = "true" ]; then
                echo -e "  恢复训练:   ${BOLD}是${NC} ($RESUME_CHECKPOINT)"
            fi
            echo -e "  设备:       ${BOLD}${TRAIN_DEVICE}${NC}"
            echo -e "  wandb:      ${BOLD}$([ "$USE_WANDB" = "true" ] && echo "启用" || echo "禁用 (tensorboard)")${NC}"
            ;;
        sim)
            echo -e "  模式:       ${BOLD}sim${NC}"
            echo -e "  任务:       ${BOLD}${TASK}${NC}"
            echo -e "  策略:       ${BOLD}${CHECKPOINT:-$POLICY policy}${NC}"
            echo -e "  环境数:     ${BOLD}${NUM_ENVS_SIM}${NC}"
            echo -e "  设备:       ${BOLD}${SIM_DEVICE}${NC}"
            echo -e "  Viser:      ${BOLD}$([ "$HEADLESS" = "true" ] && echo "禁用" || echo "http://localhost:$VISER_PORT")${NC}"
            if [ "$INJECT_COMMANDS" = "true" ] && [ "$HEADLESS" != "true" ]; then
                echo -e "  命令注入:   ${BOLD}启用${NC} (${COMMAND_NAME})"
            fi
            # 显示 DDS 配置
            if [ "$COMMAND_SOURCE" != "gui" ] && [ "$COMMAND_SOURCE" != "" ]; then
                echo -e "  DDS 注入:   ${BOLD}启用${NC} (${COMMAND_SOURCE}, key=${ROBOT_KEY}, domain=${DDS_DOMAIN})"
            fi
            if [ -n "$MAX_STEPS" ]; then
                echo -e "  最大步数:   ${BOLD}${MAX_STEPS}${NC}"
            fi
            ;;
        list)    echo -e "  模式:       ${BOLD}list${NC}" ;;
        test)    echo -e "  模式:       ${BOLD}test${NC}" ;;
        doctor)  echo -e "  模式:       ${BOLD}doctor${NC}" ;;
    esac
    echo -e "  Python:     ${BOLD}${VENV_LABEL}${NC}"
    echo ""
}

# ══════════════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════════════
# deps 模式用于安装环境, 此时可能还没有 python → 跳过 detect_python
# 其他模式 (train/sim/list/test/version) 必须有 python
# build_pythonpath 不依赖 python, 任何模式都可调用 (只设 PYTHONPATH)
build_pythonpath

case "${1:-}" in
    deps|-h|--help|help)
        # 给 stub 避免 run_cmd 等函数引用空变量
        : "${PYTHON_BIN:=python3}"
        : "${PIP_BIN:=pip3}"
        : "${VENV_LABEL:=系统 Python (无 venv)}"
        : "${VENV_DIR:=$PROJECT_DIR/.venv}"
        ;;
    *)
        detect_python
        ;;
esac

if [ $# -eq 0 ]; then
    # ── 交互式 ──
    log_banner "═══════════════════════════════════════════════════════════════"
    log_banner "         unitree_rl_mjlab_viser — 启动入口"
    log_banner "═══════════════════════════════════════════════════════════════"
    echo ""
    log_info "Python:    $VENV_LABEL"
    log_info "兄弟项目:  $SIBLING_RL_DIR"
    echo ""
    check_core_deps || log_warn "依赖不全, 可用: $0 doctor"
    echo ""
    select_mode
    case "$MODE" in
        train)   config_train ;;
        sim)     config_sim ;;
        doctor)  ;;
    esac
    confirm_and_start
    run_cmd
else
    # ── 非交互式 ──
    case "${1:-}" in
        train|sim|list|test|doctor) MODE="$1"; shift ;;
        -h|--help|help) show_usage; exit 0 ;;
        *) log_error "未知模式: $1"; show_usage; exit 1 ;;
    esac
    while [ $# -gt 0 ]; do
        case "$1" in
            # 通用
            --task)             TASK="$2"; shift 2 ;;
            --viser-port)       VISER_PORT="$2"; shift 2 ;;
            --viser-fps)        VISER_FPS="$2"; shift 2 ;;
            --viser-env-idx)    VISER_ENV_IDX="$2"; shift 2 ;;
            --headless)         HEADLESS="true"; shift 1 ;;
            # num-envs 根据模式映射到不同变量
            --num-envs)
                if [ "$MODE" = "sim" ]; then NUM_ENVS_SIM="$2"; else NUM_ENVS="$2"; fi
                shift 2 ;;
            # train
            --max-iterations)   MAX_ITERATIONS="$2"; shift 2 ;;
            --seed)             SEED="$2"; shift 2 ;;
            --save-interval)    SAVE_INTERVAL="$2"; shift 2 ;;
            --motion-file)      MOTION_FILE="$2"; shift 2 ;;
            --enable-control)   ENABLE_CONTROL="true"; shift 1 ;;
            --resume)           RESUME="true"; shift 1 ;;
            --checkpoint)       CHECKPOINT="$2"; RESUME_CHECKPOINT="$2"; shift 2 ;;
            --log-root)         LOG_ROOT="$2"; shift 2 ;;
            --use-wandb)        USE_WANDB="true"; shift 1 ;;
            # sim
            --policy)           POLICY="$2"; shift 2 ;;
            --no-inject-commands) INJECT_COMMANDS="false"; shift 1 ;;
            --inject-commands)  INJECT_COMMANDS="true"; shift 1 ;;
            --command-name)     COMMAND_NAME="$2"; shift 2 ;;
            # DDS 命令注入
            --command-source)   COMMAND_SOURCE="$2"; shift 2 ;;
            --dds-domain)       DDS_DOMAIN="$2"; shift 2 ;;
            --dds-interface)    DDS_INTERFACE="$2"; shift 2 ;;
            --robot-key)        ROBOT_KEY="$2"; shift 2 ;;
            --dds-timeout)      DDS_TIMEOUT="$2"; shift 2 ;;
            --max-steps)        MAX_STEPS="$2"; shift 2 ;;
            # device (train/sim 通用)
            --device)
                if [ "$MODE" = "sim" ]; then SIM_DEVICE="$2"; else TRAIN_DEVICE="$2"; fi
                shift 2 ;;
            # doctor (无额外参数)
            -h|--help|help)     show_usage; exit 0 ;;
            # help
            -h|--help|help)     show_usage; exit 0 ;;
            *)
                # 非交互 train/sim: 第一个未知参数视为任务 ID
                if [ -z "$TASK" ] && { [ "$MODE" = "train" ] || [ "$MODE" = "sim" ]; }; then
                    TASK="$1"; shift 1
                else
                    log_warn "忽略未知参数: $1"; shift 1
                fi
                ;;
        esac
    done
    # train 模式下若 --checkpoint 同时启用 resume
    if [ "$MODE" = "train" ] && [ -n "$CHECKPOINT" ] && [ "$RESUME" != "true" ]; then
        RESUME="true"
        RESUME_CHECKPOINT="$CHECKPOINT"
    fi
    if [ "$MODE" = "train" ] || [ "$MODE" = "sim" ]; then
        if [ -z "$TASK" ]; then
            log_error "train/sim 模式需要任务 ID: $0 $MODE <TaskID> [args]"
            log_info "可用任务列表: $0 list"
            show_usage; exit 1
        fi
        check_task_registry
        # 校验 task ID 是否在已知列表里 (不阻断, 仅警告)
        scan_tasks
        if [ ${#TASK_ARRAY[@]} -gt 0 ] && ! printf '%s\n' "${TASK_ARRAY[@]}" | grep -qx "$TASK"; then
            log_warn "任务 '$TASK' 不在已知列表 (共 ${#TASK_ARRAY[@]} 个)"
            log_warn "如不确定, 跑 '$0 list' 查看或去掉 <TaskID> 参数走交互菜单"
        fi
    fi
    confirm_and_start
    run_cmd
fi
