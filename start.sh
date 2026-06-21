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
#   deps    检查/补全依赖
#   version 显示项目版本和 venv 信息
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
#   ./start.sh deps --install                        # 自动装依赖
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
VISER_FPS="10"
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

# sim 专用
CHECKPOINT=""
POLICY="zero"          # zero | random
NUM_ENVS_SIM="1"
INJECT_COMMANDS="true"
COMMAND_NAME="twist"
MAX_STEPS=""

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
        VENV_LABEL="本地 venv (${PYTHON_BIN})"
    elif [ -n "$SIBLING_GR00T_DIR" ] && [ -x "$SIBLING_GR00T_DIR/.venv/bin/python" ]; then
        PYTHON_BIN="$SIBLING_GR00T_DIR/.venv/bin/python"
        PIP_BIN="$SIBLING_GR00T_DIR/.venv/bin/pip"
        VENV_LABEL="gr00t_mjlab_autodl venv (${PYTHON_BIN})"
    elif command -v python3.12 &>/dev/null; then
        PYTHON_BIN="$(command -v python3.12)"
        PIP_BIN="$(command -v pip3.12)"
        VENV_LABEL="系统 Python 3.12"
    else
        PYTHON_BIN="$(command -v python3)"
        PIP_BIN="$(command -v pip3)"
        VENV_LABEL="系统 Python ($(basename "$PYTHON_BIN"))"
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
build_pythonpath() {
    local pp="$SRC_DIR"
    [ -n "$SIBLING_RL_DIR" ] && pp="$pp:$SIBLING_RL_DIR/src"
    PYTHONPATH="$pp"
    export PYTHONPATH
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
  deps      检查/补全依赖
  version   显示版本和环境信息

通用参数:
  --task <id>              任务 ID (e.g. Unitree-G1-Flat)
  --viser-port <N>         Viser HTTP/WS 端口 (默认: 20006, 0 = 禁用)
  --viser-fps <F>          Viser 渲染 FPS (默认: 10)
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

依赖参数 (deps):
  --install                自动 pip install 缺失依赖
  --full                   装完整重型依赖 (torch + mjlab + mujoco-warp, ~3GB)

示例:
  $0 list
  $0 version
  $0 train Unitree-G1-Flat --num-envs 64 --max-iterations 100
  $0 train Unitree-G1-Flat --viser-port 0 --num-envs 2048   # 无 Viser 训练
  $0 train Unitree-G1-Tracking-No-State-Estimation --motion-file dance1.npz
  $0 sim Unitree-G1-Flat                                    # 零动作
  $0 sim Unitree-G1-Flat --checkpoint logs/.../model_500.pt
  $0 sim Unitree-G1-Flat --policy random
  $0 test
  $0 deps --install
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
            ;;
        sim)
            CMD_ARGS="--task $TASK"
            _opt viser-port "$VISER_PORT"
            _opt viser-env-idx "$VISER_ENV_IDX"
            _flag headless "$HEADLESS"
            _opt num-envs "$NUM_ENVS_SIM"
            _opt checkpoint "$CHECKPOINT"
            _opt policy "$POLICY"
            # tyro 的 inject_commands: bool=True, 配套 --no-inject-commands
            [ "${INJECT_COMMANDS:-true}" = "false" ] && CMD_ARGS="$CMD_ARGS --no-inject-commands"
            _opt command-name "$COMMAND_NAME"
            _opt max-steps "$MAX_STEPS"
            ;;
        list)   CMD_ARGS="" ;;
        test)   CMD_ARGS="" ;;
        deps)   CMD_ARGS="" ;;
        version) CMD_ARGS="" ;;
    esac
}

run_cmd() {
    build_cmd
    case "$MODE" in
        train)  log_info "启动训练: unitree-viser cli train-args $CMD_ARGS" ;;
        sim)    log_info "启动仿真: unitree-viser cli sim-args $CMD_ARGS" ;;
        list)   log_info "列出可用任务" ;;
        test)   log_info "运行 smoke test" ;;
        deps)   log_info "检查依赖" ;;
        version) log_info "显示版本信息" ;;
    esac

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
            # 优先用任务注册表 API;失败时 grep Python 文件作为备用
            if ! "$PYTHON_BIN" -c "
import sys
sys.path.insert(0, '$SIBLING_RL_DIR/src')
import mjlab.tasks
import src.tasks
from mjlab.tasks.registry import list_tasks
tasks = list_tasks()
print(f'\\n可用任务 ({len(tasks)} 个):\\n')
for i, t in enumerate(tasks, 1):
    print(f'  {i:3d}. {t}')
print()
" 2>/dev/null; then
                # 备用: 离线扫描兄弟项目,列出所有 unitree_<robot>_* 任务
                log_warn "任务注册表加载失败 (常见原因: mjlab 1.2.0 需要 Python 3.12)"
                log_warn "切换到离线扫描模式..."
                echo ""
                if [ -d "$SIBLING_RL_DIR/src/tasks" ]; then
                    # 找所有 task ID: <robot>_<task>_<terrain>[_no_state_estimation]
                    find "$SIBLING_RL_DIR/src/tasks" -name "env_cfgs.py" -path "*/config/*" \
                        | sed -E 's|.*/config/([^/]+)/env_cfgs.py|\1|' \
                        | sort -u | head -50
                else
                    log_error "找不到 $SIBLING_RL_DIR/src/tasks"
                fi
            fi
            ;;
        test)
            exec "$PYTHON_BIN" tests/test_smoke.py
            ;;
        deps)
            if [ "$DEPS_INSTALL" = "true" ]; then
                "$PIP_BIN" install viser tyro
                if [ "$DEPS_FULL" = "true" ]; then
                    "$PIP_BIN" install -e "$SIBLING_RL_DIR"
                else
                    "$PIP_BIN" install --no-deps mjlab==1.2.0
                fi
            else
                check_core_deps || log_warn "用 --install 自动补全"
            fi
            # 显示详情
            "$PIP_BIN" list 2>/dev/null | grep -iE "viser|tyro|mjlab|mujoco|warp|rsl|torch|nvidia" | head -20
            ;;
        version)
            echo ""
            log_banner "═══ unitree_rl_mjlab_viser ═══"
            "$PYTHON_BIN" -c "import unitree_viser; print(f'  version: {unitree_viser.__version__}')"
            echo "  Python:  $($PYTHON_BIN --version 2>&1)"
            echo "  venv:    $VENV_LABEL"
            echo "  sibling: $SIBLING_RL_DIR"
            [ -n "$SIBLING_GR00T_DIR" ] && echo "  gr00t:   $SIBLING_GR00T_DIR"
            echo ""
            log_banner "═══ 关键依赖 ═══"
            for pkg in viser tyro mjlab torch mujoco warp rsl_rl_lib mujoco_warp; do
                # pip 包名 vs import 名映射
                case "$pkg" in
                    rsl_rl_lib) import_pkg="rsl_rl"; metadata_pkg="rsl-rl-lib" ;;
                    warp)       import_pkg="warp"; metadata_pkg="warp-lang" ;;
                    mjlab)      import_pkg="mjlab"; metadata_pkg="mjlab" ;;
                    *)          import_pkg="$pkg"; metadata_pkg="$pkg" ;;
                esac
                ver=$("$PYTHON_BIN" -c "
import importlib, importlib.metadata as md
# 优先 metadata (不需要 import,避免 mjlab 触发 warp 依赖)
try:
    print(md.version('$metadata_pkg'))
    raise SystemExit(0)
except Exception:
    pass
# 回退到 import __version__
try:
    m = importlib.import_module('$import_pkg')
    print(getattr(m, '__version__', 'unknown'))
except Exception as e:
    print(f'✗ {type(e).__name__}: {e}')
" 2>/dev/null)
                printf "  %-15s %s\n" "$pkg:" "$ver"
            done
            echo ""
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
        "deps   — 检查/补全依赖" \
        "version — 显示版本和环境信息")
    case $idx in
        0) MODE="train" ;;
        1) MODE="sim" ;;
        2) MODE="list" ;;
        3) MODE="test" ;;
        4) MODE="deps" ;;
        5) MODE="version" ;;
    esac
    log_info "已选择: $MODE"
}

# 列出可用任务,让用户选择 (或回退到手动输入)
select_task() {
    log_info "正在加载任务列表..."
    local tasks=""
    if tasks=$("$PYTHON_BIN" -c "
import sys
sys.path.insert(0, '$SIBLING_RL_DIR/src')
import mjlab.tasks
import src.tasks
from mjlab.tasks.registry import list_tasks
for t in list_tasks(): print(t)
" 2>/dev/null) && [ -n "$tasks" ]; then
        local task_array=()
        while IFS= read -r t; do task_array+=("$t"); done <<< "$tasks"
        if [ ${#task_array[@]} -eq 0 ]; then
            log_warn "未发现任何任务,使用手动输入"
            TASK=$(prompt_input "请输入任务 ID (e.g. Unitree-G1-Flat)" "" true)
            return
        fi
        # 验证 --task 指定
        if [ -n "$TASK" ]; then
            if printf '%s\n' "${task_array[@]}" | grep -qx "$TASK"; then
                log_info "使用指定任务: $TASK"
                return
            else
                log_warn "任务 '$TASK' 不存在, 请重新选择"
            fi
        fi
        local idx=$(prompt_select "请选择任务 (共 ${#task_array[@]} 个):" "${task_array[@]}")
        TASK="${task_array[$idx]}"
        log_info "已选择任务: $TASK"
    else
        # 回退: 手动输入
        log_warn "无法加载任务注册表 (常见原因: mjlab 1.2.0 需要 Python 3.12)"
        log_info "已知任务示例:"
        echo "  • Unitree-Go2-Flat / Unitree-Go2-Rough"
        echo "  • Unitree-G1-Flat / Unitree-G1-Rough"
        echo "  • Unitree-G1-Tracking-No-State-Estimation"
        echo ""
        TASK=$(prompt_input "请手动输入任务 ID" "" true)
        log_info "已选择任务: $TASK"
    fi
}

config_train() {
    select_task

    # Viser 选项
    if [ "$(prompt_yn "启用 Viser 浏览器可视化?" "y")" = "true" ]; then
        VISER_PORT=$(prompt_input "Viser 端口" "20006")
        VISER_FPS=$(prompt_input "Viser FPS" "10")
        if [ "$(prompt_yn "启用训练控制 (暂停/单步/速度)?" "y")" = "true" ]; then
            ENABLE_CONTROL="true"
        fi
    else
        HEADLESS="true"
    fi

    # 训练超参
    echo ""
    local idx=$(prompt_select "选择配置模板:" "快速 (64 envs, 100 iters)" "小规模 (512, 5000)" "标准 (1024, 10000)" "大规模 (2048, 20000)" "自定义")
    case $idx in
        0) NUM_ENVS=64;  MAX_ITERATIONS=100  ;;
        1) NUM_ENVS=512; MAX_ITERATIONS=5000 ;;
        2) NUM_ENVS=1024; MAX_ITERATIONS=10000 ;;
        3) NUM_ENVS=2048; MAX_ITERATIONS=20000 ;;
        4)
            NUM_ENVS=$(prompt_input "环境数" "1024")
            MAX_ITERATIONS=$(prompt_input "最大迭代" "10000")
            SEED=$(prompt_input "随机种子" "42")
            SAVE_INTERVAL=$(prompt_input "保存间隔" "100")
            ;;
    esac

    # tracking 任务需要 motion file
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
            ;;
        sim)
            echo -e "  模式:       ${BOLD}sim${NC}"
            echo -e "  任务:       ${BOLD}${TASK}${NC}"
            echo -e "  策略:       ${BOLD}${CHECKPOINT:-$POLICY policy}${NC}"
            echo -e "  环境数:     ${BOLD}${NUM_ENVS_SIM}${NC}"
            echo -e "  Viser:      ${BOLD}$([ "$HEADLESS" = "true" ] && echo "禁用" || echo "http://localhost:$VISER_PORT")${NC}"
            if [ "$INJECT_COMMANDS" = "true" ] && [ "$HEADLESS" != "true" ]; then
                echo -e "  命令注入:   ${BOLD}启用${NC}"
            fi
            if [ -n "$MAX_STEPS" ]; then
                echo -e "  最大步数:   ${BOLD}${MAX_STEPS}${NC}"
            fi
            ;;
        list)    echo -e "  模式:       ${BOLD}list${NC}" ;;
        test)    echo -e "  模式:       ${BOLD}test${NC}" ;;
        deps)    echo -e "  模式:       ${BOLD}deps${NC}  install=$DEPS_INSTALL full=$DEPS_FULL" ;;
        version) echo -e "  模式:       ${BOLD}version${NC}" ;;
    esac
    echo -e "  Python:     ${BOLD}${VENV_LABEL}${NC}"
    echo ""
}

# ══════════════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════════════
detect_python
build_pythonpath

if [ $# -eq 0 ]; then
    # ── 交互式 ──
    log_banner "═══════════════════════════════════════════════════════════════"
    log_banner "         unitree_rl_mjlab_viser — 启动入口"
    log_banner "═══════════════════════════════════════════════════════════════"
    echo ""
    log_info "Python:    $VENV_LABEL"
    log_info "兄弟项目:  $SIBLING_RL_DIR"
    [ -n "$SIBLING_GR00T_DIR" ] && log_info "复用 venv: $SIBLING_GR00T_DIR"
    echo ""
    check_core_deps || log_warn "依赖不全, 可用: $0 deps --install"
    echo ""
    select_mode
    case "$MODE" in
        train)   config_train ;;
        sim)     config_sim ;;
        deps)    config_deps ;;
    esac
    confirm_and_start
    run_cmd
else
    # ── 非交互式 ──
    case "${1:-}" in
        train|sim|list|test|deps|version) MODE="$1"; shift ;;
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
            # sim
            --policy)           POLICY="$2"; shift 2 ;;
            --no-inject-commands) INJECT_COMMANDS="false"; shift 1 ;;
            --inject-commands)  INJECT_COMMANDS="true"; shift 1 ;;
            --command-name)     COMMAND_NAME="$2"; shift 2 ;;
            --max-steps)        MAX_STEPS="$2"; shift 2 ;;
            # deps
            --install)          DEPS_INSTALL="true"; shift 1 ;;
            --full)             DEPS_FULL="true"; shift 1 ;;
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
            show_usage; exit 1
        fi
        check_task_registry
    fi
    confirm_and_start
    run_cmd
fi
