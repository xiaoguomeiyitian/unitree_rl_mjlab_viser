#!/usr/bin/env bash
# ============================================================================
# train_viser.sh — 一键启动训练 + 打开 Viser 浏览器
#
# 用法:
#   ./scripts/train_viser.sh                              # 默认任务 + 浏览器
#   ./scripts/train_viser.sh Unitree-G1-Flat              # 指定任务
#   ./scripts/train_viser.sh Unitree-Go2-Flat --num-envs 1024
#   ./scripts/train_viser.sh Unitree-G1-Flat --headless   # 不开浏览器
#   ./scripts/train_viser.sh Unitree-G1-Flat --enable-control  # 带训练控制
#
# 前置条件:
#   1. unitree_rl_mjlab 已安装 (pip install -e ../unitree_rl_mjlab)
#   2. mjlab 1.2.0 + viser + torch 已安装
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 颜色
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[1;34m'; NC='\033[0m'

# 默认任务
TASK="${1:-Unitree-Go2-Flat}"
shift || true

echo -e "${BLUE}=== unitree-viser train ===${NC}"
echo -e "  Task: ${GREEN}${TASK}${NC}"
echo -e "  剩余参数: ${YELLOW}$*${NC}"
echo ""

cd "$PROJECT_ROOT"

# 决定 Viser 端口
PORT=20006
EXTRA_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--viser-port" ]]; then
        shift; PORT="$1"; shift
    fi
done

# 启动训练
python -m unitree_viser.cli train "$TASK" "$@"

# 提示浏览器访问 (如果有 client 端)
echo ""
echo -e "${GREEN}训练已完成. 日志查看 logs/viser/${TASK}/<时间戳>/${NC}"
