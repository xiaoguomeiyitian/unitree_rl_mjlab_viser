#!/usr/bin/env bash
# ============================================================================
# sim_viser.sh — 一键启动仿真 (从浏览器驱动虚拟机器人)
#
# 用法:
#   ./scripts/sim_viser.sh Unitree-G1-Flat                            # zero policy
#   ./scripts/sim_viser.sh Unitree-G1-Flat --policy random            # random policy
#   ./scripts/sim_viser.sh Unitree-G1-Flat --checkpoint path/to/model.pt
#   ./scripts/sim_viser.sh Unitree-G1-Flat --no-inject                 # 不开命令注入
#   ./scripts/sim_viser.sh Unitree-G1-Flat --headless --max-steps 100  # 不开浏览器
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[1;34m'; NC='\033[0m'

TASK="${1:-Unitree-G1-Flat}"
shift || true

echo -e "${BLUE}=== unitree-viser sim ===${NC}"
echo -e "  Task: ${GREEN}${TASK}${NC}"
echo -e "  剩余参数: ${YELLOW}$*${NC}"
echo ""

cd "$PROJECT_ROOT"
python -m unitree_viser.cli sim "$TASK" "$@"
