"""end_to_end_test.py - DDS 端到端测试 (Publisher + Subscriber).

测试流程:
1. 启动一个 viser sim 的 DDS 订阅端 (本仓库的 DdsCommandInjector)
2. 启动一个 DDS Publisher (用 unitree_sdk2py 直接发消息)
3. Publisher 发 3 组不同 ly/lx/rx 值
4. 验证订阅端 _pending 收到正确换算结果
5. inject 写入 vel_command_b

用法 (开两个终端):

    # 终端 1: 启动 viser sim DDS 订阅端
    ./start.sh sim Unitree-Go2-Flat --headless --command-source dds \
        --robot-key test_e2e --num-envs 1 --max-steps 30

    # 终端 2: 运行此脚本发 DDS 消息
    .venv/bin/python tests/end_to_end_test.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# 必要的 sys.path 修复 (与 cli.py 一致)
sys.path[:] = [p for p in sys.path if p != ""]
PROJECT_ROOT = Path(__file__).resolve().parent.parent
UNITREE_RL_MJLAB_SRC = PROJECT_ROOT.parent / "unitree_rl_mjlab" / "src"
if UNITREE_RL_MJLAB_SRC.is_dir():
    sys.path.insert(0, str(UNITREE_RL_MJLAB_SRC))
sys.modules.pop("src", None)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher  # noqa: E402
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_  # noqa: E402
from unitree_sdk2py.idl.default import unitree_go_msg_dds__WirelessController_  # noqa: E402


TOPIC = "rt/test_e2e/wirelesscontroller"


def main() -> int:
    print("=" * 60)
    print("DDS End-to-End Test")
    print(f"Topic: {TOPIC}")
    print("=" * 60)

    # 1. 初始化 DDS (与订阅端相同的 domain_id 和 interface)
    # 注: ChannelFactoryInitialize 用位置参数 (domain, interface)
    ChannelFactoryInitialize(0, "lo")
    print("[OK] DDS factory initialized (domain=0, interface=lo)")

    # 2. 创建 Publisher
    publisher = ChannelPublisher(TOPIC, WirelessController_)
    publisher.Init()
    print(f"[OK] Publisher initialized for topic: {TOPIC}")

    # 3. 测试用例: (lx, ly, rx, ry, keys) → 期望 (vx, vy, wz)
    # 换算: vx=ly, vy=-lx, yaw=-rx
    test_cases = [
        # (lx, ly, rx, keys, expected_vx, expected_vy, expected_wz, label)
        (0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, "zero"),
        (0.0, 1.0, 0.0, 0, 1.0, 0.0, 0.0, "forward 1 m/s"),
        (0.0, -1.0, 0.0, 0, -1.0, 0.0, 0.0, "backward 1 m/s"),
        (0.5, 0.0, 0.0, 0, 0.0, -0.5, 0.0, "right 0.5 m/s"),
        (-0.5, 0.0, 0.0, 0, 0.0, 0.5, 0.0, "left 0.5 m/s"),
        (0.0, 0.0, 0.5, 0, 0.0, 0.0, -0.5, "turn right 0.5 rad/s"),
        (0.0, 0.0, -0.5, 0, 0.0, 0.0, 0.5, "turn left 0.5 rad/s"),
        (0.3, 0.8, -0.5, 0, 0.8, -0.3, 0.5, "combined forward+right+turn"),
    ]

    # 4. 发送每条消息, 间隔 1.5s (超过订阅端 0.5s 超时)
    for lx, ly, rx, keys, evx, evy, ewz, label in test_cases:
        msg = unitree_go_msg_dds__WirelessController_()
        msg.lx = lx
        msg.ly = ly
        msg.rx = rx
        msg.ry = 0.0
        msg.keys = keys
        publisher.Write(msg)
        print(
            f"[SEND] {label}: lx={lx} ly={ly} rx={rx} → "
            f"期望 (vx={evx}, vy={evy}, wz={ewz})"
        )
        time.sleep(1.5)

    # 5. 发送 zero 消息 (验证归零路径)
    msg = unitree_go_msg_dds__WirelessController_()
    publisher.Write(msg)
    print("[SEND] zero (归零测试)")

    print()
    print("=" * 60)
    print("发送完成!")
    print("请检查订阅端 (viser sim) 的 vel_command_b 是否按预期变化")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())