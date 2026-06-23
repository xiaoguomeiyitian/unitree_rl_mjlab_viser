"""DdsCommandInjector 单元测试.

不真订阅 DDS.重点测试:
- _axis_to_vel 换算所有边界
- inject() 写入 vel_command_b (通过 mock term)
- 线程安全 (Lock 保护 _pending)
- _on_message 回调更新
- get_pending 返回副本
- stop() 设置 _running
- 无 SDK 时 mock 降级
"""

from __future__ import annotations

import os
import sys
import threading
from unittest.mock import MagicMock, patch

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


# ── _axis_to_vel 换算 ────────────────────────────────────────────────────


def test_axis_to_vel_forward():
    """ly=1 → vx=1 (前进)."""
    from unitree_viser.sim.dds_command_injection import DdsCommandInjector

    inst = DdsCommandInjector.__new__(DdsCommandInjector)
    vx, _, _ = inst._axis_to_vel(lx=0.0, ly=1.0, rx=0.0)
    assert vx == 1.0


def test_axis_to_vel_backward():
    """ly=-1 → vx=-1 (后退)."""
    from unitree_viser.sim.dds_command_injection import DdsCommandInjector

    inst = DdsCommandInjector.__new__(DdsCommandInjector)
    vx, _, _ = inst._axis_to_vel(lx=0.0, ly=-1.0, rx=0.0)
    assert vx == -1.0


def test_axis_to_vel_right_joystick():
    """lx=+0.5 → vy=-0.5 (物理左移; 注意反向)."""
    from unitree_viser.sim.dds_command_injection import DdsCommandInjector

    inst = DdsCommandInjector.__new__(DdsCommandInjector)
    _, vy, _ = inst._axis_to_vel(lx=0.5, ly=0.0, rx=0.0)
    assert vy == -0.5


def test_axis_to_vel_left_joystick():
    """lx=-0.5 → vy=+0.5 (物理右移)."""
    from unitree_viser.sim.dds_command_injection import DdsCommandInjector

    inst = DdsCommandInjector.__new__(DdsCommandInjector)
    _, vy, _ = inst._axis_to_vel(lx=-0.5, ly=0.0, rx=0.0)
    assert vy == 0.5


def test_axis_to_vel_turn_right():
    """rx=+0.5 → yaw=-0.5 (物理右转; 注意反向)."""
    from unitree_viser.sim.dds_command_injection import DdsCommandInjector

    inst = DdsCommandInjector.__new__(DdsCommandInjector)
    _, _, yaw = inst._axis_to_vel(lx=0.0, ly=0.0, rx=0.5)
    assert yaw == -0.5


def test_axis_to_vel_turn_left():
    """rx=-0.5 → yaw=+0.5 (物理左转)."""
    from unitree_viser.sim.dds_command_injection import DdsCommandInjector

    inst = DdsCommandInjector.__new__(DdsCommandInjector)
    _, _, yaw = inst._axis_to_vel(lx=0.0, ly=0.0, rx=-0.5)
    assert yaw == 0.5


def test_axis_to_vel_combined():
    """ly=0.8, lx=0.3, rx=-0.5 → vx=0.8, vy=-0.3, yaw=0.5."""
    from unitree_viser.sim.dds_command_injection import DdsCommandInjector

    inst = DdsCommandInjector.__new__(DdsCommandInjector)
    vx, vy, yaw = inst._axis_to_vel(lx=0.3, ly=0.8, rx=-0.5)
    assert vx == 0.8
    assert vy == -0.3
    assert yaw == 0.5


def test_axis_to_vel_zero():
    """全 0 → 全 0."""
    from unitree_viser.sim.dds_command_injection import DdsCommandInjector

    inst = DdsCommandInjector.__new__(DdsCommandInjector)
    assert inst._axis_to_vel(0, 0, 0) == (0.0, 0.0, 0.0)


# ── 构造与配置 ──────────────────────────────────────────────────────────


def test_init_validates_command_term():
    """构造时若 command_name 不在 terms 里应抛 ValueError."""
    from unitree_viser.sim.dds_command_injection import DdsCommandInjector

    env = MagicMock()
    env.command_manager._terms = {"other": MagicMock()}
    with pytest.raises(ValueError, match="not in env.command_manager._terms"):
        DdsCommandInjector(
            server=None, env=env, command_name="twist",
            dds_domain=0, dds_interface="lo", robot_key="go2_0",
        )


def test_init_creates_pending_and_lock():
    """构造后 _pending 是 0, _lock 存在."""
    from unitree_viser.sim.dds_command_injection import DdsCommandInjector

    env = MagicMock()
    env.device = "cpu"
    env.num_envs = 1
    env.command_manager._terms = {"twist": MagicMock()}
    inj = DdsCommandInjector(
        server=None, env=env, command_name="twist",
        dds_domain=0, dds_interface="lo", robot_key="go2_0",
    )
    assert inj._pending == {"vx": 0.0, "vy": 0.0, "wz": 0.0}
    assert isinstance(inj._lock, type(threading.Lock()))
    assert inj._topic == "rt/go2_0/wirelesscontroller"


def test_robot_key_changes_topic():
    """robot_key 决定 topic 后缀."""
    from unitree_viser.sim.dds_command_injection import DdsCommandInjector

    env = MagicMock()
    env.device = "cpu"
    env.num_envs = 1
    env.command_manager._terms = {"twist": MagicMock()}

    inj = DdsCommandInjector(
        server=None, env=env, robot_key="g1_0",
    )
    assert inj._topic == "rt/g1_0/wirelesscontroller"


# ── inject() 写 vel_command_b ─────────────────────────────────────────────


def test_inject_writes_vel_command_b():
    """inject() 把 _pending 写入 vel_command_b."""
    import torch

    from unitree_viser.sim.dds_command_injection import DdsCommandInjector

    # 用真实 torch tensor 作为 vel_command_b
    vcb = torch.zeros(1, 3)

    class _Term:
        vel_command_b = vcb

    env = MagicMock()
    env.device = "cpu"
    env.num_envs = 1
    env.command_manager._terms = {"twist": _Term()}

    inj = DdsCommandInjector(server=None, env=env, command_name="twist")
    inj._pending = {"vx": 0.7, "vy": -0.2, "wz": 0.4}

    # inject() 内部 import torch as _torch, 直接用真实 torch
    inj.inject()

    # 验证 vel_command_b 被写入
    assert vcb[0, 0].item() == pytest.approx(0.7)
    assert vcb[0, 1].item() == pytest.approx(-0.2)
    assert vcb[0, 2].item() == pytest.approx(0.4)


def test_inject_falls_back_to_command_buf():
    """无 vel_command_b 时, 回退写 _command_buf."""
    import torch

    from unitree_viser.sim.dds_command_injection import DdsCommandInjector

    buf = torch.zeros(1, 3)

    class _Term:
        _command_buf = buf

    env = MagicMock()
    env.device = "cpu"
    env.num_envs = 1
    env.command_manager._terms = {"twist": _Term()}

    inj = DdsCommandInjector(server=None, env=env, command_name="twist")
    inj._pending = {"vx": 1.0, "vy": -0.5, "wz": 0.3}

    inj.inject()

    assert buf[0, 0].item() == pytest.approx(1.0)
    assert buf[0, 1].item() == pytest.approx(-0.5)
    assert buf[0, 2].item() == pytest.approx(0.3)


def test_inject_thread_safe_with_lock():
    """inject() 在 _lock 保护下读取 _pending (多线程不崩溃)."""
    import torch

    from unitree_viser.sim.dds_command_injection import DdsCommandInjector

    vcb = torch.zeros(1, 3)

    class _Term:
        vel_command_b = vcb

    env = MagicMock()
    env.device = "cpu"
    env.num_envs = 1
    env.command_manager._terms = {"twist": _Term()}

    inj = DdsCommandInjector(server=None, env=env, command_name="twist")
    inj._pending = {"vx": 1.0, "vy": 0.0, "wz": 0.0}

    # 多线程同时 inject, 不应崩溃
    threads = [threading.Thread(target=inj.inject) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=2.0)


# ── _on_message 回调 ────────────────────────────────────────────────────


def test_on_message_updates_pending():
    """DDS 回调触发 _pending 更新."""
    from unitree_viser.sim.dds_command_injection import DdsCommandInjector

    env = MagicMock()
    env.device = "cpu"
    env.num_envs = 1
    env.command_manager._terms = {"twist": MagicMock()}

    inj = DdsCommandInjector(server=None, env=env, command_name="twist")

    msg = MagicMock()
    msg.lx = 0.3
    msg.ly = 0.8
    msg.rx = -0.5

    inj._on_message(msg)

    p = inj.get_pending()
    # vx=ly=0.8, vy=-lx=-0.3, wz=-rx=0.5
    assert p["vx"] == 0.8
    assert p["vy"] == -0.3
    assert p["wz"] == 0.5


def test_on_message_swallows_exceptions():
    """_on_message 内部异常不应崩溃."""
    from unitree_viser.sim.dds_command_injection import DdsCommandInjector

    env = MagicMock()
    env.device = "cpu"
    env.num_envs = 1
    env.command_manager._terms = {"twist": MagicMock()}

    inj = DdsCommandInjector(server=None, env=env, command_name="twist")

    class _BadMsg:
        @property
        def lx(self):
            raise RuntimeError("bad message")

    # 不应抛
    inj._on_message(_BadMsg())


# ── get_pending 线程安全 ─────────────────────────────────────────────────


def test_get_pending_returns_copy():
    """get_pending 返回副本, 修改不影响内部."""
    from unitree_viser.sim.dds_command_injection import DdsCommandInjector

    env = MagicMock()
    env.device = "cpu"
    env.num_envs = 1
    env.command_manager._terms = {"twist": MagicMock()}

    inj = DdsCommandInjector(server=None, env=env, command_name="twist")
    p = inj.get_pending()
    p["vx"] = 999.0
    assert inj._pending["vx"] == 0.0


# ── stop / 运行标志 ───────────────────────────────────────────────────────


def test_stop_sets_running_false():
    """stop() 应把 _running 设为 False."""
    from unitree_viser.sim.dds_command_injection import DdsCommandInjector

    env = MagicMock()
    env.device = "cpu"
    env.num_envs = 1
    env.command_manager._terms = {"twist": MagicMock()}

    inj = DdsCommandInjector(server=None, env=env, command_name="twist")
    assert inj._running is True
    inj.stop()
    assert inj._running is False


# ── 模拟 SDK 不可用时的降级 ─────────────────────────────────────────────


def test_start_falls_back_when_sdk_unavailable(monkeypatch):
    """当 unitree_sdk2py 不可用, start() 不抛异常, 标记为 mock."""
    import unitree_viser.sim.dds_command_injection as mod
    from unitree_viser.sim.dds_command_injection import DdsCommandInjector

    monkeypatch.setattr(mod, "_DDS_AVAILABLE", False)
    monkeypatch.setattr(mod, "_factory_initialized", False)

    env = MagicMock()
    env.device = "cpu"
    env.num_envs = 1
    env.command_manager._terms = {"twist": MagicMock()}

    inj = DdsCommandInjector(server=None, env=env, command_name="twist")
    # 不抛异常
    inj.start()
    assert inj._started is True
    assert inj._subscriber is None


def test_start_idempotent():
    """start() 重复调用不会再次初始化."""
    import unitree_viser.sim.dds_command_injection as mod
    from unitree_viser.sim.dds_command_injection import DdsCommandInjector

    env = MagicMock()
    env.device = "cpu"
    env.num_envs = 1
    env.command_manager._terms = {"twist": MagicMock()}

    inj = DdsCommandInjector(server=None, env=env, command_name="twist")
    with patch.object(mod, "_DDS_AVAILABLE", False):
        inj.start()
        # 第二次调用应该立即返回
        inj.start()
    assert inj._started is True
