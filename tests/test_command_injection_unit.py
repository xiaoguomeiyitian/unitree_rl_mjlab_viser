"""CommandInjector 单元测试.

CommandInjector 注入逻辑 (不真启动 GUI):
- inject() 写 vel_command_b
- inject() 写 _command_buf fallback
- 禁用时不写
- get_pending() 返回当前 pending
- 构造时检查 term 存在
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def _make_env_with_vel_command_b():
    """构造 env + term + vel_command_b mock."""
    term = MagicMock()
    term.vel_command_b = MagicMock()
    term.vel_command_b.dtype = MagicMock()
    term.vel_command_b.shape = (1, 3)
    # 让 dtype 比较工作
    term.vel_command_b.dtype.__eq__ = lambda self, other: True

    env = MagicMock()
    env.device = "cpu"
    env.num_envs = 1
    env.command_manager._terms = {"twist": term}

    server = MagicMock()
    server.gui.add_folder.return_value.__enter__ = MagicMock(return_value=MagicMock())
    server.gui.add_folder.return_value.__exit__ = MagicMock(return_value=False)
    server.gui.add_checkbox.return_value = MagicMock()
    server.gui.add_slider.return_value = MagicMock()
    server.gui.add_button.return_value = MagicMock()

    return env, server, term


def test_init_validates_command_term():
    """构造时若 command_name 不在 terms 里应抛 ValueError."""
    from unitree_viser.sim.command_injection import CommandInjector

    env = MagicMock()
    env.command_manager._terms = {"other": MagicMock()}
    server = MagicMock()

    with pytest.raises(ValueError, match="not in env.command_manager._terms"):
        CommandInjector(server=server, env=env, command_name="twist")


def test_init_succeeds_with_valid_term():
    """构造时 term 存在则不抛."""
    from unitree_viser.sim.command_injection import CommandInjector

    env, server, _ = _make_env_with_vel_command_b()
    inj = CommandInjector(server=server, env=env, command_name="twist")
    assert inj._command_name == "twist"
    assert inj._enabled is True
    # pending 初始为 0
    assert inj.get_pending() == {"vx": 0.0, "vy": 0.0, "wz": 0.0}


def test_inject_writes_vel_command_b():
    """inject() 把 pending 写入 vel_command_b."""
    import torch

    env, server, term = _make_env_with_vel_command_b()
    # 用真的 tensor 替换 MagicMock, 验证 [:] = 真的 tensor.unsqueeze(0).expand(...)
    vcb = torch.zeros(1, 3, dtype=torch.float32)
    term.vel_command_b = vcb

    from unitree_viser.sim.command_injection import CommandInjector
    inj = CommandInjector(server=server, env=env, command_name="twist")
    inj._pending = {"vx": 0.5, "vy": -0.3, "wz": 0.1}

    inj.inject()

    # vel_command_b 已被写入 (除 wz 永远是 0, 因 _torch.tensor([vx, vy, 0.0]))
    assert float(vcb[0, 0]) == pytest.approx(0.5)
    assert float(vcb[0, 1]) == pytest.approx(-0.3)
    assert float(vcb[0, 2]) == pytest.approx(0.0)


def test_inject_skipped_when_disabled():
    """禁用时不调用写."""
    from unitree_viser.sim.command_injection import CommandInjector

    env, server, term = _make_env_with_vel_command_b()
    inj = CommandInjector(server=server, env=env, command_name="twist")
    inj._enabled = False
    inj.inject()

    # vel_command_b 不应被访问
    term.vel_command_b.__setitem__.assert_not_called()


def test_inject_falls_back_to_command_buf():
    """term 没有 vel_command_b 时降级到 _command_buf."""
    from unitree_viser.sim.command_injection import CommandInjector

    buf = MagicMock()
    buf.shape = (1, 4)

    class _TermNoVel:
        _command_buf = buf

    env = MagicMock()
    env.device = "cpu"
    env.num_envs = 1
    env.command_manager._terms = {"twist": _TermNoVel()}
    server = MagicMock()
    server.gui.add_folder.return_value.__enter__ = MagicMock(return_value=MagicMock())
    server.gui.add_folder.return_value.__exit__ = MagicMock(return_value=False)
    server.gui.add_checkbox.return_value = MagicMock()
    server.gui.add_slider.return_value = MagicMock()
    server.gui.add_button.return_value = MagicMock()

    inj = CommandInjector(server=server, env=env, command_name="twist")
    inj._pending = {"vx": 1.0, "vy": 0.5, "wz": 0.2}
    inj.inject()

    # _command_buf 应被写入
    assert buf.__setitem__.call_count >= 1


def test_inject_silent_for_motion_tracking_term():
    """无 vel_command_b 也无 _command_buf (如 motion term) 时静默跳过."""
    from unitree_viser.sim.command_injection import CommandInjector

    class _MotionTerm:
        # 既无 vel_command_b 也无 _command_buf
        pass

    env = MagicMock()
    env.device = "cpu"
    env.num_envs = 1
    env.command_manager._terms = {"motion": _MotionTerm()}
    server = MagicMock()
    server.gui.add_folder.return_value.__enter__ = MagicMock(return_value=MagicMock())
    server.gui.add_folder.return_value.__exit__ = MagicMock(return_value=False)
    server.gui.add_checkbox.return_value = MagicMock()
    server.gui.add_slider.return_value = MagicMock()
    server.gui.add_button.return_value = MagicMock()

    inj = CommandInjector(server=server, env=env, command_name="motion")
    # 不抛
    inj.inject()


def test_get_pending_returns_copy():
    """get_pending 返回字典副本 (修改返回 dict 不影响内部)."""
    from unitree_viser.sim.command_injection import CommandInjector

    env, server, _ = _make_env_with_vel_command_b()
    inj = CommandInjector(server=server, env=env, command_name="twist")
    p = inj.get_pending()
    p["vx"] = 999.0
    # 内部不变
    assert inj._pending["vx"] == 0.0


def test_stop_button_resets_pending():
    """Stop 按钮清零所有命令."""
    from unitree_viser.sim.command_injection import CommandInjector

    env, server, _ = _make_env_with_vel_command_b()
    # 为 add_button 配一个可记录 callbacks 的 side_effect
    callbacks = {}
    buttons_seen = []

    def _make_button(label, *args, **kwargs):
        btn = MagicMock()
        buttons_seen.append(label)
        # on_click 用作装饰器: 把 callback 存到 callbacks
        def _register(fn):
            callbacks[label] = fn
            return btn
        btn.on_click.side_effect = _register
        return btn

    server.gui.add_button.side_effect = _make_button

    inj = CommandInjector(server=server, env=env, command_name="twist")

    # 模拟滑块更新
    inj._pending = {"vx": 1.0, "vy": 0.5, "wz": 0.2}

    # 触发 Stop 按钮回调
    assert "Stop (zero commands)" in callbacks
    callbacks["Stop (zero commands)"](None)

    assert inj._pending == {"vx": 0.0, "vy": 0.0, "wz": 0.0}