"""TrainingController 状态机测试 (不依赖 viser).

TrainingController 创建时需要 server, 这里通过 MagicMock 注入.
重点验证:
- 暂停/恢复 toggle 行为
- wait_if_paused 在各种状态下的返回
- single_step 一次走一个 iter 然后重新 pause
- request_quit 解除 wait
- speed slider 更新 multiplier
"""

from __future__ import annotations

import os
import sys
import threading
import time
from unittest.mock import MagicMock

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def _make_controller():
    """构造一个绕过 GUI 的 TrainingController."""
    from unitree_viser.train.training_controller import TrainingController

    server = MagicMock()
    # 模拟 server.gui.add_folder 返回一个 context manager 风格的 MagicMock
    server.gui.add_folder.return_value.__enter__ = MagicMock(return_value=MagicMock())
    server.gui.add_folder.return_value.__exit__ = MagicMock(return_value=False)

    # add_button / add_slider / add_html 直接返回 MagicMock (含 on_click/on_update)
    server.gui.add_button.return_value = MagicMock()
    server.gui.add_slider.return_value = MagicMock()
    server.gui.add_html.return_value = MagicMock()

    # 调用 __init__ 会触发 _build_gui, 但因为都是 MagicMock, 不会出错
    controller = TrainingController(server=server, fps=10.0)
    return controller, server


def test_initial_state_not_paused():
    """初始不暂停."""
    controller, _ = _make_controller()
    assert controller.is_paused() is False


def test_toggle_pause_flips_state():
    """toggle_pause 切换 pause 状态."""
    controller, _ = _make_controller()
    assert controller.is_paused() is False
    controller.toggle_pause()
    assert controller.is_paused() is True
    controller.toggle_pause()
    assert controller.is_paused() is False


def test_wait_if_paused_returns_false_when_running():
    """未暂停时 wait_if_paused 返回 False."""
    controller, _ = _make_controller()
    assert controller.wait_if_paused() is False


def test_wait_if_paused_blocks_when_paused():
    """暂停时 wait_if_paused 阻塞,直到 toggle_pause."""
    controller, _ = _make_controller()
    controller.toggle_pause()  # 进入暂停

    result = [None]

    def _waiter():
        result[0] = controller.wait_if_paused()

    t = threading.Thread(target=_waiter)
    t.start()
    time.sleep(0.1)
    assert t.is_alive(), "wait_if_paused 应该阻塞线程"

    controller.toggle_pause()  # 解除暂停
    t.join(timeout=1.0)
    assert not t.is_alive()
    assert result[0] is False


def test_request_quit_unblocks_and_returns_true():
    """request_quit 解除 wait 并返回 True."""
    controller, _ = _make_controller()
    controller.toggle_pause()  # 进入暂停

    result = [None]

    def _waiter():
        result[0] = controller.wait_if_paused()

    t = threading.Thread(target=_waiter)
    t.start()
    time.sleep(0.1)
    assert t.is_alive()

    controller.request_quit()
    t.join(timeout=1.0)
    assert not t.is_alive()
    assert result[0] is True


def test_single_step_returns_one_iter_then_pauses():
    """single_step 模式: 让 runner 走 1 个 iter, 然后自动重新 pause."""
    controller, _ = _make_controller()
    # 请求单步 (不主动 set pause_event)
    controller.request_single_step()

    # runner 调用 wait_if_paused 应该立刻返回 (走 1 个 iter)
    assert controller.wait_if_paused() is False
    # 但 pause_event 现在应被 set (单步后自动重新暂停)
    assert controller.is_paused() is True

    # 再次调用 wait_if_paused 会阻塞
    result = [None]

    def _waiter():
        result[0] = controller.wait_if_paused()

    t = threading.Thread(target=_waiter)
    t.start()
    time.sleep(0.1)
    assert t.is_alive(), "单步之后应自动重新暂停"
    controller.toggle_pause()  # 解除
    t.join(timeout=1.0)
    assert not t.is_alive()


def test_speed_multiplier_default_and_update():
    """默认 speed=1.0, 可更新."""
    controller, _ = _make_controller()
    assert controller.get_speed_multiplier() == 1.0

    # 直接修改内部状态模拟滑块更新
    controller._state.speed_multiplier = 2.5
    assert controller.get_speed_multiplier() == 2.5


def test_toggle_pause_updates_button_label():
    """toggle_pause 应更新按钮 label/icon."""
    controller, server = _make_controller()
    pause_button = controller._pause_button
    # 初始: _build_gui 中调用 add_button("Pause", icon=...)
    add_button_calls = server.gui.add_button.call_args_list
    assert len(add_button_calls) == 2  # pause + step buttons
    pause_init = add_button_calls[0]
    assert pause_init.args[0] == "Pause"

    # 第一次 toggle: 进入暂停, label 应变为 "Resume"
    controller.toggle_pause()
    assert pause_button.label == "Resume"
    # 第二次 toggle: 退出暂停, label 回到 "Pause"
    controller.toggle_pause()
    assert pause_button.label == "Pause"