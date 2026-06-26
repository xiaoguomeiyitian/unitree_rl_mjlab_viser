"""ViserRenderThread._render_loop 单元测试.

测试渲染主循环的关键逻辑:
- 帧率控制
- 客户端连接/断开检测
- 连续错误退出
- GUI 更新频率控制
"""

from __future__ import annotations

import os
import sys
import time as _time
import types
from unittest.mock import MagicMock, patch

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Stub mediapy
_stub = types.ModuleType("mediapy")
_stub.__file__ = "(test-stub)"
_stub.set_ffmpeg = lambda _path: None
sys.modules["mediapy"] = _stub


def test_render_loop_no_clients_no_render():
    """无客户端时渲染线程不渲染."""
    from unitree_viser.render.async_render import ViserRenderThread

    server = MagicMock()
    server.get_clients.return_value = []

    with patch("mujoco.MjData", return_value=MagicMock()):
        thread = ViserRenderThread(
            server, MagicMock(), 0, MagicMock(), MagicMock(), MagicMock(), target_fps=20.0
        )

    with patch("mujoco.mj_forward") as mock_forward:
        thread.start()
        try:
            _time.sleep(0.3)
            mock_forward.assert_not_called()
        finally:
            thread.stop(timeout=1.0)


def test_render_loop_renders_when_clients_connected():
    """有客户端时渲染线程执行 mj_forward."""
    from unitree_viser.render.async_render import ViserRenderThread

    server = MagicMock()
    server.get_clients.return_value = [MagicMock()]

    with patch("mujoco.MjData", return_value=MagicMock()):
        thread = ViserRenderThread(
            server, MagicMock(), 0, MagicMock(), MagicMock(), MagicMock(), target_fps=20.0
        )

    with patch("mujoco.mj_forward") as mock_forward:
        thread.start()
        try:
            _time.sleep(0.2)
            assert mock_forward.call_count >= 1
        finally:
            thread.stop(timeout=1.0)


def test_render_loop_error_count_increments():
    """渲染错误时 error_count 递增."""
    from unitree_viser.render.async_render import ViserRenderThread

    server = MagicMock()
    server.get_clients.return_value = [MagicMock()]

    mj_data_b = MagicMock()
    mj_data_b.qpos = MagicMock()
    mj_data_b.qpos.__setitem__ = MagicMock()

    with patch("mujoco.MjData", return_value=mj_data_b):
        thread = ViserRenderThread(
            server, MagicMock(), 0, MagicMock(), MagicMock(), MagicMock(), target_fps=20.0
        )

    # mj_forward 第 1 次抛异常, 之后正常
    call_count = [0]
    def mock_mj_forward(m, d):
        call_count[0] += 1
        if call_count[0] == 1:
            raise TypeError("render error")

    with patch("mujoco.mj_forward", side_effect=mock_mj_forward):
        thread.start()
        try:
            _time.sleep(0.3)
            # 第 1 次错误后 error_count 应 >= 1
            assert thread._error_count >= 1
            # 第 2 次成功后 error_count 应被重置
            _time.sleep(0.3)
            assert thread._error_count == 0
        finally:
            thread.stop(timeout=1.0)


def test_render_loop_updates_gui_once_per_second():
    """GUI 更新频率控制 (每秒最多 1 次)."""
    from unitree_viser.render.async_render import ViserRenderThread

    server = MagicMock()
    server.get_clients.return_value = [MagicMock()]

    with patch("mujoco.MjData", return_value=MagicMock()):
        thread = ViserRenderThread(
            server, MagicMock(), 0, MagicMock(), MagicMock(), MagicMock(), target_fps=30.0
        )

    thread.set_gui_state({"info_html": MagicMock(), "reward_plotter": MagicMock()})

    with patch("mujoco.mj_forward"), \
         patch("unitree_viser.render.viser_setup.update_training_info") as mock_info, \
         patch("unitree_viser.render.viser_setup.push_reward_to_plot") as mock_plot:
        thread.start()
        try:
            _time.sleep(1.5)
            # GUI 更新应被调用, 但不超过 2 次 (1.5 秒内)
            assert mock_info.call_count <= 2
            assert mock_plot.call_count <= 2
        finally:
            thread.stop(timeout=1.0)


def test_render_loop_client_connect_disconnect():
    """客户端连接/断开时打印状态变化."""
    from unitree_viser.render.async_render import ViserRenderThread

    server = MagicMock()
    # 第 1 次: 有客户端, 第 2 次: 无客户端, 第 3 次: 有客户端
    server.get_clients.side_effect = [
        [MagicMock()],
        [],
        [MagicMock()],
        [MagicMock()],
    ]

    with patch("mujoco.MjData", return_value=MagicMock()):
        thread = ViserRenderThread(
            server, MagicMock(), 0, MagicMock(), MagicMock(), MagicMock(), target_fps=20.0
        )

    with patch("mujoco.mj_forward"):
        thread.start()
        try:
            _time.sleep(0.5)
            # 线程应仍在运行 (有重连)
            assert thread.is_alive
        finally:
            thread.stop(timeout=1.0)
