"""async_render 单元测试.

测试 ViserRenderThread 新 API + 向后兼容的函数式 API.
重点测试:
- ViserRenderThread: start/stop 幂等, is_alive, get_stats, set_training_state
- make_viser_handle 字段完整性
- start_viser_render_thread (deprecated) 向后兼容
"""

from __future__ import annotations

import os
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def test_make_viser_handle_keys():
    """make_viser_handle 返回的 dict 含所有必需字段."""
    from unitree_viser.render.async_render import make_viser_handle

    handle = make_viser_handle(
        server=MagicMock(),
        scene=MagicMock(),
        env_idx=3,
        mj_model=MagicMock(),
        mj_data=MagicMock(),
        sim=MagicMock(),
    )

    assert handle["server"] is not None
    assert handle["scene"] is not None
    assert handle["env_idx"] == 3
    assert handle["mj_model"] is not None
    assert handle["mj_data"] is not None
    assert handle["sim"] is not None
    # 占位字段初始为 None
    assert handle["_render_thread"] is None
    assert handle["_render_stop"] is None


def test_make_viser_handle_different_env_idx():
    """env_idx 正确传递."""
    from unitree_viser.render.async_render import make_viser_handle

    h = make_viser_handle(MagicMock(), MagicMock(), 42, MagicMock(), MagicMock(), MagicMock())
    assert h["env_idx"] == 42


def test_start_thread_skips_render_when_no_clients():
    """无客户端时, 渲染线程不应调用 mjwarp.get_data_into."""
    from unitree_viser.render import async_render

    server = MagicMock()
    # 模拟 get_clients 返回空列表
    server.get_clients.return_value = []

    handle = async_render.make_viser_handle(
        server=server,
        scene=MagicMock(),
        env_idx=0,
        mj_model=MagicMock(),
        mj_data=MagicMock(),
        sim=MagicMock(),
    )

    with patch("mujoco_warp.get_data_into") as mock_get_data, \
         patch("mujoco.MjData", return_value=MagicMock()):
        async_render.start_viser_render_thread(handle, target_fps=20.0)
        try:
            time.sleep(0.3)  # 让线程跑几次循环
            # 无客户端 → 不应触发任何 GPU→CPU 同步
            mock_get_data.assert_not_called()
        finally:
            async_render.stop_viser_render_thread(handle, timeout=1.0)


def test_start_thread_is_idempotent():
    """重复调用 start 不会同时运行多个线程."""
    from unitree_viser.render import async_render

    server = MagicMock()
    server.get_clients.return_value = [MagicMock()]  # 有客户端

    handle = async_render.make_viser_handle(
        server, MagicMock(), 0, MagicMock(), MagicMock(), MagicMock()
    )

    with patch("mujoco_warp.get_data_into"), \
         patch("mujoco.MjData", return_value=MagicMock()):
        async_render.start_viser_render_thread(handle, target_fps=20.0)
        thread1 = handle["_render_thread"]
        async_render.start_viser_render_thread(handle, target_fps=20.0)
        thread2 = handle["_render_thread"]
        try:
            # 不应同时有两个存活线程
            alive_threads = [t for t in [thread1, thread2] if t is not None and t.is_alive()]
            assert len(alive_threads) <= 1, "不应同时运行多个渲染线程"
        finally:
            async_render.stop_viser_render_thread(handle, timeout=1.0)


def test_stop_thread_is_idempotent():
    """stop 多次调用不会出错."""
    from unitree_viser.render import async_render

    server = MagicMock()
    server.get_clients.return_value = [MagicMock()]

    handle = async_render.make_viser_handle(
        server, MagicMock(), 0, MagicMock(), MagicMock(), MagicMock()
    )

    with patch("mujoco_warp.get_data_into"), \
         patch("mujoco.MjData", return_value=MagicMock()):
        async_render.start_viser_render_thread(handle, target_fps=20.0)
        async_render.stop_viser_render_thread(handle)
        # 第二次调用不应抛异常
        async_render.stop_viser_render_thread(handle)
        # 线程已停止
        t = handle["_render_thread"]
        assert t is None or not t.is_alive()


def test_stop_thread_without_start_is_safe():
    """没启动线程就 stop 也不出错."""
    from unitree_viser.render import async_render

    handle = async_render.make_viser_handle(
        MagicMock(), MagicMock(), 0, MagicMock(), MagicMock(), MagicMock()
    )
    # 不 start, 直接 stop - 不应抛
    async_render.stop_viser_render_thread(handle)
    assert handle["_render_thread"] is None
    assert handle["_render_stop"] is None


def test_render_thread_survives_exceptions():
    """渲染线程遇到异常不应崩溃."""
    from unitree_viser.render import async_render

    server = MagicMock()
    # 始终返回客户端
    server.get_clients.return_value = [MagicMock()]

    handle = async_render.make_viser_handle(
        server, MagicMock(), 0, MagicMock(), MagicMock(), MagicMock()
    )

    # mj_forward 第 1 次抛异常, 第 2 次起正常
    call_count = [0]
    def mock_mj_forward(m, d):
        call_count[0] += 1
        if call_count[0] == 1:
            raise TypeError("mj_forward boom")

    with patch("mujoco_warp.get_data_into"), \
         patch("mujoco.MjData", return_value=MagicMock()), \
         patch("mujoco.mj_forward", side_effect=mock_mj_forward):
        async_render.start_viser_render_thread(handle, target_fps=20.0)
        try:
            time.sleep(0.5)
            # 通过 _render_thread_obj 访问线程对象
            thread_obj = handle.get("_render_thread_obj")
            assert thread_obj is not None and thread_obj.is_alive, "线程遇到异常应继续存活"
        finally:
            async_render.stop_viser_render_thread(handle, timeout=1.0)


# ── ViserRenderThread 新 API 测试 ─────────────────────────────────────────


def test_viser_render_thread_creation():
    """ViserRenderThread 创建时属性正确."""
    from unitree_viser.render.async_render import ViserRenderThread

    server = MagicMock()
    scene = MagicMock()
    with patch("mujoco.MjData", return_value=MagicMock()):
        thread = ViserRenderThread(
            server=server,
            scene=scene,
            env_idx=0,
            mj_model=MagicMock(),
            mj_data=MagicMock(),
            sim=MagicMock(),
            target_fps=15.0,
        )

    assert thread.is_alive is False
    assert thread.get_stats()["alive"] is False
    assert thread.get_stats()["total_frames"] == 0
    assert thread.get_stats()["error_count"] == 0


def test_viser_render_thread_set_training_state():
    """set_training_state 正确设置内部状态."""
    from unitree_viser.render.async_render import ViserRenderThread
    from unitree_viser.render.shared_state import TrainingState

    with patch("mujoco.MjData", return_value=MagicMock()):
        thread = ViserRenderThread(
            MagicMock(), MagicMock(), 0, MagicMock(), MagicMock(), MagicMock()
        )

    state = TrainingState()
    thread.set_training_state(state)
    # 通过消费验证状态已设置
    state.update(iteration=10, mean_reward=5.0)
    # 内部 _consume_training_state 应该能消费
    assert thread._training_state is state


def test_viser_render_thread_start_stop():
    """start/stop 基本功能."""
    from unitree_viser.render.async_render import ViserRenderThread

    server = MagicMock()
    server.get_clients.return_value = [MagicMock()]

    with patch("mujoco_warp.get_data_into"), \
         patch("mujoco.MjData", return_value=MagicMock()):
        thread = ViserRenderThread(
            server, MagicMock(), 0, MagicMock(), MagicMock(), MagicMock(), target_fps=20.0
        )

        thread.start()
        assert thread.is_alive is True
        thread.stop(timeout=1.0)
        assert thread.is_alive is False


def test_viser_render_thread_idempotent_start():
    """重复 start 是幂等的."""
    from unitree_viser.render.async_render import ViserRenderThread

    server = MagicMock()
    server.get_clients.return_value = [MagicMock()]

    with patch("mujoco_warp.get_data_into"), \
         patch("mujoco.MjData", return_value=MagicMock()):
        thread = ViserRenderThread(
            server, MagicMock(), 0, MagicMock(), MagicMock(), MagicMock(), target_fps=20.0
        )

        thread.start()
        thread.start()  # 不应创建第二个线程
        assert thread.is_alive is True
        thread.stop(timeout=1.0)


def test_viser_render_thread_stop_without_start():
    """未 start 时 stop 不报错."""
    from unitree_viser.render.async_render import ViserRenderThread

    with patch("mujoco.MjData", return_value=MagicMock()):
        thread = ViserRenderThread(
            MagicMock(), MagicMock(), 0, MagicMock(), MagicMock(), MagicMock()
        )
    thread.stop()  # 不应抛异常


def test_viser_render_thread_no_clients_no_render():
    """无客户端时, 渲染线程不应调用 mjwarp.get_data_into."""
    from unitree_viser.render.async_render import ViserRenderThread

    server = MagicMock()
    server.get_clients.return_value = []

    with patch("mujoco_warp.get_data_into") as mock_get_data, \
         patch("mujoco.MjData", return_value=MagicMock()):
        thread = ViserRenderThread(
            server, MagicMock(), 0, MagicMock(), MagicMock(), MagicMock(), target_fps=20.0
        )

        thread.start()
        try:
            time.sleep(0.3)
            mock_get_data.assert_not_called()
        finally:
            thread.stop(timeout=1.0)


def test_training_state_update_consume():
    """TrainingState update/consume 基本逻辑."""
    from unitree_viser.render.shared_state import TrainingState

    state = TrainingState()
    assert state.consume() is None  # 初始无 dirty

    state.update(iteration=5, mean_reward=3.14, episode_length=100.0)
    result = state.consume()
    assert result is not None
    assert result["iteration"] == 5
    assert result["mean_reward"] == 3.14
    assert result["episode_length"] == 100.0

    # 消费后 dirty 清除
    assert state.consume() is None

    # 更新部分字段
    state.update(iteration=6)
    result = state.consume()
    assert result["iteration"] == 6
    assert result["mean_reward"] == 3.14  # 保持不变