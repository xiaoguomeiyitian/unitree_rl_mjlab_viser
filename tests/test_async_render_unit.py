"""async_render 单元测试.

不真启动线程 (会浪费资源),重点测试:
- make_viser_handle 字段完整性
- start_viser_render_thread 无客户端时不调用 mjwarp.get_data_into
- start_viser_render_thread 已启动时是幂等
- stop_viser_render_thread 是幂等的
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

    with patch("mujoco_warp.get_data_into") as mock_get_data:
        async_render.start_viser_render_thread(handle, target_fps=20.0)
        try:
            time.sleep(0.3)  # 让线程跑几次循环
            # 无客户端 → 不应触发任何 GPU→CPU 同步
            mock_get_data.assert_not_called()
        finally:
            async_render.stop_viser_render_thread(handle, timeout=1.0)


def test_start_thread_is_idempotent():
    """重复调用 start 不会启动多个线程."""
    from unitree_viser.render import async_render

    server = MagicMock()
    server.get_clients.return_value = [MagicMock()]  # 有客户端

    handle = async_render.make_viser_handle(
        server, MagicMock(), 0, MagicMock(), MagicMock(), MagicMock()
    )

    with patch("mujoco_warp.get_data_into"):
        async_render.start_viser_render_thread(handle, target_fps=20.0)
        thread1 = handle["_render_thread"]
        async_render.start_viser_render_thread(handle, target_fps=20.0)
        thread2 = handle["_render_thread"]
        try:
            assert thread1 is thread2, "重复调用 start 不应创建新线程"
            assert thread1.is_alive()
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

    with patch("mujoco_warp.get_data_into"):
        async_render.start_viser_render_thread(handle, target_fps=20.0)
        async_render.stop_viser_render_thread(handle)
        # 第二次调用不应抛异常
        async_render.stop_viser_render_thread(handle)
        assert not handle["_render_thread"].is_alive()


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
    # 第 1 次返回客户端 (触发渲染), 第 2 次起抛异常
    server.get_clients.side_effect = [
        [MagicMock()],
        Exception("boom"),
        [MagicMock()],
        [MagicMock()],
    ]

    handle = async_render.make_viser_handle(
        server, MagicMock(), 0, MagicMock(), MagicMock(), MagicMock()
    )

    with patch("mujoco_warp.get_data_into"):
        async_render.start_viser_render_thread(handle, target_fps=20.0)
        try:
            time.sleep(0.5)
            assert handle["_render_thread"].is_alive(), "线程遇到异常应继续存活"
        finally:
            async_render.stop_viser_render_thread(handle, timeout=1.0)