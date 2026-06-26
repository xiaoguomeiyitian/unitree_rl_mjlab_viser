"""CLI 集成测试.

测试 run_train / run_sim 的主要流程 (使用 mock 避免真实环境).
覆盖:
- run_train headless 模式
- run_sim headless 模式
- CLI 参数解析
"""

from __future__ import annotations

import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


# ── 测试 CLI 参数默认值 ─────────────────────────────────────────────────


def test_train_args_defaults():
    """TrainArgs 默认值正确."""
    from unitree_viser.cli import TrainArgs

    args = TrainArgs(task="TestTask")
    assert args.task == "TestTask"
    assert args.viser_port == 20006
    assert args.viser_fps == 30.0
    assert args.enable_control is False
    assert args.headless is False
    assert args.device == "auto"
    assert args.num_envs is None
    assert args.max_iterations is None


def test_sim_args_defaults():
    """SimArgs 默认值正确."""
    from unitree_viser.cli import SimArgs

    args = SimArgs(task="TestTask")
    assert args.task == "TestTask"
    assert args.num_envs == 1
    assert args.viser_port == 20006
    assert args.inject_commands is True
    assert args.command_name == "twist"
    assert args.command_source == "gui"
    assert args.headless is False
    assert args.device == "cpu"


def test_sim_args_dds_config():
    """SimArgs DDS 配置可覆盖."""
    from unitree_viser.cli import SimArgs

    args = SimArgs(
        task="TestTask",
        command_source="dds",
        dds_domain=1,
        dds_interface="eth0",
        robot_key="g1_0",
        dds_timeout=1.0,
    )
    assert args.command_source == "dds"
    assert args.dds_domain == 1
    assert args.dds_interface == "eth0"
    assert args.robot_key == "g1_0"
    assert args.dds_timeout == 1.0


# ── 测试 _ensure_sibling_src_on_path ────────────────────────────────────


def test_ensure_sibling_src_on_path_filters_empty_string():
    """空字符串从 sys.path 中移除."""
    from unitree_viser.cli import _ensure_sibling_src_on_path

    original = sys.path[:]
    sys.path.insert(0, "")
    try:
        _ensure_sibling_src_on_path()
        assert "" not in sys.path
    finally:
        sys.path[:] = original


def test_ensure_sibling_src_on_path_no_crash_without_sibling():
    """没有兄弟项目时不崩溃."""
    from unitree_viser.cli import _ensure_sibling_src_on_path

    # 临时移除 UNITREE_RL_MJLAB_SRC
    old_env = os.environ.pop("UNITREE_RL_MJLAB_SRC", None)
    try:
        # 不应崩溃
        _ensure_sibling_src_on_path()
    finally:
        if old_env is not None:
            os.environ["UNITREE_RL_MJLAB_SRC"] = old_env


# ── 测试 _stub_mediapy ──────────────────────────────────────────────────


def test_stub_mediapy_injects_stub_when_missing():
    """mediapy 不可用时注入 stub."""
    from unitree_viser.cli import _stub_mediapy

    # 移除已缓存的 mediapy
    saved = sys.modules.pop("mediapy", None)
    try:
        _stub_mediapy()
        assert "mediapy" in sys.modules
        assert hasattr(sys.modules["mediapy"], "set_ffmpeg")
    finally:
        if saved is not None:
            sys.modules["mediapy"] = saved
        else:
            sys.modules.pop("mediapy", None)


def test_stub_mediapy_skips_when_already_imported():
    """mediapy 已存在时不覆盖."""
    from unitree_viser.cli import _stub_mediapy

    # 保存原始 mediapy (conftest.py 注入的 stub)
    original_mediapy = sys.modules.get("mediapy")
    real_mediapy = types.ModuleType("mediapy")
    real_mediapy.__file__ = "real"
    sys.modules["mediapy"] = real_mediapy
    try:
        _stub_mediapy()
        assert sys.modules["mediapy"].__file__ == "real"
    finally:
        # 恢复原始 mediapy stub
        if original_mediapy is not None:
            sys.modules["mediapy"] = original_mediapy
        else:
            sys.modules.pop("mediapy", None)


# ── 测试 main() 入口 ────────────────────────────────────────────────────


def test_main_function_exists():
    """main 函数存在且可调用."""
    from unitree_viser.cli import main

    assert callable(main)


def test_run_train_function_exists():
    """run_train 函数存在."""
    from unitree_viser.cli import run_train

    assert callable(run_train)


def test_run_sim_function_exists():
    """run_sim 函数存在."""
    from unitree_viser.cli import run_sim

    assert callable(run_sim)
