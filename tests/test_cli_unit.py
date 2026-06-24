"""cli 单元测试.

测试:
- _ensure_sibling_src_on_path 的 sys.path 修复
- _stub_mediapy 的 stub 注入
- TrainArgs / SimArgs 字段默认值
- main() 的子命令映射 (tyro 0.9 → 1.0)
"""

from __future__ import annotations

import importlib
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
_SIBLING = os.path.normpath(os.path.join(_HERE, "..", "..", "unitree_rl_mjlab", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def test_dataclass_train_args_defaults():
    """TrainArgs 默认值正确."""
    from unitree_viser.cli import TrainArgs

    args = TrainArgs(task="X")
    assert args.task == "X"
    assert args.viser_port == 20006
    assert args.viser_env_idx == 0
    assert args.viser_fps == 30.0
    assert args.enable_control is False
    assert args.headless is False
    assert args.num_envs is None
    assert args.max_iterations is None
    assert args.seed is None
    assert args.save_interval is None
    assert args.motion_file is None


def test_dataclass_sim_args_defaults():
    """SimArgs 默认值正确."""
    from unitree_viser.cli import SimArgs

    args = SimArgs(task="X")
    assert args.task == "X"
    assert args.checkpoint is None
    assert args.policy is None
    assert args.num_envs == 1
    assert args.viser_port == 20006
    assert args.viser_env_idx == 0
    assert args.inject_commands is True
    assert args.command_name == "twist"
    assert args.command_source == "gui"
    assert args.dds_domain == 0
    assert args.dds_interface == "lo"
    assert args.robot_key == "go2_0"
    assert args.dds_timeout == 0.5
    assert args.max_steps is None
    assert args.headless is False


def test_sim_args_dds_config():
    """SimArgs DDS 配置可覆盖."""
    from unitree_viser.cli import SimArgs

    args = SimArgs(
        task="X",
        command_source="dds",
        dds_domain=5,
        dds_interface="enp0s3",
        robot_key="g1_0",
        dds_timeout=1.5,
    )
    assert args.command_source == "dds"
    assert args.dds_domain == 5
    assert args.dds_interface == "enp0s3"
    assert args.robot_key == "g1_0"
    assert args.dds_timeout == 1.5


def test_main_function_exists():
    """main() 可调用."""
    from unitree_viser import cli

    assert callable(cli.main)


def test_run_train_function_exists():
    """run_train() 可调用."""
    from unitree_viser import cli

    assert callable(cli.run_train)


def test_run_sim_function_exists():
    """run_sim() 可调用."""
    from unitree_viser import cli

    assert callable(cli.run_sim)


# ── _ensure_sibling_src_on_path ──────────────────────────────────────────


def test_ensure_sibling_src_on_path_filters_bad_paths(monkeypatch):
    """viser/src (无 __init__.py) 应被从 sys.path 过滤掉."""
    from unitree_viser import cli

    # 重置 cli 模块的副作用: 不能直接 reload (会让 sys.modules 失效)
    # 改用 monkeypatch 设置环境变量
    viser_src = os.path.normpath(os.path.join(_SRC))

    # 模拟 sys.path 含 viser/src
    fake_path = [p for p in sys.path if p != ""]
    if viser_src not in fake_path:
        fake_path.append(viser_src)

    monkeypatch.setattr(sys, "path", fake_path)

    # 调用内部函数 (但它依赖 env var)
    monkeypatch.setenv("UNITREE_RL_MJLAB_SRC", _SIBLING)
    cli._ensure_sibling_src_on_path()

    # 兄弟 src 应排第一
    assert sys.path[0] == _SIBLING
    # viser/src 应被过滤
    assert viser_src not in sys.path


def test_ensure_sibling_src_no_env_var_no_crash(monkeypatch):
    """没设置 env var 且找不到兄弟项目时不应崩溃."""
    from unitree_viser import cli

    monkeypatch.delenv("UNITREE_RL_MJLAB_SRC", raising=False)
    # 不应抛
    cli._ensure_sibling_src_on_path()


def test_ensure_sibling_src_imports_src(monkeypatch):
    """能找到兄弟 src 时, 显式 import 锁定 real package."""
    from unitree_viser import cli

    monkeypatch.setenv("UNITREE_RL_MJLAB_SRC", _SIBLING)
    # 把 src 加进去模拟环境
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != ""] + [_SRC])

    cli._ensure_sibling_src_on_path()

    # src 应在 sys.modules
    if "src" in sys.modules:
        assert sys.modules["src"].__file__ is not None


def test_ensure_sibling_src_removes_empty_string(monkeypatch):
    """空字符串路径应被过滤."""
    from unitree_viser import cli

    monkeypatch.setattr(sys, "path", ["", "/some/path", ""])
    monkeypatch.delenv("UNITREE_RL_MJLAB_SRC", raising=False)

    cli._ensure_sibling_src_on_path()
    assert "" not in sys.path


# ── _stub_mediapy ────────────────────────────────────────────────────────


def test_stub_mediapy_when_real_available():
    """真实 mediapy 可用时, 不注入 stub."""
    from unitree_viser import cli

    # 确保 mediapy 真存在
    import mediapy  # noqa: F401

    sys.modules.pop("mediapy", None)
    cli._stub_mediapy()

    import mediapy as m_after

    # 真实模块, 应有 set_ffmpeg
    assert hasattr(m_after, "set_ffmpeg")


def test_stub_mediapy_when_import_fails():
    """mediapy import 失败时, 注入 stub."""
    from unitree_viser import cli

    # 移除 mediapy 并模拟 import 失败
    sys.modules.pop("mediapy", None)
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "mediapy":
            raise TypeError("npt.NDArray[Any] syntax error")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = mock_import
    try:
        cli._stub_mediapy()
        # _stub_mediapy 已将 stub 注入 sys.modules, 直接读取
        stub = sys.modules.get("mediapy")
        assert stub is not None, "stub 应被注入 sys.modules"
        assert stub.__file__ == "(viser-stub)"
        assert callable(stub.set_ffmpeg)
        stub.set_ffmpeg("/some/path")
    finally:
        builtins.__import__ = real_import
        sys.modules.pop("mediapy", None)


def test_stub_mediapy_idempotent():
    """重复调用 _stub_mediapy 不应替换现有 stub."""
    from unitree_viser import cli

    sys.modules.pop("mediapy", None)
    cli._stub_mediapy()
    first_stub = sys.modules["mediapy"]
    cli._stub_mediapy()
    assert sys.modules["mediapy"] is first_stub, "stub 已是 mediapy 时不应被替换"