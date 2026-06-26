"""SimViewer 集成测试.

测试 SimViewer 的完整生命周期 (使用 mock 避免真实 Viser 服务器).
覆盖:
- _extract_actor_obs
- _zero_policy / _random_policy / _make_typed_policy
- SimViewer 初始化
"""

from __future__ import annotations

import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Stub mediapy
_stub = types.ModuleType("mediapy")
_stub.__file__ = "(test-stub)"
_stub.set_ffmpeg = lambda _path: None
sys.modules["mediapy"] = _stub


# ── 测试 _extract_actor_obs ─────────────────────────────────────────────


def test_extract_actor_obs_with_actor_key():
    """dict 含 'actor' 键时返回 actor tensor."""
    from unitree_viser.sim.sim_viewer import _extract_actor_obs

    actor = torch.zeros(1, 12)
    obs = {"actor": actor, "critic": torch.zeros(1, 47)}
    result = _extract_actor_obs(obs)
    assert result is actor


def test_extract_actor_obs_plain_tensor():
    """obs 本身是 tensor 时原样返回."""
    from unitree_viser.sim.sim_viewer import _extract_actor_obs

    tensor = torch.zeros(1, 12)
    assert _extract_actor_obs(tensor) is tensor


def test_extract_actor_obs_dict_without_actor_picks_first_tensor():
    """无 'actor' 键时取第一个 tensor 类型的值."""
    from unitree_viser.sim.sim_viewer import _extract_actor_obs

    t = torch.zeros(1, 12)
    obs = {"critic": t, "other": "not_tensor"}
    assert _extract_actor_obs(obs) is t


def test_extract_actor_obs_dict_no_tensor_raises():
    """dict 中无 tensor 值时抛 ValueError."""
    from unitree_viser.sim.sim_viewer import _extract_actor_obs

    obs = {"a": "string", "b": 42}
    with pytest.raises(ValueError, match="找不到 tensor"):
        _extract_actor_obs(obs)


# ── 测试 _zero_policy / _random_policy ──────────────────────────────────


def test_zero_policy_shape():
    """zero_policy 输出 (n, 12) 形状."""
    from unitree_viser.sim.sim_viewer import _zero_policy

    obs = torch.zeros(4, 12)
    result = _zero_policy(obs)
    assert result.shape == (4, 12)
    assert (result == 0).all()


def test_zero_policy_dict_obs():
    """zero_policy 支持 dict obs."""
    from unitree_viser.sim.sim_viewer import _zero_policy

    obs = {"actor": torch.zeros(2, 12)}
    result = _zero_policy(obs)
    assert result.shape == (2, 12)
    assert (result == 0).all()


def test_random_policy_bounds():
    """random_policy 输出在 [-1, 1] 范围内."""
    from unitree_viser.sim.sim_viewer import _random_policy

    obs = {"actor": torch.zeros(3, 12)}
    result = _random_policy(obs)
    assert result.shape == (3, 12)
    assert (result >= -1.0).all()
    assert (result <= 1.0).all()


def test_make_typed_policy_enforces_num_actions():
    """_make_typed_policy 强制输出 (n, num_actions) 形状."""
    from unitree_viser.sim.sim_viewer import _make_typed_policy, _zero_policy

    typed = _make_typed_policy(_zero_policy, 23)
    obs = torch.zeros(1, 23)
    result = typed(obs)
    assert result.shape == (1, 23)


def test_make_typed_policy_handles_wrong_output_dim():
    """_make_typed_policy 处理 base_policy 输出维度不匹配."""
    from unitree_viser.sim.sim_viewer import _make_typed_policy, _zero_policy

    # zero_policy 输出 12 维, 但 typed 要求 23 维
    typed = _make_typed_policy(_zero_policy, 23)
    obs = torch.zeros(1, 12)
    result = typed(obs)
    assert result.shape == (1, 23)


# ── 测试 SimViewer 初始化 ───────────────────────────────────────────────


def test_sim_viewer_init_defaults():
    """SimViewer 初始化默认值."""
    from unitree_viser.sim.sim_viewer import SimViewer

    env = MagicMock()
    viewer = SimViewer(env=env)

    assert viewer._env is env
    assert viewer._port == 20006
    assert viewer._env_idx == 0
    assert viewer._inject_commands is True
    assert viewer._command_name == "twist"
    assert viewer._command_source == "gui"


def test_sim_viewer_init_custom():
    """SimViewer 自定义参数."""
    from unitree_viser.sim.sim_viewer import SimViewer

    env = MagicMock()
    viewer = SimViewer(
        env=env,
        port=30000,
        env_idx=2,
        inject_commands=False,
        command_name="vel",
    )

    assert viewer._port == 30000
    assert viewer._env_idx == 2
    assert viewer._inject_commands is False
    assert viewer._command_name == "vel"


def test_sim_viewer_invalid_command_source():
    """无效 command_source 抛 ValueError."""
    from unitree_viser.sim.sim_viewer import SimViewer

    env = MagicMock()
    with pytest.raises(ValueError, match="command_source"):
        SimViewer(env=env, command_source="invalid")


def test_sim_viewer_close_stops_server():
    """close() 停止 Viser 服务器."""
    from unitree_viser.sim.sim_viewer import SimViewer

    env = MagicMock()
    viewer = SimViewer(env=env)
    server = MagicMock()
    viewer._server = server
    viewer.close()
    server.stop.assert_called_once()
    assert viewer._server is None


def test_sim_viewer_close_no_server():
    """close() 无服务器时不报错."""
    from unitree_viser.sim.sim_viewer import SimViewer

    env = MagicMock()
    viewer = SimViewer(env=env)
    viewer._server = None
    viewer.close()  # 不应抛异常
