"""sim_viewer 单元测试.

测试:
- _extract_actor_obs 各种输入形式
- _zero_policy / _random_policy 形状/dtype/device
- _make_typed_policy 强制 num_actions 形状
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


# ── _extract_actor_obs ────────────────────────────────────────────────────


def test_extract_actor_obs_with_actor_key():
    """obs dict 含 'actor' 键时返回该值."""
    from unitree_viser.sim.sim_viewer import _extract_actor_obs

    t = torch.zeros(1, 12)
    obs = {"actor": t, "critic": torch.zeros(1, 47), "other": "z"}
    assert _extract_actor_obs(obs) is t


def test_extract_actor_obs_plain_tensor():
    """obs 本身是 tensor 时原样返回."""
    from unitree_viser.sim.sim_viewer import _extract_actor_obs

    t = torch.zeros(1, 12)
    assert _extract_actor_obs(t) is t


def test_extract_actor_obs_dict_without_actor_picks_first_tensor():
    """obs dict 无 'actor' 键时, 取第一个 tensor 值."""
    from unitree_viser.sim.sim_viewer import _extract_actor_obs

    t = torch.zeros(1, 12)
    obs = {"critic": t, "other": "z"}
    assert _extract_actor_obs(obs) is t


def test_extract_actor_obs_dict_no_tensor_raises():
    """obs dict 中完全没有 tensor 时应抛 ValueError."""
    from unitree_viser.sim.sim_viewer import _extract_actor_obs

    obs = {"foo": "bar", "baz": 42}
    with pytest.raises(ValueError, match="obs dict 中找不到 tensor"):
        _extract_actor_obs(obs)


# ── _zero_policy ──────────────────────────────────────────────────────────


def test_zero_policy_shape():
    """zero_policy 输出 (n, 12) 零张量."""
    from unitree_viser.sim.sim_viewer import _zero_policy

    obs = torch.zeros(4, 12)
    out = _zero_policy(obs)
    assert out.shape == (4, 12)
    assert out.sum().item() == 0.0


def test_zero_policy_device():
    """zero_policy 输出与输入同 device."""
    from unitree_viser.sim.sim_viewer import _zero_policy

    obs = torch.zeros(2, 12)
    out = _zero_policy(obs)
    assert out.device == obs.device


def test_zero_policy_with_dict_obs():
    """zero_policy 接受 dict obs."""
    from unitree_viser.sim.sim_viewer import _zero_policy

    obs = {"actor": torch.zeros(3, 12), "critic": torch.zeros(3, 47)}
    out = _zero_policy(obs)
    assert out.shape == (3, 12)


# ── _random_policy ────────────────────────────────────────────────────────


def test_random_policy_bounds():
    """random_policy 输出在 [-1, 1] 范围内."""
    from unitree_viser.sim.sim_viewer import _random_policy

    obs = torch.zeros(100, 12)
    out = _random_policy(obs)
    assert out.shape == (100, 12)
    assert (out >= -1.0).all()
    assert (out <= 1.0).all()


def test_random_policy_with_dict_obs():
    """random_policy 接受 dict obs."""
    from unitree_viser.sim.sim_viewer import _random_policy

    obs = {"actor": torch.zeros(5, 12)}
    out = _random_policy(obs)
    assert out.shape == (5, 12)


# ── _make_typed_policy ────────────────────────────────────────────────────


def test_make_typed_policy_enforces_num_actions():
    """_make_typed_policy 强制输出 (n, num_actions) 形状."""
    from unitree_viser.sim.sim_viewer import _make_typed_policy, _zero_policy

    typed = _make_typed_policy(_zero_policy, 24)
    obs = torch.zeros(2, 12)
    out = typed(obs)
    assert out.shape == (2, 24)


def test_make_typed_policy_with_dict():
    """_make_typed_policy 包装后接受 dict obs."""
    from unitree_viser.sim.sim_viewer import _make_typed_policy, _random_policy

    typed = _make_typed_policy(_random_policy, 12)
    obs = {"actor": torch.zeros(3, 12)}
    out = typed(obs)
    assert out.shape == (3, 12)
    assert (out >= -1.0).all() and (out <= 1.0).all()
