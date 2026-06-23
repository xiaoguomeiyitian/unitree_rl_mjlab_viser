"""ViserRunner / runner_subclass 单元测试.

覆盖:
- make_viser_runner_cls 工厂
- 钩子注册/清理 (通过实例调用)
- _post_iter_hooks 列表语义
- ViserRunner 类属性 (viser_gui_state / training_controller)
- _default_post_iter_hook 在 viser_gui_state=None / 存在时行为
- viser_handle TypedDict 字段完整性
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# 确保 import 路径
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


# ── 测试 make_viser_runner_cls 工厂 ────────────────────────────────────────


class _BaseRunner:
    """替身 rsl_rl.OnPolicyRunner."""

    def __init__(self):
        self.env = MagicMock()
        self.alg = MagicMock()
        self.logger = MagicMock()
        self.device = "cpu"
        self.cfg = {"save_interval": 100, "num_steps_per_env": 24, "algorithm": {}, "check_for_nan": True}
        self.current_learning_iteration = 0
        self.is_distributed = False


def _make_runner():
    """创建 ViserRunner 实例 (通过动态子类化避免依赖 MjlabOnPolicyRunner)."""
    from unitree_viser.train.runner_subclass import ViserRunner, make_viser_runner_cls
    Cls = make_viser_runner_cls(_BaseRunner)
    return Cls()


def test_make_viser_runner_cls_returns_subclass():
    """make_viser_runner_cls 返回的类同时继承 ViserRunner 与基类."""
    from unitree_viser.train.runner_subclass import (
        ViserRunner,
        make_viser_runner_cls,
    )

    Cls = make_viser_runner_cls(_BaseRunner)

    # 必须同时继承两边
    assert issubclass(Cls, ViserRunner)
    assert issubclass(Cls, _BaseRunner)
    # 动态生成类的 __name__ 应是 ViserRunner
    assert Cls.__name__ == "ViserRunner"
    # docstring 含基类名
    assert "_BaseRunner" in (Cls.__doc__ or "")


def test_make_viser_runner_cls_different_bases():
    """不同基类应生成不同类 (互不污染)."""
    from unitree_viser.train.runner_subclass import make_viser_runner_cls

    class _A:
        pass

    class _B:
        pass

    ClsA = make_viser_runner_cls(_A)
    ClsB = make_viser_runner_cls(_B)

    assert ClsA is not ClsB
    assert issubclass(ClsA, _A)
    assert issubclass(ClsB, _B)
    assert not issubclass(ClsA, _B)


# ── 钩子 API (通过实例调用) ──────────────────────────────────────────────


def test_post_iter_hooks_initially_empty():
    """类级 _post_iter_hooks 初始为空列表."""
    from unitree_viser.train.runner_subclass import ViserRunner

    # 不污染其他测试: 备份
    saved = list(ViserRunner._post_iter_hooks)
    ViserRunner._post_iter_hooks = []
    try:
        assert isinstance(ViserRunner._post_iter_hooks, list)
        assert len(ViserRunner._post_iter_hooks) == 0
    finally:
        ViserRunner._post_iter_hooks = saved


def test_register_post_iter_hook_appends():
    """register_post_iter_hook 把回调加到列表."""
    runner = _make_runner()
    saved = list(runner._post_iter_hooks)
    runner._post_iter_hooks = []
    try:
        hook = MagicMock()
        runner.register_post_iter_hook(hook)
        assert hook in runner._post_iter_hooks
    finally:
        runner._post_iter_hooks = saved


def test_clear_post_iter_hooks_empties():
    """clear_post_iter_hooks 清空列表."""
    runner = _make_runner()
    saved = list(runner._post_iter_hooks)
    runner._post_iter_hooks = [MagicMock(), MagicMock()]
    try:
        runner.clear_post_iter_hooks()
        assert runner._post_iter_hooks == []
    finally:
        runner._post_iter_hooks = saved


def test_register_multiple_preserves_order():
    """多次注册保持 FIFO 顺序."""
    runner = _make_runner()
    saved = list(runner._post_iter_hooks)
    runner._post_iter_hooks = []
    try:
        h1, h2, h3 = MagicMock(), MagicMock(), MagicMock()
        runner.register_post_iter_hook(h1)
        runner.register_post_iter_hook(h2)
        runner.register_post_iter_hook(h3)
        assert runner._post_iter_hooks == [h1, h2, h3]
    finally:
        runner._post_iter_hooks = saved


# ── ViserRunner 类属性默认值 ──────────────────────────────────────────────


def test_viser_gui_state_default_none():
    """viser_gui_state 类属性默认 None."""
    from unitree_viser.train.runner_subclass import ViserRunner

    assert ViserRunner.viser_gui_state is None


def test_training_controller_default_none():
    """training_controller 类属性默认 None."""
    from unitree_viser.train.runner_subclass import ViserRunner

    assert ViserRunner.training_controller is None


def test_learn_method_exists():
    """learn 方法存在且签名兼容 rsl_rl."""
    from unitree_viser.train.runner_subclass import (
        ViserRunner,
        make_viser_runner_cls,
    )

    Cls = make_viser_runner_cls(_BaseRunner)
    assert hasattr(Cls, "learn")
    # 接受 num_learning_iterations 与 init_at_random_ep_len
    import inspect

    sig = inspect.signature(Cls.learn)
    params = list(sig.parameters.keys())
    assert "num_learning_iterations" in params
    assert "init_at_random_ep_len" in params


# ── _default_post_iter_hook 行为 ──────────────────────────────────────────


def test_default_hook_skips_when_gui_state_none():
    """viser_gui_state 为 None 时不调用任何 update 函数."""
    from unitree_viser.train.runner_subclass import ViserRunner

    inst = ViserRunner.__new__(ViserRunner)
    inst.viser_gui_state = None

    # 即使 _post_iter_hooks 有项, 也不该调用 update
    with patch(
        "unitree_viser.render.viser_setup.update_training_info"
    ) as mock_update, patch(
        "unitree_viser.render.viser_setup.push_reward_to_plot"
    ) as mock_push:
        inst._default_post_iter_hook(0, 1.23, 100.0)
        mock_update.assert_not_called()
        mock_push.assert_not_called()


def test_default_hook_calls_update_with_gui_state():
    """viser_gui_state 存在时调用 update_training_info 和 push_reward_to_plot."""
    from unitree_viser.train.runner_subclass import ViserRunner

    inst = ViserRunner.__new__(ViserRunner)
    inst.viser_gui_state = {"info_html": MagicMock(), "reward_plotter": MagicMock()}

    with patch(
        "unitree_viser.render.viser_setup.update_training_info"
    ) as mock_update, patch(
        "unitree_viser.render.viser_setup.push_reward_to_plot"
    ) as mock_push:
        inst._default_post_iter_hook(5, 1.23, 100.0)
        mock_update.assert_called_once()
        mock_push.assert_called_once()
        # 验证参数
        call_kwargs = mock_update.call_args.kwargs
        assert call_kwargs["iteration"] == 5
        assert call_kwargs["mean_reward"] == 1.23


def test_default_hook_episode_length_none():
    """episode_length 为 None 时 push_reward 不传 episode_length."""
    from unitree_viser.train.runner_subclass import ViserRunner

    inst = ViserRunner.__new__(ViserRunner)
    inst.viser_gui_state = {"info_html": MagicMock(), "reward_plotter": MagicMock()}

    with patch(
        "unitree_viser.render.viser_setup.push_reward_to_plot"
    ) as mock_push:
        inst._default_post_iter_hook(0, 0.5, None)
        mock_push.assert_called_once()
        call_kwargs = mock_push.call_args.kwargs
        assert call_kwargs["episode_length"] is None


# ── viser_handle TypedDict ────────────────────────────────────────────────


def test_viser_handle_fields():
    """ViserHandle 包含所有必需字段."""
    from unitree_viser.render.async_render import ViserHandle, make_viser_handle

    h = make_viser_handle(
        server=MagicMock(),
        scene=MagicMock(),
        env_idx=3,
        mj_model=MagicMock(),
        mj_data=MagicMock(),
        sim=MagicMock(),
    )
    assert h["env_idx"] == 3
    assert "_render_thread" in h
    assert "_render_stop" in h
    assert h["_render_thread"] is None
    assert h["_render_stop"] is None
