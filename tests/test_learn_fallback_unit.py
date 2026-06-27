"""ViserRunner._learn_fallback 测试.

测试 fallback 训练循环 (当 monkey-patch 失败时的备用路径).
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


class _MockAlg:
    """模拟 PPO algorithm."""

    def __init__(self):
        self.learning_rate = 0.001
        self.rnd = MagicMock()
        self.rnd.weight = None

    def act(self, obs):
        return torch.zeros(2, 12)

    def process_env_step(self, obs, rewards, dones, extras):
        pass

    def compute_returns(self, obs):
        pass

    def update(self):
        return {"loss": 0.5}

    def get_policy(self):
        p = MagicMock()
        p.output_std = 0.1
        return p

    def train_mode(self):
        pass

    def broadcast_parameters(self):
        pass


class _MockLogger:
    """模拟 logger."""

    def __init__(self):
        self.writer = None
        self.log_dir = "/tmp/test_logs"
        self._log_calls = []

    def log(self, **kwargs):
        self._log_calls.append(kwargs)

    def process_env_step(self, rewards, dones, extras, intrinsic_rewards):
        pass

    def init_logging_writer(self):
        pass

    def stop_logging_writer(self):
        pass


class _MockEnv:
    """模拟环境."""

    def __init__(self):
        self.num_envs = 2
        self.device = "cpu"
        self.max_episode_length = 100
        self.episode_length_buf = torch.zeros(2, dtype=torch.long)
        self.unwrapped = MagicMock()

    def get_observations(self):
        return torch.zeros(2, 47)

    def step(self, actions):
        obs = torch.zeros(2, 47)
        rewards = torch.ones(2)
        dones = torch.zeros(2, dtype=torch.bool)
        extras = {}
        return obs, rewards, dones, extras


class _BaseRunner:
    """模拟 MjlabOnPolicyRunner."""

    def __init__(self):
        self.env = _MockEnv()
        self.alg = _MockAlg()
        self.logger = _MockLogger()
        self.device = "cpu"
        self.cfg = {
            "num_steps_per_env": 2,
            "algorithm": {},
            "check_for_nan": False,
            "save_interval": 100,
            "clip_actions": 1.0,
        }
        self.current_learning_iteration = 0
        self.is_distributed = False

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        """模拟 super().learn() 抛异常以触发 fallback."""
        raise RuntimeError("simulated monkey-patch failure")

    def save(self, path):
        pass


def test_learn_fallback_runs_without_error():
    """_learn_fallback 能正常执行训练循环."""
    from unitree_viser.train.runner_subclass import ViserRunner, make_viser_runner_cls

    Cls = make_viser_runner_cls(_BaseRunner)
    runner = Cls()

    # 不应抛异常
    runner._learn_fallback(num_learning_iterations=2, init_at_random_ep_len=False)

    # 验证 logger.log 被调用
    assert len(runner.logger._log_calls) == 2


def test_learn_fallback_calls_save_at_interval():
    """_learn_fallback 在 save_interval 时保存."""
    from unitree_viser.train.runner_subclass import ViserRunner, make_viser_runner_cls

    Cls = make_viser_runner_cls(_BaseRunner)
    runner = Cls()
    runner.cfg["save_interval"] = 1  # 每次迭代都保存
    runner.logger.writer = MagicMock()  # 启用保存

    with patch.object(runner, "save") as mock_save:
        runner._learn_fallback(num_learning_iterations=2)
        # save 在 it % save_interval == 0 时调用 (it=0, 1) + 结束时 1 次 = 3
        assert mock_save.call_count >= 2


def test_learn_fallback_updates_current_iteration():
    """_learn_fallback 更新 current_learning_iteration."""
    from unitree_viser.train.runner_subclass import ViserRunner, make_viser_runner_cls

    Cls = make_viser_runner_cls(_BaseRunner)
    runner = Cls()
    runner.current_learning_iteration = 0

    runner._learn_fallback(num_learning_iterations=3)

    # 循环 range(0, 3) → it = 0, 1, 2, 最后 current_learning_iteration = 2
    assert runner.current_learning_iteration == 2


def test_learn_fallback_respects_post_iter_hooks():
    """_learn_fallback 调用 post_iter_hooks."""
    from unitree_viser.train.runner_subclass import ViserRunner, make_viser_runner_cls

    Cls = make_viser_runner_cls(_BaseRunner)
    runner = Cls()
    hook = MagicMock()
    runner.register_post_iter_hook(hook)

    runner._learn_fallback(num_learning_iterations=2)

    # hook 应被调用 2 次
    assert hook.call_count == 2


def test_learn_fallback_handles_hook_exception():
    """hook 异常不中断训练."""
    from unitree_viser.train.runner_subclass import ViserRunner, make_viser_runner_cls

    Cls = make_viser_runner_cls(_BaseRunner)
    runner = Cls()

    def bad_hook(it, env, loss_dict):
        raise RuntimeError("hook error")

    runner.register_post_iter_hook(bad_hook)

    # 不应抛异常
    runner._learn_fallback(num_learning_iterations=2)
