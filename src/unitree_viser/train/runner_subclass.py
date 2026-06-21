"""ViserRunner — MjlabOnPolicyRunner 子类, 加 on_iter 钩子.

为什么需要这个
==============
rsl_rl 的 ``OnPolicyRunner.learn()`` 是一个**内联** for 循环, **没有**任何
回调/钩子机制. mjlab 的 ``MjlabOnPolicyRunner`` 只重写了 ``save/load/onnx-export``,
``learn()`` 是从 rsl_rl 完整继承的.

我们想要在训练时往 Viser 注入数据(折线图、Info 面板), 必须在某个时刻插入代码.
三个选项:

1. (a) **完全复制** learn() 循环 — trainBot 的方案 (~80 行重复)
2. (b) **子类化 + 重写** learn() — 复制主体, 在关键位置插入 1-2 行
3. (c) 包装 env.step() — 复杂, 多次 step/iter 会重复触发

本模块采用 **(b)**: 复制 learn() 主体 (rsl_rl 5.x 约 60 行), 改动:

- 在 ``self.logger.log(...)`` 之后插入 ``_post_iter_hook``
- ``_post_iter_hook`` 列表是用户注册回调的容器
- 同时把 ``TrainingController.wait_if_paused()`` 集成进去
- 保留所有原始功能 (init_at_random_ep_len, distributed, save, log_dir)

升级策略
========
如果 rsl_rl / mjlab 更新改变了 learn() 主体, 需要重新对照同步.
我们把循环的"原版"作为 reference, 改动处用 ``# [VISER]`` 标记方便对比.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import torch

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
    from unitree_viser.train.training_controller import TrainingController


# 钩子函数签名: (iteration: int, env: ManagerBasedRlEnv, loss_dict: dict) -> None
PostIterHook = Callable[[int, "ManagerBasedRlEnv", dict], None]


class ViserRunner:
    """Mixin 类 - 给 MjlabOnPolicyRunner 加 on_iter 钩子 + 暂停支持.

    由于 MjlabOnPolicyRunner 的 ``learn()`` 是从 rsl_rl 继承的, 我们不直接改它,
    而是在 ``ViserRunner.learn()`` 里复制一份原版循环, 加上 on_iter 钩子.
    """

    # 由调用方注入
    viser_gui_state: dict | None = None
    training_controller: "TrainingController | None" = None
    """可选的训练控制器 (暂停/单步/速度). 注入后才生效."""

    _post_iter_hooks: list[PostIterHook] = []

    def register_post_iter_hook(self, hook: PostIterHook) -> None:
        """注册一个 iter 结束后的钩子."""
        self._post_iter_hooks.append(hook)

    def clear_post_iter_hooks(self) -> None:
        self._post_iter_hooks.clear()

    # ── learn() — 复制 rsl_rl 主体, 加 2 处 [VISER] 改动 ────────────────────

    def learn(  # type: ignore[override]
        self,
        num_learning_iterations: int,
        init_at_random_ep_len: bool = False,
    ) -> None:
        """复制 rsl_rl.OnPolicyRunner.learn(), 加 Viser 钩子.

        改动位置 (用 ``# [VISER]`` 标记):

        1. 每个 iter 结束后调用 ``_post_iter_hook``
        2. 暂停/单步控制
        """
        # 初始化 NaN 检查函数
        try:
            from mjlab.utils.torch import check_nan
        except ImportError:
            def check_nan(obs, rewards, dones):  # type: ignore[no-redef]
                pass

        # 运行时导入 torch (本模块顶层的 import 是 TYPE_CHECKING, 避免无 torch 时 import 失败)
        import torch as _torch

        # ── 原版: 随机化初始 episode 长度 ──
        if init_at_random_ep_len:
            self.env.episode_length_buf = _torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        if self.is_distributed:
            self.alg.broadcast_parameters()
        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations

        # 收集最近一次 iter 的指标 (供 Viser 钩子用)
        last_mean_reward: float = 0.0
        last_episode_length: float = 0.0

        for it in range(start_it, total_it):
            start_time = time.time()

            # ── Rollout (原版) ──
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs, rewards, dones = (
                        obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    self.alg.process_env_step(obs, rewards, dones, extras)
                intrinsic_rewards = (
                    self.alg.intrinsic_rewards if self.cfg["algorithm"].get("rnd_cfg") else None
                )
                self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)
                last_mean_reward = float(rewards.mean().item())
                if hasattr(extras, "get"):
                    last_episode_length = float(extras.get("episode_length", last_episode_length))
                collect_time = time.time() - start_time
                start_time = time.time()
                self.alg.compute_returns(obs)

            # ── Update (原版) ──
            loss_dict = self.alg.update()
            learn_time = time.time() - start_time
            self.current_learning_iteration = it

            # ── Log (原版) ──
            rnd_weight = (
                self.alg.rnd.weight
                if self.cfg["algorithm"].get("rnd_cfg")
                else None
            )
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=rnd_weight,
            )

            # ── Save (原版) ──
            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

            # ── [VISER] 训练控制: 暂停/单步 ──
            if self.training_controller is not None:
                should_quit = self.training_controller.wait_if_paused()
                if should_quit:
                    print("[CTRL] 收到退出请求, 停止训练")
                    break

            # ── [VISER] 触发 on_iter 钩子 ──
            if self._post_iter_hooks:
                env_unwrapped = self.env.unwrapped
                for hook in self._post_iter_hooks:
                    try:
                        hook(it, env_unwrapped, loss_dict)
                    except Exception as e:
                        print(f"[VISER] post_iter_hook 失败: {e}")

            # ── [VISER] 默认钩子: 更新 Info 面板 + 折线图 ──
            self._default_post_iter_hook(it, last_mean_reward, last_episode_length)

        # 训练结束
        if self.logger.writer is not None:
            self.save(
                os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt")
            )
            self.logger.stop_logging_writer()

    def _default_post_iter_hook(
        self,
        iteration: int,
        mean_reward: float,
        episode_length: float,
    ) -> None:
        """如果注册了 viser_gui_state, 默认更新 Info 面板和折线图."""
        from unitree_viser.render.viser_setup import (
            push_reward_to_plot,
            update_training_info,
        )

        if self.viser_gui_state is None:
            return

        # 取 render 线程的 FPS (如果有)
        current_fps = None
        render_thread = self.viser_gui_state.get("render_thread")
        if render_thread is not None and hasattr(render_thread, "_render_loop"):
            # 简单估算: 用最近 1 秒的计数
            current_fps = self._sample_render_fps()

        update_training_info(
            self.viser_gui_state,
            iteration=iteration,
            mean_reward=mean_reward,
            current_fps=current_fps,
        )
        push_reward_to_plot(
            self.viser_gui_state,
            iteration=iteration,
            mean_reward=mean_reward,
            episode_length=episode_length,
        )

    # ── FPS 估算 ────────────────────────────────────────────────────────────

    _fps_sample: list[float] | None = None

    def _sample_render_fps(self) -> float | None:
        """估算后台渲染线程的 FPS."""
        if self._fps_sample is None:
            return None
        now = time.time()
        self._fps_sample.append(now)
        # 保留最近 1 秒
        while self._fps_sample and now - self._fps_sample[0] > 1.0:
            self._fps_sample.pop(0)
        if len(self._fps_sample) < 2:
            return None
        return (len(self._fps_sample) - 1) / (self._fps_sample[-1] - self._fps_sample[0])


# ── 工厂: 构造带 Viser 能力的 Runner ───────────────────────────────────────


def make_viser_runner_cls(base_runner_cls: type) -> type:
    """动态创建一个 ViserRunner, 继承 ``base_runner_cls`` (通常是 MjlabOnPolicyRunner).

    这样我们不用复制 MjlabOnPolicyRunner 的 __init__/save/load/onnx-export,
    只需要替换 learn().

    Usage::

        from mjlab.rl.runner import MjlabOnPolicyRunner
        from unitree_viser.train.runner_subclass import make_viser_runner_cls

        ViserRunnerCls = make_viser_runner_cls(MjlabOnPolicyRunner)
        runner = ViserRunnerCls(env, agent_cfg, log_dir, device)
    """
    return type(
        "ViserRunner",
        (ViserRunner, base_runner_cls),
        {
            "__doc__": (
                f"Combined ViserRunner + {base_runner_cls.__name__}. "
                "Adds on_iter hooks and pause/step/speed control."
            ),
        },
    )
