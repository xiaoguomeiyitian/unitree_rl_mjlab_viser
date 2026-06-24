"""ViserRunner — 通过 monkey-patch 给任意 Runner 注入 on_iter 钩子.

v2 改进: 不再完整复制 rsl_rl.OnPolicyRunner.learn(), 而是 monkey-patch
``self.logger.log`` 在原始 log 之后触发 hooks. 上游 rsl_rl 升级时自动兼容.
如果 monkey-patch 失败, 会 fallback 到 v1 的完整复制模式.
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


PostIterHook = Callable[[int, "ManagerBasedRlEnv", dict], None]


class ViserRunner:
    """Mixin 类 - 给任意 Runner 加 on_iter 钩子 + 暂停支持.

    使用方式:
        ViserRunnerCls = make_viser_runner_cls(base_runner_cls)
        runner = ViserRunnerCls(env, agent_cfg, log_dir, device)
        runner.viser_gui_state = gui_state
        runner.register_post_iter_hook(my_hook)
    """

    viser_gui_state: dict | None = None
    training_controller: "TrainingController | None" = None
    _post_iter_hooks: list[PostIterHook] = []

    def register_post_iter_hook(self, hook: PostIterHook) -> None:
        """注册一个 iter 结束后的钩子."""
        self._post_iter_hooks.append(hook)

    def clear_post_iter_hooks(self) -> None:
        self._post_iter_hooks.clear()

    # ── learn() — monkey-patch 方式 ─────────────────────────────────────────

    def learn(  # type: ignore[override]
        self,
        num_learning_iterations: int,
        init_at_random_ep_len: bool = False,
    ) -> None:
        """在原始 learn() 基础上注入 Viser 钩子 (monkey-patch 方式)."""
        # 保存原始 log 方法
        original_log = self.logger.log
        viser_self = self

        def patched_log(**kwargs) -> None:
            original_log(**kwargs)
            it = kwargs.get("it", 0)
            loss_dict = kwargs.get("loss_dict", {})
            viser_self._trigger_post_iter_hooks(it, loss_dict)

        self.logger.log = patched_log  # type: ignore[assignment]

        try:
            # 调用父类的原始 learn()
            super().learn(num_learning_iterations, init_at_random_ep_len)
        except Exception:
            # monkey-patch 失败时 fallback 到 v1 完整复制
            self.logger.log = original_log  # type: ignore[assignment]
            self._learn_fallback(num_learning_iterations, init_at_random_ep_len)
            return
        finally:
            self.logger.log = original_log  # type: ignore[assignment]

    def _trigger_post_iter_hooks(self, iteration: int, loss_dict: dict) -> None:
        """触发所有注册的 post_iter hooks."""
        self._default_post_iter_hook(iteration, 0.0, 0.0)

        if self.training_controller is not None:
            should_quit = self.training_controller.wait_if_paused()
            if should_quit:
                return

        if self._post_iter_hooks:
            env_unwrapped = self.env.unwrapped
            for hook in self._post_iter_hooks:
                try:
                    hook(iteration, env_unwrapped, loss_dict)
                except Exception as e:
                    print(f"[VISER] post_iter_hook 失败: {e}")

    def _default_post_iter_hook(
        self,
        iteration: int,
        mean_reward: float,
        episode_length: float,
    ) -> None:
        """如果注册了 viser_gui_state, 更新 Info 面板和折线图."""
        from unitree_viser.render.viser_setup import (
            push_reward_to_plot,
            update_training_info,
        )

        if self.viser_gui_state is None:
            return

        update_training_info(
            self.viser_gui_state,
            iteration=iteration,
            mean_reward=mean_reward,
            current_fps=None,
        )
        push_reward_to_plot(
            self.viser_gui_state,
            iteration=iteration,
            mean_reward=mean_reward,
            episode_length=episode_length,
        )

    def _learn_fallback(
        self,
        num_learning_iterations: int,
        init_at_random_ep_len: bool = False,
    ) -> None:
        """Fallback: 完整复制 rsl_rl.OnPolicyRunner.learn() (v1 逻辑)."""
        try:
            from mjlab.utils.torch import check_nan
        except ImportError:
            def check_nan(obs, rewards, dones):  # type: ignore[no-redef]
                pass

        import torch as _torch

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

        last_mean_reward: float = 0.0
        last_episode_length: float = 0.0

        for it in range(start_it, total_it):
            start_time = time.time()

            with _torch.inference_mode():
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

            loss_dict = self.alg.update()
            learn_time = time.time() - start_time
            self.current_learning_iteration = it

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

            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

            if self.training_controller is not None:
                should_quit = self.training_controller.wait_if_paused()
                if should_quit:
                    print("[CTRL] 收到退出请求, 停止训练")
                    break

            if self._post_iter_hooks:
                env_unwrapped = self.env.unwrapped
                for hook in self._post_iter_hooks:
                    try:
                        hook(it, env_unwrapped, loss_dict)
                    except Exception as e:
                        print(f"[VISER] post_iter_hook 失败: {e}")

            self._default_post_iter_hook(it, last_mean_reward, last_episode_length)

        if self.logger.writer is not None:
            self.save(
                os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt")
            )
            self.logger.stop_logging_writer()


def make_viser_runner_cls(base_runner_cls: type) -> type:
    """动态创建一个 ViserRunner, 继承 ``base_runner_cls``."""
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
