"""SimViewer - 仿真模式 (无训练) 的 Viser 浏览器.

与 mjlab 自带的 ``ViserPlayViewer`` 不同, SimViewer:

- **不**继承 play/pause/step 按钮 (用户已在 sim 模式, 持续 stepping)
- **支持**命令注入 (从 Viser GUI 控制 vx/vy/wz)
- **支持**加载训练好的策略 (checkpoint) 或用 random/zero policy
- **单环境** (num_envs=1, 仿真不需要 batch)

用法::

    viewer = SimViewer(env=env, policy=policy, port=20006)
    viewer.setup()
    viewer.run()
"""

from __future__ import annotations

import time
from typing import Any, Protocol

import numpy as np
import torch
import viser

from unitree_viser.sim.command_injection import CommandInjector


class PolicyLike(Protocol):
    """策略协议 - 接受 obs tensor 或 dict, 返回 actions tensor.

    接受 dict 是为了兼容 mjlab 的 ManagerBasedRlEnv.reset()/step() 返回的
    嵌套 dict 观测 (含 'actor' / 'critic' 等子项).
    """


    def __call__(self, obs: Any) -> torch.Tensor: ...


def _extract_actor_obs(obs: Any) -> torch.Tensor:
    """从 obs dict 提取 actor 输入张量. obs 本身是 tensor 时原样返回."""
    if isinstance(obs, dict):
        if "actor" in obs:
            return obs["actor"]
        # 退化: 取第一个 tensor 值
        for v in obs.values():
            if isinstance(v, torch.Tensor):
                return v
        raise ValueError(f"obs dict 中找不到 tensor 类型的值: {list(obs.keys())}")
    return obs


def _zero_policy(obs: Any) -> torch.Tensor:
    """Zero policy - 输出零动作 (站立).

    不能用 env.action_manager.total_action_dim, 因为这函数不在 env 上下文里.
    调用方负责传入正确的 num_actions. 默认 12 维 (Unitree Go2 12 DOF).
    """
    actor_obs = _extract_actor_obs(obs)
    n = actor_obs.shape[0]
    # 默认 12 DOF; 真实维度由 _make_typed_policy 包装时决定
    return torch.zeros(n, 12, device=actor_obs.device)


def _random_policy(obs: Any) -> torch.Tensor:
    """Random policy - 输出 [-1, 1] 之间的随机动作 (默认 12 DOF)."""
    actor_obs = _extract_actor_obs(obs)
    return torch.rand(actor_obs.shape[0], 12, device=actor_obs.device) * 2 - 1


def _make_typed_policy(base_policy, num_actions: int):
    """包装 base_policy, 强制输出形状为 (n, num_actions)."""
    def typed_policy(obs: Any) -> torch.Tensor:
        actor_obs = _extract_actor_obs(obs)
        n = actor_obs.shape[0]
        # 调 base, 然后 reshape
        try:
            out = base_policy(obs)
            if out.shape[-1] != num_actions:
                out = out.expand(n, num_actions).contiguous()
        except Exception:
            out = torch.zeros(n, num_actions, device=actor_obs.device)
        return out
    return typed_policy


class SimViewer:
    """简单的仿真模式 Viser viewer (单环境 + 命令注入).

    主循环::

        while not should_stop:
            actions = policy(obs)
            injector.inject()                  # 写入 vel_command_b
            obs, reward, terminated, truncated, extras = env.step(actions)
            sim_scene_update(...)             # 推送新状态到 Viser
    """

    def __init__(
        self,
        env: Any,
        policy: PolicyLike | None = None,
        port: int = 20006,
        env_idx: int = 0,
        inject_commands: bool = True,
        command_name: str = "twist",
    ) -> None:
        """Args:
            env: ``ManagerBasedRlEnv`` (通常用 ``env.unwrapped``)
            policy: 可调用对象; ``None`` 表示 zero policy
            port: Viser HTTP/WS 端口
            env_idx: 显示的环境索引
            inject_commands: 是否启用命令注入 GUI
            command_name: 命令 term 名称 (默认 ``twist``)
        """
        self._env = env
        # 包装 policy, 强制输出形状 (n, num_actions)
        raw_policy = policy or _zero_policy
        try:
            num_actions = int(self._env.action_manager.total_action_dim)
        except Exception:
            num_actions = 12
        self._policy = _make_typed_policy(raw_policy, num_actions)
        self._port = port
        self._env_idx = env_idx

        self._server: viser.ViserServer | None = None
        self._scene: Any = None
        self._injector: CommandInjector | None = None

        self._should_stop = False
        self._step_count = 0
        self._inject_commands = inject_commands
        self._command_name = command_name

    def setup(self) -> None:
        """初始化 Viser 服务器 + 场景 + 注入器."""
        from mjlab.viewer.viser.scene import ViserMujocoScene

        sim = self._env.sim
        self._server = viser.ViserServer(host="0.0.0.0", port=self._port, label="Sim")
        self._scene = ViserMujocoScene.create(
            server=self._server,
            mj_model=sim.mj_model,
            num_envs=self._env.num_envs,
        )
        self._scene.env_idx = self._env_idx
        self._scene.camera_tracking_enabled = True
        self._scene.create_visualization_gui(
            camera_distance=3.0,
            camera_azimuth=45.0,
            camera_elevation=20.0,
        )

        if self._inject_commands:
            try:
                self._injector = CommandInjector(
                    server=self._server,
                    env=self._env,
                    command_name=self._command_name,
                )
            except ValueError as e:
                print(f"[SIM] 跳过命令注入: {e}")
                self._injector = None

        # 仿真控制 GUI
        self._build_sim_control_gui()

        @self._server.on_client_connect
        def _on_connect(client: viser.ClientHandle) -> None:
            client.camera.position = np.array([3.0, -3.0, 2.0])
            client.camera.look_at = np.array([0.0, 0.0, 0.3])

    def _build_sim_control_gui(self) -> None:
        assert self._server is not None
        with self._server.gui.add_folder("Sim Control"):
            reset_button = self._server.gui.add_button(
                "Reset Environment",
                icon=viser.Icon.REFRESH,
            )

            @reset_button.on_click
            def _(_) -> None:
                self._env.reset()
                self._step_count = 0
                print("[SIM] 环境已重置")

            quit_button = self._server.gui.add_button(
                "Quit",
                icon=viser.Icon.X,
            )

            @quit_button.on_click
            def _(_) -> None:
                self._should_stop = True
                print("[SIM] 收到退出请求")

            self._status_html = self._server.gui.add_html(
                "<div>Steps: <b>0</b></div>"
            )

    def run(self, max_steps: int | None = None) -> None:
        """主循环 - 持续 stepping + 推送到 Viser.

        Args:
            max_steps: 最多多少步 (None 表示无限, 直到用户 Quit)
        """
        import mujoco_warp as mjwarp

        assert self._scene is not None, "call setup() first"

        obs, _ = self._env.reset()
        sim = self._env.sim
        target_fps = 30.0
        last_render_t = 0.0

        try:
            while not self._should_stop:
                if max_steps is not None and self._step_count >= max_steps:
                    print(f"[SIM] 达到 max_steps={max_steps}, 退出")
                    break

                with torch.inference_mode():
                    actions = self._policy(obs)

                if self._injector is not None:
                    self._injector.inject()

                obs, reward, terminated, truncated, extras = self._env.step(actions)
                self._step_count += 1

                # 限流推送 3D 状态
                now = time.time()
                if now - last_render_t >= 1.0 / target_fps:
                    try:
                        mjwarp.get_data_into(
                            sim.mj_data,
                            sim.mj_model,
                            sim.wp_data,
                            world_id=self._env_idx,
                        )
                        self._scene.update_from_mjdata(sim.mj_data)
                    except Exception as e:
                        print(f"[SIM] 渲染失败: {e}")
                    last_render_t = now

                if self._step_count % 10 == 0:
                    self._update_status_html(reward, terminated, truncated)

        except KeyboardInterrupt:
            print("\n[SIM] 用户中断 (Ctrl+C)")

        print(f"[SIM] 仿真结束, 总步数: {self._step_count}")

    def _update_status_html(
        self,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        if self._status_html is None:
            return
        mean_reward = float(reward.mean().item()) if reward.numel() > 0 else 0.0
        n_done = int((terminated | truncated).sum().item())
        self._status_html.content = (
            "<div style='font-family:monospace;'>"
            f"Steps: <b>{self._step_count}</b><br/>"
            f"Mean Reward: <b>{mean_reward:.3f}</b><br/>"
            f"Done envs: <b>{n_done}/{self._env.num_envs}</b>"
            "</div>"
        )

    def close(self) -> None:
        if self._server is not None:
            try:
                self._server.stop()
            except Exception:
                pass
            self._server = None


# 导出
__all__ = ["SimViewer", "PolicyLike", "_zero_policy", "_random_policy"]
