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
from typing import Any, Callable, Protocol

import numpy as np
import torch
import viser

from unitree_viser.sim.command_injection import CommandInjector


class PolicyLike(Protocol):
    """策略协议 - 接受 obs tensor, 返回 actions tensor."""

    def __call__(self, obs: torch.Tensor) -> torch.Tensor: ...


def _zero_policy(obs: torch.Tensor) -> torch.Tensor:
    """Zero policy - 输出零动作 (站立)."""
    n = obs.shape[0]
    # 默认 12 DOF (Go2) - 不够就用 n_envs * 0
    return torch.zeros(n, 1, device=obs.device)


def _random_policy(obs: torch.Tensor) -> torch.Tensor:
    """Random policy - 输出 [-1, 1] 之间的随机动作."""
    return torch.rand(obs.shape[0], 1, device=obs.device) * 2 - 1


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
        self._policy = policy or _zero_policy
        self._port = port
        self._env_idx = env_idx

        # Viser 资源
        self._server: viser.ViserServer | None = None
        self._scene: Any = None
        self._injector: CommandInjector | None = None

        # 状态
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

        # 命令注入
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

        # 仿真控制按钮 (Stop / Reset)
        self._build_sim_control_gui()

        # 默认相机
        @self._server.on_client_connect
        def _on_connect(client: viser.ClientHandle) -> None:
            client.camera.position = np.array([3.0, -3.0, 2.0])
            client.camera.look_at = np.array([0.0, 0.0, 0.3])

    def _build_sim_control_gui(self) -> None:
        assert self._server is not None
        with self._server.gui.add_folder("Sim Control"):
            # Reset
            reset_button = self._server.gui.add_button(
                "Reset Environment",
                icon=viser.Icon.REFRESH,
            )

            @reset_button.on_click
            def _(_) -> None:
                self._env.reset()
                self._step_count = 0
                print("[SIM] 环境已重置")

            # Quit
            quit_button = self._server.gui.add_button(
                "Quit",
                icon=viser.Icon.X,
            )

            @quit_button.on_click
            def _(_) -> None:
                self._should_stop = True
                print("[SIM] 收到退出请求")

            # 状态显示
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

        # 初始 obs
        obs = self._env.get_observations()
        sim = self._env.sim
        target_fps = 30.0
        last_render_t = 0.0

        try:
            while not self._should_stop:
                if max_steps is not None and self._step_count >= max_steps:
                    print(f"[SIM] 达到 max_steps={max_steps}, 退出")
                    break

                # 1. 计算动作
                with torch.inference_mode():
                    actions = self._policy(obs)

                # 2. 注入命令 (在 env.step 之前)
                if self._injector is not None:
                    self._injector.inject()

                # 3. 推进 env
                obs, reward, terminated, truncated, extras = self._env.step(actions)
                self._step_count += 1

                # 4. 限流推送 3D 状态
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

                # 5. 更新状态 HTML
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
