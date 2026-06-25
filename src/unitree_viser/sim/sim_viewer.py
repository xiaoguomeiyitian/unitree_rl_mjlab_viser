"""SimViewer - 仿真模式 (无训练) 的 Viser 浏览器.

与 mjlab 自带的 ViserPlayViewer 不同, SimViewer:

- 不继承 play/pause/step 按钮
- 支持命令注入 (GUI 滑块 / DDS 遥控器)
- 支持加载训练好的策略 (checkpoint) 或用 random/zero policy
- 单环境 (num_envs=1)
"""

from __future__ import annotations

import time
from typing import Any, Protocol

import numpy as np
import torch
import viser

from unitree_viser.sim.command_injection import CommandInjector


class PolicyLike(Protocol):
    """策略协议 - 接受 obs tensor 或 dict, 返回 actions tensor."""

    def __call__(self, obs: Any) -> torch.Tensor: ...


def _extract_actor_obs(obs: Any) -> torch.Tensor:
    """从 obs dict 提取 actor 输入张量. obs 本身是 tensor 时原样返回."""
    if isinstance(obs, dict):
        if "actor" in obs:
            return obs["actor"]
        for v in obs.values():
            if isinstance(v, torch.Tensor):
                return v
        raise ValueError(f"obs dict 中找不到 tensor 类型的值: {list(obs.keys())}")
    return obs


def _zero_policy(obs: Any) -> torch.Tensor:
    """Zero policy - 输出零动作 (站立)."""
    actor_obs = _extract_actor_obs(obs)
    n = actor_obs.shape[0]
    return torch.zeros(n, 12, device=actor_obs.device)


def _random_policy(obs: Any, num_actions: int = 12) -> torch.Tensor:
    """Random policy - 输出 [-1, 1] 之间的随机动作.

    Args:
        obs: 观测值
        num_actions: 动作维度 (默认 12 DOF, 适用于 G1/Go2)
    """
    actor_obs = _extract_actor_obs(obs)
    return torch.rand(actor_obs.shape[0], num_actions, device=actor_obs.device) * 2 - 1


def _make_typed_policy(base_policy, num_actions: int):
    """包装 base_policy, 强制输出形状为 (n, num_actions)."""
    def typed_policy(obs: Any) -> torch.Tensor:
        actor_obs = _extract_actor_obs(obs)
        n = actor_obs.shape[0]
        try:
            out = base_policy(obs)
            if out.shape[-1] != num_actions:
                out = out.expand(n, num_actions).contiguous()
        except Exception:
            out = torch.zeros(n, num_actions, device=actor_obs.device)
        return out
    return typed_policy


class SimViewer:
    """仿真模式 Viser viewer (单环境 + 命令注入)."""

    def __init__(
        self,
        env: Any,
        policy: PolicyLike | None = None,
        port: int = 20006,
        env_idx: int = 0,
        inject_commands: bool = True,
        command_name: str = "twist",
        command_source: str = "gui",
        dds_domain: int = 0,
        dds_interface: str = "lo",
        robot_key: str = "go2_0",
        dds_timeout: float = 0.5,
    ) -> None:
        """Args:
            env: ``ManagerBasedRlEnv``
            policy: 可调用对象; ``None`` 表示 zero policy
            port: Viser HTTP/WS 端口
            env_idx: 显示的环境索引
            inject_commands: 是否启用命令注入 GUI
            command_name: 命令 term 名称 (默认 ``twist``)
            command_source: ``gui``/``dds``/``both``
            dds_domain: CycloneDDS 域 ID
            dds_interface: DDS 网络接口
            robot_key: DDS topic 后缀
            dds_timeout: DDS 消息超时秒数
        """
        self._env = env
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
        self._injector_gui: CommandInjector | None = None
        self._injector_dds: Any = None

        self._should_stop = False
        self._step_count = 0
        self._inject_commands = inject_commands
        self._command_name = command_name

        if command_source not in ("gui", "dds", "both"):
            raise ValueError(
                f"command_source must be 'gui', 'dds', or 'both'; got {command_source!r}"
            )
        self._command_source = command_source
        self._dds_domain = dds_domain
        self._dds_interface = dds_interface
        self._robot_key = robot_key
        self._dds_timeout = dds_timeout

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

        if self._inject_commands and self._command_source in ("gui", "both"):
            try:
                self._injector_gui = CommandInjector(
                    server=self._server,
                    env=self._env,
                    command_name=self._command_name,
                )
            except ValueError as e:
                print(f"[SIM] 跳过 GUI 命令注入: {e}")
                self._injector_gui = None

        if self._command_source in ("dds", "both"):
            try:
                from unitree_viser.sim.dds_command_injection import (
                    DdsCommandInjector,
                )
                self._injector_dds = DdsCommandInjector(
                    server=self._server,
                    env=self._env,
                    command_name=self._command_name,
                    dds_domain=self._dds_domain,
                    dds_interface=self._dds_interface,
                    robot_key=self._robot_key,
                    timeout_s=self._dds_timeout,
                )
                self._injector_dds.start()
            except (ValueError, ImportError) as e:
                print(f"[SIM] 跳过 DDS 命令注入: {e}")
                self._injector_dds = None

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

        使用 mj_forward (CPU) 同步渲染数据, 与训练路径 ViserRenderThread 一致.
        避免 mjwarp.get_data_into() 的全量 GPU→CPU 传输, 减少延迟.
        """
        import mujoco as mj

        assert self._scene is not None, "call setup() first"

        obs, _ = self._env.reset()
        sim = self._env.sim
        target_fps = 30.0
        last_render_t = 0.0

        # 创建独立的 mj_data 用于渲染, 避免与训练/仿真线程冲突
        mj_data_render = mj.MjData(sim.mj_model)

        try:
            while not self._should_stop:
                if max_steps is not None and self._step_count >= max_steps:
                    print(f"[SIM] 达到 max_steps={max_steps}, 退出")
                    break

                with torch.inference_mode():
                    actions = self._policy(obs)

                if self._injector_gui is not None:
                    self._injector_gui.inject()
                if self._injector_dds is not None:
                    self._injector_dds.inject()

                obs, reward, terminated, truncated, extras = self._env.step(actions)
                self._step_count += 1

                now = time.time()
                if now - last_render_t >= 1.0 / target_fps:
                    try:
                        # 轻量同步: 拷贝 qpos + mj_forward 计算派生量
                        mj_data_render.qpos[:] = sim.mj_data.qpos
                        mj.mj_forward(sim.mj_model, mj_data_render)
                        self._scene.update_from_mjdata(mj_data_render)
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
