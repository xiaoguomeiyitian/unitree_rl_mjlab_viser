"""CommandInjector - 从 Viser GUI 注入命令到 env.command_manager.

仿真模式时, 用户通过 Viser UI 设置 vx/vy/wz, 在 ``env.step()`` 之前写入
``env.command_manager._terms[<name>].vel_command_b``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import viser
    from mjlab.envs import ManagerBasedRlEnv


class CommandInjector:
    """把 Viser GUI 滑块映射到 env.command_manager 的内部 buffer."""

    def __init__(
        self,
        server: "viser.ViserServer",
        env: "ManagerBasedRlEnv",
        command_name: str = "twist",
    ) -> None:
        """Args:
            server: viser.ViserServer 实例
            env: mjlab ManagerBasedRlEnv
            command_name: ``env.command_manager._terms`` 中的 key
        """
        self._server = server
        self._env = env
        self._command_name = command_name

        if command_name not in env.command_manager._terms:
            raise ValueError(
                f"Command '{command_name}' not in env.command_manager._terms. "
                f"Available: {list(env.command_manager._terms.keys())}"
            )
        self._term = env.command_manager._terms[command_name]

        self._pending = {
            "vx": 0.0,
            "vy": 0.0,
            "wz": 0.0,
        }
        self._enabled = True

        self._build_gui()

    def _build_gui(self) -> None:
        with self._server.gui.add_folder("Command Injection"):
            self._enable_cb = self._server.gui.add_checkbox(
                "Enable",
                initial_value=True,
                hint="勾选后注入 GUI 设置的命令",
            )

            @self._enable_cb.on_update
            def _(event) -> None:
                self._enabled = bool(event.target.value)

            self._vx_slider = self._server.gui.add_slider(
                "vx (forward, m/s)",
                min=-1.5,
                max=1.5,
                step=0.05,
                initial_value=0.0,
            )

            @self._vx_slider.on_update
            def _(event) -> None:
                self._pending["vx"] = float(event.target.value)

            self._vy_slider = self._server.gui.add_slider(
                "vy (lateral, m/s)",
                min=-1.0,
                max=1.0,
                step=0.05,
                initial_value=0.0,
            )

            @self._vy_slider.on_update
            def _(event) -> None:
                self._pending["vy"] = float(event.target.value)

            self._wz_slider = self._server.gui.add_slider(
                "wz (yaw rate, rad/s)",
                min=-1.5,
                max=1.5,
                step=0.05,
                initial_value=0.0,
            )

            @self._wz_slider.on_update
            def _(event) -> None:
                self._pending["wz"] = float(event.target.value)

            self._stop_button = self._server.gui.add_button("Stop (zero commands)")

            @self._stop_button.on_click
            def _(_) -> None:
                self._vx_slider.value = 0.0
                self._vy_slider.value = 0.0
                self._wz_slider.value = 0.0
                self._pending.update(vx=0.0, vy=0.0, wz=0.0)
                print("[INJECT] 已清零所有命令")

    def inject(self) -> None:
        """在 env.step() 前调用, 把 GUI 设置的值写入 env 的命令 buffer."""
        if not self._enabled:
            return

        import torch as _torch

        if hasattr(self._term, "vel_command_b"):
            v = _torch.tensor(
                [self._pending["vx"], self._pending["vy"], 0.0],
                device=self._env.device,
                dtype=self._term.vel_command_b.dtype,
            )
            self._term.vel_command_b[:] = v.unsqueeze(0).expand(
                self._env.num_envs, 3
            )
        elif hasattr(self._term, "_command_buf"):
            buf = self._term._command_buf
            if buf.shape[-1] >= 3:
                buf[..., 0] = self._pending["vx"]
                buf[..., 1] = self._pending["vy"]
                buf[..., 2] = self._pending["wz"]

    def get_pending(self) -> dict[str, float]:
        """返回当前 pending 的命令 (调试用)."""
        return dict(self._pending)
