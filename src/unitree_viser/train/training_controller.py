"""训练控制 - 暂停 / 单步 / 速度滑块.

通过 Viser GUI 按钮控制训练循环:

- **Pause/Resume**: 切换 ``pause_event``, runner 在 _post_iter_hook 等待
- **Step**: 临时清除 ``pause_event`` 让 runner 走一个 iter, 然后重新 set
- **Speed**: 0.25x ~ 4x 滑块, 实际效果是让 async render 线程按更慢/更快的频率更新

线程模型
========
- 训练主循环: ``MjlabOnPolicyRunner.learn()`` 在主线程
- Viser server: 在 viser 内部线程 (按 viser 库的实现)
- 我们的 controller: 只持有 ``threading.Event``, 无自己的线程
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import viser


@dataclass
class _ControlState:
    """线程间共享的控制状态."""

    pause_event: threading.Event
    """set 时表示训练暂停; runner 在 _post_iter_hook 中等待这个 event clear"""

    single_step: threading.Event
    """set 时表示"单步请求": runner 应走 1 个 iter, 然后重新 pause"""

    speed_multiplier: float = 1.0
    """渲染速度倍率 (0.25x ~ 4x)"""

    quit_requested: bool = False
    """用户点 Quit 时设为 True, runner 应在下一个 iter 后退出"""


class TrainingController:
    """封装 Viser GUI 与训练循环的暂停/单步/速度控制.

    用法::

        server = viser.ViserServer()
        controller = TrainingController(server=server, fps=10.0)
        # ... 在 ViserRunner._post_iter_hook 中:
        controller.wait_if_paused()
    """

    def __init__(self, server: "viser.ViserServer", fps: float = 10.0) -> None:
        self._server = server
        self._state = _ControlState(
            pause_event=threading.Event(),
            single_step=threading.Event(),
        )
        # 训练开始时不暂停, 让用户先点 Pause 再观察
        self._state.pause_event.clear()
        self._state.single_step.clear()

        self._build_gui(initial_fps=fps)

    def _build_gui(self, initial_fps: float) -> None:
        import viser

        with self._server.gui.add_folder("Training Control"):
            self._pause_button = self._server.gui.add_button(
                "Pause",
                icon=viser.Icon.PLAYER_PAUSE,
            )

            @self._pause_button.on_click
            def _(_) -> None:
                self.toggle_pause()

            self._step_button = self._server.gui.add_button(
                "Step (1 iter)",
                icon=viser.Icon.PLAYER_TRACK_NEXT,
            )

            @self._step_button.on_click
            def _(_) -> None:
                self.request_single_step()

            self._speed_slider = self._server.gui.add_slider(
                "Render Speed (x)",
                min=0.25,
                max=4.0,
                step=0.25,
                initial_value=1.0,
                hint="调整 Viser 渲染频率倍率 (不影响训练速度)",
            )

            @self._speed_slider.on_update
            def _(event) -> None:
                self._state.speed_multiplier = float(event.target.value)

            self._status_html = self._server.gui.add_html(
                f"<div>Target FPS: <b>{initial_fps:.1f}</b></div>"
            )

    # ── 公共 API (供 runner 钩子调用) ──────────────────────────────────────

    def wait_if_paused(self) -> bool:
        """如果处于暂停状态, 阻塞当前线程直到恢复.

        Returns:
            True 如果用户请求退出 (runner 应停止训练), False 继续.
        """
        if self._state.quit_requested:
            return True

        if self._state.single_step.is_set():
            # 单步模式: 走一个 iter, 然后自动重新 pause
            self._state.single_step.clear()
            self._state.pause_event.set()
            return False

        # 暂停时阻塞; clear 后返回
        self._state.pause_event.wait()
        return self._state.quit_requested

    def is_paused(self) -> bool:
        return self._state.pause_event.is_set()

    def toggle_pause(self) -> None:
        if self._state.pause_event.is_set():
            self._state.pause_event.clear()
            self._update_pause_button(paused=False)
            print("[CTRL] 训练已恢复")
        else:
            self._state.pause_event.set()
            self._update_pause_button(paused=True)
            print("[CTRL] 训练已暂停")

    def request_single_step(self) -> None:
        self._state.single_step.set()
        self._state.pause_event.clear()
        print("[CTRL] 单步请求已发出")

    def get_speed_multiplier(self) -> float:
        return self._state.speed_multiplier

    def request_quit(self) -> None:
        self._state.quit_requested = True
        self._state.pause_event.set()  # 解除 wait

    # ── 内部 ────────────────────────────────────────────────────────────────

    def _update_pause_button(self, paused: bool) -> None:
        import viser

        self._pause_button.label = "Resume" if paused else "Pause"
        self._pause_button.icon = (
            viser.Icon.PLAYER_PLAY if paused else viser.Icon.PLAYER_PAUSE
        )
