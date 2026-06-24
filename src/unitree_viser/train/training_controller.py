"""训练控制 - 暂停 / 单步 / 速度滑块."""

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
    single_step: threading.Event
    wakeup_event: threading.Event
    speed_multiplier: float = 1.0
    quit_requested: bool = False


class TrainingController:
    """封装 Viser GUI 与训练循环的暂停/单步/速度控制."""

    def __init__(self, server: "viser.ViserServer", fps: float = 30.0) -> None:
        self._server = server
        self._state = _ControlState(
            pause_event=threading.Event(),
            single_step=threading.Event(),
            wakeup_event=threading.Event(),
        )
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
                hint="调整 Viser 渲染频率倍率",
            )

            @self._speed_slider.on_update
            def _(event) -> None:
                self._state.speed_multiplier = float(event.target.value)

            self._status_html = self._server.gui.add_html(
                f"<div>Target FPS: <b>{initial_fps:.1f}</b></div>"
            )

    def wait_if_paused(self) -> bool:
        """如果处于暂停状态, 阻塞直到恢复.

        Returns:
            True 如果用户请求退出.
        """
        if self._state.quit_requested:
            return True

        if self._state.single_step.is_set():
            self._state.single_step.clear()
            self._state.pause_event.set()
            return False

        while self._state.pause_event.is_set() and not self._state.quit_requested:
            self._state.wakeup_event.wait(timeout=0.05)
            self._state.wakeup_event.clear()
        return self._state.quit_requested

    def is_paused(self) -> bool:
        return self._state.pause_event.is_set()

    def toggle_pause(self) -> None:
        if self._state.pause_event.is_set():
            self._state.pause_event.clear()
            self._state.wakeup_event.set()
            self._update_pause_button(paused=False)
            print("[CTRL] 训练已恢复")
        else:
            self._state.pause_event.set()
            self._update_pause_button(paused=True)
            print("[CTRL] 训练已暂停")

    def request_single_step(self) -> None:
        self._state.single_step.set()
        self._state.pause_event.clear()
        self._state.wakeup_event.set()
        print("[CTRL] 单步请求已发出")

    def get_speed_multiplier(self) -> float:
        return self._state.speed_multiplier

    def request_quit(self) -> None:
        self._state.quit_requested = True
        self._state.pause_event.set()
        self._state.wakeup_event.set()

    def _update_pause_button(self, paused: bool) -> None:
        import viser

        self._pause_button.label = "Resume" if paused else "Pause"
        self._pause_button.icon = (
            viser.Icon.PLAYER_PLAY if paused else viser.Icon.PLAYER_PAUSE
        )
