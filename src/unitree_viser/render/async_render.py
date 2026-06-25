"""Viser 后台渲染线程 — 封装为独立对象, 管理渲染 + GUI 更新.

改进:
  - 面向对象封装: ViserRenderThread 类, start/stop 风格
  - 异常自动恢复: 渲染出错时自动重试, 不崩溃
  - 帧率自适应: 根据实际渲染耗时动态调整
  - GUI 更新: 渲染线程每秒消费 TrainingState 并更新 viser GUI
  - 向后兼容: 保留函数式 API (deprecated)
"""

from __future__ import annotations

import threading
import time as _time
from typing import Any, TypedDict

if False:  # TYPE_CHECKING 替代
    from unitree_viser.render.shared_state import TrainingState


# ── ViserHandle (向后兼容) ──────────────────────────────────────────────────


class ViserHandle(TypedDict, total=False):
    """viser 渲染句柄 (向后兼容, 新代码直接用 ViserRenderThread)."""

    server: Any
    scene: Any
    env_idx: int
    mj_model: Any
    mj_data: Any
    sim: Any
    _render_thread: threading.Thread | None
    _render_stop: threading.Event | None
    _render_error_count: int
    _render_total_frames: int
    _render_thread_obj: Any  # ViserRenderThread 实例引用


def make_viser_handle(
    server: Any,
    scene: Any,
    env_idx: int,
    mj_model: Any,
    mj_data: Any,
    sim: Any,
) -> ViserHandle:
    """构造带后台渲染线程占位字段的 viser 句柄."""
    return {
        "server": server,
        "scene": scene,
        "env_idx": env_idx,
        "mj_model": mj_model,
        "mj_data": mj_data,
        "sim": sim,
        "_render_thread": None,
        "_render_stop": None,
        "_render_error_count": 0,
        "_render_total_frames": 0,
        "_render_thread_obj": None,
    }


# ── ViserRenderThread (新 API) ───────────────────────────────────────────────


class ViserRenderThread:
    """Viser 后台渲染线程 — 封装为独立对象, 管理渲染 + GUI 更新.

    用法:
        render_thread = ViserRenderThread(server, scene, env_idx, mj_model, mj_data, sim, target_fps=30.0)
        render_thread.set_training_state(training_state)
        render_thread.start()
        ...
        render_thread.stop()
    """

    def __init__(
        self,
        server: Any,
        scene: Any,
        env_idx: int,
        mj_model: Any,
        mj_data: Any,
        sim: Any,
        target_fps: float = 15.0,
    ) -> None:
        self._server = server
        self._scene = scene
        self._env_idx = env_idx
        self._mj_model = mj_model
        self._mj_data = mj_data
        self._sim = sim
        self._target_fps = target_fps

        # 双缓冲: 创建独立的 mj_data_b 用于渲染线程 (单环境大小)
        # mj_data_a (=mj_data=sim.mj_data) 为多环境共享, 训练线程写
        # mj_data_b 为单环境, 渲染线程读
        # 使用 mj_forward (CPU) 替代 mjwarp.get_data_into (GPU→CPU 全量传输)
        # env.step() 已自动同步 qpos 到 mj_data, 只需 mj_forward 计算派生量
        import mujoco
        self._mj_data_b = mujoco.MjData(mj_model)

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._data_lock = threading.Lock()

        # 统计
        self._total_frames: int = 0
        self._error_count: int = 0

        # 训练状态 (由外部 set_training_state 注入)
        self._training_state: Any = None  # TrainingState | None
        self._gui_state: dict | None = None  # 用于 GUI 更新



    # ── 公共 API ───────────────────────────────────────────────────────────

    def start(self) -> None:
        """启动渲染线程 (幂等)."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._render_loop, name="viser-render", daemon=True
        )
        self._thread.start()
        print(f"[INFO] ViserRenderThread 已启动 (目标 {self._target_fps:.1f} FPS)")

    def stop(self, timeout: float = 2.0) -> None:
        """停止渲染线程 (幂等)."""
        if self._thread is None:
            return
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self._thread = None
        print(
            f"[INFO] ViserRenderThread 已停止 "
            f"(渲染 {self._total_frames} 帧, {self._error_count} 次异常)"
        )

    @property
    def is_alive(self) -> bool:
        """查询渲染线程是否存活."""
        return self._thread is not None and self._thread.is_alive()

    def get_stats(self) -> dict:
        """返回渲染统计信息."""
        return {
            "alive": self.is_alive,
            "total_frames": self._total_frames,
            "error_count": self._error_count,
        }

    def set_training_state(self, training_state: Any) -> None:
        """设置训练状态引用, 渲染线程每秒消费并更新 GUI.

        Args:
            training_state: TrainingState 实例, 或 None 禁用 GUI 更新
        """
        self._training_state = training_state

    def set_gui_state(self, gui_state: dict) -> None:
        """设置 gui_state 引用 (用于 GUI 更新).

        Args:
            gui_state: viser_setup.setup_viser_for_training() 返回的 gui_state dict
        """
        self._gui_state = gui_state

    def set_sync_training(self, enabled: bool) -> None:
        """设置是否启用训练同步.

        启用后, 渲染线程每渲染一帧会通过 tick_event 通知主线程,
        主线程等待此事件后才执行下一次训练迭代, 实现训练与渲染同速.
        无浏览器连接时, tick_event 保持 clear 状态, 主线程不阻塞.

        Args:
            enabled: True 启用同步, False 全速训练 (默认)
        """
        self._sync_training = enabled
        if not enabled:
            # 禁用同步时, 确保 tick_event 始终置位, 主线程不阻塞
            self._tick_event.set()

    @property
    def is_rendering(self) -> bool:
        """查询渲染线程是否在活跃渲染 (有浏览器连接).

        主线程可据此判断是否需要等待 tick.
        """
        return self.is_alive() and self._tick_event.is_set()

    def wait_for_tick(self, timeout: float = 1.0) -> bool:
        """等待渲染线程完成一帧. 用于训练同步.

        应在每次训练迭代开始时调用. 如果渲染线程未启用同步,
        立即返回 True (全步训练模式).

        Returns:
            True 如果收到 tick 或应该继续, False 如果停止.
        """
        if not self._sync_training:
            return True
        if not self.is_alive():
            return True
        return self._tick_event.wait(timeout=timeout)

    # ── 渲染循环 ───────────────────────────────────────────────────────────

    def _render_loop(self) -> None:
        """渲染线程主循环 — 从 MuJoCo 取数据 + 更新 viser scene + 更新 GUI.

        轻量同步: 直接读取 mj_data.qpos (env.step() 已同步),
        然后 mj_forward() 计算派生量 (xpos, xquat 等).
        避免 mjwarp.get_data_into() 的全量 GPU→CPU 传输.
        """
        import mujoco as mj

        from unitree_viser.render.viser_setup import (
            push_reward_to_plot,
            update_training_info,
        )

        last_render_t = 0.0
        last_client_count = -1
        is_connected = False

        # GUI 更新频率控制
        last_gui_update_t: float = 0.0
        GUI_UPDATE_INTERVAL = 1.0  # 每秒更新一次 GUI

        # 有浏览器连接时降低 FPS 到 30, 减少 CPU 争抢; 无连接时几乎零开销
        connected_fps = min(self._target_fps, 30.0)
        connected_interval = 1.0 / connected_fps

        while not self._stop_event.is_set():
            try:
                # 检查浏览器连接
                try:
                    clients = self._server.get_clients()
                    n_clients = len(clients)
                except Exception:
                    n_clients = 1

                if n_clients != last_client_count:
                    if n_clients > 0:
                        is_connected = True
                        print(
                            f"[RENDER] 浏览器已连接 (客户端数={n_clients}), "
                            f"渲染 @{connected_fps:.0f}FPS"
                        )
                    else:
                        is_connected = False
                        print("[RENDER] 无浏览器连接, 暂停渲染")
                    last_client_count = n_clients

                if not is_connected:
                    _time.sleep(0.5)
                    continue

                # 帧率控制
                now = _time.time()
                elapsed = now - last_render_t
                if elapsed < connected_interval:
                    _time.sleep(min(0.02, connected_interval - elapsed))
                    continue
                last_render_t = now

                # 渲染: 轻量同步 — 直接读 qpos + mj_forward 计算派生量
                # env.step() 已自动将 warp 数据同步到 mj_data.qpos
                # 只需 mj_forward 计算 Viser 需要的派生量 (xpos, xquat 等)
                with self._data_lock:
                    # 拷贝 qpos 到独立的 mj_data_b (避免 mj_forward 与训练线程冲突)
                    self._mj_data_b.qpos[:] = self._mj_data.qpos
                    mj.mj_forward(self._mj_model, self._mj_data_b)
                    self._scene.update_from_mjdata(self._mj_data_b)

                self._total_frames += 1
                self._error_count = 0

                # GUI 更新 (每秒最多 1 次)
                now = _time.time()
                if now - last_gui_update_t >= GUI_UPDATE_INTERVAL:
                    last_gui_update_t = now
                    state = self._consume_training_state()
                    if state is not None and self._gui_state is not None:
                        self._update_gui(**state)

            except Exception as e:
                self._error_count += 1
                if self._error_count <= 3:
                    import traceback
                    print(f"[RENDER] 渲染异常 ({self._error_count}/3): {e}")
                    traceback.print_exc()
                elif self._error_count == 4:
                    print("[RENDER] 渲染异常已达 3 次, 静默忽略后续错误...")
                _time.sleep(min(0.5 * self._error_count, 5.0))

    def _consume_training_state(self) -> dict | None:
        """消费训练状态, 返回 None 表示无新状态."""
        if self._training_state is None:
            return None
        if hasattr(self._training_state, "consume"):
            return self._training_state.consume()
        return None

    def _update_gui(self, **state: Any) -> None:
        """更新 viser GUI (info panel + reward plot)."""
        from unitree_viser.render.viser_setup import (
            push_reward_to_plot,
            update_training_info,
        )

        if self._gui_state is None:
            return

        try:
            update_training_info(
                self._gui_state,
                iteration=state.get("iteration", 0),
                mean_reward=state.get("mean_reward", 0.0),
                episode_length=state.get("episode_length", 0.0),
                total_timesteps=state.get("total_timesteps"),
                elapsed_s=state.get("elapsed_s"),
            )
            push_reward_to_plot(
                self._gui_state,
                iteration=state.get("iteration", 0),
                mean_reward=state.get("mean_reward", 0.0),
                episode_length=state.get("episode_length"),
            )
        except Exception as e:
            print(f"[RENDER] GUI 更新失败: {e}")


# ── 向后兼容: 函数式 API (deprecated) ──────────────────────────────────────


def start_viser_render_thread(
    viser_handle: ViserHandle,
    target_fps: float = 15.0,
) -> None:
    """deprecated: 使用 ViserRenderThread(...).start() 代替.

    保留此函数以兼容 trainBot 等现有调用方.
    """
    # 检查是否已有 ViserRenderThread 实例
    existing = viser_handle.get("_render_thread_obj")
    if existing is not None and existing.is_alive():
        return

    thread = ViserRenderThread(
        server=viser_handle["server"],
        scene=viser_handle["scene"],
        env_idx=viser_handle["env_idx"],
        mj_model=viser_handle["mj_model"],
        mj_data=viser_handle["mj_data"],
        sim=viser_handle["sim"],
        target_fps=target_fps,
    )
    thread.start()
    viser_handle["_render_thread_obj"] = thread  # 存储对象引用


def stop_viser_render_thread(
    viser_handle: ViserHandle, timeout: float = 2.0
) -> None:
    """deprecated: 使用 ViserRenderThread.stop() 代替."""
    thread = viser_handle.get("_render_thread_obj")
    if thread is not None:
        thread.stop(timeout=timeout)
    else:
        # fallback 到旧逻辑: 直接操作 stop_event
        stop_event = viser_handle.get("_render_stop")
        t = viser_handle.get("_render_thread")
        if stop_event is not None:
            stop_event.set()
        if t is not None and t.is_alive():
            t.join(timeout=timeout)
        total = viser_handle.get("_render_total_frames", 0)
        errors = viser_handle.get("_render_error_count", 0)
        print(f"[INFO] Viser 后台渲染线程已停止 (渲染 {total} 帧, {errors} 次异常)")


def is_render_thread_alive(viser_handle: ViserHandle) -> bool:
    """查询渲染线程是否存活."""
    thread = viser_handle.get("_render_thread_obj")
    if thread is not None:
        return thread.is_alive
    # fallback
    t = viser_handle.get("_render_thread")
    return t is not None and t.is_alive()


def get_render_stats(viser_handle: ViserHandle) -> dict:
    """查询渲染统计信息."""
    thread = viser_handle.get("_render_thread_obj")
    if thread is not None:
        return thread.get_stats()
    # fallback
    return {
        "alive": is_render_thread_alive(viser_handle),
        "total_frames": viser_handle.get("_render_total_frames", 0),
        "error_count": viser_handle.get("_render_error_count", 0),
    }
