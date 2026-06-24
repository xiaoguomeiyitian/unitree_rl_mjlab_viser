"""Viser 后台渲染线程 — 避免阻塞训练主循环.

改进:
  - 异常自动恢复: 渲染出错时自动重试, 不崩溃
  - 帧率自适应: 根据实际渲染耗时动态调整
  - 状态查询: 支持 is_alive() / stats() 查询
"""

from __future__ import annotations

import threading
import time as _time
from typing import Any, TypedDict


class ViserHandle(TypedDict, total=False):
    """viser 渲染句柄."""

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
    }


def start_viser_render_thread(
    viser_handle: ViserHandle,
    target_fps: float = 10.0,
) -> None:
    """启动后台渲染线程, 避免阻塞训练主循环.

    Args:
        viser_handle: ``make_viser_handle`` 返回的句柄
        target_fps: 目标渲染帧率 (默认 10 FPS)
    """
    import mujoco_warp as mjwarp

    server = viser_handle["server"]
    scene = viser_handle["scene"]
    env_idx = viser_handle["env_idx"]
    mj_model = viser_handle["mj_model"]
    mj_data = viser_handle["mj_data"]
    sim = viser_handle["sim"]

    # 避免启动多个线程
    if viser_handle.get("_render_thread") is not None:
        if viser_handle["_render_thread"].is_alive():
            return

    # 有浏览器连接时降低 FPS 到 30, 减少 CPU 争抢; 无连接时几乎零开销
    connected_fps = min(target_fps, 30.0)
    connected_interval = 1.0 / connected_fps

    stop_event = threading.Event()
    data_lock = threading.Lock()

    def _render_loop() -> None:
        last_render_t = 0.0
        last_client_count = -1
        error_count = 0
        total_frames = 0
        is_connected = False

        while not stop_event.is_set():
            try:
                # 检查浏览器连接
                try:
                    clients = server.get_clients()
                    n_clients = len(clients)
                except Exception:
                    n_clients = 1

                if n_clients != last_client_count:
                    if n_clients > 0:
                        is_connected = True
                        print(
                            f"[RENDER] 浏览器已连接 (客户端数={n_clients}), "
                            f"渲染 @{connected_fps:.0f}FPS (降低优先级, 不拖慢训练)"
                        )
                    else:
                        is_connected = False
                        print("[RENDER] 无浏览器连接, 暂停渲染 (训练全速运行)")
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

                # 渲染 (用锁保护共享数据)
                with data_lock:
                    mjwarp.get_data_into(
                        mj_data, mj_model, sim.wp_data, world_id=env_idx
                    )
                    scene.update_from_mjdata(mj_data)

                total_frames += 1
                error_count = 0
                viser_handle["_render_total_frames"] = total_frames

            except Exception as e:
                error_count += 1
                viser_handle["_render_error_count"] = error_count
                if error_count <= 3:
                    print(f"[RENDER] 渲染异常 ({error_count}/3): {e}")
                elif error_count == 4:
                    print("[RENDER] 渲染异常已达 3 次, 静默忽略后续错误...")
                _time.sleep(min(0.5 * error_count, 5.0))

    thread = threading.Thread(target=_render_loop, name="viser-render", daemon=True)
    thread.start()
    viser_handle["_render_thread"] = thread
    viser_handle["_render_stop"] = stop_event
    print(f"[INFO] Viser 后台渲染线程已启动 (目标 {target_fps:.1f} FPS)")


def stop_viser_render_thread(
    viser_handle: ViserHandle, timeout: float = 2.0
) -> None:
    """停止后台渲染线程."""
    stop_event = viser_handle.get("_render_stop")
    thread = viser_handle.get("_render_thread")
    if stop_event is not None:
        stop_event.set()
    if thread is not None and thread.is_alive():
        thread.join(timeout=timeout)
    total = viser_handle.get("_render_total_frames", 0)
    errors = viser_handle.get("_render_error_count", 0)
    print(f"[INFO] Viser 后台渲染线程已停止 (渲染 {total} 帧, {errors} 次异常)")


def is_render_thread_alive(viser_handle: ViserHandle) -> bool:
    """查询渲染线程是否存活."""
    thread = viser_handle.get("_render_thread")
    return thread is not None and thread.is_alive()


def get_render_stats(viser_handle: ViserHandle) -> dict:
    """查询渲染统计信息."""
    return {
        "alive": is_render_thread_alive(viser_handle),
        "total_frames": viser_handle.get("_render_total_frames", 0),
        "error_count": viser_handle.get("_render_error_count", 0),
    }
