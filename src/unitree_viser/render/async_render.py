"""Viser 后台渲染线程 — 避免阻塞训练主循环.

从 trainBot/unitree_rl_mjlab_viser 移植, 仅改 import 路径.

渲染移到独立 daemon 线程, 主训练循环零阻塞.
默认 10 FPS, 无浏览器连接时直接 sleep, 几乎零开销.
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

    min_interval = 1.0 / max(target_fps, 0.1)
    stop_event = threading.Event()
    data_lock = threading.Lock()

    def _render_loop() -> None:
        last_render_t = 0.0
        last_client_count = -1
        while not stop_event.is_set():
            try:
                try:
                    clients = server.get_clients()
                    n_clients = len(clients)
                except Exception:
                    n_clients = 1

                if n_clients != last_client_count:
                    if n_clients > 0:
                        print(
                            f"[RENDER] 浏览器已连接 (客户端数={n_clients}), "
                            f"开始渲染 @{target_fps:.1f}FPS"
                        )
                    else:
                        print(
                            "[RENDER] 无浏览器连接, 暂停渲染 (训练全速运行)"
                        )
                    last_client_count = n_clients

                if n_clients == 0:
                    _time.sleep(0.5)
                    continue

                now = _time.time()
                if now - last_render_t < min_interval:
                    _time.sleep(0.02)
                    continue
                last_render_t = now

                with data_lock:
                    mjwarp.get_data_into(
                        mj_data, mj_model, sim.wp_data, world_id=env_idx
                    )
                    scene.update_from_mjdata(mj_data)
            except Exception:
                _time.sleep(0.1)

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
    print("[INFO] Viser 后台渲染线程已停止")
