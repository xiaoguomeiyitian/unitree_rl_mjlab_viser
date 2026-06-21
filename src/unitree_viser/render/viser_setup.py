"""Viser 服务器 + 场景 + GUI 初始化工厂.

负责:

1. 创建 ``viser.ViserServer`` (HTTP/WS 服务, 浏览器访问)
2. 创建 ``ViserMujocoScene`` (3D 场景句柄)
3. 注册 GUI 控件 (Info 面板, 训练控制, 折线图)
4. 返回 ``ViserHandle`` 给 ``async_render`` 线程使用

设计
====
- 这是训练模式的 viewer, **不**驱动 ``env.step()`` (由 rsl_rl runner 驱动).
- 只读取 ``env.sim`` 状态推送到浏览器, 不修改 env.
- 训练控制(暂停/单步/速度)通过 ``TrainingController`` 注入到 runner 钩子.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import viser

from unitree_viser.render.async_render import ViserHandle, make_viser_handle
from unitree_viser.render.term_plots import ViserTermPlotter

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.viser.scene import ViserMujocoScene


def _create_viser_server(port: int, label: str = "RL Training") -> viser.ViserServer:
    """创建 Viser 服务器, 自动绑定到 0.0.0.0."""
    return viser.ViserServer(
        host="0.0.0.0",
        port=port,
        label=label,
    )


def _create_scene(
    server: viser.ViserServer, env: "ManagerBasedRlEnv"
) -> "ViserMujocoScene":
    """创建 3D 场景."""
    from mjlab.viewer.viser.scene import ViserMujocoScene

    sim = env.sim
    return ViserMujocoScene.create(
        server=server,
        mj_model=sim.mj_model,
        num_envs=env.num_envs,
    )


def setup_viser_for_training(
    env: "ManagerBasedRlEnv",
    port: int = 20006,
    env_idx: int = 0,
    enable_control: bool = False,
    fps: float = 10.0,
) -> tuple[viser.ViserServer, "ViserMujocoScene", ViserHandle, dict[str, Any]]:
    """初始化训练模式的 Viser 可视化.

    Args:
        env: mjlab ``ManagerBasedRlEnv`` 实例 (通常用 ``env.unwrapped``)
        port: HTTP/WS 端口 (浏览器访问 ``http://localhost:<port>``)
        env_idx: 默认显示的环境索引
        enable_control: 是否启用训练控制面板 (暂停/单步/速度)
        fps: 异步渲染线程目标帧率

    Returns:
        ``(server, scene, viser_handle, gui_state)`` 四元组

        - ``server``: ``viser.ViserServer`` 实例
        - ``scene``: ``ViserMujocoScene`` 实例
        - ``viser_handle``: 传给 ``start_viser_render_thread``
        - ``gui_state``: dict 含 ``reward_plotter``, ``info_html``, ``controller`` 等
    """
    sim = env.sim

    # 1. 服务器 + 场景
    server = _create_viser_server(port=port, label="Unitree RL Training")
    scene = _create_scene(server, env)
    scene.env_idx = env_idx
    scene.camera_tracking_enabled = True

    # 2. 视口/环境选择 GUI
    scene.create_visualization_gui(
        camera_distance=3.0,
        camera_azimuth=45.0,
        camera_elevation=20.0,
    )

    # 3. Info 面板 (HTML)
    gui_state: dict[str, Any] = {}

    with server.gui.add_folder("Training Info"):
        info_html = server.gui.add_html(
            "<div style='font-family:monospace;'>"
            "Iteration: <b>--</b><br/>"
            "Mean Reward: <b>--</b><br/>"
            "FPS: <b>--</b>"
            "</div>"
        )
        # 折线图占位 (按需创建, 避免启动慢)
        reward_plotter = ViserTermPlotter(
            server=server,
            term_names=["Mean Reward", "Episode Length"],
            name="Training",
        )

    gui_state["info_html"] = info_html
    gui_state["reward_plotter"] = reward_plotter

    # 4. 训练控制 (可选)
    if enable_control:
        from unitree_viser.train.training_controller import TrainingController

        controller = TrainingController(server=server, fps=fps)
        gui_state["controller"] = controller
    else:
        gui_state["controller"] = None

    # 5. 默认相机
    @server.on_client_connect
    def _on_connect(client: viser.ClientHandle) -> None:
        client.camera.position = np.array([3.0, -3.0, 2.0])
        client.camera.look_at = np.array([0.0, 0.0, 0.3])

    # 6. 构造 ViserHandle (供后台渲染线程使用)
    viser_handle = make_viser_handle(
        server=server,
        scene=scene,
        env_idx=env_idx,
        mj_model=sim.mj_model,
        mj_data=sim.mj_data,
        sim=sim,
    )

    return server, scene, viser_handle, gui_state


def update_training_info(
    gui_state: dict[str, Any],
    iteration: int,
    mean_reward: float | None,
    current_fps: float | None,
    extra_lines: list[str] | None = None,
) -> None:
    """更新 Info 面板的 HTML 内容.

    Args:
        gui_state: ``setup_viser_for_training`` 返回的 ``gui_state``
        iteration: 当前训练迭代次数
        mean_reward: 最近一次迭代的平均 reward
        current_fps: 当前渲染 FPS
        extra_lines: 额外的 HTML 行 (可选)
    """
    info_html = gui_state.get("info_html")
    if info_html is None:
        return

    lines = [
        f"Iteration: <b>{iteration}</b>",
        f"Mean Reward: <b>{mean_reward:.3f}</b>" if mean_reward is not None else "Mean Reward: <b>--</b>",
        f"Render FPS: <b>{current_fps:.1f}</b>" if current_fps is not None else "Render FPS: <b>--</b>",
    ]
    if extra_lines:
        lines.extend(extra_lines)

    info_html.content = (
        "<div style='font-family:monospace; font-size:0.9em;'>"
        + "<br/>".join(lines)
        + "</div>"
    )


def push_reward_to_plot(
    gui_state: dict[str, Any],
    iteration: int,
    mean_reward: float,
    episode_length: float | None = None,
) -> None:
    """把当前 iter 的 reward/length 推入 ViserTermPlotter."""
    plotter = gui_state.get("reward_plotter")
    if plotter is None:
        return

    values = {"Mean Reward": mean_reward}
    if episode_length is not None:
        values["Episode Length"] = episode_length

    plotter.update(iteration=iteration, values=values)
