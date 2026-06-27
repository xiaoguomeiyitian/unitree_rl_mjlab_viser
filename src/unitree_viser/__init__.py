"""unitree_rl_mjlab_viser — Viser 浏览器训练/仿真.

- ``unitree-viser train <TaskID>`` — 训练时实时观察
- ``unitree-viser sim   <TaskID>`` — 仿真模式, 从浏览器控制虚拟机器人
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    # async_render
    "ViserHandle",
    "make_viser_handle",
    "ViserRenderThread",
    "start_viser_render_thread",
    "stop_viser_render_thread",
    # viser_setup
    "setup_viser_for_training",
    "update_training_info",
    "push_reward_to_plot",
    # term_plots
    "ViserTermPlotter",
    # train
    "ViserRunner",
    "TrainingController",
    "make_viser_runner_cls",
    # sim
    "CommandInjector",
    "DdsCommandInjector",
    "SimViewer",
    "PolicyLike",
    "_zero_policy",
    "_random_policy",
    "_make_typed_policy",
]
