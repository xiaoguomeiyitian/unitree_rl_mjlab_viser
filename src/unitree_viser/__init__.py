"""unitree_rl_mjlab_viser — Viser 浏览器训练/仿真

为 unitree_rl_mjlab 提供基于 Viser 的浏览器可视化能力:

- ``unitree-viser train <TaskID>`` — 训练时实时观察 (3D 场景 + 折线图 + 可选控制)
- ``unitree-viser sim   <TaskID>`` — 仿真模式, 从浏览器控制虚拟机器人

不修改 unitree_rl_mjlab 源码, 纯 import + 子类化。
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "make_viser_handle",
    "start_viser_render_thread",
    "stop_viser_render_thread",
    "ViserRunner",
]
