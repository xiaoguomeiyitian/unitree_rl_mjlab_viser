"""Render 工具集 - Viser 后台渲染 + 视图类封装.

延迟导入 viser 相关模块, 避免在仅有 stdlib 的环境 (如 smoke test)
import 时失败. 用户从 ``unitree_viser.render`` 显式调用函数才触发实际 import.
"""

from __future__ import annotations


def __getattr__(name):
    """PEP 562: 延迟属性访问 - 按需 import.

    让 ``from unitree_viser.render import make_viser_handle`` 这样的语句
    在调用时才真正 import 相关模块 (含 viser), 避免本子包 import 阶段失败.
    """
    if name in ("ViserHandle", "make_viser_handle", "start_viser_render_thread",
                "stop_viser_render_thread"):
        from unitree_viser.render.async_render import (
            ViserHandle as _ViserHandle,
            make_viser_handle as _make,
            start_viser_render_thread as _start,
            stop_viser_render_thread as _stop,
        )
        return {
            "ViserHandle": _ViserHandle,
            "make_viser_handle": _make,
            "start_viser_render_thread": _start,
            "stop_viser_render_thread": _stop,
        }[name]

    if name in ("setup_viser_for_training", "update_training_info", "push_reward_to_plot"):
        from unitree_viser.render.viser_setup import (
            push_reward_to_plot as _push,
            setup_viser_for_training as _setup,
            update_training_info as _update,
        )
        return {
            "setup_viser_for_training": _setup,
            "update_training_info": _update,
            "push_reward_to_plot": _push,
        }[name]

    if name == "ViserTermPlotter":
        from unitree_viser.render.term_plots import ViserTermPlotter as _VTP
        return _VTP

    raise AttributeError(f"module 'unitree_viser.render' has no attribute {name!r}")


__all__ = [
    "ViserHandle",
    "make_viser_handle",
    "start_viser_render_thread",
    "stop_viser_render_thread",
    "ViserTermPlotter",
    "setup_viser_for_training",
    "update_training_info",
    "push_reward_to_plot",
]
