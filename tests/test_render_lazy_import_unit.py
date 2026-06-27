"""render 包的 PEP 562 延迟导入测试.

测试:
- unitree_viser.render.ViserHandle 等属性的延迟导入
- 不存在属性抛 AttributeError
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def test_render_lazy_imports_async_render():
    """访问 ViserHandle 时才真正 import async_render."""
    import unitree_viser.render as r

    # 触发延迟导入
    handle = r.ViserHandle
    assert handle is not None
    # 同时导出的还有 start/stop/make
    assert callable(r.make_viser_handle)
    assert callable(r.start_viser_render_thread)
    assert callable(r.stop_viser_render_thread)


def test_render_lazy_imports_viser_setup():
    """setup_viser_for_training / update_training_info / push_reward_to_plot."""
    import unitree_viser.render as r

    assert callable(r.setup_viser_for_training)
    assert callable(r.update_training_info)
    assert callable(r.push_reward_to_plot)


def test_render_lazy_imports_term_plotter():
    """ViserTermPlotter 延迟导入."""
    import unitree_viser.render as r

    plotter_cls = r.ViserTermPlotter
    assert plotter_cls is not None


def test_render_unknown_attribute_raises():
    """未知属性应抛 AttributeError, 不静默."""
    import unitree_viser.render as r

    with __import__("pytest").raises(AttributeError, match="has no attribute"):
        r.nonexistent_attr


def test_render_all_exports():
    """__all__ 包含所有公开符号."""
    import unitree_viser.render as r

    assert "ViserHandle" in r.__all__
    assert "make_viser_handle" in r.__all__
    assert "start_viser_render_thread" in r.__all__
    assert "stop_viser_render_thread" in r.__all__
    assert "ViserTermPlotter" in r.__all__
    assert "setup_viser_for_training" in r.__all__
    assert "update_training_info" in r.__all__
    assert "push_reward_to_plot" in r.__all__


def test_lazy_import_idempotent():
    """多次访问同一个属性应返回同一对象 (缓存)."""
    import unitree_viser.render as r

    a = r.ViserHandle
    b = r.ViserHandle
    assert a is b

    c = r.setup_viser_for_training
    d = r.setup_viser_for_training
    assert c is d