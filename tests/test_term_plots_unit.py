"""ViserTermPlotter 单元测试.

测试:
- mjlab 自带 ViserTermPlotter 可用时优先用
- 不可用时降级到 fallback (自己用 add_plot)
- update() 推送数据
- 缓冲区超过 1000 截断
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def test_uses_mjlab_plotter_when_available():
    """mjlab 自带 ViserTermPlotter 可用时优先用."""
    mock_mjlab_plotter = MagicMock()
    server = MagicMock()
    server.gui.add_folder.return_value.__enter__ = MagicMock(return_value=MagicMock())
    server.gui.add_folder.return_value.__exit__ = MagicMock(return_value=False)

    with patch(
        "unitree_viser.render.term_plots._try_import_mjlab_plotter",
        return_value=mock_mjlab_plotter,
    ):
        from unitree_viser.render.term_plots import ViserTermPlotter

        plotter = ViserTermPlotter(server, ["Reward"], name="test")
        assert plotter._mode == "mjlab"
        mock_mjlab_plotter.assert_called_once_with(
            server=server, term_names=["Reward"], name="test"
        )


def test_fallback_when_mjlab_unavailable():
    """mjlab 不可用时降级到 add_plot fallback."""
    server = MagicMock()
    server.gui.add_folder.return_value.__enter__ = MagicMock(return_value=MagicMock())
    server.gui.add_folder.return_value.__exit__ = MagicMock(return_value=False)
    server.gui.add_plot.return_value = MagicMock()

    with patch(
        "unitree_viser.render.term_plots._try_import_mjlab_plotter",
        return_value=None,
    ):
        from unitree_viser.render.term_plots import ViserTermPlotter

        plotter = ViserTermPlotter(server, ["A", "B"], name="fb")
        assert plotter._mode == "fallback"
        # 应创建 2 个 add_plot
        assert server.gui.add_plot.call_count == 2
        # 缓冲区初始化
        assert set(plotter._buffers.keys()) == {"A", "B"}


def test_update_delegates_to_mjlab_impl():
    """mjlab 模式下 update 透传给底层."""
    server = MagicMock()
    with patch(
        "unitree_viser.render.term_plots._try_import_mjlab_plotter",
        return_value=MagicMock(),
    ):
        from unitree_viser.render.term_plots import ViserTermPlotter

        plotter = ViserTermPlotter(server, ["X"])
        plotter.update(iteration=5, values={"X": 1.5})
        plotter._impl.update.assert_called_once_with(
            iteration=5, values={"X": 1.5}
        )


def test_fallback_update_appends_and_assigns_data():
    """fallback 模式下 update 累积数据并赋值给 plot.data."""
    import numpy as np

    server = MagicMock()
    server.gui.add_folder.return_value.__enter__ = MagicMock(return_value=MagicMock())
    server.gui.add_folder.return_value.__exit__ = MagicMock(return_value=False)

    plot_handle = MagicMock()
    server.gui.add_plot.return_value = plot_handle

    with patch(
        "unitree_viser.render.term_plots._try_import_mjlab_plotter",
        return_value=None,
    ):
        from unitree_viser.render.term_plots import ViserTermPlotter

        plotter = ViserTermPlotter(server, ["loss"])
        plotter.update(iteration=0, values={"loss": 0.5})

    xs, ys = plotter._buffers["loss"]
    assert xs == [0]
    assert ys == [0.5]
    plot_handle.data = (np.array([0]), np.array([0.5]))
    # 调用赋值成功 (不报错)


def test_fallback_buffer_caps_at_1000():
    """fallback 模式下缓冲区超过 1000 会截断最旧."""
    server = MagicMock()
    server.gui.add_folder.return_value.__enter__ = MagicMock(return_value=MagicMock())
    server.gui.add_folder.return_value.__exit__ = MagicMock(return_value=False)
    plot_handle = MagicMock()
    server.gui.add_plot.return_value = plot_handle

    with patch(
        "unitree_viser.render.term_plots._try_import_mjlab_plotter",
        return_value=None,
    ):
        from unitree_viser.render.term_plots import ViserTermPlotter

        plotter = ViserTermPlotter(server, ["m"])

        # 推 1500 个点
        for i in range(1500):
            plotter.update(iteration=i, values={"m": float(i)})

    xs, ys = plotter._buffers["m"]
    assert len(xs) == 1000
    assert xs[0] == 500  # 最早剩 500
    assert xs[-1] == 1499


def test_fallback_update_ignores_unknown_term():
    """update 传入未注册的 term 应被忽略 (不抛)."""
    server = MagicMock()
    server.gui.add_folder.return_value.__enter__ = MagicMock(return_value=MagicMock())
    server.gui.add_folder.return_value.__exit__ = MagicMock(return_value=False)
    server.gui.add_plot.return_value = MagicMock()

    with patch(
        "unitree_viser.render.term_plots._try_import_mjlab_plotter",
        return_value=None,
    ):
        from unitree_viser.render.term_plots import ViserTermPlotter

        plotter = ViserTermPlotter(server, ["known"])
        # 不抛异常
        plotter.update(iteration=0, values={"unknown": 1.0})
        # 已知 buffer 不被污染
        assert plotter._buffers["known"] == ([], [])


def test_fallback_update_swallows_data_assign_exception():
    """plot.data 赋值失败被吞 (viser API 兼容)."""
    server = MagicMock()
    server.gui.add_folder.return_value.__enter__ = MagicMock(return_value=MagicMock())
    server.gui.add_folder.return_value.__exit__ = MagicMock(return_value=False)
    plot_handle = MagicMock()
    type(plot_handle).data = property(
        lambda self: None,
        lambda self, v: (_ for _ in ()).throw(Exception("old viser")),
    )
    server.gui.add_plot.return_value = plot_handle

    with patch(
        "unitree_viser.render.term_plots._try_import_mjlab_plotter",
        return_value=None,
    ):
        from unitree_viser.render.term_plots import ViserTermPlotter

        plotter = ViserTermPlotter(server, ["t"])
        # 不应抛
        plotter.update(iteration=0, values={"t": 1.0})


def test_try_import_mjlab_plotter_returns_none_on_import_error():
    """_try_import_mjlab_plotter 在 ImportError 时返回 None."""
    from unitree_viser.render import term_plots

    with patch.dict(sys.modules, {"mjlab.viewer.viser.term_plotter": None}):
        # 直接调用, 触发 ImportError 分支
        result = term_plots._try_import_mjlab_plotter()
    assert result is None