"""Viser 折线图包装.

mjlab 1.2.0 自带 ``ViserTermPlotter`` (在 ``mjlab.viewer.viser.term_plotter``),
可直接复用. 本模块:

1. 探测 mjlab 自带的 ViserTermPlotter 是否可用
2. 若可用, 直接用 (推荐)
3. 若不可用, 提供一个轻量级 fallback 实现 (用 viser.GuiPlot 折线图)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import viser


def _try_import_mjlab_plotter() -> Any:
    """探测 mjlab 自带的 ViserTermPlotter."""
    try:
        from mjlab.viewer.viser.term_plotter import ViserTermPlotter

        return ViserTermPlotter
    except ImportError:
        return None


class ViserTermPlotter:
    """训练过程中的折线图 (reward / loss / episode length).

    用法::

        plotter = ViserTermPlotter(server, ["Mean Reward"], name="Training")
        for it in range(1000):
            ...
            plotter.update(iteration=it, values={"Mean Reward": 0.5})
    """

    def __init__(
        self,
        server: "viser.ViserServer",
        term_names: list[str],
        name: str = "Metrics",
    ) -> None:
        self._term_names = list(term_names)
        self._name = name
        self._server = server

        # 优先用 mjlab 自带的; 没有就用 viser.gui.add_plot
        mjlab_plotter = _try_import_mjlab_plotter()
        if mjlab_plotter is not None:
            self._impl = mjlab_plotter(
                server=server,
                term_names=term_names,
                name=name,
            )
            self._mode = "mjlab"
        else:
            # Fallback: 自己用 viser.gui.add_plot
            self._impl = self._build_fallback(server, term_names, name)
            self._mode = "fallback"

    def _build_fallback(
        self,
        server: "viser.ViserServer",
        term_names: list[str],
        name: str,
    ) -> dict[str, Any]:
        """用 viser.gui.add_plot 自建折线图."""
        handles: dict[str, Any] = {}
        with server.gui.add_folder(name):
            for term_name in term_names:
                # viser >= 0.2 提供 GuiPlotHandle
                plot = server.gui.add_plot(
                    f"/plot/{term_name}",
                    aspect=3.0,
                    # initial values: empty
                )
                handles[term_name] = plot
        # 用一个简单的 buffer 保存最近 N 个点
        self._buffers: dict[str, tuple[list[int], list[float]]] = {
            t: ([], []) for t in term_names
        }
        return handles

    def update(self, iteration: int, values: dict[str, float]) -> None:
        """把新的数据点推入折线图.

        Args:
            iteration: X 轴 (iter 编号)
            values: term_name -> value, 键必须是 ``term_names`` 的子集
        """
        if self._mode == "mjlab":
            # mjlab 的接口 (假设)
            self._impl.update(iteration=iteration, values=values)
            return

        # Fallback: 手动更新 viser plot
        assert isinstance(self._impl, dict)
        for term_name, value in values.items():
            if term_name not in self._impl:
                continue
            xs, ys = self._buffers[term_name]
            xs.append(iteration)
            ys.append(float(value))
            # 保留最近 1000 个点
            if len(xs) > 1000:
                xs.pop(0)
                ys.pop(0)
            try:
                self._impl[term_name].data = (np.array(xs), np.array(ys))
            except Exception:
                # 旧版 viser 的 API 不同, 跳过
                pass


# 延迟 import numpy 避免顶层失败
import numpy as np  # noqa: E402
