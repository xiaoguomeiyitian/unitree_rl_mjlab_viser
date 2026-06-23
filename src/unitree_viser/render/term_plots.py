"""Viser 折线图包装.

探测 mjlab 自带的 ViserTermPlotter 是否可用, 若不可用则提供 fallback 实现.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

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
    """训练过程中的折线图 (reward / loss / episode length)."""

    def __init__(
        self,
        server: "viser.ViserServer",
        term_names: list[str],
        name: str = "Metrics",
    ) -> None:
        self._term_names = list(term_names)
        self._name = name
        self._server = server

        mjlab_plotter = _try_import_mjlab_plotter()
        if mjlab_plotter is not None:
            self._impl = mjlab_plotter(
                server=server,
                term_names=term_names,
                name=name,
            )
            self._mode = "mjlab"
        else:
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
                plot = server.gui.add_plot(
                    f"/plot/{term_name}",
                    aspect=3.0,
                )
                handles[term_name] = plot
        self._buffers: dict[str, tuple[list[int], list[float]]] = {
            t: ([], []) for t in term_names
        }
        return handles

    def update(self, iteration: int, values: dict[str, float]) -> None:
        """把新的数据点推入折线图."""
        if self._mode == "mjlab":
            self._impl.update(iteration=iteration, values=values)
            return

        assert isinstance(self._impl, dict)
        for term_name, value in values.items():
            if term_name not in self._impl:
                continue
            xs, ys = self._buffers[term_name]
            xs.append(iteration)
            ys.append(float(value))
            if len(xs) > 1000:
                xs.pop(0)
                ys.pop(0)
            try:
                self._impl[term_name].data = (np.array(xs), np.array(ys))
            except Exception:
                pass
