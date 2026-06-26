"""viser_setup 单元测试.

测试:
- update_training_info 把内容写到 HTML
- push_reward_to_plot 调用 plotter.update
- 空值/None 处理
"""

from __future__ import annotations

import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# Stub mediapy to avoid Python 3.12 + numpy 2.x incompatibility
_stub = types.ModuleType("mediapy")
_stub.__file__ = "(test-stub)"
_stub.set_ffmpeg = lambda _path: None
sys.modules["mediapy"] = _stub

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.normpath(os.path.join(_HERE, "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def test_update_training_info_writes_html():
    """update_training_info 把内容写入 info_html."""
    from unitree_viser.render.viser_setup import update_training_info

    info_html = MagicMock()
    gui_state = {"info_html": info_html}

    update_training_info(
        gui_state,
        iteration=42,
        mean_reward=1.234,
        current_fps=12.5,
    )

    # info_html.content 被赋值
    assert info_html.content
    content = info_html.content
    assert "42" in content
    assert "1.234" in content
    assert "12.5" in content


def test_update_training_info_handles_none_values():
    """mean_reward / current_fps 为 None 时不显示该字段."""
    from unitree_viser.render.viser_setup import update_training_info

    info_html = MagicMock()
    gui_state = {"info_html": info_html}

    update_training_info(
        gui_state,
        iteration=10,
        mean_reward=None,
        current_fps=None,
    )
    content = info_html.content
    assert "10" in content
    # None 值被跳过，不显示任何数值
    assert "Mean Reward" not in content
    assert "Render FPS" not in content


def test_update_training_info_no_html_safe():
    """gui_state 没有 info_html 时不抛."""
    from unitree_viser.render.viser_setup import update_training_info

    # 不应抛
    update_training_info({}, iteration=0, mean_reward=0.0, current_fps=10.0)


def test_update_training_info_with_extra_lines():
    """extra_lines 应附加到 HTML."""
    from unitree_viser.render.viser_setup import update_training_info

    info_html = MagicMock()
    gui_state = {"info_html": info_html}

    update_training_info(
        gui_state,
        iteration=0,
        mean_reward=0.0,
        current_fps=10.0,
        extra_lines=["Custom line 1", "Custom line 2"],
    )
    content = info_html.content
    assert "Custom line 1" in content
    assert "Custom line 2" in content


# ── push_reward_to_plot ─────────────────────────────────────────────────


def test_push_reward_to_plot_calls_plotter_update():
    """push_reward_to_plot 调用 plotter.update."""
    from unitree_viser.render.viser_setup import push_reward_to_plot

    plotter = MagicMock()
    gui_state = {"reward_plotter": plotter}

    push_reward_to_plot(
        gui_state,
        iteration=42,
        mean_reward=1.5,
        episode_length=200.0,
    )

    plotter.update.assert_called_once_with(
        iteration=42,
        values={"Mean Reward": 1.5, "Episode Length": 200.0},
    )


def test_push_reward_to_plot_without_episode_length():
    """不传 episode_length 时, values 不含该 key."""
    from unitree_viser.render.viser_setup import push_reward_to_plot

    plotter = MagicMock()
    gui_state = {"reward_plotter": plotter}

    push_reward_to_plot(gui_state, iteration=0, mean_reward=0.5)
    plotter.update.assert_called_once_with(
        iteration=0, values={"Mean Reward": 0.5}
    )


def test_push_reward_to_plot_no_plotter_safe():
    """gui_state 没有 reward_plotter 时不抛."""
    from unitree_viser.render.viser_setup import push_reward_to_plot

    push_reward_to_plot({}, iteration=0, mean_reward=0.5)


# ── setup_viser_for_training (high-level mock test) ─────────────────────


def test_setup_viser_for_training_returns_4tuple():
    """setup_viser_for_training 返回 (server, scene, handle, gui_state) 四元组."""
    from unitree_viser.render.viser_setup import setup_viser_for_training

    env = MagicMock()
    env.sim.mj_model = MagicMock()
    env.num_envs = 1
    env.sim.mj_data = MagicMock()
    env.sim.wp_data = MagicMock()

    server = MagicMock()
    server.gui.add_folder.return_value.__enter__ = MagicMock(return_value=MagicMock())
    server.gui.add_folder.return_value.__exit__ = MagicMock(return_value=False)
    server.gui.add_html.return_value = MagicMock()
    server.gui.add_plot.return_value = MagicMock()

    scene = MagicMock()

    with patch("viser.ViserServer", return_value=server), patch(
        "unitree_viser.render.viser_setup._create_scene", return_value=scene
    ), patch("mujoco.MjData", return_value=MagicMock()):
        result = setup_viser_for_training(env=env, port=20006, env_idx=0)

    assert len(result) == 4
    server_out, scene_out, handle, gui_state = result
    assert server_out is server
    assert scene_out is scene
    assert "info_html" in gui_state
    assert "reward_plotter" in gui_state


def test_setup_viser_for_training_with_control():
    """enable_control=True 时 gui_state 包含 controller."""
    from unitree_viser.render.viser_setup import setup_viser_for_training

    env = MagicMock()
    env.sim.mj_model = MagicMock()
    env.num_envs = 1
    env.sim.mj_data = MagicMock()

    server = MagicMock()
    server.gui.add_folder.return_value.__enter__ = MagicMock(return_value=MagicMock())
    server.gui.add_folder.return_value.__exit__ = MagicMock(return_value=False)
    server.gui.add_html.return_value = MagicMock()
    server.gui.add_plot.return_value = MagicMock()
    server.gui.add_button.return_value = MagicMock()
    server.gui.add_slider.return_value = MagicMock()

    scene = MagicMock()

    with patch("viser.ViserServer", return_value=server), patch(
        "unitree_viser.render.viser_setup._create_scene", return_value=scene
    ), patch("mujoco.MjData", return_value=MagicMock()):
        _, _, _, gui_state = setup_viser_for_training(
            env=env, enable_control=True, fps=10.0
        )

    assert gui_state["controller"] is not None


def test_setup_viser_for_training_no_control():
    """enable_control=False 时 controller 是 None."""
    from unitree_viser.render.viser_setup import setup_viser_for_training

    env = MagicMock()
    env.sim.mj_model = MagicMock()
    env.num_envs = 1
    env.sim.mj_data = MagicMock()

    server = MagicMock()
    server.gui.add_folder.return_value.__enter__ = MagicMock(return_value=MagicMock())
    server.gui.add_folder.return_value.__exit__ = MagicMock(return_value=False)
    server.gui.add_html.return_value = MagicMock()
    server.gui.add_plot.return_value = MagicMock()

    scene = MagicMock()

    with patch("viser.ViserServer", return_value=server), patch(
        "unitree_viser.render.viser_setup._create_scene", return_value=scene
    ), patch("mujoco.MjData", return_value=MagicMock()):
        _, _, _, gui_state = setup_viser_for_training(env=env, enable_control=False)

    assert gui_state["controller"] is None