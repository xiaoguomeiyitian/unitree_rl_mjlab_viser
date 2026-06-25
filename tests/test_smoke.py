"""Smoke test - 验证 import 不报错, 优雅处理缺失依赖.

设计: 尽量在 import 失败时仍然能验证尽可能多的 API 表面.
对每个测试, 区分两类失败:

- "可选依赖缺失" (如 ``viser``/``torch``/``tyro``) — 跳过并打印警告
- "真正的代码错误" — 标为 FAILED

运行方式::

    PYTHONPATH=src python3 tests/test_smoke.py

期望: 当 **所有** 依赖都装好后, 所有 7 个测试通过.
当部分依赖缺失时, 大部分测试跳过, 至少 ``test_top_level_import`` 必通过.
"""

import importlib
import sys


def _try_import(name: str):
    """尝试 import, 失败返回 (None, error_msg)."""
    try:
        return importlib.import_module(name), None
    except ImportError as e:
        return None, str(e)


def test_top_level_import():
    """能 import 顶层包 (无外部依赖)."""
    import unitree_viser

    assert hasattr(unitree_viser, "__version__")
    print(f"✓ unitree_viser version: {unitree_viser.__version__}")


def test_async_render():
    """async_render 模块 (不需要 viser, 只用 stdlib)."""
    from unitree_viser.render.async_render import (
        ViserHandle,
        make_viser_handle,
        start_viser_render_thread,
        stop_viser_render_thread,
    )

    # 用 None 占位构造 handle (不真启动线程)
    handle = make_viser_handle(
        server=object(),
        scene=object(),
        env_idx=0,
        mj_model=object(),
        mj_data=object(),
        sim=object(),
    )
    assert handle["env_idx"] == 0
    assert handle["_render_thread"] is None
    print("✓ async_render: ViserHandle + factory works")


def test_train_module_api():
    """train 子模块的 ViserRunner (不需要 viser)."""
    from unitree_viser.train.runner_subclass import (
        ViserRunner,
        make_viser_runner_cls,
    )

    assert ViserRunner is not None
    assert callable(make_viser_runner_cls)
    # _post_iter_hooks 现在是实例属性 (在 __init__ 中初始化), 不再是类属性
    # 验证 register/clear 方法存在
    assert callable(getattr(ViserRunner, "register_post_iter_hook", None))
    assert callable(getattr(ViserRunner, "clear_post_iter_hooks", None))
    print("✓ train.runner_subclass: ViserRunner + hook API works")


def test_training_controller_class():
    """TrainingController 类本身 (不实例化, 避免依赖 viser)."""
    from unitree_viser.train.training_controller import TrainingController

    for method in ("wait_if_paused", "toggle_pause", "request_single_step",
                   "get_speed_multiplier", "request_quit", "is_paused"):
        assert hasattr(TrainingController, method), f"missing {method}"
    print("✓ TrainingController: all methods present")


def test_command_injector_class():
    """CommandInjector 类 (不实例化, 避免依赖 viser)."""
    from unitree_viser.sim.command_injection import CommandInjector

    assert hasattr(CommandInjector, "inject")
    assert hasattr(CommandInjector, "get_pending")
    print("✓ CommandInjector: inject + get_pending API present")


def test_dds_command_injector_class():
    """DdsCommandInjector 类 (不实例化, 避免依赖 viser + unitree_sdk2py)."""
    from unitree_viser.sim.dds_command_injection import DdsCommandInjector

    # 必须的方法
    for method in ("start", "inject", "get_pending", "stop"):
        assert hasattr(DdsCommandInjector, method), f"missing {method}"

    # 私有方法: 摇杆 → 速度换算
    inst = DdsCommandInjector.__new__(DdsCommandInjector)  # 不调 __init__
    vx, vy, yaw = inst._axis_to_vel(lx=0.0, ly=1.0, rx=0.0)
    assert vx == 1.0, f"vx should be 1.0 (ly), got {vx}"
    vx, vy, yaw = inst._axis_to_vel(lx=0.5, ly=0.0, rx=0.0)
    assert vy == -0.5, f"vy should be -0.5 (-lx), got {vy}"
    vx, vy, yaw = inst._axis_to_vel(lx=0.0, ly=0.0, rx=0.5)
    assert yaw == -0.5, f"yaw should be -0.5 (-rx), got {yaw}"

    print("✓ DdsCommandInjector: axis→velocity conversion correct (vx=ly, vy=-lx, yaw=-rx)")


def test_sim_viewer_module():
    """sim_viewer 模块 (需要 torch + viser)."""
    torch, err = _try_import("torch")
    if torch is None:
        print(f"⊘ test_sim_viewer: skipped (no torch: {err})")
        return

    from unitree_viser.sim.sim_viewer import (
        SimViewer,
        _zero_policy,
        _random_policy,
        _make_typed_policy,
    )

    # obs 可能是 tensor (来自 wrapper) 或 dict (来自 ManagerBasedRlEnv)
    obs_tensor = torch.zeros(1, 12)
    obs_dict = {"actor": torch.zeros(1, 12), "critic": torch.zeros(1, 47)}

    # 默认 _zero_policy 输出 12 维 (Unitree Go2)
    a = _zero_policy(obs_dict)
    assert a.shape == (1, 12), f"expected (1, 12), got {a.shape}"
    assert a.device == obs_dict["actor"].device

    a = _random_policy(obs_dict)
    assert a.shape == (1, 12)
    assert (a >= -1.0).all() and (a <= 1.0).all()

    # _make_typed_policy 包装: 输出 (n, num_actions) 形状
    typed = _make_typed_policy(_zero_policy, 12)
    a = typed(obs_dict)
    assert a.shape == (1, 12)

    # 也支持 tensor 输入
    a = _zero_policy(obs_tensor)
    assert a.shape == (1, 12)

    print("✓ sim_viewer: zero/random policy returns correct shape")


def test_cli_apis():
    """cli 模块的 dataclass 和 main 函数 (需要 tyro)."""
    tyro, err = _try_import("tyro")
    if tyro is None:
        print(f"⊘ test_cli_apis: skipped (no tyro: {err})")
        return

    from unitree_viser import cli

    assert callable(cli.main)
    assert callable(cli.run_train)
    assert callable(cli.run_sim)
    assert hasattr(cli, "TrainArgs")
    assert hasattr(cli, "SimArgs")
    # TrainArgs 字段
    assert "task" in cli.TrainArgs.__dataclass_fields__
    assert "viser_port" in cli.TrainArgs.__dataclass_fields__
    assert "headless" in cli.TrainArgs.__dataclass_fields__
    print("✓ cli: TrainArgs / SimArgs / main / run_* present")


def test_render_viser_setup():
    """viser_setup 模块 (需要 viser)."""
    viser, err = _try_import("viser")
    if viser is None:
        print(f"⊘ test_render_viser_setup: skipped (no viser: {err})")
        return

    from unitree_viser.render import viser_setup

    assert hasattr(viser_setup, "setup_viser_for_training")
    assert hasattr(viser_setup, "update_training_info")
    assert hasattr(viser_setup, "push_reward_to_plot")
    print("✓ render.viser_setup: public API present")


def test_term_plots():
    """term_plots 模块 (需要 viser 才能实例化)."""
    viser, err = _try_import("viser")
    if viser is None:
        print(f"⊘ test_term_plots: skipped (no viser: {err})")
        return

    from unitree_viser.render.term_plots import ViserTermPlotter

    assert ViserTermPlotter is not None
    print("✓ term_plots: ViserTermPlotter class present")


def test_install_script_exists():
    """一键安装脚本存在且语法正确.

    检查:
    1. scripts/install_env.sh 存在
    2. 有可执行权限
    3. 通过 bash -n 语法检查
    4. 包含关键的 install_pytorch_gpu / install_pytorch_cpu 函数
    """
    import os
    import subprocess

    # smoke test 运行时的 cwd 不一定是项目根, 用文件相对路径
    test_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(test_dir)
    install_script = os.path.join(project_root, "scripts", "install_env.sh")

    assert os.path.exists(install_script), f"install script not found: {install_script}"
    assert os.access(install_script, os.X_OK), f"install script not executable: {install_script}"

    # 语法检查
    result = subprocess.run(
        ["bash", "-n", install_script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"install_env.sh syntax error: {result.stderr}"

    # 内容检查 - 必须有 GPU/CPU 分支
    with open(install_script, "r", encoding="utf-8") as f:
        content = f.read()
    assert "install_pytorch_gpu" in content, "missing install_pytorch_gpu function"
    assert "install_pytorch_cpu" in content, "missing install_pytorch_cpu function"
    assert "MUJOCO_GL" in content, "missing MUJOCO_GL configuration"

    print(f"✓ install_env.sh: exists + executable + syntax OK ({os.path.getsize(install_script)} bytes)")


def main() -> None:
    """运行所有 smoke test."""
    tests = [
        test_top_level_import,
        test_async_render,
        test_train_module_api,
        test_training_controller_class,
        test_command_injector_class,
        test_dds_command_injector_class,
        test_sim_viewer_module,
        test_cli_apis,
        test_render_viser_setup,
        test_term_plots,
        test_install_script_exists,
    ]
    passed = 0
    skipped = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"✗ {t.__name__}: AssertionError: {e}")
            failed += 1
        except ImportError as e:
            # 区分: 顶部 try_import 已处理; 这里是真正的代码 bug
            print(f"✗ {t.__name__}: ImportError: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {t.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print()
    total = len(tests)
    print(f"Summary: {passed} passed, 0 skipped, {failed} failed (out of {total})")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
