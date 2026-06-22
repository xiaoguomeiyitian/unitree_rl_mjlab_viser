"""unitree-viser - 主 CLI 入口.

两个子命令:

- ``unitree-viser train <TaskID>`` — 训练模式 (后台异步渲染)
- ``unitree-viser sim <TaskID>``  — 仿真模式 (浏览器驱动虚拟机器人)

使用 tyro 做参数解析, 继承自 unitree_rl_mjlab 的任务注册表.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union

import tyro

# mjlab / unitree_rl_mjlab 不在顶层 import, 以便 smoke test 能在没有 mjlab 的环境下通过.


# ── sys.path 修复 (必须在 import mjlab 之前) ────────────────────────────
# 当从 viser 项目目录运行 `python -m unitree_viser.cli` 时, sys.path 包含
# `/root/unitree/unitree_rl_mjlab_viser/src` (因为 cli.py 在那里). 该目录
# 虽然没有 `__init__.py`, 但 Python 会把它当作 namespace package, 抢占
# `import src.tasks` 对兄弟项目 unitree_rl_mjlab/src 的解析.
# 解决: 1) 把兄弟项目 src 强制放到 sys.path 最前;
#       2) **关键**: 删掉或遮蔽 viser/src 这个会触发 namespace 抢占的路径;
#       3) 显式 `import src` 一次, 锁住 real package.
def _ensure_sibling_src_on_path() -> None:
    # 0. 删掉空字符串条目 (代表 cwd)
    sys.path[:] = [p for p in sys.path if p != ""]

    # 1. 找兄弟项目 src 路径
    sib = os.environ.get("UNITREE_RL_MJLAB_SRC")
    if not (sib and Path(sib).is_dir()):
        here = Path(__file__).resolve()
        for parent in here.parents:
            candidate = parent / "unitree_rl_mjlab" / "src"
            if candidate.is_dir() and (candidate / "tasks" / "__init__.py").exists():
                sib = str(candidate)
                break

    if not sib:
        return

    # 2. 移除会触发 namespace 抢占的 viser/src 路径
    # 关键: 任何 sys.path 下的路径, 如果它本身是一个会被当作 `src` namespace package
    # 的目录 (即: 没有 __init__.py 但 Python 会扫到), 都要过滤掉.
    here = Path(__file__).resolve()
    viser_pkg_dir = here.parent  # /root/unitree/unitree_rl_mjlab_viser/src/unitree_viser
    viser_src_dir = viser_pkg_dir.parent  # /root/unitree/unitree_rl_mjlab_viser/src

    # 收集要过滤掉的路径
    bad_paths = set()
    for p in sys.path:
        pp = Path(p).resolve()
        # 跳过 viser/src 目录 (会触发 src namespace 抢占)
        if pp == viser_src_dir:
            bad_paths.add(p)
        # 跳过 viser/src/unitree_viser 目录
        if pp == viser_pkg_dir:
            bad_paths.add(p)
        # 跳过任何含空 src/ 子目录的路径 (兜底)
        if (pp / "src").is_dir() and not (pp / "src" / "__init__.py").exists():
            # 这是个可疑的命名空间冲突路径
            # 但只对那些 src 目录下没有 unitree_rl_mjlab 等内容的删
            cand = pp / "src"
            if not any((cand / name).exists() for name in ["assets", "tasks", "__init__.py"]):
                bad_paths.add(p)

    sys.path[:] = [p for p in sys.path if p not in bad_paths]

    # 3. 把兄弟项目 src 放到 sys.path 最前
    if sib in sys.path:
        sys.path.remove(sib)
    sys.path.insert(0, sib)

    # 4. 关键: 清掉 sys.modules 里可能已经被错误加载的 `src`, 然后重新 import
    sys.modules.pop("src", None)
    sys.modules.pop("src.assets", None)
    sys.modules.pop("src.tasks", None)
    try:
        import src  # noqa: F401
    except ImportError:
        pass  # 找不到就放过, 后续调用方自己报错

    # DEBUG: 打印 sys.path 和 src 解析状态
    if os.environ.get("UNITREE_VISER_DEBUG_PATH"):
        print(f"[DEBUG-PATH] sys.path[:5]: {sys.path[:5]}", file=sys.stderr)
        if "src" in sys.modules:
            print(f"[DEBUG-PATH] sys.modules['src'].__file__: {sys.modules['src'].__file__!r}", file=sys.stderr)
        else:
            print(f"[DEBUG-PATH] src NOT in sys.modules", file=sys.stderr)


def _stub_mediapy() -> None:
    """mediapy 1.2.6 在 Python 3.12 + numpy 2.x 下无法导入 (npt.NDArray[Any] 报错).
    mjlab 1.2.0 的 _configure_mediapy() 强制 import mediapy, 失败会让 sim/train 起不来.
    这里提前注入一个最小 stub, 让 mjlab 完成初始化 (它只用 set_ffmpeg, 其他不用).
    """
    if "mediapy" in sys.modules:
        return
    try:
        import mediapy  # noqa: F401
    except (ImportError, TypeError):
        import types
        stub = types.ModuleType("mediapy")
        stub.__file__ = "(viser-stub)"  # 标记为 stub
        def _set_ffmpeg(_path): pass
        stub.set_ffmpeg = _set_ffmpeg
        sys.modules["mediapy"] = stub


_ensure_sibling_src_on_path()
_stub_mediapy()


# ── Train 子命令 ──────────────────────────────────────────────────────────


@dataclass
class TrainArgs:
    """训练模式参数."""

    task: str
    """任务 ID, e.g. ``Unitree-G1-Flat``, ``Unitree-Go2-Rough``"""

    # ── Viser 选项 ──
    viser_port: int = 20006
    """Viser HTTP/WS 端口 (0 = 不启动 Viser, 仅训练)"""
    viser_env_idx: int = 0
    """显示哪个环境"""
    viser_fps: float = 10.0
    """Viser 渲染目标 FPS"""
    enable_control: bool = False
    """是否启用训练控制 (暂停/单步/速度滑块)"""

    # ── 训练超参 (透传到 mjlab TrainConfig) ──
    num_envs: int | None = None
    """覆盖 env_cfg.scene.num_envs"""
    max_iterations: int | None = None
    """覆盖 agent_cfg.max_iterations"""
    seed: int | None = None
    """覆盖 agent_cfg.seed"""
    save_interval: int | None = None
    """覆盖 agent_cfg.save_interval"""
    motion_file: str | None = None
    """跟踪任务必填, 指向 .npz 文件"""

    # ── Headless 模式 ──
    headless: bool = False
    """完全跳过 Viser, 用 unitree_rl_mjlab 原版 train.py (用于 CI)"""


def run_train(args: TrainArgs) -> None:
    """训练模式主函数."""
    # 延迟 import - 需要 mjlab / unitree_rl_mjlab 已安装
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.rl.runner import MjlabOnPolicyRunner

    from unitree_viser.render.viser_setup import setup_viser_for_training
    from unitree_viser.render.async_render import (
        start_viser_render_thread,
        stop_viser_render_thread,
    )
    from unitree_viser.train.runner_subclass import make_viser_runner_cls

    # 1. 加载配置
    env_cfg = load_env_cfg(args.task)
    agent_cfg = load_rl_cfg(args.task)
    runner_cls_base = load_runner_cls(args.task)

    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    if args.max_iterations is not None:
        agent_cfg["max_iterations"] = args.max_iterations
    if args.seed is not None:
        agent_cfg["seed"] = args.seed
    if args.save_interval is not None:
        agent_cfg["save_interval"] = args.save_interval

    # 跟踪任务需要 motion_file
    is_tracking = "motion" in env_cfg.commands
    if is_tracking and not args.motion_file and not args.headless:
        raise ValueError(
            f"Task '{args.task}' is a tracking task. "
            "Please provide --motion-file /path/to/motion.npz"
        )
    if args.motion_file and is_tracking:
        env_cfg.commands["motion"].motion_file = args.motion_file

    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.get("clip_actions", 1.0))

    ViserRunnerCls = make_viser_runner_cls(runner_cls_base or MjlabOnPolicyRunner)

    from datetime import datetime
    from pathlib import Path
    log_dir = Path("logs") / "viser" / args.task / datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)

    runner = ViserRunnerCls(env, agent_cfg, str(log_dir), device)

    viser_handle = None
    if not args.headless and args.viser_port > 0:
        try:
            _, _, viser_handle, gui_state = setup_viser_for_training(
                env=env.unwrapped,
                port=args.viser_port,
                env_idx=args.viser_env_idx,
                enable_control=args.enable_control,
                fps=args.viser_fps,
            )
            runner.viser_gui_state = gui_state
            if args.enable_control and "controller" in gui_state:
                from unitree_viser.train.training_controller import TrainingController
                assert isinstance(gui_state["controller"], TrainingController)
                runner.training_controller = gui_state["controller"]
            print(f"\n[VISER] 浏览器访问: http://localhost:{args.viser_port}\n")
        except Exception as e:
            print(f"[VISER] 启动失败, 继续训练但无可视化: {e}")
            viser_handle = None

    try:
        if viser_handle is not None:
            start_viser_render_thread(viser_handle, target_fps=args.viser_fps)
        runner.learn(
            num_learning_iterations=agent_cfg["max_iterations"],
            init_at_random_ep_len=True,
        )
    finally:
        if viser_handle is not None:
            stop_viser_render_thread(viser_handle)
        print(f"\n[TRAIN] 完成. 日志目录: {log_dir}")


# ── Sim 子命令 ────────────────────────────────────────────────────────────


@dataclass
class SimArgs:
    """仿真模式参数."""

    task: str
    """任务 ID"""

    checkpoint: str | None = None
    """训练好的策略 .pt 文件 (None = zero policy)"""
    policy: Literal["zero", "random"] | None = None
    """无 checkpoint 时使用的 policy (zero/random)"""

    num_envs: int = 1
    """仿真环境数 (默认 1, 简单)"""

    viser_port: int = 20006
    """Viser HTTP/WS 端口"""
    viser_env_idx: int = 0
    """显示哪个环境"""

    inject_commands: bool = True
    """是否启用命令注入 GUI"""
    command_name: str = "twist"
    """命令 term 名称"""

    max_steps: int | None = None
    """最多多少步 (None = 无限, 直到用户 Quit)"""

    headless: bool = False
    """不启动 Viser, 仅仿真 (用于测试)"""


def run_sim(args: SimArgs) -> None:
    """仿真模式主函数."""
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.envs import ManagerBasedRlEnv

    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = args.num_envs

    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

    # 加载策略
    policy = None
    if args.checkpoint:
        from mjlab.rl import MjlabOnPolicyRunner
        from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
        from unitree_viser.train.runner_subclass import make_viser_runner_cls

        env_wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
        ViserRunnerCls = make_viser_runner_cls(MjlabOnPolicyRunner)
        runner = ViserRunnerCls(env_wrapped, {"max_iterations": 0}, None, device)
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=device)
        print(f"[SIM] 加载策略: {args.checkpoint}")
    else:
        from unitree_viser.sim.sim_viewer import (
            _zero_policy, _random_policy, _make_typed_policy,
        )
        base = _zero_policy if args.policy != "random" else _random_policy
        try:
            num_actions = int(env.action_manager.total_action_dim)
        except Exception:
            num_actions = 12
        policy = _make_typed_policy(base, num_actions)
        print(f"[SIM] 使用 {args.policy or 'zero'} policy ({num_actions} DOF)")

    # 启动 Viser
    if args.headless:
        print("[SIM] Headless 模式: 不启动 Viser")
        # 简单 step 循环
        obs, _ = env.reset()
        for i in range(args.max_steps or 100):
            with torch.inference_mode():
                actions = policy(obs)
            obs, _, _, _, _ = env.step(actions)
            if i % 10 == 0:
                print(f"[SIM] step {i}")
        return

    from unitree_viser.sim.sim_viewer import SimViewer

    viewer = SimViewer(
        env=env,
        policy=policy,
        port=args.viser_port,
        env_idx=args.viser_env_idx,
        inject_commands=args.inject_commands,
        command_name=args.command_name,
    )
    viewer.setup()
    try:
        viewer.run(max_steps=args.max_steps)
    finally:
        viewer.close()


# ── Main 入口 ─────────────────────────────────────────────────────────────


def main() -> None:
    """tyro CLI 入口.

    兼容 tyro 0.9 (子命令 train/sim) 和 tyro 1.0+ (子命令 train-args/sim-args):
    1. tyro 0.9 子命令名直接是类名小写: train / sim
    2. tyro 1.0 子命令名是类名 kebab-case: train-args / sim-args
    3. 本函数先尝试用 tyro 1.0 解析;若首参是 train/sim, 重写为新语法再解析
    """
    if len(sys.argv) > 1 and sys.argv[1] in ("train", "sim"):
        # 把 train → train-args, sim → sim-args (兼容老用法)
        sys.argv[1] = sys.argv[1] + "-args"

    cli = tyro.cli(
        Union[TrainArgs, SimArgs],
        description="Unitree RL Mjlab - Viser 浏览器训练/仿真",
    )
    if isinstance(cli, TrainArgs):
        run_train(cli)
    elif isinstance(cli, SimArgs):
        run_sim(cli)
    else:
        print(f"Unknown command: {cli}")
        sys.exit(1)


if __name__ == "__main__":
    main()
