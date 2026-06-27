"""unitree-viser - 主 CLI 入口.

子命令:

- ``unitree-viser train <TaskID>`` — 训练模式 (后台异步渲染)
- ``unitree-viser sim <TaskID>``  — 仿真模式 (浏览器驱动虚拟机器人)

使用 tyro 做参数解析, 继承自 unitree_rl_mjlab 的任务注册表.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union

import tyro

logger = logging.getLogger("unitree_viser.cli")


# ── sys.path 修复 ────────────────────────────────────────────────────────
def _ensure_sibling_src_on_path() -> None:
    sys.path[:] = [p for p in sys.path if p != ""]

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

    here = Path(__file__).resolve()
    viser_pkg_dir = here.parent
    viser_src_dir = viser_pkg_dir.parent

    bad_paths = set()
    for p in sys.path:
        pp = Path(p).resolve()
        if pp == viser_src_dir:
            bad_paths.add(p)
        if pp == viser_pkg_dir:
            bad_paths.add(p)
        if (pp / "src").is_dir() and not (pp / "src" / "__init__.py").exists():
            cand = pp / "src"
            if not any((cand / name).exists() for name in ["assets", "tasks", "__init__.py"]):
                bad_paths.add(p)

    sys.path[:] = [p for p in sys.path if p not in bad_paths]

    if sib in sys.path:
        sys.path.remove(sib)
    sys.path.insert(0, sib)

    sys.modules.pop("src", None)
    sys.modules.pop("src.assets", None)
    sys.modules.pop("src.tasks", None)
    try:
        import src  # noqa: F401
    except ImportError:
        pass

    if os.environ.get("UNITREE_VISER_DEBUG_PATH"):
        print(f"[DEBUG-PATH] sys.path[:5]: {sys.path[:5]}", file=sys.stderr)
        if "src" in sys.modules:
            print(f"[DEBUG-PATH] sys.modules['src'].__file__: {sys.modules['src'].__file__!r}", file=sys.stderr)
        else:
            print(f"[DEBUG-PATH] src NOT in sys.modules", file=sys.stderr)


def _stub_mediapy() -> None:
    """mediapy 1.2.6 在 Python 3.12 + numpy 2.x 下无法导入, 注入最小 stub.
    mjlab 1.2.0 的 _configure_mediapy() 强制 import mediapy, 此处提前注入 stub.
    """
    if "mediapy" in sys.modules:
        return
    try:
        import mediapy  # noqa: F401
    except (ImportError, TypeError):
        import types
        stub = types.ModuleType("mediapy")
        stub.__file__ = "(viser-stub)"
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

    viser_port: int = 20006
    """Viser HTTP/WS 端口 (0 = 不启动 Viser)"""
    viser_env_idx: int = 0
    """显示哪个环境"""
    viser_fps: float = 30.0
    """Viser 渲染目标 FPS"""
    enable_control: bool = False
    """是否启用训练控制面板 (暂停/单步/速度滑块)"""

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

    headless: bool = False
    """完全跳过 Viser, 用 unitree_rl_mjlab 原版 train.py (用于 CI)"""

    device: str = "auto"
    """设备选择: ``auto`` (自动检测 GPU), ``cuda:0`` (强制 GPU), ``cpu`` (强制 CPU)"""

    use_wandb: bool = False
    """是否启用 wandb 日志记录 (默认关闭, 使用 tensorboard)"""

    resume: bool = False
    """是否从 checkpoint 恢复训练"""
    checkpoint: str | None = None
    """恢复训练时加载的 .pt checkpoint 路径 (必须与 --resume 同时使用)"""
    resume_log_dir: str | None = None
    """恢复训练时继续使用的日志目录 (None = 自动创建新目录)"""


def run_train(args: TrainArgs) -> None:
    """训练模式主函数."""
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

    # 0. 参数验证
    if not (0 <= args.viser_port <= 65535):
        raise ValueError(f"viser_port 必须在 0-65535 范围内, 收到 {args.viser_port}")
    if args.viser_fps <= 0:
        raise ValueError(f"viser_fps 必须为正数, 收到 {args.viser_fps}")

    # 1. 加载配置
    env_cfg = load_env_cfg(args.task)
    agent_cfg = load_rl_cfg(args.task)
    runner_cls_base = load_runner_cls(args.task)

    from dataclasses import asdict
    agent_cfg = asdict(agent_cfg)

    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    if args.max_iterations is not None:
        agent_cfg["max_iterations"] = args.max_iterations
    if args.seed is not None:
        agent_cfg["seed"] = args.seed
    if args.save_interval is not None:
        agent_cfg["save_interval"] = args.save_interval

    is_tracking = "motion" in env_cfg.commands
    if is_tracking and not args.motion_file and not args.headless:
        raise ValueError(
            f"Task '{args.task}' is a tracking task. "
            "Please provide --motion-file /path/to/motion.npz"
        )
    if args.motion_file and is_tracking:
        env_cfg.commands["motion"].motion_file = args.motion_file

    # 默认禁用 wandb (使用 tensorboard), 除非用户显式开启
    if not args.use_wandb and agent_cfg.get("logger", "tensorboard") == "wandb":
        agent_cfg["logger"] = "tensorboard"
        logger.info("wandb 已默认禁用, 使用 tensorboard 日志 (--use-wandb 可开启)")

    import torch
    if args.device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info("使用设备: %s", device)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.get("clip_actions", 1.0))

    ViserRunnerCls = make_viser_runner_cls(runner_cls_base or MjlabOnPolicyRunner)

    from datetime import datetime
    from pathlib import Path
    if args.resume and args.resume_log_dir:
        log_dir = Path(args.resume_log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
    else:
        log_dir = Path("logs") / "viser" / args.task / datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir.mkdir(parents=True, exist_ok=True)

    runner = ViserRunnerCls(env, agent_cfg, str(log_dir), device)

    # 恢复训练: 加载 checkpoint
    if args.resume and args.checkpoint:
        import torch as _torch
        logger.info("从 checkpoint 恢复: %s", args.checkpoint)
        runner.load(_torch.load(args.checkpoint, map_location=device, weights_only=False))
        logger.info("恢复完成, 从 iteration %d 继续", runner.current_learning_iteration)

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
            logger.info("浏览器访问: http://localhost:%d", args.viser_port)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.warning("Viser 启动失败, 继续训练但无可视化: %s", e)
            viser_handle = None

    try:
        if viser_handle is not None:
            start_viser_render_thread(viser_handle, target_fps=args.viser_fps)
            # 将 render_thread 传给 runner (仅用于统计，不影响训练速度)
            render_thread = gui_state.get("_render_thread")
            logger.info("Viser 已启动: 浏览器仅用于观察, 训练全速运行")
        runner.learn(
            num_learning_iterations=agent_cfg["max_iterations"],
            init_at_random_ep_len=True,
        )
    finally:
        if viser_handle is not None:
            stop_viser_render_thread(viser_handle)
        logger.info("训练完成. 日志目录: %s", log_dir)


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
    """仿真环境数 (默认 1)"""

    viser_port: int = 20006
    """Viser HTTP/WS 端口"""
    viser_env_idx: int = 0
    """显示哪个环境"""

    inject_commands: bool = True
    """是否启用命令注入 GUI"""
    command_name: str = "twist"
    """命令 term 名称"""

    command_source: Literal["gui", "dds", "both"] = "gui"
    """命令注入源: gui/dds/both"""
    dds_domain: int = 0
    """CycloneDDS 域 ID (与遥控器端一致)"""
    dds_interface: str = "lo"
    """CycloneDDS 网络接口 (lo=本机, enp*=以太网)"""
    robot_key: str = "go2_0"
    """DDS topic 后缀"""
    dds_timeout: float = 0.5
    """DDS 消息超时 (秒), 超时后归零; 设 0 禁用"""

    max_steps: int | None = None
    """最多多少步 (None = 无限)"""

    headless: bool = False
    """不启动 Viser, 仅仿真 (用于测试)"""

    device: str = "cpu"
    """设备选择: ``auto`` (自动检测), ``cuda:0`` (强制 GPU), ``cpu`` (强制 CPU, 默认不占 GPU 显存)"""


def run_sim(args: SimArgs) -> None:
    """仿真模式主函数."""
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.envs import ManagerBasedRlEnv

    # 参数验证
    if not (0 <= args.viser_port <= 65535):
        raise ValueError(f"viser_port 必须在 0-65535 范围内, 收到 {args.viser_port}")
    if args.num_envs < 1:
        raise ValueError(f"num_envs 必须 >= 1, 收到 {args.num_envs}")
    if args.max_steps is not None and args.max_steps < 0:
        raise ValueError(f"max_steps 必须 >= 0, 收到 {args.max_steps}")

    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = args.num_envs

    import torch
    if args.device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info("使用设备: %s", device)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

    policy = None
    if args.checkpoint:
        from mjlab.rl import MjlabOnPolicyRunner
        from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper
        from mjlab.tasks.registry import load_rl_cfg
        from unitree_viser.train.runner_subclass import make_viser_runner_cls
        from dataclasses import asdict

        env_wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
        # 加载完整 agent_cfg (包含 algorithm 等键), 统一转为 dict
        agent_cfg = asdict(load_rl_cfg(args.task))
        ViserRunnerCls = make_viser_runner_cls(MjlabOnPolicyRunner)
        runner = ViserRunnerCls(env_wrapped, agent_cfg, None, device)
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device=device)
        logger.info("加载策略: %s", args.checkpoint)
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
        logger.info("使用 %s policy (%d DOF)", args.policy or "zero", num_actions)

    if args.headless:
        logger.info("Headless 模式: 不启动 Viser")
        dds_injector = None
        if args.command_source in ("dds", "both"):
            try:
                from unitree_viser.sim.dds_command_injection import (
                    DdsCommandInjector,
                )
                dds_injector = DdsCommandInjector(
                    server=None,
                    env=env,
                    command_name=args.command_name,
                    dds_domain=args.dds_domain,
                    dds_interface=args.dds_interface,
                    robot_key=args.robot_key,
                    timeout_s=args.dds_timeout,
                )
                dds_injector.start()
                logger.info(
                    "DDS 已订阅: %s (domain=%d, interface=%s)",
                    dds_injector._topic, args.dds_domain, args.dds_interface,
                )
            except Exception as e:
                print(f"[SIM] 跳过 DDS 命令注入: {e}")
                dds_injector = None

        obs, _ = env.reset()
        total_steps = args.max_steps or 100
        try:
            for i in range(total_steps):
                with torch.inference_mode():
                    actions = policy(obs)
                if dds_injector is not None:
                    dds_injector.inject()
                obs, _, _, _, _ = env.step(actions)
                if i % 10 == 0:
                    pending = dds_injector.get_pending() if dds_injector else None
                    cmd_str = (
                        f" cmd=({pending['vx']:.2f},{pending['vy']:.2f},{pending['wz']:.2f})"
                        if pending is not None
                        else ""
                    )
                    print(f"[SIM] step {i}{cmd_str}")
        finally:
            if dds_injector is not None:
                dds_injector.stop()
        logger.info("仿真结束")
        return

    from unitree_viser.sim.sim_viewer import SimViewer

    viewer = SimViewer(
        env=env,
        policy=policy,
        port=args.viser_port,
        env_idx=args.viser_env_idx,
        inject_commands=args.inject_commands,
        command_name=args.command_name,
        command_source=args.command_source,
        dds_domain=args.dds_domain,
        dds_interface=args.dds_interface,
        robot_key=args.robot_key,
        dds_timeout=args.dds_timeout,
    )
    viewer.setup()
    try:
        viewer.run(max_steps=args.max_steps)
    finally:
        viewer.close()


def main() -> None:
    """tyro CLI 入口."""
    if len(sys.argv) > 1 and sys.argv[1] in ("train", "sim"):
        sys.argv[1] = sys.argv[1] + "-args"

    cli = tyro.cli(
        Union[TrainArgs, SimArgs],
        description="Unitree RL Mjlab - Viser 浏览器训练/仿真",
    )
    if isinstance(cli, TrainArgs):
        run_train(cli)
    elif isinstance(cli, SimArgs):
        run_sim(cli)


if __name__ == "__main__":
    main()
