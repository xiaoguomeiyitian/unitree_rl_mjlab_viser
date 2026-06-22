"""DdsCommandInjector — 从 DDS 遥控器消息推算 vx/vy/wz, 写入 env 的 vel_command_b.

与 Viser GUI 滑块注入器 (CommandInjector) 平行存在, 接口完全一致 (``inject()``).
目的是让 unitree_remote_ctrl 的 Web 遥控器 / 物理手柄能驱动 viser sim 中的机器人.

通信链路::

    浏览器遥控器 → WebSocket → Flask → DDS Publisher (100Hz)
                                → rt/{robot_key}/wirelesscontroller (WirelessController_ 消息)
                                → ChannelSubscriber (本模块订阅)
                                → lx/ly/rx → 推算 (vx, vy, wz)
                                → env.step() 前写入 vel_command_b

Joystick → velocity 换算 (与 JoystickSimulator.target_velocity 一致):
    vx  = ly        (左摇杆 Y 上推 = 前进)
    vy  = -lx       (左摇杆 X 右推 = 负向 → 真实左移)
    yaw = -rx       (右摇杆 X 右推 = 负向 → 真实右转)

线程模型
========
- DDS 订阅回调: 由 unitree_sdk2py 在内部线程触发
- env.step() / inject(): 在主线程
- 共享状态: ``_pending`` dict + ``_lock`` (Lock 保护跨线程读写)

超时保护
========
- 0.5s 无新消息 → 轴值归零 (与 DdsJoystickSource 行为一致)
- 防止遥控器断连/网页关闭后机器人持续以最后指令运动

降级方案
========
- ``unitree_sdk2py`` 不可用时, 退化为 Mock 注入器 (订阅永不触发, 速度恒为 0)
- 让 smoke test 在干净环境也能跑 (不强制 unitree_sdk2py)
"""

from __future__ import annotations

import threading
import time as _time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import viser  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv


# ── unitree_sdk2py 可用性检测 ──────────────────────────────────────────────
# 与 joystick_publisher.py 的降级策略保持一致.
try:
    from unitree_sdk2py.core.channel import (  # type: ignore[import-not-found]
        ChannelFactoryInitialize,
        ChannelSubscriber,
    )
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import (  # type: ignore[import-not-found]
        WirelessController_,
    )
    from unitree_sdk2py.idl.default import (  # type: ignore[import-not-found]
        unitree_go_msg_dds__WirelessController_,
    )

    _DDS_AVAILABLE = True
except ImportError:
    _DDS_AVAILABLE = False

    # ── Mock 降级 (smoke test / 无 DDS 环境) ──
    class WirelessController_:  # type: ignore[no-redef]
        def __init__(self) -> None:
            self.lx: float = 0.0
            self.ly: float = 0.0
            self.rx: float = 0.0
            self.ry: float = 0.0
            self.keys: int = 0

    def unitree_go_msg_dds__WirelessController_():  # type: ignore[no-redef]
        return WirelessController_()

    def ChannelFactoryInitialize(domain_id: int, interface: str) -> None:
        """Mock: 无操作."""
        pass

    class ChannelSubscriber:  # type: ignore[no-redef]
        def __init__(self, topic: str, msg_type: Any) -> None:
            self._topic = topic

        def Init(self, handler: Any, queue_size: int) -> None:
            """Mock: 不实际订阅, 仅记录."""


# ── 模块级 DDS factory 标志 ────────────────────────────────────────────────
# ChannelFactoryInitialize 进程级单例, 避免多次 init 报错.
_factory_initialized = False


class DdsCommandInjector:
    """从 DDS 订阅 WirelessController_ 消息, 推算 vx/vy/wz, 写入 env vel_command_b.

    Args:
        server: Viser server (可选, 仅日志用)
        env: mjlab ManagerBasedRlEnv
        command_name: env.command_manager._terms 中的 key (默认 "twist")
        dds_domain: CycloneDDS 域 ID (默认 0)
        dds_interface: 网络接口 ("lo"=本机回环, "enp*"=以太网)
        robot_key: DDS topic 后缀, 决定订阅 rt/{robot_key}/wirelesscontroller
        timeout_s: DDS 消息超时 (秒), 超时后归零. 默认 0.5s, 设 0 禁用.
    """

    # Joystick → velocity 换算常量 (来自 JoystickSimulator.target_velocity)
    # 注: 不 import JoystickSimulator 避免依赖 unitree_remote_ctrl.
    def _axis_to_vel(self, lx: float, ly: float, rx: float) -> tuple[float, float, float]:
        """lx/ly/rx → (vx, vy, yaw).

        vx  = ly        左摇杆 Y 上推 → 前进 (m/s)
        vy  = -lx       左摇杆 X 右推 → 物理左移 (m/s) [反向]
        yaw = -rx       右摇杆 X 右推 → 物理右转 (rad/s) [反向]
        """
        return (float(ly), float(-lx), float(-rx))

    def __init__(
        self,
        server: "viser.ViserServer | None",
        env: "ManagerBasedRlEnv",
        command_name: str = "twist",
        dds_domain: int = 0,
        dds_interface: str = "lo",
        robot_key: str = "go2_0",
        timeout_s: float = 0.5,
    ) -> None:
        self._env = env
        self._command_name = command_name
        self._dds_domain = dds_domain
        self._dds_interface = dds_interface
        self._robot_key = robot_key
        self._timeout_s = timeout_s

        # ── 解析 command term ──
        if command_name not in env.command_manager._terms:
            raise ValueError(
                f"Command '{command_name}' not in env.command_manager._terms. "
                f"Available: {list(env.command_manager._terms.keys())}"
            )
        self._term = env.command_manager._terms[command_name]

        # ── 共享状态 (DDS 线程写, 主线程读) ──
        self._pending: dict[str, float] = {"vx": 0.0, "vy": 0.0, "wz": 0.0}
        self._lock = threading.Lock()
        self._last_msg_time: float = 0.0

        # ── DDS 订阅 ──
        self._subscriber: ChannelSubscriber | None = None
        self._topic: str = f"rt/{robot_key}/wirelesscontroller"
        self._started: bool = False
        self._running: bool = True

    def start(self) -> None:
        """初始化 DDS factory 并启动订阅线程.

        注意: 此方法必须在主线程调用一次 (ChannelFactoryInitialize 进程单例).
        """
        if self._started:
            return
        self._started = True

        if not _DDS_AVAILABLE:
            print(
                f"[DDS-INJECT] ⚠️ unitree_sdk2py 不可用, DdsCommandInjector 退化为 mock "
                f"(topic={self._topic}, 速度恒为 0)"
            )
            return

        # 初始化 DDS factory (进程单例)
        global _factory_initialized
        if not _factory_initialized:
            try:
                ChannelFactoryInitialize(self._dds_domain, self._dds_interface)
                _factory_initialized = True
                print(
                    f"[DDS-INJECT] DDS factory 初始化完成 "
                    f"(domain={self._dds_domain}, interface={self._dds_interface})"
                )
                # CycloneDDS lo 接口 unicast 发现需要时间
                _time.sleep(0.3)
            except Exception as e:
                print(f"[DDS-INJECT] ❌ DDS factory 初始化失败: {e}")
                return

        # 创建订阅
        try:
            self._subscriber = ChannelSubscriber(self._topic, WirelessController_)
            self._subscriber.Init(self._on_message, 1)  # 1 帧缓存
            print(
                f"[DDS-INJECT] ✅ 已订阅: {self._topic} "
                f"(超时={self._timeout_s}s 归零)"
            )
            self._last_msg_time = _time.time()
        except Exception as e:
            print(f"[DDS-INJECT] ❌ 订阅失败: {e}")
            self._subscriber = None

        # 启动超时监控线程
        if self._timeout_s > 0:
            t = threading.Thread(target=self._timeout_monitor, name="dds-inject-timeout", daemon=True)
            t.start()

    def _on_message(self, msg: Any) -> None:
        """DDS 消息回调 (由 unitree_sdk2py 在内部线程触发).

        把 lx/ly/rx 推算成 (vx, vy, wz) 写入 _pending.
        """
        try:
            vx, vy, yaw = self._axis_to_vel(msg.lx, msg.ly, msg.rx)
            with self._lock:
                self._pending["vx"] = vx
                self._pending["vy"] = vy
                self._pending["wz"] = yaw
                self._last_msg_time = _time.time()
        except Exception as e:
            # 不让 DDS 线程崩
            print(f"[DDS-INJECT] 消息处理失败: {e}")

    def _timeout_monitor(self) -> None:
        """监控 DDS 消息超时, 超时后归零 (防止断连后机器人持续运动)."""
        check_interval = max(self._timeout_s * 0.5, 0.1)
        while self._running:
            _time.sleep(check_interval)
            if not self._running:
                break
            if self._last_msg_time == 0.0:
                # 还没收到过消息, 不触发归零
                continue
            elapsed = _time.time() - self._last_msg_time
            if elapsed > self._timeout_s:
                with self._lock:
                    if (
                        self._pending["vx"] != 0.0
                        or self._pending["vy"] != 0.0
                        or self._pending["wz"] != 0.0
                    ):
                        self._pending["vx"] = 0.0
                        self._pending["vy"] = 0.0
                        self._pending["wz"] = 0.0
                        print(
                            f"[DDS-INJECT] ⚠️ DDS 超时 ({elapsed:.1f}s), 已归零"
                        )

    def inject(self) -> None:
        """在 env.step() 前调用, 把 _pending 写入 env vel_command_b.

        接口与 CommandInjector.inject() 一致, SimViewer 无需区分调用对象.
        """
        import torch as _torch

        with self._lock:
            vx, vy, wz = (
                self._pending["vx"],
                self._pending["vy"],
                self._pending["wz"],
            )

        if hasattr(self._term, "vel_command_b"):
            # UniformVelocityCommand 风格 (mjlab 速度任务的标准 term)
            v = _torch.tensor(
                [vx, vy, wz],
                device=self._env.device,
                dtype=self._term.vel_command_b.dtype,
            )
            self._term.vel_command_b[:] = v.unsqueeze(0).expand(
                self._env.num_envs, 3
            )
        elif hasattr(self._term, "_command_buf"):
            # 通用 fallback
            buf = self._term._command_buf
            if buf.shape[-1] >= 3:
                buf[..., 0] = vx
                buf[..., 1] = vy
                buf[..., 2] = wz
        # Motion tracking 任务不支持命令注入, 静默跳过

    def get_pending(self) -> dict[str, float]:
        """返回当前 pending (调试用)."""
        with self._lock:
            return dict(self._pending)

    def stop(self) -> None:
        """停止超时线程 (DDS 订阅无法主动取消, 进程退出时自然清理)."""
        self._running = False


__all__ = ["DdsCommandInjector", "WirelessController_"]