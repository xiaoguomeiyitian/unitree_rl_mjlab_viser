"""DdsCommandInjector — 从 DDS 遥控器消息推算 vx/vy/wz, 写入 env 的 vel_command_b.

与 CommandInjector 接口一致, 让 unitree_remote_ctrl 的 Web 遥控器能驱动 sim 机器人.

Joystick → velocity:
    vx  = ly        (左摇杆 Y 上推 = 前进)
    vy  = -lx       (左摇杆 X 右推 = 物理左移)
    yaw = -rx       (右摇杆 X 右推 = 物理右转)

超时保护: 0.5s 无新消息 → 轴值归零.
降级: unitree_sdk2py 不可用时退化为 Mock 注入器.
"""

from __future__ import annotations

import logging
import threading
import time as _time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import viser  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv

logger = logging.getLogger("unitree_viser.sim.dds")


# ── Module-level constants ─────────────────────────────────────────────────
_RECONNECT_INTERVAL = 5.0  # seconds
_RECONNECT_BACKOFF_MAX = 60.0  # seconds
_TIMEOUT_CHECK_INTERVAL_MIN = 0.1  # seconds
_FACTORY_INIT_DELAY = 0.3  # seconds


# ── unitree_sdk2py 可用性检测 ──────────────────────────────────────────────
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
        pass

    class ChannelSubscriber:  # type: ignore[no-redef]
        def __init__(self, topic: str, msg_type: Any) -> None:
            self._topic = topic

        def Init(self, handler: Any, queue_size: int) -> None:
            pass


_factory_initialized = False
_factory_lock = threading.Lock()


class DdsCommandInjector:
    """从 DDS 订阅 WirelessController_ 消息, 推算 vx/vy/wz, 写入 env vel_command_b."""

    def _axis_to_vel(self, lx: float, ly: float, rx: float) -> tuple[float, float, float]:
        """lx/ly/rx → (vx, vy, yaw)."""
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

        if command_name not in env.command_manager._terms:
            raise ValueError(
                f"Command '{command_name}' not in env.command_manager._terms. "
                f"Available: {list(env.command_manager._terms.keys())}"
            )
        self._term = env.command_manager._terms[command_name]

        self._pending: dict[str, float] = {"vx": 0.0, "vy": 0.0, "wz": 0.0}
        self._lock = threading.Lock()
        self._last_msg_time: float = 0.0

        self._subscriber: ChannelSubscriber | None = None
        self._topic: str = f"rt/{robot_key}/wirelesscontroller"
        self._started: bool = False
        self._running: bool = True
        self._reconnect_interval: float = _RECONNECT_INTERVAL
        self._max_reconnect_attempts: int = 0  # 0 = 无限重连
        self._timeout_thread: threading.Thread | None = None
        self._reconnect_thread: threading.Thread | None = None

    def _do_subscribe(self) -> bool:
        """执行一次订阅, 返回是否成功."""
        global _factory_initialized
        if not _factory_initialized:
            with _factory_lock:
                # Double-check after acquiring lock
                if not _factory_initialized:
                    try:
                        ChannelFactoryInitialize(self._dds_domain, self._dds_interface)
                        _factory_initialized = True
                        logger.info(
                            "DDS factory 初始化完成 (domain=%d, interface=%s)",
                            self._dds_domain, self._dds_interface,
                        )
                        _time.sleep(_FACTORY_INIT_DELAY)
                    except Exception as e:
                        logger.error("DDS factory 初始化失败: %s", e)
                        return False

        try:
            self._subscriber = ChannelSubscriber(self._topic, WirelessController_)
            self._subscriber.Init(self._on_message, 1)
            logger.info("已订阅: %s (超时=%.1fs 归零)", self._topic, self._timeout_s)
            self._last_msg_time = _time.time()
            return True
        except Exception as e:
            logger.error("订阅失败: %s", e)
            self._subscriber = None
            return False

    def start(self) -> None:
        """初始化 DDS factory 并启动订阅线程 (含自动重连)."""
        if self._started:
            return
        self._started = True

        if not _DDS_AVAILABLE:
            logger.warning("unitree_sdk2py 不可用, 退化为 mock (topic=%s)", self._topic)
            return

        self._do_subscribe()

        if self._timeout_s > 0:
            self._timeout_thread = threading.Thread(target=self._timeout_monitor, name="dds-inject-timeout", daemon=True)
            self._timeout_thread.start()

        # 启动重连线程
        self._reconnect_thread = threading.Thread(target=self._reconnect_loop, name="dds-reconnect", daemon=True)
        self._reconnect_thread.start()

    def _on_message(self, msg: Any) -> None:
        """DDS 消息回调 (由 unitree_sdk2py 在内部线程触发)."""
        try:
            vx, vy, yaw = self._axis_to_vel(msg.lx, msg.ly, msg.rx)
            with self._lock:
                self._pending["vx"] = vx
                self._pending["vy"] = vy
                self._pending["wz"] = yaw
                self._last_msg_time = _time.time()
        except Exception as e:
            logger.warning("消息处理失败: %s", e)

    def _reconnect_loop(self) -> None:
        """后台线程: 检测订阅断开并重连 (指数退避).

        成功后重置 attempt 计数器, 确保恢复后立即响应.
        """
        if not _DDS_AVAILABLE:
            return
        attempt = 0
        while self._running:
            _time.sleep(self._reconnect_interval)
            if not self._running:
                break
            if self._subscriber is None:
                attempt += 1
                if self._max_reconnect_attempts > 0 and attempt > self._max_reconnect_attempts:
                    logger.error("重连次数已达上限 (%d), 停止重连", self._max_reconnect_attempts)
                    break
                backoff = min(self._reconnect_interval * (2 ** (attempt - 1)), _RECONNECT_BACKOFF_MAX)
                logger.info("尝试重连 (%d): %s (等待 %.1fs)", attempt, self._topic, backoff)
                _time.sleep(backoff)
                success = self._do_subscribe()
                if success:
                    attempt = 0  # 成功后重置退避计数器

    def _timeout_monitor(self) -> None:
        """监控 DDS 消息超时, 超时后归零."""
        check_interval = max(self._timeout_s * 0.5, _TIMEOUT_CHECK_INTERVAL_MIN)
        while self._running:
            _time.sleep(check_interval)
            if not self._running:
                break
            if self._last_msg_time == 0.0:
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
                        logger.warning("DDS 超时 (%.1fs), 已归零", elapsed)

    def inject(self) -> None:
        """在 env.step() 前调用, 把 _pending 写入 env vel_command_b."""
        import torch as _torch

        with self._lock:
            vx, vy, wz = (
                self._pending["vx"],
                self._pending["vy"],
                self._pending["wz"],
            )

        if hasattr(self._term, "vel_command_b"):
            v = _torch.tensor(
                [vx, vy, wz],
                device=self._env.device,
                dtype=self._term.vel_command_b.dtype,
            )
            self._term.vel_command_b[:] = v.unsqueeze(0).expand(
                self._env.num_envs, 3
            )
        elif hasattr(self._term, "_command_buf"):
            buf = self._term._command_buf
            if buf.shape[-1] >= 3:
                buf[..., 0] = vx
                buf[..., 1] = vy
                buf[..., 2] = wz

    def get_pending(self) -> dict[str, float]:
        """返回当前 pending (调试用)."""
        with self._lock:
            return dict(self._pending)

    def stop(self) -> None:
        """停止后台线程并释放资源."""
        self._running = False
        if self._timeout_thread is not None and self._timeout_thread.is_alive():
            self._timeout_thread.join(timeout=1.0)
        if self._reconnect_thread is not None and self._reconnect_thread.is_alive():
            self._reconnect_thread.join(timeout=1.0)
        self._subscriber = None
        self._timeout_thread = None
        self._reconnect_thread = None


__all__ = ["DdsCommandInjector", "WirelessController_"]