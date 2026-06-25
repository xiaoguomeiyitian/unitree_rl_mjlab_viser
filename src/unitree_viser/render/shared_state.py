"""训练状态共享缓冲区 — 主线程写, 渲染线程读."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class TrainingState:
    """主线程写入的最新训练状态, 渲染线程读取并更新到 viser GUI.

    线程安全: update() 和 consume() 通过 _lock 保护 _dirty 标记,
    确保在 ARM 等弱内存模型上也能正确同步.
    """

    iteration: int = 0
    mean_reward: float = 0.0
    episode_length: float = 0.0
    total_timesteps: int = 0
    elapsed_s: float = 0.0
    # 标记是否有新状态 (渲染线程消费后重置)
    _dirty: bool = field(default=False, repr=False)
    # 线程安全锁
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, **kwargs) -> None:
        """主线程调用: 更新状态并标记为 dirty."""
        for k, v in kwargs.items():
            if hasattr(self, k) and not k.startswith("_"):
                setattr(self, k, v)
        with self._lock:
            self._dirty = True

    def consume(self) -> dict | None:
        """渲染线程调用: 读取并清除 dirty 标记. 返回 None 表示无新状态."""
        with self._lock:
            if not self._dirty:
                return None
            self._dirty = False
        return {
            "iteration": self.iteration,
            "mean_reward": self.mean_reward,
            "episode_length": self.episode_length,
            "total_timesteps": self.total_timesteps,
            "elapsed_s": self.elapsed_s,
        }
