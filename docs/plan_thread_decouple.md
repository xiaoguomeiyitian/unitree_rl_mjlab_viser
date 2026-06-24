# 方案：训练与 Viser 渲染完全线程隔离

> 目标：主线程（训练）与 Viser 渲染/连接完全解耦，浏览器连接/断开不影响训练速度。

---

## 1. 问题分析

### 1.1 当前线程架构

训练启动后共 4 个线程：

| # | 线程名 | Daemon | 职责 |
|---|--------|--------|------|
| 1 | 主线程 | No | 运行 `runner.learn()` 训练循环 |
| 2 | `viser-render` | Yes | 从 MuJoCo 取数据 → 更新 viser scene |
| 3 | `_server_thread` (viser) | Yes | asyncio 事件循环：HTTP + WebSocket I/O |
| 4 | DDS 订阅线程 (可选) | Yes | 监听遥控器 DDS 消息 |

此外，viser 内部有一个 `ThreadPoolExecutor(max_workers=32)` 处理 GUI 回调和 `on_client_connect`。

### 1.2 问题根因

主线程**每迭代**调用 `_trigger_post_iter_hooks()`，其中包含：

1. `update_training_info()` → 操作 viser HTML 控件
2. `push_reward_to_plot()` → 操作 viser 折线图
3. `training_controller.wait_if_paused()` → 检查暂停 Event

这些 viser GUI 操作在 viser 的 `ThreadPoolExecutor` 中执行。虽然不直接阻塞主线程，但：

- **高频调用**：每迭代一次调用一次（训练 100-500 it/s = 每秒 100-500 次 GUI 操作）
- **线程池争抢**：浏览器连接时触发的 GUI 初始化回调（camera、scene 加载等）占用同一个 `ThreadPoolExecutor`，导致训练 hook 的 GUI 更新排队等待
- **训练抖动**：迭代间隔不稳定，影响采集数据质量

### 1.3 现象解释

> "初始没有浏览器连接主线程在全力训练，导致后续浏览器无法连接 viser"

- 无浏览器连接时，渲染线程 0.5s sleep，主线程全速训练
- 浏览器连接时，viser server 线程触发 `on_client_connect` 回调（在 ThreadPoolExecutor 中执行），大量 GUI 初始化操作占用线程池
- 主线程的 `_default_post_iter_hook` 也提交 GUI 操作到同一个 ThreadPoolExecutor，排队等待
- 极端情况下，高频训练迭代产生的 GUI 操作堆积，进一步挤占线程池资源

---

## 2. 方案

### 2.1 核心思路

**主线程只写共享缓冲区，所有 viser GUI 操作移到渲染线程。**

```
Before:
  主线程(训练) ──每迭代──→ viser GUI 更新 (ThreadPoolExecutor)
  渲染线程 ──30FPS──→ viser scene 更新

After:
  主线程(训练) ──每迭代──→ shared_state 缓冲区 (写)
  渲染线程 ──30FPS──→ viser scene 更新 + 每秒1-2次 GUI 更新
```

### 2.2 修改步骤

#### Step 1: 新增共享状态缓冲区

**文件**: `src/unitree_viser/render/shared_state.py`（新增）

```python
"""训练状态共享缓冲区 — 主线程写, 渲染线程读."""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class TrainingState:
    """主线程写入的最新训练状态, 渲染线程读取并更新到 viser GUI."""

    iteration: int = 0
    mean_reward: float = 0.0
    episode_length: float = 0.0
    total_timesteps: int = 0
    elapsed_s: float = 0.0
    # 标记是否有新状态 (渲染线程消费后重置)
    _dirty: bool = field(default=False, repr=False)

    def update(self, **kwargs) -> None:
        """主线程调用: 更新状态并标记为 dirty."""
        for k, v in kwargs.items():
            if hasattr(self, k) and not k.startswith("_"):
                setattr(self, k, v)
        self._dirty = True

    def consume(self) -> dict | None:
        """渲染线程调用: 读取并清除 dirty 标记. 返回 None 表示无新状态."""
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
```

**设计决策**：
- 不用 `threading.Lock`：Python GIL 保证简单类型赋值原子性，单写单读足够
- `_dirty` 标记避免渲染线程重复更新 GUI

#### Step 2: 修改 `runner_subclass.py` — 主线程零 viser 调用

**文件**: `src/unitree_viser/train/runner_subclass.py`

修改 `_trigger_post_iter_hooks()`：

```python
def _trigger_post_iter_hooks(self, iteration: int, loss_dict: dict) -> None:
    """触发所有注册的 post_iter hooks — 只写共享缓冲区, 不调 viser."""
    # 更新共享状态 (渲染线程负责更新 viser GUI)
    if self.viser_gui_state is not None and hasattr(self.viser_gui_state, "_training_state"):
        self.viser_gui_state._training_state.update(
            iteration=iteration,
            mean_reward=0.0,  # TODO: 从 loss_dict 提取
            episode_length=0.0,
        )

    # 训练控制保留在主线程 (需要阻塞主线程)
    if self.training_controller is not None:
        should_quit = self.training_controller.wait_if_paused()
        if should_quit:
            return

    # 用户自定义 hooks
    if self._post_iter_hooks:
        env_unwrapped = self.env.unwrapped
        for hook in self._post_iter_hooks:
            try:
                hook(iteration, env_unwrapped, loss_dict)
            except Exception as e:
                print(f"[VISER] post_iter_hook 失败: {e}")
```

移除 `_default_post_iter_hook()` 中对 `update_training_info` 和 `push_reward_to_plot` 的调用（或标记为 deprecated）。

#### Step 3: 修改 `async_render.py` — 渲染线程承担 GUI 更新

**文件**: `src/unitree_viser/render/async_render.py`

在渲染循环中增加 GUI 更新逻辑：

```python
# 在 _render_loop 中:
last_gui_update_t = 0.0
GUI_UPDATE_INTERVAL = 1.0  # 每秒更新一次 GUI

# ... 现有渲染逻辑 ...

# GUI 更新 (每秒最多 1 次)
now = _time.time()
if now - last_gui_update_t >= GUI_UPDATE_INTERVAL:
    last_gui_update_t = now
    state = training_state.consume() if training_state else None
    if state is not None:
        update_training_info(gui_state, **state)
        push_reward_to_plot(
            gui_state,
            state["iteration"],
            state["mean_reward"],
            state["episode_length"],
        )
```

#### Step 4: 修改 `viser_setup.py` — 挂载共享状态

**文件**: `src/unitree_viser/render/viser_setup.py`

在 `setup_viser_for_training()` 中创建并挂载 `TrainingState`：

```python
from unitree_viser.render.shared_state import TrainingState

# 在 setup_viser_for_training() 中:
training_state = TrainingState()
gui_state["_training_state"] = training_state
```

### 2.3 修改文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `src/unitree_viser/render/shared_state.py` | 新增 | 共享状态缓冲区 |
| `src/unitree_viser/train/runner_subclass.py` | 修改 | hook 改为写缓冲区 |
| `src/unitree_viser/render/async_render.py` | 修改 | 渲染线程增加 GUI 更新 |
| `src/unitree_viser/render/viser_setup.py` | 修改 | 挂载 TrainingState |

### 2.4 数据流

```
主线程 (训练循环)                    渲染线程 (viser-render)
─────────────────                   ─────────────────────
  env.step()                          while not stop:
  ↓                                     ├─ 检查浏览器连接
  algorithm.update()                    ├─ 30FPS: scene.update_from_mjdata()
  ↓                                     └─ 每秒1次: update_training_info()
  logger.log() ← monkey-patch          ↑
  ↓                                  shared_state.consume()
  _trigger_post_iter_hooks()
  ↓
  shared_state.update(iteration=...)
  ↓
  training_controller.wait_if_paused()
```

### 2.5 效果

- **主线程零 viser 调用**：浏览器连接/断开不影响训练速度
- **GUI 更新频率降低**：从 "每迭代" 降到 "每秒 1-2 次"，viser 内部压力减小
- **渲染线程统一管理**：scene 更新 + GUI 更新在同一线程，无跨线程竞争
- **训练抖动消除**：迭代间隔稳定，采集数据质量提升

### 2.6 验证方法

1. `python3 -m py_compile` 所有修改的 .py 文件通过
2. 启动训练后，无浏览器连接时主线程全速运行（无 viser 调用开销）
3. 浏览器连接后，渲染线程正常 30FPS 渲染 + 每秒更新 GUI
4. 浏览器断开/重连不影响训练速度
5. `bash -n start.sh` 通过

### 2.7 后续考虑

1. 如果未来需要更复杂的共享状态（如多 env 的 per-env reward），可考虑用 `queue.Queue` 或 `threading.Lock`
2. `training_controller.wait_if_paused()` 当前用 `Event.wait(timeout=0.05)` 已是非阻塞，影响极小，暂保留在主线程
3. 可考虑将 `update_training_info` 中的 HTML 字符串拼接优化为模板缓存
