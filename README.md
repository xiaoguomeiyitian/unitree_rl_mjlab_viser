# unitree_rl_mjlab_viser

> Viser 浏览器训练/仿真 — 为 [unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) 提供实时可视化

不修改 unitree_rl_mjlab 源码。纯 import + 子类化, 在外面包一层 Viser 浏览器 UI。

---

## 功能

| 模式 | 命令 | 用途 |
|---|---|---|
| **训练** | `unitree-viser train <TaskID>` | 训练时浏览器实时观察 3D 场景 + 折线图 + 训练控制(暂停/单步/速度) |
| **仿真** | `unitree-viser sim <TaskID>` | 仿真模式, 从浏览器控制虚拟机器人 (vx/vy/wz 命令注入) |
| **Headless** | `--headless` | 不启动 Viser, 仅训练/仿真 (用于 CI) |

---

## 项目结构

```
unitree_rl_mjlab_viser/
├── pyproject.toml                         # 依赖: viser, tyro, unitree_rl_mjlab, mjlab
├── README.md                              # 本文件
├── src/unitree_viser/
│   ├── __init__.py
│   ├── cli.py                             # tyro CLI 入口 (train / sim 子命令)
│   ├── render/
│   │   ├── __init__.py                    # PEP 562 延迟导入
│   │   ├── async_render.py                # ★ 后台渲染线程 (从 trainBot 移植, 165 行)
│   │   ├── viser_setup.py                 # 训练模式 viewer (服务器+场景+GUI)
│   │   └── term_plots.py                  # Viser 折线图包装 (mjlab + fallback)
│   ├── train/
│   │   ├── __init__.py
│   │   ├── runner_subclass.py             # ★ MjlabOnPolicyRunner 子类 + on_iter 钩子
│   │   └── training_controller.py         # 训练控制 (暂停/单步/速度滑块)
│   └── sim/
│       ├── __init__.py
│       ├── command_injection.py           # ★ Viser GUI → env.command_manager 桥
│       └── sim_viewer.py                  # 仿真模式 viewer (单环境 + 命令注入)
├── scripts/
│   ├── train_viser.sh                     # 一键启动训练 + 浏览器
│   └── sim_viser.sh                       # 一键启动仿真
└── tests/
    └── test_smoke.py                      # 9 个 smoke test, 容错处理缺失依赖
```

---

## 安装

### 快速开始 (推荐)

`scripts/install_env.sh` 会**自动检测 NVIDIA GPU**, 选择对应的安装路径:

| 环境 | 安装路径 | 运行时 |
|---|---|---|
| **有 NVIDIA GPU** | PyTorch `+cu128` + nvidia 运行时 + `libegl1-mesa` | `MUJOCO_GL=egl` |
| **无 GPU / AMD 集显** | PyTorch `+cpu` + `libosmesa6` | `CUDA_VISIBLE_DEVICES=""` + `MUJOCO_GL=osmesa` |

```bash
# 1. 克隆本项目 (与 unitree_rl_mjlab 同级)
cd /home/kxy/work/unitree
git clone <this-repo> unitree_rl_mjlab_viser
cd unitree_rl_mjlab_viser

# 2. 一键安装 (自动检测 GPU/CPU)
./scripts/install_env.sh

# 3. 验证
./start.sh version      # 显示版本 + GPU/CPU 模式 + 关键依赖
./start.sh test         # 跑 smoke test (10 passed)
./start.sh list         # 列出所有可用任务
```

### 安装模式选项

```bash
# 强制 GPU 模式 (服务器: 已知有 NVIDIA 卡)
./scripts/install_env.sh --force-gpu

# 强制 CPU 模式 (笔记本没独显 / WSL2 无 GPU 透传)
./scripts/install_env.sh --force-cpu

# 国内用户: 用阿里云镜像加速
./scripts/install_env.sh --mirror cn

# 重建环境 (修改 Python 版本 / 重装)
./scripts/install_env.sh --recreate --force-gpu

# 跳过 apt (自己装过系统依赖)
./scripts/install_env.sh --no-apt
```

### 委托自 start.sh

`start.sh deps` 现在自动委托给 `install_env.sh`:

```bash
./start.sh deps              # 等价于 ./scripts/install_env.sh
./start.sh deps --force-cpu  # 强制 CPU 模式
```

### 手动安装 (不推荐)

如果不想用一键脚本, 可手动:

```bash
# 1. 创建 venv
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade "pip>=23" "wheel" "setuptools<82"

# 2. 装 PyTorch (二选一)
pip install torch==2.11.0+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# 或者 CPU-only:
# pip install torch==2.11.0+cpu torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. 装其他依赖
pip install -e ../unitree_rl_mjlab  # mjlab 1.2.0 + mujoco-warp 3.5.0 + scipy
pip install "mujoco>=3.5.0,<3.6.0" "warp-lang>=1.12.0,<1.13.0"
pip install -e .

# 4. 验证
PYTHONPATH=src python3 tests/test_smoke.py
```

### CPU 模式注意事项

> ⚠️ mjlab 1.2.0 的 CPU 模式是 **experimental**: 上游有 `skipif(not torch.cuda.is_available())` 的测试标记 ("Likely bug on CPU MjWarp").  
> **功能可用**, 但部分高级特性 (大 batch 并行 raycast sensor 等) 可能不稳定.  
> 适合: 调试 / 小规模仿真 / 教学演示. **生产训练请用 GPU**.

### 关键依赖

```toml
[project]
dependencies = [
    "viser>=0.2.0",                        # 浏览器 3D 可视化
    "tyro>=0.9.0",                         # CLI 解析
    "torch>=2.0",                          # 具体 wheel 由 install_env.sh 选择
    "scipy>=1.11.0",                       # mjlab.terrains 依赖
    "mujoco>=3.5.0,<3.6.0",                # mjlab 物理后端
    "warp-lang>=1.12.0,<1.13.0",           # mjlab 底层 kernel 引擎
    "setuptools<82",                       # PyTorch 2.11.0 要求
    "unitree_rl_mjlab",                    # 来自 ../unitree_rl_mjlab (editable)
    "mjlab==1.2.0",
    "mujoco-warp==3.5.0",
]
```

---

## 使用

### 训练模式

```bash
# 基本训练 (开 Viser, 默认 20006 端口)
unitree-viser train Unitree-Go2-Flat

# 带训练控制 (暂停/单步/速度)
unitree-viser train Unitree-G1-Flat --enable-control

# 指定端口 + 减小环境数
unitree-viser train Unitree-Go2-Flat --viser-port 30000 --num-envs 1024

# Headless (不启动 Viser, 等同原版 train.py)
unitree-viser train Unitree-G1-Flat --headless --max-iterations 100

# 跟踪任务 (G1 dance)
unitree-viser train Unitree-G1-Tracking-No-State-Estimation \
    --motion-file src/assets/motions/g1/dance1_subject2.npz

# 一键脚本
./scripts/train_viser.sh Unitree-G1-Flat --enable-control
```

打开浏览器 → `http://localhost:20006` → 看 3D 机器人 + 折线图。

### 仿真模式

```bash
# Zero policy (无训练, 仅看机器人在零力矩下表现)
unitree-viser sim Unitree-G1-Flat

# 加载训练好的 checkpoint
unitree-viser sim Unitree-G1-Flat \
    --checkpoint logs/rsl_rl/g1_velocity/.../model_500.pt

# Random policy
unitree-viser sim Unitree-G1-Flat --policy random

# 关闭命令注入
unitree-viser sim Unitree-G1-Flat --no-inject

# Headless 跑 100 步
unitree-viser sim Unitree-G1-Flat --headless --max-steps 100

# 一键脚本
./scripts/sim_viser.sh Unitree-G1-Flat --policy random
```

打开浏览器 → `http://localhost:20006` → 拖动 vx/vy/wz 滑块控制机器人。

---

## 架构

### 训练模式数据流

```
┌─────────────────────────────────────────────────────────────┐
│ 主线程: ViserRunner.learn()                                  │
│   ├─ rollout  (24× env.step + PPO collect)                  │
│   ├─ PPO update (alg.update)                                │
│   ├─ logger.log + save (every save_interval)                │
│   ├─ TrainingController.wait_if_paused()    [暂停/单步]     │
│   └─ fire _post_iter_hooks                   [Viser 数据]   │
└─────────────────────────────────────────────────────────────┘
                          ▲
                          │ 训练数据
                          │
┌─────────────────────────┴───────────────────────────────────┐
│ Daemon 线程: Viser 后台渲染                                 │
│   ├─ 检查 server.get_clients() → 0 个时 sleep (零开销)    │
│   ├─ 限流 10 FPS                                            │
│   ├─ mjwarp.get_data_into() (GPU→CPU 同步)                  │
│   └─ scene.update_from_mjdata() + server.flush()            │
└─────────────────────────────────────────────────────────────┘
                          ▲
                          │ sim.data (warp 数组)
                          │
┌─────────────────────────┴───────────────────────────────────┐
│ Viser Server (浏览器 0.0.0.0:20006)                         │
│   ├─ 3D 场景 (ViserMujocoScene)                             │
│   ├─ Info 面板 (HTML)                                       │
│   ├─ 折线图 (ViserTermPlotter)                              │
│   └─ 训练控制 (Pause/Step/Speed)                            │
└─────────────────────────────────────────────────────────────┘
```

### 仿真模式数据流

```
┌─────────────────────────────────────────────────────────────┐
│ 主线程: SimViewer.run()                                     │
│   while not should_stop:                                    │
│     ├─ actions = policy(obs)                                │
│     ├─ CommandInjector.inject()           [GUI → vel_cmd]   │
│     ├─ obs = env.step(actions)                              │
│     ├─ 限流 30 FPS                                          │
│     └─ scene.update_from_mjdata(sim.mj_data)                │
└─────────────────────────────────────────────────────────────┘
```

---

## 关键设计决策

1. **不修改 unitree_rl_mjlab 源码** — 用户明确要求, 通过 import + 子类化实现
2. **异步后台渲染线程** — 零训练开销, 客户端断开时 sleep 500ms
3. **Runner 子类 + 重写 learn()** — 比 trainBot 复制 80+ 行循环更精炼 (~50 行复制 + 2 处 [VISER] 标记)
4. **不 fork mjlab / rsl_rl** — 1.2.0 钉版, 避免上游冲突
5. **训练控制用 `threading.Event`** — 跨线程安全, 不阻塞主循环
6. **延迟导入 (PEP 562)** — render 子包不强制依赖 viser, smoke test 能在干净环境跑

---

## 与上游 mjlab/play.py 的差异

| 功能 | play.py (上游) | unitree_rl_mjlab_viser |
|---|---|---|
| 浏览器 3D 场景 | ✅ ViserPlayViewer | ✅ 重用 ViserMujocoScene |
| 暂停/单步/重置 | ✅ | ✅ (训练模式) |
| Reward 折线图 | ✅ | ✅ (训练模式实时更新) |
| 命令注入 | ❌ | ✅ (仿真模式专属) |
| **训练时实时** | ❌ | ✅ (核心特色) |
| 训练控制 | ❌ | ✅ (暂停/单步/速度) |

---

## 故障排查

### 1. 浏览器看不到画面

检查 Viser server 是否启动:

```bash
curl http://localhost:20006
# 应返回 HTML (viser 默认主页)
```

检查控制台:
- 训练启动时: `[INFO] Viser 后台渲染线程已启动 (目标 10.0 FPS)`
- 浏览器连接: `[RENDER] 浏览器已连接 (客户端数=1), 开始渲染 @10.0FPS`
- 浏览器断开: `[RENDER] 无浏览器连接, 暂停渲染 (训练全速运行)`

### 2. 训练时变慢

后台渲染线程有 client-count 检查, 无客户端时**几乎零开销**。
变慢的常见原因:

- **GPU 同步频繁**: 把 `--viser-fps` 调低 (如 5 FPS)
- **WandB / TensorBoard**: 检查是否与 Viser 资源冲突
- **同时启用 --video**: Viser 与 offscreen renderer 都会复制 `mj_data`, 不建议同时开

### 3. 端口被占用

```bash
# 找占用 20006 的进程
lsof -i :20006
# 或换端口
unitree-viser train Unitree-G1-Flat --viser-port 30000
```

### 4. `motion_file not found`

跟踪任务必填, 路径要 .npz:

```bash
# CSV → NPZ 转换
python /home/kxy/work/unitree/unitree_rl_mjlab/scripts/csv_to_npz.py \
    --input-file src/assets/motions/g1/dance1_subject2.csv \
    --output-name dance1_subject2.npz \
    --input-fps 30 --output-fps 50 --robot g1

# 然后训练
unitree-viser train Unitree-G1-Tracking-No-State-Estimation \
    --motion-file /home/kxy/work/unitree/unitree_rl_mjlab/src/motions/g1/dance1_subject2.npz
```

---

## 升级到上游 mjlab 新版本

如果 rsl_rl 升级并修改了 `OnPolicyRunner.learn()`, 你需要:

1. 打开 `src/unitree_viser/train/runner_subclass.py` 的 `ViserRunner.learn()` 方法
2. 对比 rsl_rl 上游 `learn()` 的新版本
3. 把 rsl_rl 的改动合并到 ViserRunner, **保留** 两处 `# [VISER]` 标记
4. 运行 smoke test 验证

---

## License

Apache-2.0 (与 unitree_rl_mjlab 一致)
