# r2wmp 设计文档

## 1. 项目目标

把 **ReDRAW 论文** 的两阶段训练 + Ensemble Residual 思想,移植到 **Unitree A1** 仿真环境上,实现 sim-to-real 迁移。

## 2. 核心思路

### 2.1 ReDRAW 论文关键点

| 设计点 | 描述 |
|--------|------|
| **两阶段训练** | Stage 1 预训练 WM + AC;Stage 2 冻结 WM,只训 Residual |
| **Residual 设计** | 加在 latent stochastic 上,补偿 sim-to-real gap |
| **零初始化** | Residual 严格零初始化,训练初期不破坏已收敛 RSSM |
| **stop_gradients** | Stage 1 时 Residual 不接收 WM loss 梯度 |
| **Stage 2 重建** | Stage 2 用更小的 Residual(1 层),从头训 |
| **100x lr** | Stage 2 用 1e-2 学习率,快速拟合 gap |

### 2.2 A1 任务的特殊性

A1 是**密集 reward + 高维观测 + 复杂动力学**的任务,和 ReDRAW 的 DMC 任务差异巨大:

| 维度 | DMC(ReDRAW) | A1AMP(本项目) |
|------|--------------|---------------|
| 物理仿真 | MuJoCo | Isaac Gym |
| 观测 | 64×64 RGB | 235 维向量 |
| Reward | 稀疏 0/1 | 密集复合 |
| Episode 长度 | 1000 步(~10s) | 1000 步(~20s) |
| 并行规模 | 8-32 | 4096 |
| sim-to-real gap | 物理参数扰动 | **多源 gap**(物理+动力学+接触) |

### 2.3 我们的针对性设计

针对 A1 的**多源 sim-to-real gap**,设计**两层 Residual**:

| Residual | 作用位置 | 补偿的误差类型 |
|----------|---------|---------------|
| **PhysicalResidual** | latent mean | 物理参数(电机扭矩、PD 增益、质量、摩擦) |
| **DynamicsResidual** | deter | 未建模动力学(齿轮间隙、皮带柔性、接触) |

## 3. 三个环境的精确定义

### 3.1 InaccurateSimEnv(Stage 1 训练)

**目的**: 模拟 sim-to-real gap,作为源域

| 配置项 | 值 |
|--------|---|
| num_envs | 4096 |
| 电机扭矩 | [0.65, 0.75] 随机 |
| PD 增益 Kp | [14, 18] 随机 |
| PD 增益 Kd | [0.3, 0.5] 随机 |
| 额外质量 | [+1.0, +3.0] kg |
| 摩擦 | [0.3, 1.5] 随机 |
| 外力推 | [10, 30] N,周期 [3, 8]s |
| 观测 | **完整 235 维**(proprio + privileged + heightmap) |
| 地形 | trimesh 中等难度 |

### 3.2 PseudoRealEnv(评估 + 数据采集)

**目的**: 充当"真机",代表 sim-to-real 迁移的目标

| 配置项 | 值 |
|--------|---|
| num_envs | 64(模拟真机只有少量并行) |
| 电机扭矩 | 固定 1.0(标称) |
| PD 增益 | 固定标称 |
| 摩擦 | 固定标称 |
| 质量 | 固定标称 |
| 外力推 | 无 |
| 观测 | **仅 45 维 proprio**(模拟真机传感器) |
| 地形 | **与 InaccurateSim 一致!** |

### 3.3 关键差异

| 维度 | InaccurateSim | PseudoReal |
|------|--------------|-----------|
| 物理参数 | 随机(模拟 gap) | 标称(代表真机) |
| 观测维度 | 235 维(完整) | 45 维(简化) |
| num_envs | 4096(并行训练) | 64(模拟真机) |
| **地形** | **相同** | **相同** |

## 4. 两阶段训练

### 4.1 Stage 1: Sim 预训练

```
For step in range(2_000_000):
    1. 在 InaccurateSimEnv 收集数据(4096 并行)
    2. WM 训练(每 100 步):
       - observe(real_data, with_residual=True, stop_residual_grad=True)
       - KL(post || sim+residual)  # residual stop grad
    3. AC 训练(每 100 步):
       - imagine(horizon=15, with_residual=False)  # 干净 latent
       - λ-return + AC loss
```

**Stage 1 结束时**:
- RSSM 主干训练好
- Actor / Critic 训练好
- Residual 输出 ≈ 0(零初始化 + 源域 sim 无 gap)

### 4.2 Stage 2: 伪 real 微调

```
1. 加载 Stage 1 checkpoint
2. 重新创建 Residual(1 层,Stage 1 是 3 层)
3. 冻结: encoder, RSSM, decoder, actor, critic
4. Optimizer: 只针对 Residual, lr=1e-2 (100x)

For step in range(1_000_000):
    1. 从 pseudo_real_data.npz 采样 batch
    2. observe(real_data, with_residual=True, stop_residual_grad=False)
    3. KL(post || sim+residual)
    4. 反向传播
```

**Stage 2 结束时**:
- Residual 学到 sim-to-real gap
- Actor 完全不动(Stage 1 训的)

## 5. 评估协议

### 5.1 三策略对比

| 策略 | 来源 | 含义 |
|------|------|------|
| **A. Zero-shot** | Stage 1 模型(Residual≈0) | baseline |
| **B. Residual 微调** | Stage 1 + Stage 2 Residual | sim-to-real gap 闭合 |
| **C. Upper bound** | Stage 2 全量微调 | 性能上限(可选) |

### 5.2 评估指标

| 指标 | 含义 |
|------|------|
| episode_return | 1000 步 reward 总和 |
| success_rate | return > 200 的 episode 比例 |
| fall_rate | 摔倒 episode 比例 |
| mean_episode_length | 平均 episode 长度 |

### 5.3 Gap 闭合率

```
gap_before = sim_return - A_real_return
gap_after = sim_return - B_real_return
gap_closed_pct = (gap_before - gap_after) / |gap_before| * 100%
```

**期望**: B > A, gap_closed_pct > 70% 表示算法有效。

## 6. 文件清单

```
D:\songay\sim2real\r2wmp\
├── README.md
├── configs/
│   ├── env_inaccurate_sim.yaml    # InaccurateSim 配置
│   ├── env_pseudo_real.yaml        # PseudoReal 配置
│   └── train.yaml                  # 训练超参
├── envs/
│   ├── inaccurate_sim_env.py
│   ├── pseudo_real_env.py
│   └── obs_wrapper.py
├── models/
│   ├── residual.py                 # PhysicalResidual + DynamicsResidual
│   ├── rssm.py
│   ├── encoder.py / decoder.py
│   ├── actor.py / critic.py
│   └── world_model.py
├── training/
│   ├── trainer.py
│   ├── wm_loss.py
│   ├── ac_loss.py
│   └── replay_buffer.py
├── evaluation/
│   └── eval_protocol.py
├── scripts/
│   ├── train_stage1.py
│   ├── collect_pseudo_real_data.py
│   ├── train_stage2.py
│   └── eval_compare.py
├── docs/
│   ├── design.md
│   ├── stage1.md
│   ├── stage2.md
│   └── evaluation.md
├── checkpoints/
├── datasets/
└── results/
```

## 7. 快速开始

```bash
# 1. Stage 1 训练(2M 步,需要 Isaac Gym + WMP)
python scripts/train_stage1.py --total_steps 2000000

# 2. 采集伪 real 数据(200 episodes)
python scripts/collect_pseudo_real_data.py --num_episodes 200

# 3. Stage 2 训练(1M 步)
python scripts/train_stage2.py --total_steps 1000000

# 4. 三策略对比
python scripts/eval_compare.py --num_episodes 50
```

## 8. 参考资料

- ReDRAW 论文: *Adapting World Models with Latent-State Dynamics Residuals*
- WMP: NVIDIA World Model PPO
- DreamerV3: Mastering Diverse Domains through World Models
- A1AMPCfg: D:\songay\sim2real\WMP\legged_gym\envs\a1\a1_amp_config.py