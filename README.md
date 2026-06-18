# r2wmp: ReDRAW → Unitree A1 Sim-to-Real Transfer

把 **ReDRAW 论文**(Adapting World Models with Latent-State Dynamics Residuals)的两阶段训练 + Ensemble Residual 思想,移植到 **Unitree A1** 仿真环境,实现 sim-to-real 迁移。

---

## 📋 目录

1. [项目结构](#1-项目结构)
2. [环境要求](#2-环境要求)
3. [安装步骤](#3-安装步骤)
4. [快速开始](#4-快速开始)
5. [三阶段详细流程](#5-三阶段详细流程)
6. [配置参数说明](#6-配置参数说明)
7. [常见问题](#7-常见问题)
8. [评估输出](#8-评估输出)

---

## 1. 项目结构

```
D:\songay\sim2real\r2wmp\
├── README.md                       # 本文件
├── configs/                        # 配置
│   ├── env_inaccurate_sim.yaml    # InaccurateSim 配置(Stage 1 训练)
│   ├── env_pseudo_real.yaml        # PseudoReal 配置(评估)
│   └── train.yaml                  # 训练超参
├── envs/                           # 环境层
│   ├── base_env.py                # 抽象基类
│   ├── wmp_env_base.py            # WMP LeggedRobot 包装
│   ├── inaccurate_sim_env.py      # 不准确 sim(域随机化)
│   └── pseudo_real_env.py         # 伪 real(标称物理)
├── models/                         # 模型层
│   ├── residual.py                # PhysicalResidual + DynamicsResidual
│   ├── rssm.py                    # RSSM(K-tuple categorical)
│   ├── encoder.py / decoder.py
│   ├── actor.py / critic.py      # Dreamer AC
│   └── world_model.py             # 整合
├── training/                       # 训练逻辑
│   ├── trainer.py                 # Stage 1 + Stage 2 主循环
│   ├── wm_loss.py                 # WM 损失
│   ├── ac_loss.py                 # AC 损失(λ-return)
│   └── replay_buffer.py           # Replay buffer
├── evaluation/                     # 评估
│   └── eval_protocol.py           # 三策略对比
├── scripts/                        # 可执行脚本
│   ├── train_stage1.py
│   ├── collect_pseudo_real_data.py
│   ├── train_stage2.py
│   └── eval_compare.py
├── tests/                          # 单元测试
│   └── test_mock.py
├── docs/                           # 文档
│   ├── design.md
│   ├── install.md
│   ├── troubleshooting.md
│   ├── review_checklist.md
│   ├── implementation_targets.md
│   └── claude_review_*.md
├── checkpoints/                    # 模型权重
├── datasets/                       # 数据集
├── logs/                           # 训练日志
└── results/                        # 评估结果
```

---

## 2. 环境要求

### 硬件

| 组件 | 最低 | 推荐 |
|------|------|------|
| GPU | NVIDIA RTX 3060 (12GB) | NVIDIA A100 (40GB) |
| CPU | 8 核 | 16 核 |
| 内存 | 16 GB | 32 GB |
| 磁盘 | 10 GB | 20 GB |

### 软件

| 组件 | 版本 | 备注 |
|------|------|------|
| OS | Windows 10/11 (WSL2) 或 Linux | Isaac Gym 不支持 Windows 原生 |
| CUDA | 11.7+ | 必须匹配 PyTorch |
| Python | 3.8 / 3.9 | WMP 用 3.8 |
| Isaac Gym | Preview 4 | NVIDIA 官方 |
| WMP | 本仓库 `D:\songay\sim2real\WMP` | NVIDIA WMP |
| PyTorch | 1.10 - 2.0 | 与 CUDA 匹配 |

### 关键依赖

```
torch>=1.10,<2.1
numpy>=1.20
pyyaml>=5.4
isaacgym (Preview 4)
legged_gym (from WMP)
rsl_rl (from WMP)
```

---

## 3. 安装步骤

### 步骤 1: 安装 CUDA + PyTorch

```bash
# 验证 GPU
nvidia-smi

# 安装 PyTorch(以 CUDA 11.8 为例)
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# 验证
python -c "import torch; print(torch.cuda.is_available())"
# 应该输出 True
```

### 步骤 2: 安装 Isaac Gym Preview 4

```bash
# 下载: https://developer.nvidia.com/isaac-gym

# Linux
tar -xvf isaacgym_preview4.tar.gz
cd isaacgym_preview4/python
pip install -e .

# 验证
cd ../examples
python joint_monkey.py  # 应该能看到猴子
```

### 步骤 3: 安装 WMP 依赖

```bash
# WSL2 中:/mnt/d/songay/sim2real/WMP
cd /mnt/d/songay/sim2real/WMP

# 安装 rsl_rl
cd rsl_rl && pip install -e . && cd ..

# 安装 legged_gym
cd legged_gym && pip install -e . && cd ..
```

### 步骤 4: 设置环境变量

```bash
# Linux / WSL2
export WMP_ROOT=/mnt/d/songay/sim2real/WMP
export PYTHONPATH=$PYTHONPATH:$WMP_ROOT

# Windows (PowerShell)
$env:WMP_ROOT = "D:\songay\sim2real\WMP"
$env:PYTHONPATH = "$env:PYTHONPATH;$env:WMP_ROOT"
```

### 步骤 5: 验证 WMP 导入

```bash
python -c "
import sys
sys.path.append('/mnt/d/songay/sim2real/WMP')
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.a1.a1_amp_config import A1AMPCfg
cfg = A1AMPCfg()
print(f'A1AMPCfg OK: num_envs={cfg.env.num_envs}')
"
```

### 步骤 6: 安装 r2wmp

```bash
cd /mnt/d/songay/sim2real/r2wmp
pip install pyyaml matplotlib tensorboard pytest  # 可选
```

### 步骤 7: 验证 r2wmp

```bash
# 静态验证
python tests/test_mock.py  # 需要 torch

# 期望:全部测试通过
```

---

## 4. 快速开始

### 完整流程(5 步)

```bash
# 1. Stage 1 训练(在 InaccurateSim 上)
python scripts/train_stage1.py --total_steps 2000000

# 2. 采集伪 real 数据(用 Stage 1 训出的 Actor)
python scripts/collect_pseudo_real_data.py --num_episodes 200

# 3. Stage 2 训练(冻结 WM,只训 Residual)
python scripts/train_stage2.py --total_steps 1000000

# 4. 三策略对比评估
python scripts/eval_compare.py --num_episodes 50

# 5. 查看结果
cat results/comparison.json
```

### 最小测试(验证环境)

```bash
# Stage 1 短跑(1000 步,验证能跑通)
python scripts/train_stage1.py --total_steps 1000

# Stage 2 短跑
python scripts/train_stage2.py --total_steps 1000

# 评估 10 episodes
python scripts/eval_compare.py --num_episodes 10
```

---

## 5. 三阶段详细流程

### 阶段 0:准备

```bash
# 确保 WMP 仓库存在
ls /mnt/d/songay/sim2real/WMP/   # 或 D:\songay\sim2real\WMP\

# 设置环境变量
export WMP_ROOT=/mnt/d/songay/sim2real/WMP

# 创建必要目录
cd /mnt/d/songay/sim2real/r2wmp
mkdir -p checkpoints datasets logs results
```

### 阶段 1:Sim 预训练(2M 步,~6-12 小时)

**目的**: 在 InaccurateSimEnv 上训练 World Model + Actor-Critic。Residual 零初始化,停梯度。

```bash
python scripts/train_stage1.py \
  --config configs/train.yaml \
  --num_envs 4096 \
  --total_steps 2000000 \
  --device cuda:0 \
  --headless
```

**关键参数**:
- `--num_envs 4096`: Isaac Gym 并行数
- `--total_steps 2000000`: 训练步数
- 显存不够?降低到 `--num_envs 2048`

**输出**:
- `checkpoints/stage1_step{100k,200k,...,2M}.ckpt`: 定期保存
- `checkpoints/stage1_final.ckpt`: 最终保存

**监控**:
```bash
# 看日志
tail -f logs/stage1.log

# GPU 监控
nvidia-smi -l 1
```

**预期行为**:
- `episode_return` 逐渐上升(目标 > 200)
- `kl_loss` 逐渐下降(< 5 nat)
- `Residual 输出 ≈ 0`(零初始化)

### 阶段 1.5:采集伪 real 数据(200 episodes,~1-2 小时)

**目的**: 用 Stage 1 训的 Actor 在 PseudoRealEnv 上跑,收集 transitions 作为 Stage 2 训练数据。

```bash
python scripts/collect_pseudo_real_data.py \
  --checkpoint checkpoints/stage1_final.ckpt \
  --output datasets/pseudo_real_data.npz \
  --num_episodes 200 \
  --max_steps 1000 \
  --device cuda:0
```

**输出**:
- `datasets/pseudo_real_data.npz`: 包含 obs/action/reward/next_obs/done/is_first

**注意**:
- Actor 在 PseudoReal 上可能表现差(这正是 ReDRAW 要解决的问题)
- 如果 return ≈ 0,数据质量可能不好,考虑增加 episodes 到 500

### 阶段 2:伪 real 微调 Residual(1M 步,~3-6 小时)

**目的**: 冻结主 RSSM + Actor + Encoder + Decoder,**只训练新创建的 Residual**(1 层)。

```bash
python scripts/train_stage2.py \
  --stage1_ckpt checkpoints/stage1_final.ckpt \
  --real_data datasets/pseudo_real_data.npz \
  --total_steps 1000000 \
  --device cuda:0
```

**关键行为**:
1. 加载 Stage 1 checkpoint
2. **重新创建 Residual**(从 3 层 → 1 层,零初始化)
3. 冻结主网络(`freeze_main_network`)
4. 只优化 Residual,学习率 1e-2(100x)

**输出**:
- `checkpoints/stage2_step{100k,...,1M}.ckpt`
- `checkpoints/stage2_final.ckpt`

**预期行为**:
- Residual loss 下降
- Residual 输出 ≠ 0(学到了 sim-to-real gap)

### 阶段 3:三策略对比评估(50 episodes,~30 分钟)

```bash
python scripts/eval_compare.py \
  --stage1_ckpt checkpoints/stage1_final.ckpt \
  --stage2_residual checkpoints/stage2_final.ckpt \
  --num_episodes 50
```

**输出**:
- `results/comparison.json`: 三策略指标
- 终端输出汇总

---

## 6. 配置参数说明

### `configs/train.yaml` 主要参数

```yaml
# RSSM 配置
rssm:
  deter: 512              # GRU 隐藏状态维度
  stoch: 32               # 随机变量维度
  discrete: 32            # K-tuple categorical 类别数
  hidden: 512             # MLP 隐藏层
  unimix_ratio: 0.01      # OneHot 平滑
  min_std: 0.1            # 最小 std

# Residual 配置(Stage 1 vs Stage 2)
residual:
  stage1_n_layers: 3      # Stage 1 残差层数
  stage1_hidden: 64
  stage2_n_layers: 1      # Stage 2 残差层数(更小)
  stage2_hidden: 64
  dynamics_hidden: 128     # DynamicsResidual hidden
  history_len: 4          # K 步历史
  init: 'zero'            # 严格零初始化
  stage2_lr: 1.0e-2       # 100x 学习率

# Actor-Critic
actor:
  layers: 2
  units: 512
  lr: 3.0e-5
  entropy_scale: 1.0e-3   # 探索熵权重

critic:
  layers: 2
  units: 512
  lr: 3.0e-5
  slow_target_update: 1
  slow_target_fraction: 0.02   # EMA 比例

# 训练超参
training:
  batch_size: 512
  batch_length: 50         # 序列长度
  imag_horizon: 15         # 想象 horizon
  gamma: 0.997             # 折扣因子
  lambda_: 0.95            # λ-return
  grad_clip: 1000.0
  model_lr: 1.0e-4
  adam_eps: 1.0e-8

# Stage 步数
stage1:
  total_steps: 2_000_000
  eval_every: 50_000
  ckpt_every: 100_000

stage2:
  total_steps: 1_000_000
  eval_every: 50_000
  ckpt_every: 100_000
  collect_episodes: 200
```

### `configs/env_inaccurate_sim.yaml`

域随机化配置:

```yaml
num_envs: 4096
device: cuda:0
headless: true

domain_rand:
  motor_strength_range: [0.65, 0.75]  # 电机扭矩降 30%
  kp_range: [14.0, 18.0]             # PD Kp 偏离
  kd_range: [0.3, 0.5]               # PD Kd 偏离
  added_mass_range: [1.0, 3.0]       # +1-3kg 负载
  friction_range: [0.3, 1.5]        # 摩擦 ±50%
  push_robots: true
  push_force: [10.0, 30.0]
  push_interval_s: [3.0, 8.0]
```

### `configs/env_pseudo_real.yaml`

```yaml
num_envs: 64                # 模拟真机只有少量并行
# 所有 domain_rand 关闭(代表"真机"标称物理)
```

### 命令行参数

```bash
# train_stage1.py
python scripts/train_stage1.py \
  --config configs/train.yaml \
  --num_envs 4096 \              # 并行 env 数
  --total_steps 2000000 \       # 训练步数
  --device cuda:0 \             # GPU
  --headless                    # 无头模式(必须)

# collect_pseudo_real_data.py
python scripts/collect_pseudo_real_data.py \
  --checkpoint checkpoints/stage1_final.ckpt \
  --output datasets/pseudo_real_data.npz \
  --num_episodes 200 \          # episodes 数
  --max_steps 1000 \            # 每 episode 步数
  --device cuda:0

# train_stage2.py
python scripts/train_stage2.py \
  --stage1_ckpt checkpoints/stage1_final.ckpt \
  --real_data datasets/pseudo_real_data.npz \
  --total_steps 1000000 \
  --device cuda:0

# eval_compare.py
python scripts/eval_compare.py \
  --stage1_ckpt checkpoints/stage1_final.ckpt \
  --stage2_residual checkpoints/stage2_final.ckpt \
  --num_episodes 50 \
  --device cuda:0
```

---

## 7. 常见问题

### Q1: 4096 env OOM 怎么办?

```bash
python scripts/train_stage1.py --num_envs 2048  # 降到一半
```

### Q2: Stage 2 loss 不下降?

检查:
- 数据质量: `python scripts/eval_compare.py --num_episodes 10` 看策略在 PseudoReal 上表现
- 数据量: 增加 `--num_episodes 500`
- 学习率: 修改 `configs/train.yaml` 的 `residual.stage2_lr` (1e-2 → 3e-3)

### Q3: KL loss = NaN?

```yaml
# 降低学习率
training:
  model_lr: 3e-5  # 从 1e-4
  grad_clip: 100.0  # 从 1000
```

### Q4: Actor 输出全是 0?

关掉 AC 训练,先只训 WM:
```python
# trainer.py 临时注释 AC 训练
# if step > 100 and step % 20 == 0: ...
```

### Q5: Isaac Gym 找不到?

```bash
# 检查 PYTHONPATH
echo $PYTHONPATH
# 应该包含 WMP_ROOT

# 重新设置
export WMP_ROOT=/mnt/d/songay/sim2real/WMP
export PYTHONPATH=$PYTHONPATH:$WMP_ROOT
```

更多问题见 `docs/troubleshooting.md`。

---

## 8. 评估输出

`results/comparison.json`:

```json
{
    "A_zeroshot_pseudo_real": {
        "mean_return": 120.5,
        "std_return": 80.2,
        "success_rate": 0.1,
        "fall_rate": 0.8
    },
    "B_residual_pseudo_real": {
        "mean_return": 680.3,
        "std_return": 120.5,
        "success_rate": 0.7,
        "fall_rate": 0.2
    },
    "sim_performance": 750.0,
    "zeroshot_real_performance": 120.5,
    "finetuned_real_performance": 680.3,
    "gap_before": 629.5,
    "gap_after": 69.7,
    "gap_closed_pct": 88.9
}
```

**解读**:
| gap_closed_pct | 含义 |
|----------------|------|
| > 70% | ✅ 算法有效 |
| 30-70% | ⚠️ 部分有效 |
| < 30% | ❌ 效果不佳,需要调参 |

---

## 📚 相关文档

- `docs/design.md` - 总体设计
- `docs/install.md` - 详细安装
- `docs/troubleshooting.md` - 故障排查
- `docs/implementation_targets.md` - 实施目标
- `docs/review_checklist.md` - 评审 checklist
- `docs/claude_review_final.md` - Claude Code 最终评审(5/5)

---

**文档版本**: 1.0
**更新日期**: 2026-06-18
**适用项目**: r2wmp v0.1