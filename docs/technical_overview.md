# r2wmp 项目技术方案与进度总览

> **项目名称**: r2wmp — ReDRAW World Model Porting for Unitree A1
> **版本**: v0.3 (AMP 集成完成)
> **最后更新**: 2026-06-19 16:30
> **作者**: AnthonySung
> **仓库**: https://github.com/AnthonySung/r2wmp

---

## 1. 完整任务介绍

### 1.1 项目背景

将 **ReDRAW 论文** (*Adapting World Models with Latent-State Dynamics Residuals*, JAX/Flax) 的 **两阶段训练 + Ensemble Residual** 算法,移植到 **Unitree A1** 四足机器人仿真环境(WMP, NVIDIA Isaac Gym + PyTorch),实现 sim-to-real 迁移。

**源项目对比**:

| 维度 | ReDRAW (源论文) | WMP (A1 环境) | r2wmp (本项目) |
|------|----------------|----------------|----------------|
| 框架 | JAX + Flax | PyTorch | PyTorch ✅ |
| 仿真 | MuJoCo (DMC) | Isaac Gym | Isaac Gym ✅ |
| 机器人 | 通用 DMC 任务 | Unitree A1 | Unitree A1 ✅ |
| 算法 | DreamerV3 + Ensemble Residual | AMPPPO + World Model | DreamerV3 + Ensemble Residual + AMP ✅ |
| 训练范式 | 两阶段 | 单阶段 PPO | 两阶段(Stage 1 WM + Stage 2 Residual) ✅ |

### 1.2 核心挑战

1. **框架迁移**: JAX 的 `vmap/pmap/jit` 生态 → 纯 PyTorch `nn.Module + optim.Adam`
2. **环境适配**: DMC 标准 Gym API → Isaac Gym 的 GPU 并行批量 step
3. **AMP 集成**: WMP 的 AMPPPO(基于 PPO) → r2wmp 的 Dreamer(基于 WM 想象训练)
4. **Reward 处理**: ReDRAW 用纯 task reward → A1 需要 task + AMP motion prior 混合 reward
5. **Sim-to-real gap**: DMC 的简单物理参数扰动 → A1 的多源 gap(电机扭矩/PD/摩擦/质量/接触)

### 1.3 验收标准

| 指标 | 最低 | 理想 |
|------|------|------|
| Sim-to-real gap 闭合率 | > 30% | > 70% |
| Stage 2 策略 > Stage 1 策略 | ✅ | — |
| AMP discriminator 收敛 | exp_d > 0.8, pol_d < -0.8 | exp_d > 0.95, pol_d < -0.95 |
| A1 在 PseudoReal 上 episode_return | > 20 | > 200 |
| 端到端训练 | 能跑完 Stage 1+2 | 单指令完成 |

---

## 2. 阶段划分与目标

### 阶段 0: 环境部署
**目标**: 在 autodl 容器上创建独立的 r2wmp 训练环境,不破坏 WMP 原环境

| 子任务 | 状态 | 依赖 | 实现技术点 |
|--------|------|------|-----------|
| 0.1 装独立 miniconda + r2wmp_env | ✅ | — | `/opt/miniconda3_r2wmp`, Python 3.8.20 |
| 0.2 装 PyTorch 2.0.1+cu118 | ✅ | 0.1 | `torch==2.0.1+cu118`, CUDA 11.8 toolkit |
| 0.3 装 IsaacGym 1.0rc3 | ✅ | 0.2 | `pip install -e /home/WMP/isaacgym/python` |
| 0.4 装 legged_gym + rsl_rl | ✅ | 0.3 | sys.path + PYTHONPATH,无 pip install -e |
| 0.5 装所有依赖 | ✅ | 0.2 | 钉死 numpy<1.24(兼容 np.float),typing_extensions≥4.8(兼容 torch) |
| 0.6 配置 profile.d | ✅ | 0.4 | `r2wmp_conda.sh` 含 PYTHONPATH + WMP_ROOT |
| 0.7 部署模板脚本 | ✅ | — | `scripts/new_server_*.sh` (6 个,可复用) |

**关键技术点**:
- **IsaacGym 必须先 import(在 torch 之前)**: 否则报 `PyTorch was imported before isaacgym modules`
- **WMP 的 rsl_rl/legged_gym 不是 pip 项目**: 没有 setup.py,不能 pip install -e;用 PYTHONPATH 暴露
- **numpy 版本冲突**: IsaacGym 的 torch_utils.py 用 `np.float`,numpy≥1.24 会报 AttributeError,要降级
- **typing_extensions 版本冲突**: WMP requirements.txt 钉 `typing_extensions==4.2.0`,torch 2.0 需要 ≥4.8

---

### 阶段 1: 代码基线 & 烟雾测试
**目标**: 确认 26 个 .py 文件语法正确 + 核心逻辑通过单元测试

| 子任务 | 状态 | 依赖 | 实现技术点 |
|--------|------|------|-----------|
| 1.1 Residual 零初始化验证 | ✅ | — | `nn.init.zeros_()` + assert |delta| < 1e-6 |
| 1.2 RSSM 前向 + kl_loss | ✅ | — | K-tuple categorical: `[B, stoch_dim, discrete]`, free bits |
| 1.3 WorldModel observe + imagine | ✅ | 1.1+1.2 | `with_residual/stop_residual_grad` 开关 |
| 1.4 Actor-Critic 输出 + EMA | ✅ | — | tanh 压缩 [-1,1], SlowCritic EMA |
| 1.5 ReplayBuffer 采样 + save/load | ✅ | — | 序列采样 [B, T, ...], numpy 存储省显存 |
| 1.6 λ-return 计算 | ✅ | — | 从后往前递推: `r + γ·c·[(1-λ)V + λG]` |
| 1.7 AC loss 端到端 | ✅ | 1.3+1.6 | imagine(horizon=15) → λ-return → actor/critic loss |
| 1.8 Residual Stage 1/2 流程 | ✅ | 1.1+1.3 | recreate_residual_for_stage2(3→1层), freeze_main_network |

**关键技术点**:
- **RSSM discrete 模式**: `stoch_dim × discrete` one-hot categorical,flatten 得到 `stoch_flat`
- **Residual 零初始化** 是关键:Stage 1 输出≈0,不破坏已收敛的 RSSM
- **stop_residual_grad=True**: Residual 在 Stage 1 不接收梯度
- **imagine(with_residual=False)**: Actor 训练在干净 latent 上,不依赖 Residual

---

### 阶段 2: Reward 链路修复(B0)
**目标**: 让 WM 用真实 env reward 训练 reward_head,Actor 用真实 reward 学走路

| 子任务 | 状态 | 依赖 | 实现技术点 |
|--------|------|------|-----------|
| 2.1 WorldModel 加 reward_head | ✅ | — | 2 层 MLP(hidden=512),输入 feat→输出 scalar |
| 2.2 compute_wm_loss 用真实 reward | ✅ | 2.1 | `MSE(predict_reward(feat), batch['reward'])` |
| 2.3 compute_ac_loss 用 predict_reward | ✅ | 2.1 | `imag_rewards = predict_reward(feat_seq)`,替代 heuristic |

**为什么**: 原始代码用 `-0.01 * actions²` 做 imag_rewards,actor 学的是"输出小动作",不是"走路"

---

### 阶段 3: AMP 集成(B1 + B2)
**目标**: 把 AMP motion prior 加到 WMP env step → replay → WM train → AC 训练

#### B1: AMP reward 接入
| 子任务 | 状态 | 依赖 | 实现技术点 |
|--------|------|------|-----------|
| 3.1 新建 `configs/amp.yaml` | ✅ | — | reward_coef, discr_hidden, motion_files, normalizer |
| 3.2 `_init_amp()` 初始化 3 组件 | ✅ | — | AMPDiscriminator / AMPLoader / Normalizer |
| 3.3 step() 算 AMP reward | ✅ | 3.2 | `predict_amp_reward(amp_obs, next_amp_obs, task_reward)` |
| 3.4 reset() 缓存 amp_obs | ✅ | — | 缓存 `_current_amp_obs` |
| 3.5 train_stage1.py 加 --amp_config | ✅ | 3.2 | 默认加载 configs/amp.yaml |

**关键技术点**:
- **AMP reward 公式**: `reward_coef × clamp(1 - 0.25 × (D - 1)², 0)`,D = discriminator 输出(∈R)
- **terminal_amp_states**: 已 reset env 的 next_amp_obs 用 terminal_amp_states 替换(discriminator 知道 episode 结束)
- **AMP normalizer 不是必须的**: B1 用默认(未训练)normalizer;但归一化可加速 discriminator 收敛

#### B2: Discriminator 训练
| 子任务 | 状态 | 依赖 | 实现技术点 |
|--------|------|------|-----------|
| 3.6 新建 `training/amp_trainer.py` | ✅ | 3.2 | `compute_amp_loss()` 函数 |
| 3.7 GAN-style training loop | ✅ | 3.6 | MSELoss: expert_d→1, policy_d→-1 |
| 3.8 Gradient penalty (WGAN-GP) | ✅ | 3.7 | `compute_grad_pen(expert, λ=10)` |
| 3.9 AMPNormalizer 同步更新 | ✅ | 3.6 | `normalizer.update(policy_state) + update(expert_state)` |
| 3.10 amp_obs buffer 收集 | ✅ | 3.2 | 每 step 收集 `[current_amp, next_amp]`,限制 8192 |

**关键技术点**:
- **GAN 收敛指标**: `exp_d → 1, pol_d → -1, gp ≈ 0.03-0.05`
- **feed_forward_generator 接口**: WMP 是位置参数 `(num_batch, batch_size)`,不是关键字
- **AMP obs 维度**: `30`(joint_pos 12 + lin_vel 3 + ang_vel 3 + joint_vel 12),不是 54
- **分开 B1→B2**: B1 验证链路通,B2 加训练,降低风险

#### AMP 比例调优
| 子任务 | 状态 | 说明 |
|--------|------|------|
| 3.11 reward_coef=0.3 验证 | ✅ | AMP 占 80%,太重,return 被 AMP 噪声推高 |
| 3.12 reward_coef=0.05 调整 | ✅ | AMP:Task ≈ 1:1,对齐 WMP 风格 |
| 3.13 确认比例合理 | ✅ | `reward_batch: mean=0.014`(task ~0.007 + AMP ~0.007) |

---

### 阶段 4: 诊断增强
**目标**: 训练日志可读性 + 问题快速定位

| 子任务 | 状态 | 实现 |
|--------|------|------|
| 4.1 log_every 参数 | ✅ | 支持 1000 步或 100 步 log |
| 4.2 wm/ac 指标输出 | ✅ | kl_loss, recon_loss, reward_loss, actor/critic/lambda_return |
| 4.3 episode_length 跟踪 | ✅ | avg_ep_len = 平均 episode step 数(从 ~50→217) |
| 4.4 AMP 指标输出 | ✅ | amp_loss/expert_d_mean/policy_d_mean/grad_pen |
| 4.5 reward batch debug | ✅ | min/max/mean/std + reward_pred min/max |

---

### 阶段 5: 评估工具
**目标**: 能评估 Stage 1 checkpoint 在 PseudoRealEnv 上的 Zero-shot 表现

| 子任务 | 状态 | 实现 |
|--------|------|------|
| 5.1 新建 `scripts/eval_stage1.py` | ✅ | 自动推断 WorldModel 参数(从 state_dict shape) |
| 5.2 修 eval_protocol.py action shape | ✅ | `action.dim()==1 → unsqueeze(0)` |
| 5.3 修 PseudoRealEnv amp_config | ✅ | 传 AMP config 给 super().__init__ |
| 5.4 跑出评估结果 | ✅ | 5 eps: return=10.98, ep_len=217, fall_rate=1.0 |

---

### 阶段 6: 未完成
**目标**: 完整训练 + 三策略对比

| 子任务 | 阻塞原因 | 预期时间 | 优先级 |
|--------|---------|---------|--------|
| 6.1 Stage 1 完整 2M 步 | 时间(50h GPU) | 50h(12h with 4096 env) | 🔴 P0 |
| 6.2 collect_pseudo_real_data.py | 依赖 6.1 | 1-2h | 🔴 P0 |
| 6.3 Stage 2 训 Residual 1M 步 | 依赖 6.2 | 3-6h | 🔴 P0 |
| 6.4 三策略对比(eval_compare.py) | 依赖 6.3 | 30min | 🟡 P1 |
| 6.5 修 eval_compare.py WorldModel 签名 | — | 15min | 🟡 P1 |
| 6.6 AMP 单元测试(test_mock.py) | — | 30min | 🟢 P2 |

---

## 3. 技术架构图

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      r2wmp 项目 (纯 PyTorch)                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐         │
│  │  Stage 1: 在 InaccurateSimEnv 训练 (AMP + Task + Residual 零初始化)  │         │
│  │  4096 envs, domain_rand: motor[-30%], PD[-20%], mass[+1~3kg]        │         │
│  └─────────────┬───────────────────────────────────────────────────────┘         │
│                │ checkpoint (stage1_final.ckpt)                                 │
│                ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐         │
│  │  Stage 2: 在 PseudoReal 数据上微调 Residual (冻结 WM,只训 Residual)  │         │
│  │  1 层 MLP + 100x lr (1e-2) + GAN/freeze_wm                        │         │
│  └─────────────┬───────────────────────────────────────────────────────┘         │
│                │ checkpoint (stage2_final.ckpt)                                 │
│                ▼                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐         │
│  │  三策略对比评估 (Zero-shot vs Residual 微调 vs 真实上界)              │         │
│  └─────────────────────────────────────────────────────────────────────┘         │
│                                                                                  │
│  ┌────────────────────────────┐   ┌────────────────────────────┐                │
│  │  WMP A1AMP Env            │   │  AMP 模块                  │                │
│  │  (IsaacGym LeggedRobot)   │   │  AMPDiscriminator (复用了   │                │
│  │  → task_reward + amp_obs  │   │  WMP 的,GAN+grad_pen)     │                │
│  └────────────┬───────────────┘   │  AMPLoader (mocap 数据)    │                │
│               │                   │  AMPNormalizer (RunningMeanStd) │           │
│               ▼                   └────────────┬───────────────┘                │
│  ┌────────────────────────────┐                │                                │
│  │  Dreamer World Model      │                │                                │
│  │  reward_head(feat)→scalar │◄───────────────┘                                │
│  │  → predict task+AMP reward│               reward_buf                         │
│  │  → AC 用真实 reward 训练   │                                                 │
│  └────────────────────────────┘                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. 关键成果速览

### 4.1 服务器环境
- ✅ 独立 conda + Python 3.8.20 + torch 2.0.1+cu118 + IsaacGym 1.0rc3 + 全部依赖
- ✅ RTX 3090 / 24GB,空闲可用

### 4.2 代码指标

| 统计项 | 数值 |
|--------|------|
| 源文件 | 28 个 .py + 4 个 config + 9 个 docs |
| 新增文件 | 2 个(amp_trainer.py, eval_stage1.py) |
| 修改文件 | 10 个(含 models/training/envs/scripts/evaluation) |
| 单元测试 | 8/8 ✅ |
| 端到端测试 | 1000 步 ✅(stage1_final.ckpt 保存成功) |

### 4.3 训练数据

| 版本 | 1000 步 avg_return | avg_ep_len | reward mean | AMP exp_d | AMP pol_d |
|------|-------------------|-----------|-------------|-----------|-----------|
| 无 AMP | 0.36 | ~50 | 0.007 | — | — |
| B1 (coef=0.3) | 64.92 | 236.9 | 0.255 | 0.92 | -0.98 |
| **B2 (coef=0.05)** | **4.24** | **235.1** | **0.014** | **0.94** | **-1.00** |

### 4.4 评估数据
- **Stage 1 Zero-shot @ PseudoRealEnv**: return = 10.98 ± 8.22, ep_len = 217.4, fall_rate = 1.0

---

## 5. 启动指南

```bash
# SSH 到服务器
ssh -p 31310 root@connect.nmb2.seetacloud.com

# 激活环境
source /etc/profile.d/r2wmp_conda.sh
cd /root/r2wm/r2wmp

# 跑烟雾测试
/opt/miniconda3_r2wmp/envs/r2wmp_env/bin/python tests/test_mock.py

# 跑 1000 步短测(确认链路通)
python scripts/train_stage1.py --total_steps 1000 --num_envs 4096 --headless

# 跑完整 Stage 1 (2M 步,6-12 小时)
python scripts/train_stage1.py --total_steps 2000000 --num_envs 4096 --headless &

# 评估 Stage 1 checkpoint (10 episodes,~5 分钟)
python scripts/eval_stage1.py --num_episodes 10 --env_name pseudo_real

# 关闭 AMP(用纯 task reward)
python scripts/train_stage1.py --no_amp --total_steps 1000 --num_envs 4096 --headless
```

---

## 6. 关键技术决策记录

| 决策 | 选择 | 替代方案 | 理由 |
|------|------|---------|------|
| AMP 来源 | 复用 WMP 原版类 | 自写 discriminator | WMP 代码已稳定,NVIDIA 维护 |
| AMP 比例 | coef=0.05 | coef=0.3 → 0.01 | 平衡 AMP+Task,对齐 Dreamer 风格 |
| reward_head 结构 | 2 层 MLP(hidden=512) | 1 层/3 层 | 够表达,不过拟合 |
| AMP 训练频率 | 每 step | 每 N 步 | 1000 步内 GAN 已经收敛 |
| WorldModel 构造 | 自动推断(state_dict) | 硬编码 | 评估时不需要改代码 |
| Stage 2 顺序 | Stage 1→collect→Stage 2 | — | ReDRAW 流程标准 |
| 三策略 | Zero-shot / Residual / Upper bound | 只对比前两种 | ReDRAW 原协议 |

---

> **本文档**: 项目技术方案 + 进度总览(13.6 KB + 补充)
> **git 路径**: `docs/technical_overview.md`
> **下次维护窗口**: Stage 1 2M 步跑完时更新训练数据
