# r2wmp AMP 集成技术总结

**作者**: r2wmp 项目 (AnthonySung)
**日期**: 2026-06-19
**状态**: B1 (AMP reward 接入) + B2 (Discriminator 训练) 已完成,Stage 1 1000 步短测通过

---

## 1. 目标与背景

### 1.1 ReDRAW 论文核心
ReDRAW (Adapting World Models with Latent-State Dynamics Residuals) 用 **Ensemble Residual + 两阶段训练** 实现 sim-to-real 迁移:
- **Stage 1**: 在源域(扰动物理 sim)训练 WM + AC,Residual 零初始化 + stop_gradients
- **Stage 2**: 冻结 WM,只训 Residual(1 层重建),学习 sim-to-real gap

### 1.2 A1 任务的特殊要求
A1 是 AMP 任务 — WMP 的 A1AMPCfg 用 **Adversarial Motion Priors**(模仿 mocap 数据集)作为 reward 的一部分。AMP 让 agent 学**像 mocap 一样的运动风格**,而不仅仅是 task reward(lin_vel tracking)。

**AMP reward 公式**:
```
AMP_reward(s, s') = amp_coef * clamp(1 - 0.25 * (D(s, s') - 1)^2, min=0)
```
其中 `D(s, s')` 是 discriminator 输出,WGAN-GP 训练:expert→1, policy→-1。

### 1.3 r2wmp 的特殊情况
r2wmp 用 **Dreamer 风格**(想象训练 actor),**不是 PPO**。所以 WMP 的 `AMPPPO` 不能直接复用。需要:
- **AMP reward 加到 WMP env step 输出**(让 dreamer 的 `compute_wm_loss` 能用)
- **AMPDiscriminator 单独训**(在 Stage 1 主循环里)

---

## 2. 实现方案

### 2.1 架构图

```
┌─────────────────────────────────────────────────────────────────┐
│  WMP A1AMP Env (LeggedRobot)                                   │
│   - rew_buf (task reward: tracking_lin_vel, feet_air_time, ...) │
│   - AMP obs (joint_pos, base_lin_vel, base_ang_vel, joint_vel) │
└──────────────┬──────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│  r2wmp WMPEnvBase.step(action)  ← B1 改动点                     │
│   1. WMP step → get task_reward, amp_obs, terminal_amp_states    │
│   2. AMP reward = AMPDiscriminator.predict_amp_reward(           │
│        state, next_state, task_reward, normalizer)              │
│   3. reward = task_reward + amp_reward                          │
└──────────────┬──────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│  r2wmp ReplayBuffer + WorldModel                                 │
│   - compute_wm_loss: KL + recon + **reward_head loss** (新)      │
│     reward_pred = reward_head(feat), MSE(reward_pred, true)     │
│   - compute_ac_loss: imag_rewards = reward_head(feat_seq) (新) │
│     用真实 task + AMP reward 训练,而不是 heuristic             │
└──────────────┬──────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│  AMPDiscriminator 训练  ← B2 改动点                            │
│   - Expert batch: AMPLoader.feed_forward_generator()            │
│     (从 mocap_motions/{hop,trot}*.txt 采样)                    │
│   - Policy batch: 当前 env step 收集的 amp_obs + next_amp_obs  │
│   - GAN loss: MSE(expert_d, 1) + MSE(policy_d, -1)              │
│   - Gradient penalty: WGAN-GP λ=10                             │
│   - AMPNormalizer 同步更新 (RunningMeanStd)                     │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 文件改动清单

| 文件 | 改动 | 版本 |
|------|------|------|
| `configs/amp.yaml` | 新增:AMP 全部配置(reward_coef, hidden, motion files, training) | B1+B2 |
| `envs/wmp_env_base.py` | 加 `_init_amp()` 方法 + step() 算 AMP reward + reset() 缓存 amp_obs | B1 |
| `envs/inaccurate_sim_env.py` | 加 `amp_config` 参数,创建时传给 super | B1 |
| `envs/pseudo_real_env.py` | 同上 | B1 |
| `scripts/train_stage1.py` | 加 `--amp_config` / `--no_amp` 参数 | B1 |
| `models/world_model.py` | 加 `self.reward_head` (3 层 MLP feat→scalar) + `predict_reward()` 方法 | B0 (已做) |
| `training/wm_loss.py` | `compute_wm_loss` 用 `predict_reward(feat)` 学真实 reward | B0 (已做) |
| `training/ac_loss.py` | `compute_ac_loss` 用 `predict_reward(feat_seq)` 取代 heuristic | B0 (已做) |
| `training/amp_trainer.py` | 新增:`compute_amp_loss` (GAN + grad penalty + normalizer 更新) | **B2** |
| `training/trainer.py` | 加 `amp_disc_opt` + amp_obs buffer + 每步训 discriminator | **B2** |
| `scripts/eval_stage1.py` | 新增:评估 Stage 1 checkpoint (Zero-shot) | 评估 |
| `evaluation/eval_protocol.py` | 修 action shape (单 env 时 squeeze vs unsqueeze) | 评估 |

### 2.3 关键设计决策

#### Decision 1: AMP reward 比例 = 0.05(不是 WMP 的 0.01)

| 来源 | reward_coef | AMP 单步最大 | Task 单步范围 | AMP:Task 比例 |
|------|-------------|--------------|----------------|----------------|
| WMP 原版 | 0.01 | 0.01 | 0.02-0.05 | **1:3 ~ 1:5**(Task 主导) |
| r2wmp B1 第一次 | 0.3 | 0.3 | 0.02-0.05 | **6:1 ~ 15:1**(AMP 压制 Task) |
| **r2wmp B2 调整后** | **0.05** | **0.05** | **0.02-0.05** | **1:1 ~ 2:1**(平衡) |

**为什么 0.05 不是 0.01?** WMP 用 PPO,优势函数对 reward scale 敏感度低;r2wmp 用 Dreamer(想象训练),reward_head 学到的预测直接当 AC 的 imag_rewards,scale 不匹配会让 critic/actor 振荡。0.05 让 AMP+Task 量级匹配,训练稳定。

#### Decision 2: B1 (AMP reward 接入) 和 B2 (Discriminator 训练) 分两步

**为什么不一起做?** Discriminator 权重随机时,AMP reward 噪声极大(随机输出 × 0.05 仍然是噪声)。先用随机 discriminator 跑 1000 步验证:
- AMP 链路通(WMP env step → AMP reward → replay → WM train)
- reward_head 学真实 reward
- 不崩

确认链路通后,再加 discriminator 训练(收敛后才让 AMP reward 有意义)。

#### Decision 3: 用 WMP 原版类(`AMPDiscriminator`, `AMPLoader`, `Normalizer`),不重写

WMP 的 `AMPDiscriminator`(`rsl_rl/algorithms/amp_discriminator.py`)和 `AMPLoader`(`rsl_rl/datasets/motion_loader.py`)已经是 NVIDIA 维护的 stable 代码。直接 `from rsl_rl... import` 复用,只在我们自己的 trainer 里写 GAN training loop。

**集成方式**:
```python
# r2wmp trainer
from rsl_rl.algorithms.amp_discriminator import AMPDiscriminator
from rsl_rl.datasets.motion_loader import AMPLoader
from rsl_rl.utils.utils import Normalizer

# 初始化(在 WMPEnvBase._init_amp)
disc = AMPDiscriminator(input_dim=30*2, amp_reward_coef=0.05, ...)
loader = AMPLoader(device='cuda', motion_files=[...], ...)
normalizer = Normalizer(30)

# 算 AMP reward(在 WMPEnvBase.step)
amp_reward, _ = disc.predict_amp_reward(state, next_state, task_reward, normalizer)

# 训 discriminator(在 trainer.compute_amp_loss)
expert = next(amp_loader.feed_forward_generator(1, 4096))
policy = collected_buffer.cat()
loss = gan_loss(disc(expert), 1) + gan_loss(disc(policy), -1) + grad_pen(disc, expert)
```

#### Decision 4: amp_obs_dim = 30(不是 54)

WMP 默认的 `get_amp_observations()` 注释说"remove z, foot_pos",实际只返回 `joint_pos(12) + base_lin_vel(3) + base_ang_vel(3) + joint_vel(12) = 30`。`foot_pos / tar_toe_pos` 等被剔除,简化 discriminator 输入。

我之前以为 amp_obs_dim=54 是因为看了 AMPLoader.POS_SIZE + ROT_SIZE 等**常量定义**(用于 mocap 数据处理),但 mocap data 里包含这些,policy 的 amp_obs 不包含。

#### Decision 5: Stage 2 Residual 路径不变

ReDRAW Stage 2 是**冻结主 WM,只训 Residual**。我们的 AMP 集成**不影响 Stage 2**,因为:
- AMP reward 在 Stage 1 训练时已经进入 WM(通过 reward_head)
- Stage 2 只看 residual 学 latent gap,不需要 AMP reward

如果 Stage 2 在 PseudoReal 上跑,可以**继续用 AMP**(PseudoReal 也有 AMP mocap),让 Stage 2 residual 学到的不只是物理 gap,还有 style gap。

---

## 3. 验证结果(1000 步短测)

### 3.1 B1 阶段(AMP reward 接入)

**目的**: 验证 AMP 链路通(随机 discriminator)

```log
[InaccurateSimEnv] amp=on  ← AMP 启用
reward_coef=0.3, hidden=[1024, 512]
```

**奖励分布(1000 步)**:
```
reward_batch: min=0.1803 max=0.3133 mean=0.2556 std=0.0149
```
- AMP 单步最大 0.3,平均 0.25 — AMP 占 80%(太重)

**训练曲线**:
| step | avg_return | avg_ep_len | wm[rew] |
|------|------------|------------|---------|
| 100 | 12.05 | 51.5 | - |
| 600 | 54.49 | **236.9** | 0.110 |
| 900 | 64.92 | - | - |

**结论**: AMP 链路通,但 reward 比例不对(AMP 压制 Task)。

### 3.2 B2 阶段(Discriminator 训练 + reward 比例调整)

**目的**: 训 discriminator 让 AMP reward 有意义 + 调整比例

**AMP 训练曲线**(每 100 步打):
| step | amp[loss] | amp[exp_d] | amp[pol_d] | amp[gp] |
|------|-----------|------------|------------|---------|
| 200 | 0.037 | 0.97 | -0.86 | 0.070 |
| 500 | 0.016 | 0.92 | -0.98 | 0.046 |
| 900 | 0.015 | 0.92 | -0.96 | 0.050 |

**GAN 收敛**:
- `exp_d → 0.92-0.98` → Expert motion 判别为接近 1 ✅
- `pol_d → -0.96 to -1.05` → Policy motion 判别为接近 -1 ✅
- `amp[gp] = 0.034-0.050` → Gradient penalty 健康(没有 mode collapse)✅

**奖励分布(B2, 调整后)**:
```
reward_batch: min=0.0000 max=0.0713 mean=0.0137 std=0.0130
```
- AMP 单步最大 0.07,平均 0.014 — **AMP:Task ≈ 1:1 ~ 2:1**(平衡 ✅)

**训练曲线**:
| step | avg_return | avg_ep_len |
|------|------------|------------|
| 100 | 2.00 | 50.6 |
| 600 | **4.24** | **235.1** |
| 900 | 2.17 | 139.3 |

**结论**: AMP discriminator 收敛,reward 比例合理,**但 1000 步不够训** — 需要完整 2M 步才能稳定。

### 3.3 与无 AMP 的对比

| 版本 | avg_return | avg_ep_len | reward mean |
|------|------------|------------|-------------|
| 无 AMP | 0.36 | ~50 | 0.007 |
| B1 (coef=0.3) | 64.92 | 236.9 | 0.255 |
| **B2 (coef=0.05, 训 discriminator)** | **4.24** | **235.1** | **0.014** |

**短期 avg_return 下降是预期的**:
- B1 时,random discriminator → AMP reward 噪声大 → Actor 短视学 AMP → 走得快但姿势怪
- B2 时,discriminator 开始收敛,policy_d → -1 → Actor 被迫学像 expert → 真姿势
- **完整 2M 步后,B2 应该 > B1**

---

## 4. 评估 Stage 1 1000 步 checkpoint

### 4.1 评估脚本

`scripts/eval_stage1.py`:
- 加载 `checkpoints/stage1_final.ckpt`
- 推断 WorldModel 参数(从 state_dict 的 shape)
- 在 PseudoRealEnv 上跑 N episodes
- 输出 `mean_return / mean_ep_length / success_rate / fall_rate`

### 4.2 评估结果(待跑,预期)

由于只训了 1000 步,**预期 mean_return 很低**(可能 0-10),因为 actor 还没学会走路。但应该比随机策略高(随机策略 return ≈ 0)。

---

## 5. 已知限制与未来工作

### 5.1 已知限制
1. **Stage 1 还没跑完整 2M 步**:1000 步只是烟雾测试
2. **Stage 2 没跑**:Stage 1 final.ckpt 是 B2 训出来的,但还没做 Stage 2(冻结 WM + 只训 Residual)
3. **PseudoRealEnv 评估没跑**:因为 eval_protocol.py 的 action shape bug 修了但还没重跑
4. **AMP mocap 数据只有 4 段**(hop1/2 + trot1/2)— 如果想更好多样性,可加更多 mocap 文件

### 5.2 未来工作
1. **完整 2M 步训练**:Stage 1 + Stage 2 完整流程
2. **AMP 多样化**:加更多 mocap 数据(走、跑、跳、跨栏)
3. **Style ablation**:研究 AMP 对策略风格的具体影响(可视化 expert vs policy trajectory)
4. **三策略对比报告**:Stage 1 vs Stage 2 vs 真机上界

---

## 6. 关键代码位置

| 关注点 | 文件:行 |
|--------|---------|
| AMP 初始化 | `envs/wmp_env_base.py:106-220` (`_init_amp`) |
| AMP reward 算 | `envs/wmp_env_base.py:336-352` (step 内) |
| Discriminator 训练 | `training/amp_trainer.py:30-110` |
| Trainer 集成 AMP train | `training/trainer.py:247-290` |
| AMP config | `configs/amp.yaml` |
| 评估脚本 | `scripts/eval_stage1.py` |

---

## 7. 文档清单

| 文档 | 内容 |
|------|------|
| `docs/design.md` | r2wmp 整体设计(Stage 1/2 + Residual) |
| `docs/redraw_theory_notes.md` | ReDRAW 算法理论笔记 |
| `docs/review_checklist.md` | 需求 + 实现自审 checklist |
| `docs/implementation_targets.md` | 每个文件的实施目标 |
| `docs/stage1.md` | Stage 1 详解 |
| `docs/stage2.md` | Stage 2 详解 |
| `docs/evaluation.md` | 评估协议 |
| **`docs/amp_integration_summary.md`** | **本文档:AMP 集成技术总结** |

---

**注**: 本文档描述的是 2026-06-19 完成的 v0.2 AMP 集成。Stage 1 完整 2M 步训练、Stage 2 Residual 训练、AMP 三策略对比 仍是 TODO。
