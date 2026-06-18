# Stage 1 详解

## 目标

在 **InaccurateSimEnv** 上预训练 World Model + Actor-Critic,作为 Stage 2 的起点。

## 关键设计(对齐 ReDRAW)

### Residual 配置

| 参数 | 值 | 来源 |
|------|---|------|
| 结构 | 3 层 MLP | ReDRAW ensemble_residual 简化版 |
| hidden | 64 | 经验值 |
| 输入 | `prev_stoch + action` | ReDRAW 公式 (16) |
| 输出 | `delta_mean` | 加在 latent mean 上 |
| 初始化 | **严格零初始化** | ReDRAW |
| Stage 1 stop_grad | **True** | ReDRAW with_res_stop_gradients |
| 想象训练参与 | **False** | ReDRAW imagine 时默认 False |

### 训练流程

```python
# 1. 数据收集(4096 envs 并行)
obs = env.reset()
for t in range(episode_length):
    with torch.no_grad():
        embed = world_model.encode(obs)
        state, _ = rssm.obs_step(state, action, embed, is_first, sample=True)
        feat = rssm.get_feat(state)
        action = actor.sample(feat)
        next_obs, reward, done, _ = env.step(action)
        replay.add(obs, action, reward, next_obs, done)
        obs = next_obs

# 2. WM 训练(每 100 步)
batch = replay.sample(batch_size=512, seq_length=50)
posts, priors = world_model.observe(
    batch['obs'], batch['action'], batch['is_first'],
    with_residual=True,           # Residual 参与前向
    stop_residual_grad=True,      # 但 stop gradient
)
# 注意:posts['mean'] = rssm_mean + physical_residual(...).detach()
#       priors['mean'] = rssm_mean + physical_residual(...).detach()
# KL(post || prior) 间接优化了 RSSM,Residual 输出 ≈ 0

kl_loss, dyn_loss, rep_loss = rssm.kl_loss(posts, priors, ...)
recon_loss = mse(decoder(posts), obs)
wm_loss = kl_loss + recon_loss

# 3. AC 训练(每 100 步,想象训练)
batch = replay.sample(batch_size=512, seq_length=1)
init_state = encode(batch['obs'][:, 0])
states, actions, log_probs, entropies = world_model.imagine(
    actor, init_state, horizon=15,
    with_residual=False,  # ← 关键: Actor 在干净 latent 训练
)
# λ-return + actor loss + critic loss
```

### Stage 1 关键开关对照表

| 时刻 | `with_residual` | `stop_residual_grad` | Residual 是否改 prior | Residual 是否接收梯度 |
|------|----------------|----------------------|------------------------|---------------------|
| **observe (真实数据)** | True | True | ✅ 是(但输出≈0) | ❌ 否 |
| **imagine (Actor 训练)** | False | N/A | ❌ 否 | N/A |
| **env step (采数据)** | False | N/A | ❌ 否 | N/A |

### 为什么 Residual 在 Stage 1 几乎不起作用?

1. **零初始化**: 所有 Linear 层的 weight=0, bias=0 → 初始输出 = 0
2. **stop_gradients**: Residual 不接收 WM loss 梯度,不会因训练而变大
3. **源域无 gap**: InaccurateSimEnv 训练数据本身就没有"未建模"的 gap(物理参数是已知的)

**结果**: Stage 1 结束时,Residual 输出 ≈ 0,等价于"不存在"。

## 预期产出

- `checkpoints/stage1_final.ckpt`: 完整的 WM + AC checkpoint
- `logs/stage1.log`: 训练日志
- 在 InaccurateSim 上: actor 能走, return > 200

## 下一步

完成后运行 `scripts/collect_pseudo_real_data.py`,采集伪 real 数据用于 Stage 2。