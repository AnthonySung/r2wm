# Stage 2 详解

## 目标

在 **PseudoReal 数据** 上微调 Residual,让 Residual 学到 sim-to-real gap,但不动 Actor。

## 关键设计(对齐 ReDRAW)

### Residual 配置变更(从 Stage 1 到 Stage 2)

| 参数 | Stage 1 | Stage 2 | 来源 |
|------|---------|---------|------|
| 结构 | 3 层 MLP | **1 层 MLP** | ReDRAW ensemble_residual_extra_small_1_member |
| 参数量 | ~200K | **~70K** | 更小容量 |
| 初始化 | 零初始化 | **重新零初始化**(从头训) | ReDRAW |
| 接收梯度 | False(stop_grad) | **True** | ReDRAW Stage 2 不 freeze residual |
| 学习率 | 1e-4(随 WM) | **1e-2**(100x) | ReDRAW 100x_wm_lr |
| 冻结 | 不冻结 | **主 RSSM 冻结,Residual 训** | ReDRAW freeze_wm |

### 训练流程

```python
# 1. 加载 Stage 1
world_model.load_state_dict(stage1_ckpt)
actor.load_state_dict(stage1_ckpt)

# 2. 重新创建 Residual(关键!)
world_model.recreate_residual_for_stage2()
# 此时 physical_residual 从 3 层变 1 层,所有权重重新零初始化

# 3. 冻结
world_model.freeze_main_network()  # 只 residual 可训
for p in actor.parameters():
    p.requires_grad = False

# 4. Optimizer
residual_params = [p for p in world_model.parameters() if p.requires_grad]
optimizer = Adam(residual_params, lr=1e-2)  # 100x

# 5. 训练循环
real_replay = ReplayBuffer.from_npz('datasets/pseudo_real_data.npz')
for step in range(1_000_000):
    batch = real_replay.sample(batch_size=512, seq_length=50)

    # 观察(Residual 接收梯度)
    posts, priors = world_model.observe(
        batch['obs'], batch['action'], batch['is_first'],
        with_residual=True,
        stop_residual_grad=False,  # ← Stage 2: Residual 接收梯度
    )

    # KL loss: 让 sim+residual 接近 real
    kl_loss, _, _ = rssm.kl_loss(posts, priors, dyn_scale=1.0, rep_scale=0.0)

    optimizer.zero_grad()
    kl_loss.backward()
    optimizer.step()
```

### Stage 2 关键开关

| 时刻 | `with_residual` | `stop_residual_grad` | 说明 |
|------|----------------|----------------------|------|
| **observe (伪 real 数据)** | True | **False** | Residual 接收梯度,快速学习 gap |

**注意**: Stage 2 训练时,**Actor 没有参与**(完全冻结),Residual 独立学习。

### Residual 学到了什么?

KL loss 推动 Residual 让 `sim_post + delta ≈ real_post`,所以 Residual 学到:

```
delta ≈ real_post - sim_post
     = sim_to_real_gap_in_latent_space
```

具体来说:
- **PhysicalResidual**: 补偿电机扭矩、PD 增益、质量、摩擦等**参数误差**
- **DynamicsResidual**: 补偿齿轮间隙、皮带柔性、接触动力学等**未建模结构**

## 预期产出

- `checkpoints/stage2_final.ckpt`: 只包含 Residual 参数的 checkpoint
- Residual 在 PseudoReal 上能让 WM 预测准确率显著提升

## 评估

运行 `scripts/eval_compare.py`,对比:
- A. Zero-shot(Stage 1 Residual≈0)
- B. Stage 2 Residual 微调

**期望**: B 的 episode_return 显著高于 A,gap 闭合率 > 70%。