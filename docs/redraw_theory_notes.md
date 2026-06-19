# ReDRAW 算法理论笔记

> Adapting World Models with Latent-State Dynamics Residuals
> 用于在 r2wmp 项目实施时作为理论参考

---

## 1. 论文核心思想(一句话)

**在 latent 空间上添加可学习的残差,补偿 sim-to-real 物理差异,实现两阶段训练而不破坏主 WM。**

---

## 2. 数学框架

### 2.1 标准 RSSM(没有 Residual)

**先验**: 从 `prev_stoch` 和 `prev_action` 预测下一时刻的 latent 分布
```
û = f_φ(z_{t-1}, α̂_{t-1}, a_{t-1})       ← 先验 logits
prior_stoch ~ Categorical(softmax(û))
```

**后验**: 用 encoder 编码真实 obs,修正先验
```
α̂_real = p_φ(· | ...)                    ← 后验分布
post_stoch ~ Categorical(softmax(α̂_real))
```

**训练目标**: 最小化后验和先验的 KL 散度
```
dyn_loss = KL(sg(post) || prior)        ← prior 接近 post
rep_loss = KL(post || sg(prior))        ← post 不要偏离 prior 太远
```

### 2.2 ReDRAW 公式(18): Residual 加在 logits 上

```
prior_logits_corrected = û + ê
ê = ψ_ensemble(z_{t-1}, a_{t-1})      ← Ensemble 残差输出
```

**关键**: 加在 logits(等价于加在 mean for one-hot),不是加在 stoch 采样上。

### 2.3 Residual 学习目标

**Residual loss(公式 20)**:
```
L_ψ = E_q[Σ_t D_KL(q_φ(z_t|x_t) || p_φ(ẑ^real_t|...))]
```

**含义**: 让 sim + residual 的预测 **尽量接近 real posterior**。

---

## 3. 两阶段训练流程

### Stage 1: Plan2Explore 预训练

| 步骤 | 内容 |
|------|------|
| 输入 | 源域数据(无扰动, 如 DMC 原始任务) |
| 训练 | WM(含 Residual) + Actor + Critic |
| Residual | 零初始化 + stop_gradients → 输出 ≈ 0 |
| 目的 | 让 RSSM 学到 sim 的 latent dynamics |

**Residual 初始化是关键**:
```python
# ReDRAW 源码(nets.py)
for m in self.members:
    for layer in m:
        if isinstance(layer, nn.Linear):
            nn.init.zeros_(layer.weight)   # 零初始化
            nn.init.zeros_(layer.bias)
```

**Stage 1 结束时 Residual 状态**:
- 训练开始: 输出 = 0(零初始化)
- 训练中期: 仍然接近 0(stop_grad)
- 训练结束: 接近 0(源域 sim 无 gap,KL loss 推动 residual → 0)

### Stage 2: 离线迁移学习

| 步骤 | 内容 |
|------|------|
| 输入 | 目标域数据(有扰动, 如 cup_catch_windy)+ 真机 |
| 操作 | **冻结主 RSSM**,**只训练 Residual** |
| Residual | **重新创建**: 1 成员 + 1 层(ensemble_residual_extra_small_1_member) |
| 学习率 | **100x normal lr**(1e-2) |
| 目的 | 让 Residual 学到 sim-to-real 物理差异 |

**Stage 2 关键操作**:
1. 加载 Stage 1 checkpoint
2. 冻结 RSSM 所有参数
3. **重建** Residual(从 7 成员 3 层 → 1 成员 1 层)
4. 只训练 Residual, lr = 1e-2

**为什么 Residual 重新创建而非继承**:
- 更小的容量(1 层 vs 3 层)→ 学到的 gap 更小、更稳定
- 零初始化 → 从零开始学目标域 gap
- 不继承 Stage 1 的 Residual 输出(Stage 1 残差 ≈ 0,继承没意义)

---

## 4. Residual 三种状态(ReDRAW 核心机制)

### 状态 A: 观察真实数据(WM 训练)
```python
post, prior = rssm.obs_step(state, action, embed, is_first)
# apply_residual:
prior_logits_corrected = prior_logits + residual(prev_stoch, action).detach()
# Residual: with_residual=True, with_res_stop_gradients=True
```

### 状态 B: 想象训练(Actor 训练)
```python
# 想象轨迹,Actor 训练
imagined_states = rssm.img_step(state, action)  # 不加 Residual
# Residual: with_residual=False
```

**关键**: 想象训练时 Actor 看到的是"干净 latent"(纯 RSSM,无 Residual)。这样 Actor 学到的策略不依赖 Residual,Residual 学到的 gap 不会破坏 Actor。

### 状态 C: 部署/评估
```python
# 真实部署时,Residual 必须参与
state = rssm.img_step(state, action, with_residual=True)
# 此时 actor 输入 = RSSM feat + Residual 修正
```

### 状态对照表

| 状态 | `with_residual` | `stop_residual_grad` | 是否修改 prior | Residual 是否接收梯度 |
|------|-----------------|---------------------|----------------|---------------------|
| **A. observe** | True | True | ✅ 是(输出≈0) | ❌ 否 |
| **B. imagine** | False | N/A | ❌ 否 | N/A |
| **C. 部署** | True | N/A | ✅ 是 | N/A |

---

## 5. Ensemble vs Single(Stage 1 vs Stage 2)

### Stage 1: Ensemble(7 成员, 3 层)

```python
class EnsembleResidual:
    def __init__(self, ensemble_size=7, n_layers=3):
        self.members = [
            nn.Sequential(... n_layers=3 ...)  # 7 个独立的 3 层 MLP
            for _ in range(7)
        ]
    
    def forward(self, prev_stoch, action):
        outputs = [m(prev_stoch, action) for m in self.members]
        return torch.stack(outputs, dim=0).mean(dim=0)  # 平均
```

**为什么 Ensemble**:
- 提供 N 个不同视角的"修正量"
- 用 KL 散度计算"成员间不一致"→ 探索奖励
- Stage 1 时输出接近 0(零初始化)

### Stage 2: Single(1 成员, 1 层)

```python
class EnsembleResidualExtraSmall1Member(EnsembleResidual):
    def __init__(self):
        super().__init__(ensemble_size=1, n_layers=1)
```

**为什么 Single + 1 层**:
- 目标域 gap 通常是**单一偏移**(如电机 -30%)
- 复杂 Ensemble 过拟合
- 1 层 ≈ 单一仿射变换,刚好够

---

## 6. KL Loss 设计(关键)

### 双向 KL

```python
# 标准 DreamerV3 公式
dyn_loss = KL(sg(post) || prior)        # ← prior 学习匹配 post
rep_loss = KL(post || sg(prior))        # ← post 不要过度偏离 prior
```

**两个 loss 作用不同**:
- `dyn_loss`: 推动 prior 更准(防止 prior 偏移)
- `rep_loss`: 推动 posterior 不要坍塌到某个特定值(保留不确定性)

### Free Bits

```python
kl_loss = max(kl_loss, 1.0)  # per latent dim
```

**作用**: 每个 latent dim 至少学 1 nat 信息,防止"惰性 latent"(学不到东西)

### Residual 不接收 WM loss 梯度

```python
# Stage 1 训练 WM 时
residual_output = residual(prev_stoch, action)
prior_corrected = prior_logits + residual_output.detach()  # ← detach
kl_loss = kl(post, prior_corrected)
# 反向传播时,residual_output 不接收梯度(stop)
```

**为什么**: 
- Residual 应该独立学习 sim-to-real gap,不是 WM 训练的一部分
- 否则 WM 会"破坏"residual 的学习目标

---

## 7. Sim-to-Real Gap 分类(对 A1 任务)

A1 任务的 gap 比 DMC 复杂。ReDRAW 的 Residual 只覆盖**参数误差**,不直接覆盖**结构误差**:

| Gap 类型 | 来源 | ReDRAW Residual 能学? |
|---------|------|----------------------|
| **电机扭矩误差** | sim 假设标称,real 实际 -30% | ✅ 核心 |
| **PD 增益偏移** | sim 假设 20,real 实际 15-25 | ✅ 核心 |
| **质量误差** | 负载变化 | ✅ 核心 |
| **摩擦误差** | 地面不同 | ✅ 核心 |
| **齿轮间隙** | sim 假设无,real 有 0.1-0.5° | ❌ 结构误差,需额外机制 |
| **皮带柔性** | sim 假设刚性 | ❌ |
| **接触动力学** | 刚性 vs 形变 | ❌ |
| **电机死区** | 小电流不响应 | ❌ |

**r2wmp 的解决方案**:
- **Layer 1 (PhysicalResidual)**: 补偿参数误差(类似 ReDRAW)
- **Layer 2 (DynamicsResidual)**: 补偿未建模结构(基于 history 的时序模式)

---

## 8. 关键设计决策的"为什么"

### 为什么 Residual 加在 mean 而不是 stoch?

| 方案 | 优点 | 缺点 |
|------|------|------|
| 加在 stoch(采样) | 直观 | 不可微,无法反传 |
| 加在 logits(ReDRAW) | 数值稳定 | 仅限 discrete |
| 加在 mean(r2wmp) | 直接,可微 | 假设方差不变 |

### 为什么 Stage 1 想象训练 Residual 不参与?

想象训练 Actor 时,如果加 Residual:
- Actor 会学到"依赖 Residual 的 latent 的策略"
- Stage 2 Residual 重新创建时,Actor 输入分布完全变了
- Actor 行为退化

不加 Residual(Actor 在干净 latent 训练):
- Actor 学到"通用"的策略
- Stage 2 Residual 修正后,Actor 还能工作(因为 Residual 输出小)

### 为什么 Stage 2 Residual 重新创建而非继承?

| 方案 | 优点 | 缺点 |
|------|------|------|
| 继承继续训 | 连续训练 | Stage 1 输出≈0,继承没意义 |
| 重新创建 1 层 | 极小容量,只学核心 gap | 丢失 Stage 1 的可能信息 |
| 重新创建 N 层 | 灵活 | 过拟合风险 |

ReDRAW 选 **重新创建 1 层**:论文实验证明这个最稳定。

---

## 9. 公式速查

```
┌─────────────────────────────────────────┐
│ 公式 (15) - 先验                        │
│ û = f_φ(z_{t-1}, α̂_{t-1}, a_{t-1})  │
└─────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────┐
│ 公式 (16) - Residual 输出               │
│ ê = ψ(z_{t-1}, a_{t-1})               │
│ (ψ 是 Ensemble MLP,7 成员,3 层)         │
└─────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────┐
│ 公式 (18) - 残差加在 logits              │
│ α̂_real = softmax(û + ê)              │
└─────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────┐
│ 公式 (20) - Residual loss               │
│ L_ψ = E_q[Σ_t KL(q_φ(z_t|x_t) || p_φ)] │
└─────────────────────────────────────────┘
```

---

## 10. r2wmp 实现的关键差异

| 维度 | ReDRAW | r2wmp | 原因 |
|------|--------|-------|------|
| **RSSM** | Discrete K-tuple | Discrete K-tuple | 一致 |
| **Obs** | 64×64 RGB | 48 维 proprio | A1 任务简化 |
| **Reward** | 稀疏 0/1 | 密集复合 | A1 dense reward |
| **Plan2Explore** | 启用 | 关闭 | A1 dense 不需要 |
| **环境并行** | 8-32 | 4096 | Isaac Gym 优势 |
| **Actor-Critic** | reparameterization | reparameterization | 一致 |
| **Residual** | 单层 logits | 双层(mean + deter) | A1 多源 gap |

---

## 11. 常见误解澄清

### 误解 1:"Residual 是 WM 的一部分"

❌ **错**: Residual 是**独立的模块**,在 Stage 1 不接收 WM loss 梯度。

### 误解 2:"Stage 1 Residual 学到了东西"

❌ **错**: Stage 1 源域 sim 无 gap,Residual 输出接近 0(零初始化 + stop_grad + 无 gap)。

### 误解 3:"Stage 2 冻结主网络 = 不更新 WM"

✅ **对**: 主 RSSM 在 Stage 2 完全冻结,只训练 Residual。这是 ReDRAW 的核心创新。

### 误解 4:"Residual 必须大才有效果"

❌ **错**: Residual 输出小才对(避免 Actor 分布漂移)。ReDRAW 论文表明小 Residual 反而效果更好。

---

## 12. 实验设计(下一步可做)

### 消融实验清单

| 实验 | 配置 | 目的 |
|------|------|------|
| **Baseline** | 没有 Residual | 证明 Residual 有效 |
| **Stage 1 only** | Stage 1 训完后直接部署 | 证明两阶段必要 |
| **ReDRAW 原版** | 1 层 Residual | 对比基线 |
| **r2wmp 双层** | Physical + Dynamics | 验证双层假设 |
| **不同 domain rand** | 弱 / 强随机化 | 找最佳 gap |

### 评估指标

- `episode_return`: 1000 步 reward 总和
- `success_rate`: 完成任务的 episode 比例
- `gap_closed_pct`: sim-to-real gap 闭合率
- **目标**: gap_closed_pct > 30%(最小),> 70%(理想)

---

**文档版本**: 1.0
**最后更新**: 2026-06-18
**用途**: r2wmp 项目实施参考