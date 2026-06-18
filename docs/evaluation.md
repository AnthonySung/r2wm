# 评估协议详解

## 目的

量化 sim-to-real gap 闭合效果,验证 ReDRAW 算法在 A1 任务上的有效性。

## 三策略对比

### 策略 A: Zero-shot(基线)

| 来源 | Stage 1 checkpoint |
|------|-------------------|
| Residual | Stage 1 训出的(≈ 0) |
| Actor | Stage 1 训出的 |
| 用途 | 衡量"没有 Residual 微调"时,在伪 real 上的表现 |

### 策略 B: Stage 2 Residual 微调

| 来源 | Stage 1 + Stage 2 Residual |
|------|---------------------------|
| Residual | Stage 2 重新训的(非 0) |
| Actor | Stage 1 训出的(**完全冻结,不动**) |
| 用途 | 衡量"只靠 Residual 微调"能闭合多少 gap |

### 策略 C: Upper bound(可选)

| 来源 | Stage 1 + Stage 2 全量微调 |
|------|--------------------------|
| Residual | Stage 2 |
| Actor | Stage 2 微调 |
| 用途 | 衡量"如果能微调 Actor"的性能上限 |

## 评估流程

### 步骤 1: 在 PseudoRealEnv 上跑 N=50 episodes

```python
metrics = evaluate_policy(
    actor, world_model, pseudo_real_env,
    num_episodes=50,
    use_residual=False,  # 策略 A
)
```

### 步骤 2: 在 PseudoRealEnv 上用 Residual 跑 N=50 episodes

```python
metrics_B = evaluate_policy(
    actor, world_model, pseudo_real_env,
    num_episodes=50,
    use_residual=True,  # 策略 B
)
```

### 步骤 3: 在 InaccurateSim 上跑(用于计算 gap)

```python
metrics_sim = evaluate_policy(
    actor, world_model, inaccurate_sim_env,
    num_episodes=20,
    use_residual=False,
)
```

### 步骤 4: 计算 gap 闭合率

```
gap_before = sim_return - A_real_return   # 没有 Residual 微调时的 gap
gap_after  = sim_return - B_real_return   # Residual 微调后的 gap
gap_closed_pct = (gap_before - gap_after) / |gap_before| * 100%
```

## 评估指标

| 指标 | 含义 | 期望 |
|------|------|------|
| `episode_return` | 1000 步 reward 总和 | B > A |
| `success_rate` | return > 200 的比例 | B > A |
| `fall_rate` | 摔倒比例 | B < A |
| `mean_episode_length` | 平均长度 | B ≈ A(不被提前终止) |

## 结果输出

保存到 `results/comparison.json`:

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

## 解读

| gap_closed_pct | 含义 |
|----------------|------|
| > 70% | ✅ 算法有效,成功闭合大部分 sim-to-real gap |
| 30-70% | ⚠️ 部分有效,Residual 学到了一些 gap |
| < 30% | ❌ 效果不佳,需要调整方案 |

## 注意事项

1. **策略 A 必须用 Stage 1 的 Residual**(≈ 0),不能用 Stage 2 的(否则不算 zero-shot)
2. **策略 B 必须冻结 Actor**(只启用 Residual),否则归因不清
3. **N=50 episodes** 是经验值,可以根据需要调整
4. **Reward 阈值 200** 是经验值,需要根据实际 reward scale 调整