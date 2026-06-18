# r2wmp 故障排查指南

> 本文档列出 r2wmp 训练中的常见问题和解决方案。

---

## 1. 环境相关问题

### 1.1 `ImportError: No module named 'isaacgym'`

**原因**: Isaac Gym 未安装,或路径不对。

**解决**:
```bash
# 检查 Isaac Gym 是否在标准位置
ls /opt/isaacgym  # Linux
ls D:/songay/sim2real/WMP/isaacgym  # Windows WSL2

# 设置 PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/opt/isaacgym/python

# 或重新安装
cd /opt/isaacgym/python
pip install -e .
```

### 1.2 `ImportError: No module named 'legged_gym'`

**原因**: WMP 未正确安装。

**解决**:
```bash
cd /mnt/d/songay/sim2real/WMP/rsl_rl
pip install -e .

cd ../legged_gym
pip install -e .
```

### 1.3 `RuntimeError: Could not find a valid isaacgym installation`

**原因**: Isaac Gym 和 PyTorch 版本不匹配。

**解决**:
```bash
# Isaac Gym Preview 4 要求:
# - PyTorch 1.10 - 2.0
# - CUDA 11.7+

# 检查
python -c "import torch; print(torch.__version__)"  # 应该是 1.10-2.0
python -c "import torch; print(torch.version.cuda)"  # 应该是 11.7+
```

### 1.4 CUDA Out of Memory

**症状**:
```
RuntimeError: CUDA out of memory. Tried to allocate 1.50 GiB
```

**解决**:
```python
# 方案 1: 降低 num_envs
python scripts/train_stage1.py --num_envs 2048  # 从 4096 降到 2048

# 方案 2: 降低 RSSM 维度
# configs/train.yaml
rssm:
  deter: 256        # 从 512 降到 256
  stoch: 16         # 从 32 降到 16
  hidden: 256       # 从 512 降到 256

# 方案 3: 关掉 heightmap
# configs/train.yaml
terrain:
  measure_heights: false  # 节省 187 维 obs
```

---

## 2. 模型相关问题

### 2.1 `RuntimeError: mat1 and mat2 shapes cannot be multiplied`

**症状**: 模型输入维度不匹配。

**排查**:
```python
# 检查 obs 维度
print(f"Obs shape: {obs.shape}")  # 应该是 (batch, 45)

# 检查 feat 维度
feat = wm.rssm.get_feat(state)
print(f"Feat shape: {feat.shape}")  # 应该是 (batch, 1024+512) = (batch, 1536)
```

**常见原因**:
- `discrete` 配置错误(影响 stoch_flat_dim)
- `deter_dim` 配置错误

**解决**: 检查 `configs/train.yaml` 中的 RSSM 配置。

### 2.2 `RuntimeError: one of the variables needed for gradient computation has been modified`

**症状**: 梯度计算异常,通常是 in-place 操作。

**排查**: 检查 `recreate_residual_for_stage2()` 后是否调用了 `freeze_main_network()`,因为重新创建 Residual 会替换原参数。

**解决**: 调用顺序应该是:
```python
# 1. 先 recreate
wm.recreate_residual_for_stage2()
# 2. 再 freeze(此时 Residual 已经是新模块)
wm.freeze_main_network()
```

### 2.3 Residual 输出不是 0

**症状**: Stage 1 训练初期 Residual 输出远大于 0(应 ≈ 0)。

**排查**:
```python
# 检查初始化
for m in wm.physical_residual.net:
    if isinstance(m, nn.Linear):
        print(f"weight max: {m.weight.abs().max()}")
        # 应该是 0
```

**解决**: 确认 `PhysicalResidual.__init__` 中 `init='zero'`(默认),不是 `'small_normal'`。

---

## 3. 训练相关问题

### 3.1 KL loss = NaN

**症状**: `kl_loss: nan`

**排查**:
```python
# 检查输入是否含 nan/inf
print(f"obs has nan: {torch.isnan(obs).any()}")
print(f"action has nan: {torch.isnan(action).any()}")
```

**常见原因**:
- 输入数据未归一化
- learning rate 太大

**解决**:
```python
# 降低学习率
configs/train.yaml:
  training:
    model_lr: 3e-5  # 从 1e-4 降到 3e-5

# 或加梯度裁剪(默认已有 1000,如果还不行,降低到 100)
training:
  grad_clip: 100.0
```

### 3.2 Episode return 一直不上升

**症状**: 训练 100K 步,return 仍是 0。

**排查**:
```python
# 1. 检查 reward 是否正常
sample_rewards = [replay._reward[i] for i in range(100)]
print(f"Reward stats: min={min(sample_rewards):.2f}, max={max(sample_rewards):.2f}")

# 2. 检查 actor 输出是否合理
action = actor.sample(feat)
print(f"Action range: [{action.min():.3f}, {action.max():.3f}]")

# 3. 检查 IMAGINATION 是否有梯度
loss, metrics = compute_ac_loss(...)
print(f"lambda_return: {metrics['lambda_return_mean']}")
```

**常见原因**:
- Actor 输出退化(全 0 或全 -1)
- Reward 信号太弱
- World Model 还没收敛就开始训 Actor

**解决**:
- 先只训 WM(关掉 AC 训练),等 KL loss 收敛后再训 AC
- 调大 entropy_scale(从 1e-3 到 1e-2)

### 3.3 训练 OOM 在想象阶段

**症状**: 训练 1 步成功,但 10 步后 OOM。

**原因**: `imag_horizon=15` 太大。

**解决**:
```yaml
training:
  imag_horizon: 10  # 从 15 降到 10
```

### 3.4 检查点加载失败

**症状**: `RuntimeError: Error(s) in loading state_dict`

**原因**: Stage 1 / Stage 2 的 Residual 结构不一致。

**解决**:
```python
# Stage 2 必须重新创建 Residual
wm.recreate_residual_for_stage2()  # 这会创建 1 层 Residual
wm.load_state_dict(stage1_ckpt, strict=False)  # strict=False 允许忽略不匹配
```

或确保加载顺序:
```python
wm = WorldModel(...)  # 默认 3 层
wm.load_state_dict(stage1_ckpt)  # 加载 3 层
wm.recreate_residual_for_stage2()  # 重建为 1 层
```

---

## 4. 数据采集问题

### 4.1 Actor 在 PseudoReal 上直接摔倒

**症状**: 采集的伪 real 数据全是 done。

**原因**: Stage 1 训练的 Actor 在 PseudoReal(物理参数不同)上表现差。

**解决**:
- 这是**预期行为**,ReDRAW 算法就是用来解决这个问题的
- 但太极端会导致 Stage 2 没有可用数据

**临时方案**(可选):
```python
# 用 PPO 在 PseudoReal 上快速训练一个 baseline,再用它的策略采数据
# 见 scripts/collect_pseudo_real_data_with_baseline.py
```

### 4.2 Replay Buffer 太大

**症状**: 磁盘占满。

**解决**:
```python
# 降低 capacity
ReplayBuffer(capacity=200_000)  # 从 1M 降到 200K
```

---

## 5. 评估问题

### 5.1 B 比 A 性能差

**症状**: Residual 微调后,gap 闭合率 < 0(负值)。

**可能原因**:
- Stage 1 没训练好(Actor 本身就很差)
- Stage 2 学习率太高(Residual 学过头)
- 伪 real 数据量太少

**调试**:
```python
# 1. 检查 Residual 输出的范数
phys_residual_norm = wm.physical_residual(prev_stoch, action).norm(dim=-1).mean()
print(f"Phys residual norm: {phys_residual_norm}")
# 应该 < 1.0

# 2. 加载不同 checkpoint 对比
# 比如 stage1_step500k, stage1_step1M, stage1_step2M
```

**解决**:
- 增加 Stage 1 训练步数
- 降低 Stage 2 学习率(从 1e-2 到 3e-3)
- 增加 Stage 2 数据量(从 200 episodes 到 500)

### 5.2 评估结果波动大

**症状**: 同一策略跑两次,return 差 100+。

**原因**: num_episodes 太少。

**解决**:
```bash
python scripts/eval_compare.py --num_episodes 100  # 从 50 增加到 100
```

---

## 6. 调试技巧

### 6.1 启用详细日志

```python
# 在 train_stage1.py 中
import logging
logging.basicConfig(level=logging.DEBUG)

# 在 trainer.py 中
print(f"[Step {step}] obs range: [{obs.min():.2f}, {obs.max():.2f}]")
print(f"[Step {step}] action range: [{action.min():.2f}, {action.max():.2f}]")
print(f"[Step {step}] reward: {reward.mean():.2f}")
```

### 6.2 可视化训练曲线

```python
# 用 tensorboard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('logs/stage1')

writer.add_scalar('train/kl_loss', kl_loss, step)
writer.add_scalar('train/actor_loss', ac_loss, step)
writer.add_scalar('eval/return', eval_return, step)
```

```bash
tensorboard --logdir logs/
```

### 6.3 检查 WMP 接口对齐

```python
# 在 WMP 仓库下
python -c "
from legged_gym.envs.a1.a1_amp_config import A1AMPCfg
cfg = A1AMPCfg()
print('domain_rand fields:')
for attr in dir(cfg.domain_rand):
    if not attr.startswith('_'):
        print(f'  {attr}')
"
```

---

## 7. 性能调优

### 7.1 训练速度慢

**检查项**:
- 是否使用了 GPU?
- num_envs 是否够大?
- batch_size 是否合理?

**优化**:
```yaml
# configs/train.yaml
training:
  batch_size: 1024  # 从 512 增大(需要更多显存)
  imag_horizon: 15  # 不变
```

### 7.2 GPU 利用率低

```bash
# 监控 GPU
nvidia-smi -l 1

# 如果利用率 < 80%,考虑:
# 1. 增大 num_envs(从 2048 到 4096)
# 2. 减小 batch_size(让训练更频繁)
```

---

## 8. 获取帮助

如果以上都不能解决:

1. **查看 GitHub Issues**(如果项目开源)
2. **打印完整错误堆栈**:
   ```python
   import traceback
   try:
       # 出错的代码
       ...
   except Exception:
       traceback.print_exc()
   ```
3. **提供最小复现脚本**

---

**文档结束**