# r2wmp 需求与设计评审

> 本文档定义项目的完整需求,并对照代码实现进行**自审**。这是给 Claude Code(或人类)做代码评审的 checklist。

---

## 1. 项目目标(需求)

### 1.1 核心目标

把 **ReDRAW 论文**(Adapting World Models with Latent-State Dynamics Residuals)的两阶段训练 + Ensemble Residual 思想,移植到 **Unitree A1** 仿真环境,实现 sim-to-real 迁移。

### 1.2 输入/输出

**输入**:
- ReDRAW 论文:Ensemble Residual + 两阶段训练 + freeze_wm + 100x lr
- WMP 仓库: A1AMP 环境(Isaac Gym + LeggedRobot)
- 数据: 源域 sim(物理参数扰动)+ 目标域 sim(标称参数,代表"真机")

**输出**:
- 在 PseudoRealEnv 上比 zero-shot 性能显著提升的策略
- 三策略对比报告(Zero-shot vs Residual 微调 vs 真实上界)

### 1.3 验收标准

| 指标 | 期望 |
|------|------|
| Sim-to-real gap 闭合率 | > 30%(最小),> 70%(理想) |
| 策略 B(return) | > 策略 A(return) |
| 代码语法 | 100% 通过 |
| WMP 接口对齐 | 100% domain_rand 字段覆盖 |
| 端到端训练 | 能跑完 Stage 1 + Stage 2 |

---

## 2. 架构设计

### 2.1 三大环境

```
┌─────────────────────────────────────────────────────┐
│  InaccurateSimEnv (num_envs=4096)                   │
│  - 域随机化: 电机 0.65-0.75, PD ±20%, 摩擦 ±50%     │
│  - 用途: Stage 1 训练                                │
├─────────────────────────────────────────────────────┤
│  PseudoRealEnv (num_envs=64)                        │
│  - 域随机化: 全部关闭(标称物理)                     │
│  - 用途: Stage 2 数据采集 + 最终评估                │
├─────────────────────────────────────────────────────┤
│  共享地形: trimesh, 相同配置                        │
│  共享观测: 45 维 proprio                            │
└─────────────────────────────────────────────────────┘
```

### 2.2 两阶段训练

```
Stage 1:  InaccurateSim → 训练 WM + AC  (2M 步)
          ↓
          checkpoint
          ↓
Stage 2:  PseudoReal 数据 → 冻结 WM, 只训 Residual  (1M 步)
          ↓
          checkpoint
          ↓
Eval:     PseudoReal 上跑 50 episodes × 3 策略对比
```

### 2.3 关键算法设计(对齐 ReDRAW)

| 设计点 | ReDRAW 论文 | r2wmp 实现 |
|--------|-------------|-----------|
| RSSM 类型 | Discrete K-tuple | **同** (stoch=32, discrete=32) |
| Residual 位置 | latent stochastic | **同** (PhysicalResidual 加在 mean) |
| Residual 初始化 | 严格零初始化 | **同** (`nn.init.zeros_`) |
| Stage 1 stop_grad | True | **同** (`stop_residual_grad=True`) |
| Stage 2 重建 Residual | 1 成员 1 层 | **同** (1 层 MLP) |
| Stage 2 lr | 100x (1e-2) | **同** |
| Actor 训练时 Residual | 不参与 | **同** (`with_residual=False` 想象) |
| 冻结主网络 | freeze_wm | **同** (`freeze_main_network()`) |

---

## 3. 代码评审 Checklist

### 3.1 环境层(`envs/`)

#### 3.1.1 `base_env.py`

- [ ] 类 `BaseEnv` 定义所有子类共享的接口
- [ ] 属性:`num_envs`, `device`, `obs_dim=45`, `action_dim=12`
- [ ] 方法:`reset()`, `step(action)`, `get_proprio_obs()`, `close()`

#### 3.1.2 `wmp_env_base.py`

- [ ] 继承 `BaseEnv`
- [ ] 通过 `sys.path.append(WMP_ROOT)` 引入 WMP
- [ ] 调用 `LeggedRobot` 创建 Isaac Gym 仿真
- [ ] `step()` 标准化 WMP 的 7 元组返回为 4 元组
- [ ] `_configure_domain_rand(cfg)` 是抽象方法(子类必须实现)
- [ ] `_configure_terrain(cfg, terrain_config)` 共享配置

**潜在问题**:
- `sim_params` 配置可能与 WMP 默认值不一致
- WMP 内部 obs 维度可能是 235 而非 33(取决于 include_history_steps)

#### 3.1.3 `inaccurate_sim_env.py`

- [ ] 继承 `WMPEnvBase`
- [ ] 加载 `A1AMPCfg`
- [ ] 配置 24 个 domain_rand 字段(全部已覆盖)
- [ ] num_envs 默认 4096

#### 3.1.4 `pseudo_real_env.py`

- [ ] 继承 `WMPEnvBase`
- [ ] 加载 `A1AMPCfg`
- [ ] **关闭**所有域随机化(24 个字段都设为标称)
- [ ] num_envs 默认 64

#### 3.1.5 `verify_terrain_consistency()`

- [ ] 函数存在
- [ ] 比较两个环境的 terrain 配置
- [ ] 返回 bool

### 3.2 模型层(`models/`)

#### 3.2.1 `residual.py`

- [ ] `PhysicalResidual` 严格零初始化(`init='zero'`)
- [ ] `DynamicsResidual` 严格零初始化
- [ ] `make_*_for_stage1/2` 工厂函数区分 3 层 / 1 层
- [ ] forward 接受 (prev_stoch, action) / (deter_history, action)

#### 3.2.2 `rssm.py`

- [ ] 离散 K-tuple categorical(`discrete=32`)
- [ ] GRU-based deter 更新
- [ ] `kl_loss()` 修复版(已删除 placeholder)
- [ ] `get_feat()` 返回 `[stoch_flat + deter]`
- [ ] `_get_dist()` 支持 discrete 和 continuous

**潜在问题**:
- `kl_loss` 第 313-314 行:用 `kl_divergence` 的方向可能不正确,需对齐 ReDRAW
- `prior['stoch']` 的 shape 在 discrete 模式下是 `[B, stoch_dim, discrete]`,flatten 操作需要小心

#### 3.2.3 `world_model.py`

- [ ] `observe()` 支持 `with_residual` 和 `stop_residual_grad` 开关
- [ ] `imagine()` 默认 `with_residual=False`
- [ ] `_apply_residual()` 加在 `post['mean']` 和 `post['deter']` 上
- [ ] `recreate_residual_for_stage2()` 重建 1 层 Residual
- [ ] `freeze_main_network()` 只保留 Residual 可训

**潜在问题**:
- `imagine()` 内部 `state` 的 `dist_type` 字段在 stacking 时需要过滤
- `_apply_residual()` 中 `prev_stoch_flat` 的 flatten 操作可能维度错

#### 3.2.4 `actor.py`

- [ ] forward 返回 `(action, mean, std, entropy)`
- [ ] `sample()` 用于 env 交互
- [ ] `get_log_prob()` 用于 AC loss
- [ ] tanh 压缩到 [-1, 1]

#### 3.2.5 `critic.py`

- [ ] `A1Critic` 简单 V(s)
- [ ] `SlowCritic` EMA target

### 3.3 训练层(`training/`)

#### 3.3.1 `replay_buffer.py`

- [ ] 支持序列采样(`sample(batch_size, seq_length)`)
- [ ] is_first 自动推断
- [ ] save/load .npz 格式

#### 3.3.2 `wm_loss.py`

- [ ] `compute_wm_loss()` 计算 KL + recon
- [ ] `compute_residual_loss()` Stage 2 用

#### 3.3.3 `ac_loss.py`

- [ ] `compute_lambda_return()` 实现 λ-return
- [ ] `compute_ac_loss()` 计算 actor + critic loss
- [ ] **关键**:imagine 时 Residual **不参与**(`with_residual=False`)

#### 3.3.4 `trainer.py`

- [ ] `train_stage1()` 数据收集 + WM 训练 + AC 训练
- [ ] `train_stage2_residual()` 冻结主网络 + 只训 Residual
- [ ] `save_checkpoint()` / `load_checkpoint()`

**潜在问题**:
- 训练循环中数据收集与训练的频率
- EMA target critic 更新时机

### 3.4 评估层(`evaluation/`)

#### 3.4.1 `eval_protocol.py`

- [ ] `evaluate_policy()` 支持 `use_residual` 开关
- [ ] `three_policy_comparison()` 三策略对比
- [ ] `save_results()` 输出 JSON

### 3.5 脚本层(`scripts/`)

- [ ] `train_stage1.py` 启动 Stage 1
- [ ] `collect_pseudo_real_data.py` 采数据
- [ ] `train_stage2.py` 启动 Stage 2
- [ ] `eval_compare.py` 评估

---

## 4. 已知风险与缓解

| 风险 | 严重度 | 缓解 |
|------|--------|------|
| WMP 在 Windows 上需要 WSL2 | 高 | 文档说明 |
| Isaac Gym 安装复杂 | 高 | 文档说明 |
| 4096 env 显存不够 | 中 | 文档提供降级方案 |
| Stage 1 Actor 在 PseudoReal 上直接摔 | 中 | 可选 PPO baseline 采数据 |
| Residual 学不到东西 | 中 | 调整 Stage 2 lr / 数据量 |
| 静态验证无法覆盖运行时问题 | 高 | **必须用户实测** |

---

## 5. 测试覆盖

### 5.1 静态验证(已完成)

- ✅ 26 个文件语法 100% 正确
- ✅ 16 个关键类 100% 存在
- ✅ WMP domain_rand 字段 100% 覆盖
- ✅ 关键方法签名 100% 正确

### 5.2 动态验证(待用户)

- ⏳ `tests/test_mock.py` 需要 torch 才能跑
- ⏳ Stage 1 训练需要 Isaac Gym + GPU
- ⏳ Stage 2 训练需要 Stage 1 checkpoint
- ⏳ 评估需要完整训练好的模型

---

## 6. Claude Code 评审指令模板

让 Claude Code 检查项目时,用以下 prompt:

```
请评审 D:\songay\sim2real\r2wmp 项目:
1. 对照 D:\songay\sim2real\r2wmp\docs\implementation_targets.md 中的需求
2. 检查代码实现是否满足需求
3. 特别关注:
   - ReDRAW 算法的核心机制是否对齐(零初始化、stop_grad、Stage 2 重建)
   - WMP 接口的字段是否对齐
   - 是否存在潜在 bug(梯度流、维度匹配、shape 不对)
4. 输出评审报告,包含:
   - 每个文件的实现状态(✅/⚠️/❌)
   - 发现的 bug 和风险
   - 改进建议
```

---

## 7. 改进优先级

### P0 (必须修复)

1. **trainer.py 训练循环逻辑** — 关键流程
2. **world_model.py 的 imagine()** — 梯度流是否正确
3. **wmp_env_base.py 的 step()** — WMP 接口对齐

### P1 (重要)

4. **scripts/eval_compare.py** — 评估逻辑
5. **scripts/collect_pseudo_real_data.py** — 数据采集
6. **rssm.py 的 KL loss 方向**

### P2 (可选)

7. 添加 tensorboard 日志
8. 添加单元测试覆盖更多边界情况
9. 添加可视化脚本

---

**文档结束**