# r2wmp 实施目标文档

> 本文档定义每个文件的实施目标、关键依赖、验收标准。**纯 PyTorch 实现**,**不依赖 JAX**。

---

## 0. 全局架构

```
┌──────────────────────────────────────────────────────────────┐
│              r2wmp 项目(纯 PyTorch)                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Stage 1                    Stage 2                         │
│  ┌─────────────┐            ┌─────────────┐                 │
│  │ Inaccurate  │            │  Pseudo     │                 │
│  │   SimEnv    │            │  RealEnv    │                 │
│  │ (4096 envs) │            │ (64 envs)   │                 │
│  └──────┬──────┘            └──────┬──────┘                 │
│         │ data                    │ data                    │
│         ↓                         ↓                         │
│  ┌─────────────────────────────────────────────┐             │
│  │         Replay Buffer (numpy/torch)         │             │
│  └──────────────────┬──────────────────────────┘             │
│                     ↓                                        │
│  ┌─────────────────────────────────────────────┐             │
│  │   World Model (PyTorch)                     │             │
│  │   ├─ Encoder (45→256)                       │             │
│  │   ├─ RSSM (Deter=512, Stoch=32×32)          │             │
│  │   ├─ PhysicalResidual (3层→1层)            │             │
│  │   ├─ DynamicsResidual (3层→1层, K=4)       │             │
│  │   └─ Decoder (256+512→45)                   │             │
│  └──────────────────┬──────────────────────────┘             │
│                     ↓                                        │
│  ┌─────────────────────────────────────────────┐             │
│  │   Actor-Critic (PyTorch)                    │             │
│  │   ├─ A1Actor (feat→action)                  │             │
│  │   └─ A1Critic (feat→V)                     │             │
│  └─────────────────────────────────────────────┘             │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**禁止使用 JAX/Jaxlib/flax**。所有代码基于 PyTorch + numpy。

---

## 1. 环境层实施目标

### 1.1 `envs/base_env.py`(新增基础类)

**目标**: 定义两个环境共用的接口和行为。

**关键属性**:
- `num_envs: int` — 并行环境数量
- `device: str` — 'cuda:0' 或 'cpu'
- `obs_dim: int = 45` — 观测维度(本体感知)
- `action_dim: int = 12` — 动作维度(12 个关节)
- `dt: float = 0.02` — policy step 时间(decimation=4 × sim_dt=0.005)
- `max_episode_steps: int = 1000` — 每 episode 步数

**关键方法**:
```python
def reset(self) -> torch.Tensor:
    """重置所有 env,返回 obs [num_envs, 45]"""

def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """执行一步,返回 (obs, reward, done, info)"""

def get_proprio_obs(self) -> torch.Tensor:
    """从完整 obs 提取 45 维 proprio"""
```

### 1.2 `envs/wmp_env_base.py`(WMP 接口包装)

**目标**: 调用 WMP 的 `LeggedRobot` 并标准化输出。

**关键依赖**:
- `import sys; sys.path.append('D:/songay/sim2real/WMP')`
- `from legged_gym.envs.base.legged_robot import LeggedRobot`
- `from isaacgym import gymapi, gymtorch`

**关键流程**:
```python
class WMPEnvBase:
    def __init__(self, cfg, num_envs, device, headless):
        # 1. 创建 Isaac Gym sim
        self.gym = gymapi.acquire_gym()
        self.sim = self.gym.create_sim(...)
        
        # 2. 创建 LeggedRobot(WMP 的)
        self.env = LeggedRobot(cfg, sim_params, gymapi.SIM_PHYSX, device, headless)
        
    def reset(self):
        # 调用 WMP LeggedRobot.reset()
        # 返回 (obs, privileged_obs)
        policy_obs, priv_obs = self.env.reset()
        return self._extract_proprio(policy_obs)
    
    def step(self, action):
        # 调用 WMP LeggedRobot.step()
        # 返回 7 元组,标准化成 (obs, reward, done, info)
        result = self.env.step(action)
        policy_obs, priv_obs, rew, reset_buf, extras, reset_ids, amp_states = result
        return self._extract_proprio(policy_obs), rew, reset_buf, extras
    
    def _extract_proprio(self, full_obs):
        """从 235 维 policy_obs 提取 45 维 proprio
        33 维 base info + 12 维 last_actions"""
        # WMP obs 顺序(根据 compute_observations):
        # base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) + 
        # commands(3) + (dof_pos-default)(12) + dof_vel(12) + 
        # actions(12) + heights(187) + privileged...
        # = 45 (前 45 维本体) + 187 + ...
        
        if full_obs.shape[-1] >= 45:
            return full_obs[..., :45]
        else:
            raise ValueError(f"WMP obs too small: {full_obs.shape}")
```

**验收标准**:
- ✅ 能成功创建 4096 个并行 env(Isaac Gym)
- ✅ `reset()` 返回 `[4096, 45]` tensor
- ✅ `step(action)` 返回正确的 obs/reward/done/info
- ✅ 能在 GPU 上跑(必须)

### 1.3 `envs/inaccurate_sim_env.py`(Stage 1 训练环境)

**目标**: 包装 WMPEnvBase + 域随机化配置。

**配置**(对齐 WMP `LeggedRobotCfg.domain_rand`):
```python
class InaccurateSimEnv(WMPEnvBase):
    def _configure_domain_rand(self):
        # 这些是 WMP 已有的 domain_rand 字段
        self.env.cfg.domain_rand.randomize_motor_strength = True
        self.env.cfg.domain_rand.motor_strength_range = [0.65, 0.75]  # 关键: -30%
        self.env.cfg.domain_rand.randomize_PD_gains = True
        self.env.cfg.domain_rand.Kp_range = [14, 18]   # 标称 20 的 70-90%
        self.env.cfg.domain_rand.Kd_range = [0.3, 0.5] # 标称 0.5 的 60-100%
        self.env.cfg.domain_rand.randomize_base_mass = True
        self.env.cfg.domain_rand.added_mass_range = [1.0, 3.0]  # +1-3kg
        self.env.cfg.domain_rand.randomize_friction = True
        self.env.cfg.domain_rand.friction_range = [0.3, 1.5]
        self.env.cfg.domain_rand.push_robots = True
        self.env.cfg.domain_rand.push_interval_s = [3, 8]
        self.env.cfg.domain_rand.push_force = [10, 30]
        
        # 观测维度:235(完整,包含 heightmap 和 privileged)
        # 实际给 policy 用的是 232(去掉 base_lin_vel)或 235(包含)
        # 我们从 235 中提取前 45 维 proprio
        
        # 关键:地形配置保持
        self.env.cfg.terrain.measure_heights = True  # 同 PseudoReal
        self.env.cfg.terrain.mesh_type = 'trimesh'
        # 其他 terrain 字段也同 PseudoReal
```

**关键不变量**(与 PseudoReal 一致):
- `num_actions = 12`
- `episode_length_s = 20`(1000 步 @ 50Hz policy)
- 地形类型(trimesh 配置)
- `action_scale = 0.25`
- `stiffness = {'joint': 20.}`
- `damping = {'joint': 0.5}`

### 1.4 `envs/pseudo_real_env.py`(评估 + 数据采集环境)

**目标**: 同样的 WMPEnvBase,但**关闭所有域随机化**(代表"真机")。

**配置**:
```python
class PseudoRealEnv(WMPEnvBase):
    def _configure_domain_rand(self):
        # 全部关闭 → 标称物理
        self.env.cfg.domain_rand.randomize_motor_strength = False
        self.env.cfg.domain_rand.motor_strength_range = [1.0, 1.0]
        self.env.cfg.domain_rand.randomize_PD_gains = False
        self.env.cfg.domain_rand.Kp_range = [20.0, 20.0]
        self.env.cfg.domain_rand.Kd_range = [0.5, 0.5]
        self.env.cfg.domain_rand.randomize_base_mass = False
        self.env.cfg.domain_rand.added_mass_range = [0.0, 0.0]
        self.env.cfg.domain_rand.randomize_friction = False
        self.env.cfg.domain_rand.friction_range = [1.0, 1.0]
        self.env.cfg.domain_rand.push_robots = False
        
        # 地形同 InaccurateSim
        self.env.cfg.terrain = self.env.cfg.terrain  # 保持一致
```

**关键**:
- 物理参数: 标称(代表"真机")
- 观测提取: 45 维 proprio(模拟真机传感器)
- num_envs: **64**(模拟真机只有少量并行,不是 4096)
- 地形: **必须与 InaccurateSim 相同**

---

## 2. 模型层实施目标(全 PyTorch)

### 2.1 `models/encoder.py`

**目标**: 把 45 维 obs 编码成 embed_dim 维向量。

**结构**:
```python
class Encoder(nn.Module):
    def __init__(self, obs_dim=45, embed_dim=256, hidden=1024, n_layers=5):
        # 5 层 MLP,每层 1024 hidden
        # symlog 输入变换
    
    def forward(self, obs):
        return self.net(symlog(obs))  # [B, embed_dim]
```

**关键点**:
- `nn.Linear` + `nn.LayerNorm` + `nn.SiLU`
- 对齐 ReDRAW `symlog_inputs=True`

### 2.2 `models/decoder.py`

**目标**: 从 feat 重建 obs。

**结构**:
```python
class Decoder(nn.Module):
    def __init__(self, feat_dim, obs_dim=45, hidden=1024, n_layers=5):
        # 5 层 MLP
        # 输出不应用 symlog 反变换(简化)
```

### 2.3 `models/rssm.py`

**目标**: 完整的 RSSM(K-tuple categorical + GRU)。

**结构**:
```python
class RSSM(nn.Module):
    def __init__(self, deter_dim=512, stoch_dim=32, discrete=32, 
                 hidden=512, action_dim=12, embed_dim=256):
        # 输入层: prev_stoch + action → img_in
        # GRU: img_in → deter
        # 输出层: deter → img_out
        # 统计层: img_out → (mean, std) 或 (logit)
        # Posterior: deter + embed → obs_out → stats
    
    def initial_state(self, batch_size, device='cuda'):
        # learned: tanh(W), zeros: torch.zeros
        return {'deter': ..., 'stoch': ...}
    
    def img_step(self, prev_state, prev_action, sample=True):
        # 先验: 从 action 预测下一状态
        return prior
    
    def obs_step(self, prev_state, prev_action, embed, is_first, sample=True):
        # Posterior: 用 embed 修正先验
        return post, prior
    
    def kl_loss(self, post, prior, free=1.0):
        # 标准 KL loss with free bits
        # dyn + rep
```

**关键**:
- discrete=32 → K-tuple categorical
- stoch_dim=32 → 32 组 categorical
- latent = stoch_flat_dim = 32 × 32 = 1024
- deter = 512
- feat = [stoch_flat + deter] = [1024 + 512] = 1536

### 2.4 `models/residual.py`

**目标**: 两层 Residual,严格零初始化。

**结构**:
```python
class PhysicalResidual(nn.Module):
    """加在 latent mean 上,补偿物理参数"""
    def __init__(self, stoch_dim=1024, action_dim=12, hidden=64, n_layers=3):
        # n_layers 层 MLP
        # 严格零初始化
    
    def forward(self, prev_stoch, action):
        return self.net(cat([prev_stoch, action], -1))

class DynamicsResidual(nn.Module):
    """加在 deter 上,补偿未建模动力学"""
    def __init__(self, deter_dim=512, action_dim=12, history_len=4, 
                 hidden=128, n_layers=3):
        # n_layers 层 MLP
        # 输入: deter_history.flatten + action
    
    def forward(self, deter_history, action):
        return self.net(cat([deter_history.flatten(-2), action], -1))
```

**关键设计**:
- **严格零初始化**:`nn.init.zeros_(layer.weight)` 和 `nn.init.zeros_(layer.bias)`
- Stage 1: n_layers=3
- Stage 2: n_layers=1

### 2.5 `models/world_model.py`

**目标**: 整合 Encoder + Decoder + RSSM + 两个 Residual。

**关键方法**:
```python
class WorldModel(nn.Module):
    def observe(self, obs_seq, action_seq, is_first_seq,
                with_residual=True, stop_residual_grad=True):
        """观察真实数据,Residual 参与(可 stop grad)
        Returns: posts, priors (dict of stacked tensors [B, T, ...])
        """
    
    def imagine(self, actor, init_state, horizon, with_residual=False):
        """在 WM 中想象轨迹
        Args:
            with_residual: 想象时是否用 Residual
                - True: 部署 / Stage 2 评估
                - False: Stage 1 训练 Actor
        """
    
    def recreate_residual_for_stage2(self):
        """Stage 2: 重新创建 Residual(1 层,零初始化)"""
    
    def freeze_main_network(self):
        """Stage 2: 冻结除 Residual 外的所有参数"""
```

**关键流程**(observe):
1. encode obs → embed [B, T, embed_dim]
2. RSSM.obs_step 循环 T 步 → posts, priors
3. 对每步应用 Residual(加在 mean / deter 上)
4. stop_residual_grad 控制是否 detach

**关键流程**(imagine):
1. 从初始 state 开始
2. actor.sample(feat) → action
3. rssm.img_step(state, action) → next_state
4. 重复 horizon 步
5. **不应用 Residual**(Stage 1 训练 Actor)

---

## 3. 训练层实施目标

### 3.1 `training/replay_buffer.py`

**目标**: 简单的 Replay Buffer(支持序列采样)。

**结构**:
```python
class ReplayBuffer:
    def __init__(self, capacity=1_000_000, obs_dim=45, action_dim=12, device='cuda'):
        # 5 个 numpy/torch 数组: obs, action, reward, next_obs, done
    
    def add(self, obs, action, reward, next_obs, done):
        """单步添加"""
    
    def add_batch(self, obs_batch, action_batch, ...):
        """批量添加"""
    
    def sample(self, batch_size=512, seq_length=50):
        """采样序列批次
        Returns:
            dict with keys: obs, action, reward, next_obs, done, is_first
            Each shape: [B, T, ...]
        """
    
    def save(self, path):
        """保存到 .npz"""
    
    def load(self, path):
        """从 .npz 加载"""
    
    @classmethod
    def from_npz(cls, path, ...):
        """从 .npz 创建实例"""
```

**关键点**:
- 使用 numpy 存储(节省 GPU 内存)
- 采样时转到 torch tensor
- 支持 is_first 自动推断(done 后是 first)

### 3.2 `training/wm_loss.py`

**目标**: World Model 训练损失。

**函数**:
```python
def compute_wm_loss(world_model, batch, free_bits=1.0):
    """
    Returns:
        total_loss, metrics_dict
    
    Loss:
        - KL loss (dyn + rep with free bits)
        - Reconstruction loss (MSE)
    """

def compute_residual_loss(world_model, batch, free_bits=1.0):
    """
    Stage 2: 只计算 Residual 的 KL loss
    Args:
        world_model: 主 RSSM 冻结,Residual 可训
    """
```

### 3.3 `training/ac_loss.py`

**目标**: Actor-Critic 训练损失(Dreamer 风格)。

**函数**:
```python
def compute_lambda_return(rewards, values, continues, lambda_=0.95, gamma=0.997):
    """λ-return 计算"""

def compute_ac_loss(actor, critic, target_critic, world_model, 
                     init_obs, init_is_first, init_action=None,
                     horizon=15, gamma=0.997, lambda_=0.95):
    """
    Dreamer 风格的 AC loss:
    1. 在 WM 中想象 horizon 步(不应用 Residual)
    2. 计算 λ-return
    3. Actor loss: 最大化 λ-return + entropy bonus
    4. Critic loss: 预测 λ-return
    """
```

### 3.4 `training/trainer.py`

**目标**: Stage 1 + Stage 2 训练主循环。

**关键方法**:
```python
class Trainer:
    def __init__(self, env, config, device='cuda'):
        # 创建 world_model, actor, critic, target_critic
        # 创建 optimizers
        # 创建 replay_buffer
    
    def train_stage1(self, total_steps):
        """Stage 1: 在 InaccurateSim 上训练 WM + AC
        
        流程(每步):
        1. 收集数据(env.step)
        2. WM 训练(每 100 步):compute_wm_loss
        3. AC 训练(每 100 步):compute_ac_loss
        4. EMA 更新 target_critic
        """
    
    def train_stage2_residual(self, real_replay, total_steps):
        """Stage 2: 在伪 real 数据上微调 Residual
        
        流程:
        1. recreate_residual_for_stage2()
        2. freeze_main_network()
        3. for step:
           - 采样伪 real batch
           - compute_residual_loss
           - 反向传播
        """
    
    def save_checkpoint(self, path):
        """保存完整 ckpt"""
    
    def load_checkpoint(self, path):
        """加载完整 ckpt"""
    
    def save_residual(self, path):
        """只保存 Residual"""
    
    def load_residual(self, path):
        """加载 Residual"""
```

---

## 4. 评估层实施目标

### 4.1 `evaluation/eval_protocol.py`

**目标**: 三策略对比评估。

**函数**:
```python
def evaluate_policy(actor, world_model, env, 
                     num_episodes=50, use_residual=False):
    """
    Args:
        use_residual: 
            - False: 策略 A(Stage 1,Residual≈0)
            - True: 策略 B(Stage 2,启用 Residual)
    Returns:
        dict with: mean_return, std_return, success_rate, fall_rate
    """

def three_policy_comparison(actor, world_model, 
                              pseudo_real_env, inaccurate_sim_env,
                              num_episodes=50):
    """运行 A/B/C 三策略对比,计算 gap 闭合率"""
```

---

## 5. 脚本层实施目标

### 5.1 `scripts/train_stage1.py`

**流程**:
```python
# 1. 加载配置
config = load_yaml('configs/train.yaml')

# 2. 创建环境
env = InaccurateSimEnv(num_envs=4096, device='cuda', headless=True)

# 3. 创建 Trainer
trainer = Trainer(env, config, device='cuda')

# 4. 训练
trainer.train_stage1(total_steps=2_000_000)

# 5. 保存
trainer.save_checkpoint('checkpoints/stage1_final.ckpt')
```

### 5.2 `scripts/collect_pseudo_real_data.py`

**流程**:
```python
# 1. 加载 Stage 1
ckpt = torch.load('checkpoints/stage1_final.ckpt')
actor = load_actor(ckpt)
world_model = load_world_model(ckpt)

# 2. 创建 PseudoRealEnv
env = PseudoRealEnv(num_envs=1, device='cuda')

# 3. 收集数据
buffer = ReplayBuffer()
for ep in range(200):
    obs = env.reset()
    for t in range(1000):
        action = actor.sample(world_model.encode(obs))
        next_obs, reward, done, _ = env.step(action)
        buffer.add(obs, action, reward, next_obs, done)
        obs = next_obs
        if done: break

# 4. 保存
buffer.save('datasets/pseudo_real_data.npz')
```

### 5.3 `scripts/train_stage2.py`

**流程**:
```python
# 1. 加载 Stage 1
trainer = Trainer(env, config, device='cuda')
trainer.load_checkpoint('checkpoints/stage1_final.ckpt')

# 2. 加载伪 real 数据
real_replay = ReplayBuffer.from_npz('datasets/pseudo_real_data.npz')

# 3. 训练 Residual
trainer.train_stage2_residual(real_replay, total_steps=1_000_000)

# 4. 保存 Residual
trainer.save_residual('checkpoints/stage2_final.ckpt')
```

### 5.4 `scripts/eval_compare.py`

**流程**:
```python
# 1. 加载模型
ckpt = torch.load('checkpoints/stage1_final.ckpt')
actor = load_actor(ckpt)
world_model = load_world_model(ckpt)
res_ckpt = torch.load('checkpoints/stage2_final.ckpt')
world_model.physical_residual.load_state_dict(res_ckpt['physical_residual'])
world_model.dynamics_residual.load_state_dict(res_ckpt['dynamics_residual'])

# 2. 创建环境
pr_env = PseudoRealEnv(num_envs=1)
sim_env = InaccurateSimEnv(num_envs=1)

# 3. 三策略对比
summary = three_policy_comparison(actor, world_model, pr_env, sim_env)
save_results(summary, 'results/comparison.json')
```

---

## 6. 关键技术细节

### 6.1 Obs 维度精确值

根据 WMP 源码(`compute_observations` 第 431-479 行):

```
privileged_obs_buf = cat([
    base_lin_vel(3) * obs_scales.lin_vel,           # [3]
    base_ang_vel(3) * obs_scales.ang_vel,            # [3]
    projected_gravity(3),                           # [3]
    commands[:, :3](3) * commands_scale,            # [3]
    (dof_pos - default)(12) * obs_scales.dof_pos,   # [12]
    dof_vel(12) * obs_scales.dof_vel,               # [12]
    actions(12),                                    # [12]
    heights(187) * obs_scales.height_measurements,  # [187] (if measure_heights)
    randomized_frictions(1) (if randomize_friction),
    randomized_restitutions(1),
    randomized_added_masses(1),
    randomized_com_pos(3),
    (randomized_p_gains/Kp - 1) * scale(12),
    (randomized_d_gains/Kd - 1) * scale(12),
    contact_force(4 * 3) = 12,  # penalised contacts × 3
    contact_flag(4),             # feet contact
])
```

**默认 A1AMP** 配置(`num_observations = 235`):
- 33 + 187 + 24 + 3 = **247 维**(超过 235)
- 实际 WMP 用 `num_observations = 235` 配置时,有些 privileged 字段不开启

**我们项目的 obs 策略**:
- **训练时**:env 返回完整 obs(235 或 247 维)
- **我们的 Encoder 输入**:只取**前 45 维 proprio**(base_lin_vel + base_ang_vel + projected_gravity + commands + (dof_pos-default) + dof_vel + actions)
- **Stage 2 伪 real** 也只取 45 维 proprio

**关键**:
- 不依赖 heightmap(简化)
- 不依赖 privileged info(简化)
- 不依赖 history(简化)
- 只有本体感知 + 命令

### 6.2 Reward 计算

**WMP 使用复杂 reward**(15+ 项),具体需要查看 `a1_amp_config.py` 和 `_prepare_reward_function()`。

**我们项目**:
- **沿用 WMP 的完整 reward 函数**(不重写)
- 这意味着 Stage 1 的 reward 和 Stage 2 的 reward 相同
- **关键差别**:Stage 1 是 InaccurateSim(物理参数随机),Stage 2 是 PseudoReal(标称物理)
- 因为物理不同,同一策略得到不同 reward

### 6.3 Reward 类型详细

WMP A1AMP 启用 `A1AMPCfg.rewards.scales`,典型配置:
```python
tracking_lin_vel: 1.0
tracking_ang_vel: 0.5
lin_vel_z: -2.0
ang_vel_xy: -0.05
torques: -0.0002
dof_acc: -2.5e-7
action_rate: -0.01
dof_pos_limits: -10.0
termination: 0  # disable
```

**合计 reward 量级**:典型 [-2, +2] / step,1000 步累计 = [-2000, +2000]

### 6.4 训练步数和硬件

| 阶段 | 步数 | GPU 内存 | 时间预估 |
|------|------|---------|---------|
| Stage 1 | 2M | ~4GB | ~6-12 小时(A100) |
| Stage 2 | 1M | ~2GB | ~3-6 小时(A100) |
| 评估 | 50 episodes × 3 策略 | < 1GB | ~30 分钟 |

---

## 7. 文件清单和代码量

| 文件 | 估计行数 | 状态 |
|------|----------|------|
| `envs/base_env.py` | 100 | 待写 |
| `envs/wmp_env_base.py` | 300 | 待写(关键) |
| `envs/inaccurate_sim_env.py` | 150 | 待写 |
| `envs/pseudo_real_env.py` | 150 | 待写 |
| `envs/obs_wrapper.py` | 80 | 已存在 |
| `models/encoder.py` | 60 | 已存在 |
| `models/decoder.py` | 60 | 已存在 |
| `models/rssm.py` | 350 | 已存在 |
| `models/residual.py` | 250 | 已存在 |
| `models/world_model.py` | 400 | 已存在 |
| `models/actor.py` | 120 | 已存在 |
| `models/critic.py` | 100 | 已存在 |
| `training/replay_buffer.py` | 200 | 已存在 |
| `training/wm_loss.py` | 100 | 已存在 |
| `training/ac_loss.py` | 180 | 已存在 |
| `training/trainer.py` | 400 | 已存在 |
| `evaluation/eval_protocol.py` | 200 | 已存在 |
| `scripts/train_stage1.py` | 60 | 已存在 |
| `scripts/collect_pseudo_real_data.py` | 100 | 已存在 |
| `scripts/train_stage2.py` | 60 | 已存在 |
| `scripts/eval_compare.py` | 80 | 已存在 |
| **总计** | **~3500 行** | 4 个待写,其他已写 |

---

## 8. 验收标准

### 8.1 环境集成(WMP 接口)

- [ ] `WMPEnvBase` 能成功创建 Isaac Gym 仿真
- [ ] `InaccurateSimEnv` 配置域随机化(电机 [0.65, 0.75] 等)
- [ ] `PseudoRealEnv` 关闭所有域随机化
- [ ] 两个环境的地形配置相同
- [ ] `reset()` 返回 `[num_envs, 45]` tensor
- [ ] `step()` 返回正确的 obs/reward/done/info

### 8.2 Stage 1 训练

- [ ] WM 训练收敛(KL loss < 5 nat)
- [ ] AC 训练收敛(actor 返回合理 action)
- [ ] 在 InaccurateSim 上 episode_return > 200
- [ ] Residual 输出 ≈ 0(零初始化 + stop grad)

### 8.3 Stage 2 训练

- [ ] 重新创建 Residual(1 层,零初始化)
- [ ] 主网络冻结(只有 Residual 接收梯度)
- [ ] Residual loss 收敛(KL < 1 nat)
- [ ] Residual 输出 ≠ 0(学到了东西)

### 8.4 三策略对比

- [ ] 策略 A 在 PseudoReal 上能跑(可能表现差)
- [ ] 策略 B 在 PseudoReal 上比 A 好
- [ ] Gap 闭合率 > 30%(理想 > 70%)
- [ ] Actor 完全没动(Stage 1 训的)

### 8.5 关键不变量(必须保持)

| 维度 | InaccurateSim | PseudoReal |
|------|---------------|------------|
| 地形 | trimesh 同配置 | trimesh 同配置 |
| 关节数 | 12 | 12 |
| 动作维度 | 12 | 12 |
| 观测维度 | 45(我们只取前 45) | 45 |
| Episode 长度 | 1000 步 | 1000 步 |
| dt | 0.02s | 0.02s |
| action_scale | 0.25 | 0.25 |

**任何环境差异都必须明确标注并说明原因**。

---

## 9. 实施顺序

按依赖关系排序:

1. **环境集成**(最优先,WMP 接口对接)
   - `envs/wmp_env_base.py`(基础)
   - `envs/inaccurate_sim_env.py`(配置域随机化)
   - `envs/pseudo_real_env.py`(关闭域随机化)

2. **环境验证脚本**
   - 验证 4096 env 能跑
   - 验证 reset/step 接口正确
   - 验证 obs 提取正确(45 维)

3. **完整训练流程测试**
   - Stage 1 短跑(1000 步)
   - Stage 2 短跑(1000 步)
   - 评估(10 episodes)

4. **正式训练**
   - Stage 1: 2M 步
   - 采集伪 real 数据:200 episodes
   - Stage 2: 1M 步
   - 三策略对比:50 episodes × 3

---

## 10. 已知风险和缓解

| 风险 | 缓解 |
|------|------|
| Isaac Gym 安装复杂 | 用 WMP 的 Docker 镜像或 conda 环境 |
| WMP 依赖 rsl_rl / amp | 不使用,只调用 env 部分 |
| 4096 env 内存占用大 | 用 A100 40GB;若不够降到 2048 |
| Stage 2 策略在 PseudoReal 上直接摔 | 先用 PPO 训练一个 baseline 在 PseudoReal 上 |
| Residual 学不到东西 | 加大 Stage 2 lr / 加大 KL loss 权重 |
| 地形不一致导致 transfer 失败 | 严格使用相同 terrain 配置 |

---

## 11. 测试检查清单

完成实施后,逐项检查:

- [ ] **依赖**: `pip list | grep -E "torch|isaacgym|numpy|pyyaml"` 全有
- [ ] **导入**: `python -c "import sys; sys.path.append('D:/songay/sim2real/WMP'); from legged_gym.envs.base.legged_robot import LeggedRobot"` 不报错
- [ ] **环境创建**: `InaccurateSimEnv(num_envs=64)` 不报错
- [ ] **reset**: `obs = env.reset(); print(obs.shape)` 输出 `(64, 45)`
- [ ] **step**: `obs, r, d, _ = env.step(torch.zeros(64, 12))` 返回正确形状
- [ ] **Stage 1 短跑**: 1000 步不报错
- [ ] **Stage 2 短跑**: 1000 步不报错
- [ ] **评估**: 10 episodes 不报错

---

**文档结束**

**下一步**: 按"实施顺序"开始写代码,先写 `envs/wmp_env_base.py`(最关键)。