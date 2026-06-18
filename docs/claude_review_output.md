# Claude Code 评审输出

**日期**: 2026-06-18
**项目**: r2wmp
**评审方式**: stdin 管道传文件内容

---

# r2wmp 项目代码评审

---

## 1. `world_model.py` — 状态:**有问题(Bug 较严重)**

### Bug 1(严重):`_apply_residual` 用了错误的 `prev_stoch`
**行号:`world_model.py:222-225`**
```python
prev_stoch = post['stoch'].detach() if stop_grad else post['stoch']
```
`post['stoch']` 是**当前步**的后验随机变量,而 PhysicalResidual 应当以**上一步状态**(`state['stoch']`)为输入(在 ReDRAW 中是用 prev_latent 修正 prior mean)。当前实现等于用"被修正后的当前 stoch"再去算修正,语义错误,会导致学习目标歧义。

**修复**:把 `state`(prev dict)作为参数传入 `_apply_residual`,在 observe 循环里:
```python
post, prior = self._apply_residual(post, prior, state, action_t, deter_history, stop_residual_grad)
```

### Bug 2(中):`imagine()` 残差应用与 `observe()` 不一致
**行号:`world_model.py:184-194`**
- `observe()` 同时应用 Physical + Dynamics 残差
- `imagine()` 即便 `with_residual=True` 也**只**应用 PhysicalResidual 到 mean,完全忽略 DynamicsResidual
- 注释"with_residual=False 默认不参与"与代码中冗余分支矛盾

**修复**:要么在 `imagine()` 中也加 DynamicsResidual(传入 history),要么直接删除 `imagine()` 中这段死代码。

### Bug 3(低):`prior` 被原地修改后用于 KL,但其 `deter` 不再被使用
**行号:`world_model.py:243-245`**
`prior['deter'] = prior['deter'] + delta_dyn` 修改了 `prior.deter`,但 KL 只看 stoch 分布(mean/std/logit),所以是无害的"看似对齐"代码,可删除以减少误导。

### ✅ 正确部分
- Residual 加到 `mean` / `deter`(对齐 ReDRAW 公式 18)— 维度正确
- `stop_residual_grad` 用 `.detach()` 实现 — 正确
- `deter_history` 在残差后更新 — 时序逻辑正确

---

## 2. `rssm.py` — 状态:**基本正确**

### ✅ KL 方向:正确
- `dyn_loss = KL(sg(post) || prior)`:训练 prior 匹配 post(动力学预测)— 正确
- `rep_loss = KL(post || sg(prior))`:训练 post 不偏离 prior(防后验坍塌)— 正确
- 代码:`rssm.py:267-275`

### ✅ Free bits 逻辑:正确
- `torch.maximum(kl, free)` per-element 然后 sum — 对齐 Dreamer/标准做法
- `dyn_scale=0.5, rep_scale=0.1` — 比例合理

### 潜在问题(低):
- **`OneHotDist.__init__` 重复 softmax**:`rssm.py:18-22` 中先 `F.softmax(logits)` 再 unimix,等价于直接对 logits 做 log-softmax 数值运算,但当前写法 OK。
- **`_compute_stoch_from_deter` 在 `initial_state` 中被调用**:`rssm.py:139-143`,但 `deter = torch.tanh(self._init_deter)` 后又过 `img_out_layers + imgs_stat_layer`。流程正确,但初始 stoch 不可采样为评估时 mode 路径多走一层,微不影响。
- **`torchd.OneHotCategorical.kl_divergence` 输出形状**:返回 `[..., stoch_dim]`(已经聚合了 discrete 维),`sum(dim=-1)` 再聚合 stoch_dim,语义 = sum over all latent dims,**正确**。

---

## 3. `wmp_env_base.py` — 状态:**有严重 bug**

### Bug 1(严重):`obs_dim=45` 与注释中的 48 维 proprio 不一致
**行号:`wmp_env_base.py:171-175`**
```python
return full_obs[..., :45]
```
注释明确列出 proprio 共 48 维(`0:3 base_lin_vel` + `3:6 ang_vel` + `6:9 gravity` + `9:12 commands` + `12:24 dof_pos` + `24:36 dof_vel` + `36:48 actions`),但代码切了 45 维。**丢了 3 维**(很可能是 `commands[:3]` 或 `last_actions[:3]`)。
这会直接导致 `world_model` 的 encoder 输入维度错误,所有训练崩溃或学错。

**修复**:
```python
# 同步修改 WorldModel 初始化
self.world_model = WorldModel(obs_dim=48, ...)  # 同步 trainer.py
return full_obs[..., :48]
```
需要打开 `WMP/legged_gym/envs/base/legged_robot.py` 的 `compute_observations` 实际确认。

### Bug 2(中):`WMP_ROOT` 硬编码绝对路径
**行号:`wmp_env_base.py:21`**
- `WMP_ROOT = 'D:/songay/sim2real/WMP'` 在 Windows 上硬编码,无法移植
- 修复:用 `os.environ.get('WMP_ROOT', ...)` + `pathlib.Path` 跨平台

### Bug 3(中):7 元组解包**未做健壮性检查**
**行号:`wmp_env_base.py:144-148`**
```python
result = self._wmp_env.step(action)
policy_obs, _, reward, reset_buf, extras, _, _ = result
```
若 WMP 版本不同时返回 5 元组或 6 元组,会立刻 `ValueError`。建议加 `assert len(result) == 7` 早期失败。

### ✅ 正确部分
- `done = reset_buf.clone()` 包含 terminate + timeout — 正确
- `episode_returns` / `episode_lengths` 维护 — 正确
- 地形默认配置 — 合理

---

## 4. `trainer.py` — 状态:**多个中等问题**

### Bug 1(严重):`is_first` 没有存入 replay buffer
**行号:`trainer.py:181-188`**
```python
self.replay.add(
    obs_np[i], action_np[i], reward_np[i],
    next_obs_np[i], done_np[i]
)
```
`done` 传给 buffer,但 `is_first` 标志没有显式存入。RSSM 训练需要在 episode 起点重置 state(`obs_step` 中 `is_first_t` 分支),如果 buffer 用 `done` 推断,需明确实现;否则 WM 训练会**跨 episode 串状态**。

**修复**:要么扩展 `ReplayBuffer.add()` 接受 `is_first`,要么明确 buffer 用 `done` shift 一下 `is_first[t+1] = done[t]`。

### Bug 2(中):数据收集时 Residual 不参与,与训练时不一致
**行号:`trainer.py:158-170`**
```python
state, _ = self.world_model.rssm.obs_step(
    state, ..., embed, ..., sample=True,
)
feat = self.world_model.rssm.get_feat(state)
action = self.actor.sample(feat)
```
`obs_step` 直接调用,没有走 `world_model.observe(..., with_residual=True, stop_residual_grad=True)`。结果:actor 看到的是**无残差**的 state 特征,但 WM 训练时是**带残差**的状态,产生分布漂移。

**修复**:用 `state, _ = self.world_model.observe_step_single(obs_t, action_t, is_first_t)` 这种统一入口。

### Bug 3(中):AC 训练频率 = WM 训练频率(都是 100 步)
**行号:`trainer.py:189-201` 与 `203-225`**
Dreamer 标准做法是 **WM 训练 1 次,AC 训练 ~5 次**。当前两者同步,actor 严重欠训练。

**修复**:AC 用 `step % 20 == 0`,WM 用 `step % 100 == 0`。

### Bug 4(中):Stage 2 重建 Residual 后没重建 `wm_opt`,但 stage 2 不再用 wm_opt,所以无实际影响
**行号:`trainer.py:262-268`**
`self.world_model.recreate_residual_for_stage2()` 替换了 residual 模块,`self.wm_opt` 仍引用旧参数对象(被 GC),但 `self.res_opt` 是新构造的 — **无 bug,但易引发后续维护误解**。建议注释说明。

### Bug 5(低):`compute_ac_loss` 中 `init_obs=batch['obs'][:, 0]` 未先 encode
**行号:`trainer.py:214`**
若 `compute_ac_loss` 内部做了 `world_model.encode(init_obs)` 则 OK;若直接喂 `obs` 给 RSSM 会维度错误(45 维 vs embed_dim)。需要核验 `ac_loss.py` 实现。

### Bug 6(低):`self.world_model = self.world_model.to(self.device)` 是 no-op
**行号:`trainer.py:269`**
无害,可删除。

### ✅ 正确部分
- Stage 1/2 优化器与冻结分离 — 正确
- 残差重建 + 100x LR — 对齐 ReDRAW
- `EMA target_critic` 更新 — 正确
- checkpoint / residual 分离保存 — 合理

---

## 📊 总体评分:**2.5 / 5**

框架思路正确(ReDRAW 风格两阶段 + 双残差),但**实现细节有 3 个会直接导致训练失败的 bug**:
1. `obs_dim=45` 与 WMP 实际 48 维不一致
2. `prev_stoch` 用了 post 而非 prev
3. `is_first` 缺失导致 RSSM 跨 episode 串扰

修复后可达 4/5。

---

## 🔥 三个最关键风险

1. **维度错位风险(立即爆炸)**:`obs_dim=45` vs WMP 48 维 proprio,encoder 输入错,WM loss 直接 NaN,无法训练。**优先级 P0**,先跑一次 `print(self._wmp_env.obs_buf.shape)` 确认。

2. **状态串扰风险(隐性失效)**:`is_first` 缺失 + `prev_stoch` 错用,WM 训练会学到"episode 之间无 reset + 自回归使用当前 stoch",Reward 看似能下降但**策略完全不可用**。**优先级 P0**,需在跑 sim2real 之前验证。

3. **训练数据-策略数据分布漂移**:数据收集无残差 vs 训练有残差,actor 与 WM 状态空间错位,Stage 2 微调 Residual 时无法收敛。**优先级 P1**,需要先统一前向入口。

---

### 建议下一步
- 先 `git checkout` 一下,确认这些文件是已提交版本(用户提供的版本与本地可能有 diff)
- 跑 `python -c "import isaacgym; from legged_gym.envs.base.legged_robot import LeggedRobot; ..."` 确认 WMP `policy_obs` 实际维度
- 修完 3 个 P0 bug 后,先在 1 个 env 上 smoke test 1k step,看 WM loss 是否下降
