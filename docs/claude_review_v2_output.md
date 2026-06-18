# Claude Code 复审输出 (修复后)

**日期**: 2026-06-18

---

# P0/P1 Bug 复审报告

## 逐项确认

| # | Bug | 状态 | 证据 |
|---|-----|------|------|
| 1 | obs_dim=45→48 | ✅ | `wmp_env_base.py:32` 注释/:`get_proprio_obs_from_full` 切 `[..., :48]`;`world_model.py:38` `obs_dim=48`;`replay_buffer.py:42` `obs_dim=48`;`trainer.py:36` 传 `env.obs_dim`(动态) |
| 2 | `_apply_residual` 用 post.stoch | ✅ | `world_model.py:252` `prev_stoch = prev_state['stoch']`,然后 flatten 给 PhysicalResidual |
| 3 | 不存 is_first / 非统一入口 | ⚠️ | `trainer.py:120` `is_first_np=is_first_env.cpu().numpy()` ✓;`add(..., is_first=...)` ✓;`is_first_env = done.clone()` ✓ — **但**数据收集仍手动调 `_apply_residual`(非统一),且调用有 bug |
| 4 | AC 频率 = WM 频率 | ✅ | `trainer.py:163` WM `step % 100 == 0`;`trainer.py:178` AC `step % 20 == 0` |
| 5 | `imagine()` prev_stoch 错 | ✅ | `world_model.py:174` `prev_state = state`(在 img_step 前保存);`:184` `prev_stoch = prev_state['stoch']` |
| 6 | WMP_ROOT 硬编码 | ✅ | `wmp_env_base.py:22` `os.environ.get('WMP_ROOT', 'D:/songay/sim2real/WMP')` |
| 7 | step() 7 元组无 assert | ✅ | `wmp_env_base.py:184-188` `assert len(result) == 7, (...)` 带详细错误信息 |
| 8 | ReplayBuffer 无 is_first | ✅ | `replay_buffer.py:53` 存 `_is_first`;`add/save/load/sample` 全链路处理;`load()` 兼容旧版 npz(`_mark_episode_starts`) |

## ⚠️ #3 修复仍有问题

`trainer.py:97-108` 手动调用 `_apply_residual` 引入 2 个新 bug:

```python
new_state, _ = self.world_model._apply_residual(
    new_state, _, state,
    torch.zeros(...action...),
    torch.zeros(self.env.num_envs, 4, self.world_model._deter_dim, ...),  # 固定 0!
    stop_residual_grad=True,  # ← 关键字错!函数签名是 stop_grad
)
```

1. **P0**: `stop_residual_grad=True` 是 kwargs,但 `_apply_residual(self, post, prior, prev_state, action, deter_history, stop_grad)` 第 6 个参数叫 `stop_grad` → 运行时 `TypeError`
2. **P1**: `deter_history` 传固定 zeros tensor,DynamicResidual 永远看不到真实历史
3. **P1**: action 永远传 zeros(应该是上一帧的 action)

**修复建议**: 在 `world_model.py` 加单步入口 `observe_step(obs, action, prev_state, is_first)`,trainer 调它,而不是手撸 RSSM + residual。

## 其他发现的小问题(不在原 P0/P1 列表)

- `rssm.py:177` `OneHotDist(logits=None, ...)` 路径在 unimix_ratio>0 时会 `None / shape` 报错(本项目只走 logits 路径,实际不会触发,但 dead code)
- `world_model.py:174-187` imagine 里只有 PhysicalResidual 没 DynamicsResidual 是正确的(想象不维护 history),注释里也说明了 ✓

## 新总体评分:**4/5**

- 8 个原 P0/P1 全部已**功能性修复**
- #3 的实现走偏:手动调用方式 + kwarg 拼写错误,会在第一次训练时崩
- 整体设计(WM/AC 分频、stoch 隔离、env 维度、buffer 完整性)都对齐了 ReDRAW/WMP
- 修掉 trainer 的 3 个新 bug 后可达 **5/5**
