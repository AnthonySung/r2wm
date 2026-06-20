"""
Trainer: Stage 1 + Stage 2 训练主循环
"""

import os
import time
import json
import torch
import numpy as np
from typing import Optional

from .replay_buffer import ReplayBuffer
from .wm_loss import compute_wm_loss, compute_residual_loss
from .ac_loss import compute_ac_loss
from .amp_trainer import compute_amp_loss
from models.world_model import WorldModel
from models.actor import A1Actor
from models.critic import A1Critic, SlowCritic


class Trainer:
    """
    Trainer 整合 Stage 1 和 Stage 2 训练。
    """

    def __init__(
        self,
        env,                  # InaccurateSimEnv
        config: dict,
        device: str = 'cuda:0',
    ):
        self.env = env
        self.config = config
        self.device = device

        # 模型
        self.world_model = WorldModel(
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            deter_dim=config['rssm']['deter'],
            stoch_dim=config['rssm']['stoch'],
            discrete=config['rssm']['discrete'],
            hidden=config['rssm']['hidden'],
            use_residual=True,
            residual_config=config['residual'],
        ).to(device)

        feat_dim = self.world_model.feat_dim
        self.actor = A1Actor(
            feat_dim=feat_dim,
            action_dim=env.action_dim,
            hidden=config['actor']['units'],
            n_layers=config['actor']['layers'],
            entropy_scale=config['actor']['entropy_scale'],
        ).to(device)

        self.critic = A1Critic(
            feat_dim=feat_dim,
            hidden=config['critic']['units'],
            n_layers=config['critic']['layers'],
        ).to(device)
        self.target_critic = SlowCritic(
            self.critic,
            update_fraction=config['critic']['slow_target_fraction'],
        )

        # Optimizer
        self.wm_opt = torch.optim.Adam(
            self.world_model.parameters(),
            lr=config['training']['model_lr'],
            eps=config['training']['adam_eps'],
        )
        self.ac_opt = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=config['actor']['lr'],
            eps=config['training']['adam_eps'],
        )

        # AMP Discriminator + Optimizer (B2)
        self.amp_enabled = getattr(self.env, '_use_amp', False)
        if self.amp_enabled:
            amp_cfg = self.env._amp_cfg
            train_cfg = amp_cfg.get('training', {})
            self._amp_train_cfg = train_cfg

            # Discriminator optimizer (WMP 风格: trunk + head 分离 weight decay)
            disc = self.env._amp_discriminator
            self.amp_disc_opt = torch.optim.Adam([
                {'params': disc.trunk.parameters(), 'weight_decay': train_cfg.get('disc_weight_decay', 1e-4)},
                {'params': disc.amp_linear.parameters(), 'weight_decay': train_cfg.get('disc_head_weight_decay', 1e-2)},
            ], lr=train_cfg.get('disc_lr', 1e-3))
            print(f"[Trainer] AMP Discriminator optimizer created (lr={train_cfg.get('disc_lr', 1e-3)})")

            # AMP obs buffer (训 discriminator 用)
            self._amp_obs_buffer = []  # list of [N, amp_obs_dim]
            self._amp_next_obs_buffer = []
            self._amp_buffer_max = 8192
        else:
            self.amp_disc_opt = None
            self._amp_obs_buffer = None
            self._amp_next_obs_buffer = None

        # Replay buffer
        self.replay = ReplayBuffer(
            capacity=1_000_000,
            obs_dim=env.obs_dim,
            action_dim=env.action_dim,
            device=device,
        )

        # 状态
        self.step = 0

    # ============================================================
    # Stage 1: Sim 预训练
    # ============================================================

    def train_stage1(self, total_steps: int, eval_fn=None, log_every: int = 1000):
        """
        Stage 1: 在 InaccurateSimEnv 上训练 WM + AC

        关键:
        - Residual 零初始化 + stop_gradients
        - 想象训练 Actor 时 Residual 不参与
        """
        print(f"[Stage 1] Starting training for {total_steps} steps")
        obs = self.env.reset()
        episode_returns = []
        episode_lengths = []
        ep_return = torch.zeros(self.env.num_envs, device=self.device)
        ep_len = torch.zeros(self.env.num_envs, device=self.device)

        # 维护 RSSM 状态(用于数据收集时的连续性)
        state = self.world_model.rssm.initial_state(self.env.num_envs, device=self.device)
        is_first_env = torch.ones(self.env.num_envs, dtype=torch.bool, device=self.device)

        # 维护 last_actions 和 deter_history(用于 Residual)
        last_actions = torch.zeros(self.env.num_envs, self.env.action_dim, device=self.device)
        # deter_history: [num_envs, history_len=4, deter_dim]
        history_len = self.world_model._residual_config.get('history_len', 4)
        deter_history = torch.zeros(
            self.env.num_envs, history_len, self.world_model._deter_dim, device=self.device
        )

        start_time = time.time()

        for step in range(total_steps):
            self.step = step

            # 1. 数据收集
            with torch.no_grad():
                # 用统一的 observe 入口(应用 Residual)
                # 这样 actor 看到的状态和训练时一致
                embed = self.world_model.encode(obs)

                # 调用 obs_step(应用 Residual,stop grad)
                new_state, _ = self.world_model.rssm.obs_step(
                    state,
                    last_actions,  # 用上一步的 action(不是 zeros)
                    embed,
                    is_first_env,
                    sample=True,
                )
                # 应用 Residual(用 prev_state 和 deter_history)
                if self.world_model._use_residual:
                    new_state, _ = self.world_model._apply_residual(
                        new_state, _, state,  # prev_state = state
                        last_actions,         # 用上一步的 action
                        deter_history,        # 用真实的 history(不是 zeros)
                        True,                 # stop_grad=Stage 1 模式
                    )

                feat = self.world_model.rssm.get_feat(new_state)
                action = self.actor.sample(feat)

                next_obs, reward, done, info = self.env.step(action)
                ep_return += reward
                ep_len += 1

                # AMP: 收集 policy amp_obs 给 discriminator 训
                if self.amp_enabled:
                    next_amp_obs = self.env._wmp_env.get_amp_observations()
                    current_amp_obs = self.env._current_amp_obs
                    self._amp_obs_buffer.append(current_amp_obs.detach().clone())
                    self._amp_next_obs_buffer.append(next_amp_obs.detach().clone())
                    # 限制 buffer 大小
                    if len(self._amp_obs_buffer) * self.env.num_envs > self._amp_buffer_max:
                        keep = self._amp_buffer_max // self.env.num_envs
                        self._amp_obs_buffer = self._amp_obs_buffer[-keep:]
                        self._amp_next_obs_buffer = self._amp_next_obs_buffer[-keep:]

                # 存储
                obs_np = obs.cpu().numpy()
                action_np = action.cpu().numpy()
                next_obs_np = next_obs.cpu().numpy()
                reward_np = reward.cpu().numpy()
                done_np = done.cpu().numpy()
                is_first_np = is_first_env.cpu().numpy()

                for i in range(self.env.num_envs):
                    self.replay.add(
                        obs_np[i], action_np[i], reward_np[i],
                        next_obs_np[i], bool(done_np[i]),
                        is_first=bool(is_first_np[i])
                    )

                # Episode 统计
                if done.any():
                    done_idx = done.cpu().numpy()
                    for i in range(self.env.num_envs):
                        if done_idx[i]:
                            episode_returns.append(ep_return[i].item())
                            episode_lengths.append(ep_len[i].item())
                            ep_return[i] = 0.0
                            ep_len[i] = 0

                # 更新 is_first, state, last_actions, deter_history
                is_first_env = done.clone()
                # Episode boundary 重置 last_actions 和 deter_history
                # (修复 Claude P0-5: reset env 的 last_actions/deter_history 不能传过去)
                if done.any():
                    done_idx = done.cpu().numpy()
                    for i in range(self.env.num_envs):
                        if done_idx[i]:
                            last_actions[i] = 0.0
                            deter_history[i] = 0.0
                # 更新 deter_history: shift left, append new_state.deter
                deter_history = torch.cat([
                    deter_history[:, 1:, :],
                    new_state['deter'].unsqueeze(1)
                ], dim=1)
                state = new_state
                last_actions = action.detach()  # 当前 action 作为下一步的 prev_action
                obs = next_obs

            # 2. WM 训练(每 100 步,Dreamer 标准)
            if step > 100 and step % 100 == 0:
                batch = self.replay.sample(
                    batch_size=self.config['training']['batch_size'],
                    seq_length=self.config['training']['batch_length'],
                )
                wm_loss, wm_metrics = compute_wm_loss(
                    self.world_model, batch,
                    free_bits=1.0,
                )
                self._last_wm_metrics = wm_metrics
                self.wm_opt.zero_grad()
                wm_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.world_model.parameters(),
                    self.config['training']['grad_clip'],
                )
                self.wm_opt.step()

            # 2.5 AMP Discriminator 训练 (B2, GAN-style + grad penalty)
            if (self.amp_enabled
                    and step > 100
                    and step % self._amp_train_cfg.get('train_every', 1) == 0
                    and len(self._amp_obs_buffer) > 0):
                try:
                    # 拼 buffer 成一个 batch
                    policy_obs_cat = torch.cat(self._amp_obs_buffer, dim=0)
                    policy_next_obs_cat = torch.cat(self._amp_next_obs_buffer, dim=0)
                    # 限制 batch 大小
                    max_batch = self._amp_train_cfg.get('policy_batch_size', 4096)
                    if policy_obs_cat.shape[0] > max_batch:
                        idx = torch.randperm(policy_obs_cat.shape[0])[:max_batch]
                        policy_obs_cat = policy_obs_cat[idx]
                        policy_next_obs_cat = policy_next_obs_cat[idx]

                    amp_loss, amp_metrics = compute_amp_loss(
                        discriminator=self.env._amp_discriminator,
                        amp_loader=self.env._amp_loader,
                        policy_amp_obs=policy_obs_cat,
                        policy_next_amp_obs=policy_next_obs_cat,
                        amp_normalizer=self.env._amp_normalizer,
                        expert_target=self._amp_train_cfg.get('expert_target', 1.0),
                        policy_target=self._amp_train_cfg.get('policy_target', -1.0),
                        grad_pen_lambda=self._amp_train_cfg.get('grad_pen_lambda', 10.0),
                        expert_batch_size=self._amp_train_cfg.get('expert_batch_size', 4096),
                        device=self.device,
                    )
                    self._last_amp_metrics = amp_metrics
                    self.amp_disc_opt.zero_grad()
                    amp_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.env._amp_discriminator.parameters(),
                        self.config['training']['grad_clip'],
                    )
                    self.amp_disc_opt.step()
                except Exception as e:
                    if step == 101:
                        print(f"[Trainer] AMP train failed at step {step}: {e}")
                        import traceback
                        traceback.print_exc()

            # 3. AC 训练(每 20 步,Dreamer 标准: WM 1 次, AC ~5 次)
            if step > 100 and step % 20 == 0:
                batch = self.replay.sample(
                    batch_size=self.config['training']['batch_size'],
                    seq_length=1,  # 只用第一步
                )
                ac_loss, ac_metrics = compute_ac_loss(
                    self.actor, self.critic, self.target_critic, self.world_model,
                    init_obs=batch['obs'][:, 0],
                    init_is_first=batch['is_first'][:, 0],
                    init_action=batch['action'][:, 0],
                    horizon=self.config['training']['imag_horizon'],
                )
                self._last_ac_metrics = ac_metrics
                self.ac_opt.zero_grad()
                ac_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.config['training']['grad_clip'],
                )
                self.ac_opt.step()

                # EMA 更新 target critic
                self.target_critic.update(self.critic)

            # 4. 日志
            if step % log_every == 0 and step > 0:
                elapsed = time.time() - start_time
                avg_return = np.mean(episode_returns[-100:]) if episode_returns else 0.0
                avg_ep_len = np.mean(episode_lengths[-100:]) if episode_lengths else 0.0
                # 详情指标(用于诊断)
                last_wm = getattr(self, '_last_wm_metrics', {})
                last_ac = getattr(self, '_last_ac_metrics', {})
                last_amp = getattr(self, '_last_amp_metrics', {})
                print(
                    f"[Stage 1 Step {step}/{total_steps}] "
                    f"avg_return={avg_return:.2f} "
                    f"avg_ep_len={avg_ep_len:.1f} "
                    f"elapsed={elapsed:.1f}s "
                    f"buffer_size={len(self.replay)} "
                    f"wm[kld]={last_wm.get('kl_loss', 0):.3f} "
                    f"wm[recon]={last_wm.get('recon_loss', 0):.3f} "
                    f"wm[rew]={last_wm.get('reward_loss', 0):.3f} "
                    f"ac[actor]={last_ac.get('actor_loss', 0):.3f} "
                    f"ac[critic]={last_ac.get('critic_loss', 0):.3f} "
                    f"ac[lambda_ret]={last_ac.get('lambda_return_mean', 0):.3f} "
                    f"amp[loss]={last_amp.get('amp_loss', 0):.3f} "
                    f"amp[exp_d]={last_amp.get('expert_d_mean', 0):.2f} "
                    f"amp[pol_d]={last_amp.get('policy_d_mean', 0):.2f} "
                    f"amp[gp]={last_amp.get('grad_pen_loss', 0):.3f}"
                )

            # 5. 评估
            if step > 0 and step % self.config['stage1']['eval_every'] == 0:
                if eval_fn:
                    metrics = eval_fn(self.actor, self.world_model)
                    print(f"[Stage 1 Step {step}] Eval: {metrics}")

            # 6. 保存
            if step > 0 and step % self.config['stage1']['ckpt_every'] == 0:
                self.save_checkpoint(f'stage1_step{step}.ckpt')

        # 最终保存
        self.save_checkpoint('stage1_final.ckpt')
        print(f"[Stage 1] Done. Saved stage1_final.ckpt")

    # ============================================================
    # Stage 2: 伪 real 数据微调 Residual
    # ============================================================

    def train_stage2_residual(
        self,
        real_replay: ReplayBuffer,
        total_steps: int = 1_000_000,
        eval_fn=None,
        log_every: int = 1000,
    ):
        """
        Stage 2: 在伪 real 数据上微调 Residual

        关键:
        - 重新创建 Residual(1 层,Stage 1 是 3 层)
        - 冻结主 RSSM + Actor
        - 只训练 Residual
        - 100x 学习率
        """
        print(f"[Stage 2] Starting residual fine-tuning for {total_steps} steps")

        # 1. 重新创建 Residual(对齐 ReDRAW)
        # 注意:这会替换 self.world_model.physical_residual 和 dynamics_residual 的 Parameter 对象
        # 所以 self.wm_opt 仍然引用旧的 Residual 参数(已被 GC)
        # 这是预期行为,因为 Stage 2 完全冻结主网络,不再用 wm_opt
        # Stage 2 训练时使用新创建的 res_opt
        self.world_model.recreate_residual_for_stage2()
        self.world_model = self.world_model.to(self.device)

        # 2. 冻结主网络
        self.world_model.freeze_main_network()
        for p in self.actor.parameters():
            p.requires_grad = False
        for p in self.critic.parameters():
            p.requires_grad = False

        # 3. Optimizer(只针对 Residual,100x lr)
        residual_params = [
            p for p in self.world_model.parameters() if p.requires_grad
        ]
        print(f"[Stage 2] Trainable params: {sum(p.numel() for p in residual_params)}")
        self.res_opt = torch.optim.Adam(
            residual_params,
            lr=self.config['residual']['stage2_lr'],  # 1e-2
            eps=self.config['training']['adam_eps'],
        )

        start_time = time.time()

        for step in range(total_steps):
            # 采样伪 real 批次
            batch = real_replay.sample(
                batch_size=self.config['training']['batch_size'],
                seq_length=self.config['training']['batch_length'],
            )

            # 计算 Residual loss
            res_loss, metrics = compute_residual_loss(self.world_model, batch)

            self.res_opt.zero_grad()
            res_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                residual_params,
                self.config['training']['grad_clip'],
            )
            self.res_opt.step()

            if step % log_every == 0:
                elapsed = time.time() - start_time
                print(
                    f"[Stage 2 Step {step}/{total_steps}] "
                    f"res_kl={metrics['residual_kl_loss']:.4f} "
                    f"elapsed={elapsed:.1f}s"
                )

            if step > 0 and step % self.config['stage2']['ckpt_every'] == 0:
                self.save_residual(f'stage2_step{step}')

        # 保存
        self.save_residual('stage2_final')
        print(f"[Stage 2] Done. Saved stage2_final")

    # ============================================================
    # 保存 / 加载
    # ============================================================

    def save_checkpoint(self, filename: str):
        """保存完整 checkpoint"""
        os.makedirs('checkpoints', exist_ok=True)
        path = os.path.join('checkpoints', filename)
        torch.save({
            'world_model': self.world_model.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'step': self.step,
        }, path)
        print(f"[Checkpoint] Saved to {path}")

    def load_checkpoint(self, filename: str):
        """加载完整 checkpoint"""
        path = os.path.join('checkpoints', filename)
        ckpt = torch.load(path, map_location=self.device)
        self.world_model.load_state_dict(ckpt['world_model'])
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.target_critic.load_state_dict(ckpt['target_critic'])
        self.step = ckpt.get('step', 0)
        print(f"[Checkpoint] Loaded from {path}")

    def save_residual(self, filename: str):
        """只保存 Residual"""
        os.makedirs('checkpoints', exist_ok=True)
        path = os.path.join('checkpoints', f'{filename}.ckpt')
        torch.save({
            'physical_residual': self.world_model.physical_residual.state_dict(),
            'dynamics_residual': self.world_model.dynamics_residual.state_dict(),
        }, path)
        print(f"[Residual] Saved to {path}")

    def load_residual(self, filename: str):
        """加载 Residual"""
        path = os.path.join('checkpoints', f'{filename}.ckpt')
        ckpt = torch.load(path, map_location=self.device)
        self.world_model.physical_residual.load_state_dict(ckpt['physical_residual'])
        self.world_model.dynamics_residual.load_state_dict(ckpt['dynamics_residual'])
        print(f"[Residual] Loaded from {path}")