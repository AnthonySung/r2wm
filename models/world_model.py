"""
WorldModel: 整合 Encoder + Decoder + RSSM + 两个 Residual
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rssm import RSSM
from .encoder import Encoder
from .decoder import Decoder
from .residual import (
    PhysicalResidual,
    DynamicsResidual,
    make_physical_residual_for_stage1,
    make_physical_residual_for_stage2,
    make_dynamics_residual_for_stage1,
    make_dynamics_residual_for_stage2,
)


class WorldModel(nn.Module):
    """
    整合的 World Model。

    关键设计(对齐 ReDRAW):
    1. Residual 零初始化
    2. observe() 时 Residual 参与前向,但 stop_gradients(stop_residual_grad=True)
    3. imagine() 时 Residual 默认不参与(with_residual=False)
    4. Stage 2 时重新创建 Residual(更小结构)
    """

    def __init__(
        self,
        obs_dim: int = 48,
        action_dim: int = 12,
        deter_dim: int = 512,
        stoch_dim: int = 32,
        discrete: int = 32,
        hidden: int = 512,
        embed_dim: int = 256,
        use_residual: bool = True,
        residual_config: dict = None,
    ):
        super().__init__()
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._deter_dim = deter_dim
        self._stoch_dim = stoch_dim
        self._discrete = discrete
        self._use_residual = use_residual

        # 计算 stoch flat dim
        self._stoch_flat_dim = stoch_dim * discrete if discrete > 0 else stoch_dim

        # Residual config
        if residual_config is None:
            residual_config = {
                'stage1_hidden': 64,
                'stage1_n_layers': 3,
                'stage2_hidden': 64,
                'stage2_n_layers': 1,
                'dynamics_hidden': 128,
                'history_len': 4,
                'init': 'zero',
            }
        self._residual_config = residual_config

        # 核心组件
        self.encoder = Encoder(obs_dim, embed_dim)
        self.decoder = Decoder(self._stoch_flat_dim + deter_dim, obs_dim)
        self.rssm = RSSM(
            deter_dim=deter_dim,
            stoch_dim=stoch_dim,
            discrete=discrete,
            hidden=hidden,
            action_dim=action_dim,
            embed_dim=embed_dim,
        )

        # Residual 模块
        if use_residual:
            self.physical_residual = make_physical_residual_for_stage1(
                stoch_dim=self._stoch_flat_dim,
                action_dim=action_dim,
                config=residual_config,
            )
            self.dynamics_residual = make_dynamics_residual_for_stage1(
                deter_dim=deter_dim,
                action_dim=action_dim,
                config=residual_config,
            )
        else:
            self.physical_residual = None
            self.dynamics_residual = None

    @property
    def feat_dim(self) -> int:
        return self._stoch_flat_dim + self._deter_dim

    # ============================================================
    # Residual 切换工具(Stage 2 用)
    # ============================================================

    def recreate_residual_for_stage2(self):
        """Stage 2: 重新创建 Residual(对齐 ReDRAW ensemble_residual_extra_small_1_member)"""
        if not self._use_residual:
            return
        self.physical_residual = make_physical_residual_for_stage2(
            stoch_dim=self._stoch_flat_dim,
            action_dim=self._action_dim,
            config=self._residual_config,
        )
        self.dynamics_residual = make_dynamics_residual_for_stage2(
            deter_dim=self._deter_dim,
            action_dim=self._action_dim,
            config=self._residual_config,
        )

    def freeze_main_network(self):
        """冻结主网络(只保留 Residual 可训)"""
        for name, p in self.named_parameters():
            if 'physical_residual' in name or 'dynamics_residual' in name:
                p.requires_grad = True
            else:
                p.requires_grad = False

    def unfreeze_all(self):
        """解冻所有(Stage 1 训练用)"""
        for p in self.parameters():
            p.requires_grad = True

    # ============================================================
    # 前向传播
    # ============================================================

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """编码 obs"""
        return self.encoder(obs)

    def decode(self, state: dict) -> torch.Tensor:
        """从 state 重建 obs"""
        feat = self.rssm.get_feat(state)
        return self.decoder(feat)

    def observe(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        is_first_seq: torch.Tensor,
        with_residual: bool = True,
        stop_residual_grad: bool = True,
    ) -> tuple:
        """
        观察真实数据序列。

        Args:
            obs_seq: [B, T, obs_dim]
            action_seq: [B, T, action_dim]
            is_first_seq: [B, T] bool
            with_residual: Residual 是否参与
            stop_residual_grad: Residual 是否 stop gradient
        Returns:
            post: dict with 'deter', 'stoch', 'mean', 'std' (last step)
            posts: dict with full sequence
        """
        B, T = obs_seq.shape[:2]
        device = obs_seq.device

        embed_seq = self.encode(obs_seq)

        # 初始状态
        state = self.rssm.initial_state(B, device=device)
        deter_history = torch.zeros(
            B, self._residual_config['history_len'], self._deter_dim, device=device
        )

        posts = {'stoch': [], 'deter': [], 'mean': [], 'std': []}
        priors = {'stoch': [], 'deter': [], 'mean': [], 'std': []}

        for t in range(T):
            embed_t = embed_seq[:, t]
            action_t = action_seq[:, t] if action_seq is not None else torch.zeros(B, self._action_dim, device=device)
            is_first_t = is_first_seq[:, t] if is_first_seq is not None else torch.zeros(B, dtype=torch.bool, device=device)

            post, prior = self.rssm.obs_step(
                state, action_t, embed_t, is_first_t, sample=True
            )

            # 应用 Residual(关键!)
            # 传入 state(上一步)和 action_t,让 Residual 用 prev_stoch 而不是 post
            if with_residual and self._use_residual:
                post, prior = self._apply_residual(
                    post, prior, state, action_t, deter_history, stop_residual_grad
                )

            # 更新 deter history
            deter_history = self.rssm.get_deter_history(
                deter_history, None, post['deter']
            )

            for key in posts:
                if key in post:
                    posts[key].append(post[key])
            for key in priors:
                if key in prior:
                    priors[key].append(prior[key])

            state = post

        # Stack 成 [B, T, ...]
        posts_stacked = {k: torch.stack(v, dim=1) for k, v in posts.items()}
        priors_stacked = {k: torch.stack(v, dim=1) for k, v in priors.items()}

        return posts_stacked, priors_stacked

    def imagine(
        self,
        actor,
        init_state: dict,
        horizon: int,
        with_residual: bool = False,
    ) -> tuple:
        """
        在 WM 中想象轨迹。

        ReDRAW 设计: 想象训练 Actor 时 Residual 不参与(with_residual=False)
        """
        states = []
        actions = []
        log_probs = []
        entropies = []

        state = {k: v for k, v in init_state.items()}  # copy

        for t in range(horizon):
            feat = self.rssm.get_feat(state)
            action, mean, std, entropy = actor(feat, sample=True)
            log_prob = actor.get_log_prob(feat, action)

            # 保留上一步状态(用于 Residual 的 prev_stoch)
            prev_state = state

            # RSSM step
            new_state = self.rssm.img_step(state, action, sample=True)

            # Residual(默认不参与)
            if with_residual and self._use_residual:
                # 想象时通常不需要 DynamicsResidual(没有 history 维护)
                # 只用 PhysicalResidual,加在 mean 上
                prev_stoch = prev_state['stoch']
                if self.rssm._is_discrete:
                    prev_stoch_flat = prev_stoch.reshape(*prev_stoch.shape[:-2], self._stoch_flat_dim)
                else:
                    prev_stoch_flat = prev_stoch
                delta_phys = self.physical_residual(prev_stoch_flat, action)
                new_state['mean'] = new_state['mean'] + delta_phys

            states.append(new_state)
            actions.append(action)
            log_probs.append(log_prob)
            entropies.append(entropy)
            state = new_state

        # Stack
        states_stacked = {
            k: torch.stack([s[k] for s in states], dim=1)
            for k in states[0].keys()
            if k != 'dist_type'
        }
        actions_stacked = torch.stack(actions, dim=1)
        log_probs_stacked = torch.stack(log_probs, dim=1)
        entropies_stacked = torch.stack(entropies, dim=1)

        return states_stacked, actions_stacked, log_probs_stacked, entropies_stacked

    def _apply_residual(
        self,
        post: dict,
        prior: dict,
        prev_state: dict,
        action: torch.Tensor,
        deter_history: torch.Tensor,
        stop_grad: bool,
    ) -> tuple:
        """
        应用 Residual(对齐 ReDRAW 公式 18)

        - PhysicalResidual: 加在 latent mean 上,输入是 prev_state.stoch
        - DynamicsResidual: 加在 deter 上,输入是 deter_history

        Args:
            post: 当前步的 posterior
            prior: 当前步的 prior
            prev_state: **上一步的状态**(用于 prev_stoch 和 prev_deter)
            action: 当前动作
            deter_history: 过去 K 步的 deter 序列
            stop_grad: 是否 stop gradient(Stage 1=True, Stage 2=False)
        """
        # PhysicalResidual: 用 **prev_state.stoch**(不是 post.stoch)
        prev_stoch = prev_state['stoch']
        if self.rssm._is_discrete:
            prev_stoch_flat = prev_stoch.reshape(*prev_stoch.shape[:-2], self._stoch_flat_dim)
        else:
            prev_stoch_flat = prev_stoch

        if stop_grad:
            prev_stoch_flat = prev_stoch_flat.detach()

        delta_phys = self.physical_residual(prev_stoch_flat, action)
        if stop_grad:
            delta_phys = delta_phys.detach()

        # 加到 mean 上(对应 ReDRAW 的 logits + correction)
        # 注意: 这里 'mean' 是 Gaussian mean,等价于 ReDRAW 的 logits
        post['mean'] = post['mean'] + delta_phys
        prior['mean'] = prior['mean'] + delta_phys

        # DynamicsResidual
        delta_dyn = self.dynamics_residual(deter_history, action)
        if stop_grad:
            delta_dyn = delta_dyn.detach()

        post['deter'] = post['deter'] + delta_dyn
        prior['deter'] = prior['deter'] + delta_dyn

        return post, prior