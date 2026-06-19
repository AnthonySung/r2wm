"""
RSSM: Recurrent State-Space Model(PyTorch 版)

对齐 ReDRAW 的设计:
- K-tuple categorical latent(stoch_dim * discrete)
- Continuous Gaussian latent(可选)
- GRU-based deter 更新
- 支持 history(用于 DynamicsResidual)

PyTorch 2.0 兼容(避免 OneHotDist.mode() 的 Tensor 问题)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as torchd


class OneHotDist(torchd.OneHotCategorical):
    """One-hot categorical distribution(对齐 ReDRAW)"""

    def __init__(self, logits=None, probs=None, unimix_ratio=0.01):
        if logits is not None:
            # Unimix trick(对齐 ReDRAW)
            probs = F.softmax(logits, dim=-1)
            if unimix_ratio > 0:
                uniform = torch.ones_like(probs) / probs.shape[-1]
                probs = (1 - unimix_ratio) * probs + unimix_ratio * uniform
            super().__init__(probs=probs)
        else:
            super().__init__(probs=probs)


class RSSM(nn.Module):
    """
    Recurrent State-Space Model

    状态组成:
    - deter: [deter_dim] GRU hidden state
    - stoch: [stoch_dim, discrete] discrete latent(或 [stoch_dim] continuous)
    """

    def __init__(
        self,
        deter_dim: int = 512,
        stoch_dim: int = 32,
        discrete: int = 32,
        hidden: int = 512,
        action_dim: int = 12,
        embed_dim: int = 256,
        unimix_ratio: float = 0.01,
        min_std: float = 0.1,
        initial: str = 'learned',
    ):
        super().__init__()
        self._deter_dim = deter_dim
        self._stoch_dim = stoch_dim
        self._discrete = discrete
        self._hidden = hidden
        self._action_dim = action_dim
        self._embed_dim = embed_dim
        self._unimix_ratio = unimix_ratio
        self._min_std = min_std
        self._initial_mode = initial

        # Effective latent dim(discrete: stoch * discrete; continuous: stoch)
        self._stoch_flat_dim = stoch_dim * discrete if discrete > 0 else stoch_dim
        self._is_discrete = discrete > 0

        # === 输入层: prev_stoch + action → img_in ===
        if self._is_discrete:
            img_in_dim = self._stoch_flat_dim + action_dim
        else:
            img_in_dim = self._stoch_dim + action_dim

        self._img_in_layers = nn.Sequential(
            nn.Linear(img_in_dim, hidden, bias=False),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )

        # === GRU cell ===
        self._gru = nn.GRUCell(hidden, deter_dim)

        # === 输出层: deter → img_out ===
        self._img_out_layers = nn.Sequential(
            nn.Linear(deter_dim, hidden, bias=False),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )

        # === 先验统计层(imagination)===
        if self._is_discrete:
            self._imgs_stat_layer = nn.Linear(hidden, self._stoch_flat_dim)
        else:
            self._imgs_stat_layer = nn.Linear(hidden, 2 * self._stoch_dim)

        # === Posterior 统计层(observation)===
        if self._is_discrete:
            self._obs_stat_layer = nn.Linear(hidden, self._stoch_flat_dim)
        else:
            self._obs_stat_layer = nn.Linear(hidden, 2 * self._stoch_dim)

        # === Obs_out: deter + embed → stats input ===
        self._obs_out_layers = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, hidden, bias=False),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )

        # === Initial state ===
        if initial == 'learned':
            self._init_deter = nn.Parameter(torch.zeros(1, deter_dim))

        # Uniform weight init(对齐 WMP)
        self._init_weights()

    def _init_weights(self):
        """权重初始化(对齐 WMP)"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.1, 0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ============================================================
    # 状态初始化
    # ============================================================

    def initial_state(self, batch_size: int, device='cpu') -> dict:
        """初始化 RSSM 状态"""
        if self._initial_mode == 'learned':
            deter = torch.tanh(self._init_deter).expand(batch_size, -1).to(device)
        else:  # zeros
            deter = torch.zeros(batch_size, self._deter_dim, device=device)

        # 通过 img_out_layers → imgs_stat_layer 得到初始 stoch(mode)
        stoch = self._compute_stoch_from_deter(deter, sample=False)

        return {
            'deter': deter,
            'stoch': stoch,
        }

    def _compute_stoch_from_deter(
        self, deter: torch.Tensor, sample: bool = True
    ) -> torch.Tensor:
        """从 deter 计算 stoch"""
        x = self._img_out_layers(deter)
        stats = self._suff_stats_layer('ims', x)
        if sample:
            dist = self._get_dist(stats)
            stoch = dist.sample()
        else:
            # 取 mode 的 argmax(避免 PyTorch 2.0 OneHotDist.mode() 行为差异)
            stoch = self._stats_to_mode(stats)
        return stoch

    def _suff_stats_layer(self, name: str, x: torch.Tensor) -> dict:
        """计算统计量(无 dist_type 字段,避免污染下游)"""
        if self._is_discrete:
            layer = self._imgs_stat_layer if name == 'ims' else self._obs_stat_layer
            logits = layer(x)
            logits = logits.reshape(*logits.shape[:-1], self._stoch_dim, self._discrete)
            return {'logit': logits}
        else:
            layer = self._imgs_stat_layer if name == 'ims' else self._obs_stat_layer
            stats = layer(x)
            mean, std = torch.split(stats, [self._stoch_dim] * 2, dim=-1)
            std = 2 * torch.sigmoid(std / 2) + self._min_std
            return {'mean': mean, 'std': std}

    def _stats_to_mode(self, stats: dict) -> torch.Tensor:
        """直接取 mode(不通过 dist.mode(),避开 PyTorch 2.0 OneHotDist 行为差异)"""
        if self._is_discrete:
            # mode of one-hot = argmax of logits → one-hot
            idx = stats['logit'].argmax(dim=-1)
            return F.one_hot(idx, num_classes=self._discrete).float()
        else:
            # mode of Normal = mean
            return stats['mean']

    def _get_dist(self, stats: dict):
        """获取分布(用 Independent 包装以支持 KL)"""
        if 'logit' in stats:  # discrete
            return torchd.Independent(
                OneHotDist(logits=stats['logit'], unimix_ratio=self._unimix_ratio),
                1
            )
        else:  # continuous
            return torchd.Independent(
                torchd.Normal(stats['mean'], stats['std']), 1
            )

    # ============================================================
    # 前向传播
    # ============================================================

    def img_step(
        self,
        prev_state: dict,
        prev_action: torch.Tensor,
        sample: bool = True,
    ) -> dict:
        """
        Imagination step: 从 prev_state + action 预测下一状态。
        """
        prev_stoch = prev_state['stoch']
        prev_deter = prev_state['deter']

        # Flatten stoch if discrete
        if self._is_discrete:
            stoch_flat = prev_stoch.reshape(*prev_stoch.shape[:-2], self._stoch_flat_dim)
        else:
            stoch_flat = prev_stoch

        # img_in
        x = torch.cat([stoch_flat, prev_action], dim=-1)
        x = self._img_in_layers(x)

        # GRU
        deter = self._gru(x, prev_deter)

        # img_out → stats
        h = self._img_out_layers(deter)
        stats = self._suff_stats_layer('ims', h)
        if sample:
            dist = self._get_dist(stats)
            stoch = dist.sample()
        else:
            stoch = self._stats_to_mode(stats)

        # 合并 stats(无 dist_type 字段)
        result = {'deter': deter, 'stoch': stoch}
        result.update(stats)
        return result

    def obs_step(
        self,
        prev_state: dict,
        prev_action: torch.Tensor,
        embed: torch.Tensor,
        is_first: torch.Tensor,
        sample: bool = True,
    ) -> tuple:
        """
        Observation step: 从 embed 推断 posterior。
        """
        # 处理 is_first(重置)
        if is_first is not None and is_first.any():
            init = self.initial_state(prev_state['deter'].shape[0], device=prev_state['deter'].device)
            # is_first_expanded 必须广播到每个 tensor 的所有维度
            # prev_state['stoch'] 是 [B, 8, 8],需要 [B, 1, 1]
            prev_state = {
                k: torch.where(
                    is_first.view(-1, *([1] * (v.dim() - 1))),
                    init[k],
                    v,
                )
                for k, v in prev_state.items()
            }
            if prev_action is not None:
                prev_action = torch.where(
                    is_first.unsqueeze(-1),
                    torch.zeros_like(prev_action),
                    prev_action,
                )

        # 先验
        prior = self.img_step(prev_state, prev_action, sample=sample)

        # Posterior: 用 embed 修正
        x = torch.cat([prior['deter'], embed], dim=-1)
        x = self._obs_out_layers(x)
        stats = self._suff_stats_layer('obs', x)
        if sample:
            dist = self._get_dist(stats)
            stoch = dist.sample()
        else:
            stoch = self._stats_to_mode(stats)

        post = {'deter': prior['deter'], 'stoch': stoch}
        post.update(stats)
        return post, prior

    # ============================================================
    # KL loss
    # ============================================================

    def kl_loss(
        self,
        post: dict,
        prior: dict,
        free: float = 1.0,
        dyn_scale: float = 0.5,
        rep_scale: float = 0.1,
    ):
        """
        计算 KL loss(dyn + rep)。
        ReDRAW 公式:
        - dyn_loss = KL(sg(post) || prior)
        - rep_loss = KL(post || sg(prior))
        free bits: KL loss 下界
        """
        # 停梯度(只对 tensor 字段,跳过 None)
        post_sg = {k: v.detach() if torch.is_tensor(v) else v
                   for k, v in post.items() if torch.is_tensor(v)}
        prior_sg = {k: v.detach() if torch.is_tensor(v) else v
                    for k, v in prior.items() if torch.is_tensor(v)}

        # 当前分布
        post_dist = self._get_dist(post)
        prior_dist = self._get_dist(prior)

        # 停梯度的分布
        post_dist_sg = self._get_dist(post_sg)
        prior_dist_sg = self._get_dist(prior_sg)

        # dyn_loss = KL(stop_grad(post) || prior)
        # PyTorch 2.0 中 Independent 没有 kl_divergence 方法,用函数版本
        dyn_loss = torchd.kl.kl_divergence(post_dist_sg, prior_dist)
        if free > 0:
            dyn_loss = torch.maximum(dyn_loss, torch.full_like(dyn_loss, free))
        dyn_loss = dyn_loss.sum(dim=-1)  # sum over stoch dims

        # rep_loss = KL(post || stop_grad(prior))
        rep_loss = torchd.kl.kl_divergence(post_dist, prior_dist_sg)
        if free > 0:
            rep_loss = torch.maximum(rep_loss, torch.full_like(rep_loss, free))
        rep_loss = rep_loss.sum(dim=-1)

        total = dyn_scale * dyn_loss.mean() + rep_scale * rep_loss.mean()
        return total, dyn_loss.mean(), rep_loss.mean()

    # ============================================================
    # 工具方法
    # ============================================================

    def get_feat(self, state: dict) -> torch.Tensor:
        """获取 feature(给 Actor 使用)"""
        stoch = state['stoch']
        if self._is_discrete:
            stoch_flat = stoch.reshape(*stoch.shape[:-2], self._stoch_flat_dim)
        else:
            stoch_flat = stoch
        return torch.cat([stoch_flat, state['deter']], dim=-1)

    def get_deter(self, state: dict) -> torch.Tensor:
        """只取 deter"""
        return state['deter']

    def get_stoch(self, state: dict) -> torch.Tensor:
        """只取 stoch"""
        return state['stoch']

    def get_deter_history(
        self,
        history: torch.Tensor,
        new_deter: torch.Tensor,
    ) -> torch.Tensor:
        """更新 deter history(用于 DynamicsResidual)"""
        return torch.cat([history[..., 1:, :], new_deter.unsqueeze(-2)], dim=-2)