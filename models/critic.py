"""
A1Critic: Dreamer 风格 Critic(V(s))
支持 EMA target network(对齐 ReDRAW 的 slow_target)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class A1Critic(nn.Module):
    """V(s) 网络"""

    def __init__(
        self,
        feat_dim: int,
        hidden: int = 512,
        n_layers: int = 2,
    ):
        super().__init__()
        self._feat_dim = feat_dim

        layers = []
        in_dim = feat_dim
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.LayerNorm(hidden))
            layers.append(nn.SiLU())
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """返回 V(s)"""
        return self.net(feat).squeeze(-1)


class SlowCritic(nn.Module):
    """带 EMA target 的 Critic"""

    def __init__(self, base_critic: A1Critic, update_fraction: float = 0.02):
        super().__init__()
        self._update_fraction = update_fraction
        self.slow_net = copy.deepcopy(base_critic)
        # 不需要梯度
        for p in self.slow_net.parameters():
            p.requires_grad = False

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.slow_net(feat)

    def update(self, critic: A1Critic):
        """EMA 更新"""
        with torch.no_grad():
            for slow_p, p in zip(self.slow_net.parameters(), critic.parameters()):
                slow_p.data.mul_(1 - self._update_fraction).add_(
                    p.data, alpha=self._update_fraction
                )