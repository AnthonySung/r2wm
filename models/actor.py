"""
A1Actor: Dreamer 风格 Actor(reparameterization + continuous action)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class A1Actor(nn.Module):
    """
    Actor 网络。

    输入: feat = [stoch + deter]
    输出: action = Normal(mean, std) 采样 + tanh 压缩到 [-1, 1]
    """

    def __init__(
        self,
        feat_dim: int,
        action_dim: int = 12,
        hidden: int = 512,
        n_layers: int = 2,
        std_min: float = 0.1,
        std_max: float = 1.0,
        entropy_scale: float = 1e-3,
    ):
        super().__init__()
        self._feat_dim = feat_dim
        self._action_dim = action_dim
        self._std_min = std_min
        self._std_max = std_max
        self._entropy_scale = entropy_scale

        layers = []
        in_dim = feat_dim
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.LayerNorm(hidden))
            layers.append(nn.SiLU())
            in_dim = hidden

        self.trunk = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden, action_dim)
        self.log_std_head = nn.Linear(hidden, action_dim)

    def forward(
        self,
        feat: torch.Tensor,
        sample: bool = True,
        deterministic: bool = False,
    ) -> tuple:
        """
        前向传播。

        Args:
            feat: [..., feat_dim]
            sample: True 时采祥 + reparameterization;False 时取 mean
            deterministic: 同 sample=False,但不计算 entropy

        Returns:
            action: [..., action_dim] 压缩到 [-1, 1]
            mean: [..., action_dim]
            std: [..., action_dim]
            entropy: [...] 标量
        """
        h = self.trunk(feat)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(-5, 2)
        std = log_std.exp().clamp(self._std_min, self._std_max)

        if sample and not deterministic:
            eps = torch.randn_like(mean)
            raw_action = mean + eps * std
        else:
            raw_action = mean

        # Tanh 压缩到 [-1, 1]
        action = torch.tanh(raw_action)

        # Entropy(粗略估计,基于 std)
        entropy = (0.5 * torch.log(2 * torch.pi * std ** 2) + 0.5).sum(dim=-1)

        return action, mean, std, entropy

    def sample(self, feat: torch.Tensor) -> torch.Tensor:
        """仅返回 action(用于 env step)"""
        action, _, _, _ = self.forward(feat, sample=True)
        return action

    def get_log_prob(
        self,
        feat: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算 log_prob(给定动作)。

        Args:
            feat: [..., feat_dim]
            action: [..., action_dim] (已经 tanh 压缩)
        Returns:
            log_prob: [...]
        """
        # 反 tanh
        raw_action = torch.atanh(action.clamp(-0.999, 0.999))

        h = self.trunk(feat)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(-5, 2)
        std = log_std.exp().clamp(self._std_min, self._std_max)

        # Normal log_prob
        log_prob = -0.5 * (((raw_action - mean) / std) ** 2 + 2 * log_std + torch.log(torch.tensor(2 * torch.pi)))
        log_prob = log_prob.sum(dim=-1)

        # Tanh 修正(change of variables)
        log_prob -= torch.log(1 - action ** 2 + 1e-6).sum(dim=-1)

        return log_prob