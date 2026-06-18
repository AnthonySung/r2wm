"""
Encoder: 把 obs 编码成 embedding
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """MLP-based encoder(45 维 proprio → embed_dim)"""

    def __init__(
        self,
        obs_dim: int,
        embed_dim: int = 256,
        hidden: int = 1024,
        n_layers: int = 5,
        act: str = 'silu',
        norm: str = 'layer',
        symlog_inputs: bool = True,
    ):
        super().__init__()
        self._obs_dim = obs_dim
        self._embed_dim = embed_dim
        self._symlog_inputs = symlog_inputs

        act_fn = nn.SiLU() if act == 'silu' else nn.ReLU()
        norm_fn = nn.LayerNorm if norm == 'layer' else nn.Identity

        layers = []
        in_dim = obs_dim
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(norm_fn(hidden))
            layers.append(act_fn)
            in_dim = hidden
        layers.append(nn.Linear(in_dim, embed_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self._symlog_inputs:
            obs = self._symlog(obs)
        return self.net(obs)

    @staticmethod
    def _symlog(x: torch.Tensor) -> torch.Tensor:
        """Symlog transform: sign(x) * log(1 + |x|)"""
        return torch.sign(x) * torch.log1p(torch.abs(x))