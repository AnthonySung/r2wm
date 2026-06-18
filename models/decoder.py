"""
Decoder: 从 feature 重建 obs
"""

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """MLP-based decoder(feat → obs)"""

    def __init__(
        self,
        feat_dim: int,
        obs_dim: int,
        hidden: int = 1024,
        n_layers: int = 5,
        act: str = 'silu',
        norm: str = 'layer',
        outscale: float = 1.0,
    ):
        super().__init__()
        self._feat_dim = feat_dim
        self._obs_dim = obs_dim
        self._outscale = outscale

        act_fn = nn.SiLU() if act == 'silu' else nn.ReLU()
        norm_fn = nn.LayerNorm if norm == 'layer' else nn.Identity

        layers = []
        in_dim = feat_dim
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(norm_fn(hidden))
            layers.append(act_fn)
            in_dim = hidden
        layers.append(nn.Linear(in_dim, obs_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        out = self.net(feat)
        return out * self._outscale