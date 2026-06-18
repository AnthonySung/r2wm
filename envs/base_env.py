"""
环境基类:定义 r2wmp 所有环境的统一接口。
"""

import torch
import numpy as np
from typing import Tuple, Dict, Optional


class BaseEnv:
    """
    r2wmp 环境基类。

    所有环境必须实现:
    - reset() -> obs [num_envs, 45]
    - step(action) -> (obs, reward, done, info)
    - get_proprio_obs() -> obs [num_envs, 45]
    - close()

    通用属性:
    - num_envs: 并行环境数
    - device: 'cuda' 或 'cpu'
    - obs_dim: 45(本体感知)
    - action_dim: 12
    - dt: 0.02s(policy step)
    - max_episode_steps: 1000
    """

    OBS_DIM = 48   # WMP policy_obs 前 48 维本体感知
    ACTION_DIM = 12  # 12 个关节

    @property
    def obs_dim(self) -> int:
        return self.OBS_DIM

    def __init__(
        self,
        num_envs: int = 4096,
        device: str = 'cuda',
        headless: bool = True,
        max_episode_steps: int = 1000,
    ):
        self._num_envs = num_envs
        self._device = device
        self._headless = headless
        self._max_episode_steps = max_episode_steps

        # 内部状态
        self._step_count = 0
        self._episode_returns = torch.zeros(num_envs, device=device)
        self._episode_lengths = torch.zeros(num_envs, device=device, dtype=torch.long)

        # 当前 obs(45 维 proprio)
        self._current_obs: Optional[torch.Tensor] = None

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def device(self) -> str:
        return self._device

    @property
    def obs_dim(self) -> int:
        return self.OBS_DIM

    @property
    def action_dim(self) -> int:
        return self.ACTION_DIM

    @property
    def dt(self) -> float:
        """Policy step 时间(秒)"""
        return 0.02  # 50Hz policy (200Hz sim × decimation=4)

    @property
    def max_episode_steps(self) -> int:
        return self._max_episode_steps

    def reset(self) -> torch.Tensor:
        """
        重置所有环境。

        Returns:
            obs: [num_envs, 45] 本体感知
        """
        raise NotImplementedError

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        执行一步。

        Args:
            action: [num_envs, 12] 关节位置目标(已缩放,[-1, 1] 范围)

        Returns:
            obs: [num_envs, 45]
            reward: [num_envs]
            done: [num_envs] bool
            info: dict
        """
        raise NotImplementedError

    def get_proprio_obs(self) -> torch.Tensor:
        """
        获取本体感知 obs。

        Returns:
            obs: [num_envs, 45]
        """
        raise NotImplementedError

    def close(self):
        """清理资源"""
        pass

    def _check_action(self, action: torch.Tensor) -> torch.Tensor:
        """检查 action 形状和范围"""
        assert action.shape == (self._num_envs, self.ACTION_DIM), \
            f"Action shape mismatch: {action.shape} != ({self._num_envs}, {self.ACTION_DIM})"
        # 限制在 [-1, 1]
        return torch.clamp(action, -1.0, 1.0)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_envs={self._num_envs}, "
            f"device={self._device}, "
            f"obs_dim={self.obs_dim}, "
            f"action_dim={self.action_dim})"
        )