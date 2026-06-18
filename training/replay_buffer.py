"""
Replay Buffer: 存储轨迹数据,支持采样
"""

import numpy as np
import torch
from typing import Optional


class ReplayBuffer:
    """
    简单的 Replay Buffer(支持连续轨迹采样)

    存储:
    - obs: [T, N, obs_dim]
    - action: [T, N, action_dim]
    - reward: [T, N]
    - next_obs: [T, N, obs_dim]
    - done: [T, N]
    """

    def __init__(
        self,
        capacity: int = 1_000_000,
        obs_dim: int = 48,
        action_dim: int = 12,
        device: str = 'cpu',
    ):
        self._capacity = capacity
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._device = device

        # 存储
        self._obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._action = np.zeros((capacity, action_dim), dtype=np.float32)
        self._reward = np.zeros((capacity,), dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._done = np.zeros((capacity,), dtype=np.float32)
        self._is_first = np.zeros((capacity,), dtype=np.float32)  # 新增:is_first 标志

        self._size = 0
        self._ptr = 0

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        is_first: bool = False,
    ):
        """添加单步数据"""
        self._obs[self._ptr] = obs
        self._action[self._ptr] = action
        self._reward[self._ptr] = reward
        self._next_obs[self._ptr] = next_obs
        self._done[self._ptr] = float(done)
        self._is_first[self._ptr] = float(is_first)

        self._ptr = (self._ptr + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def add_batch(
        self,
        obs_batch: np.ndarray,
        action_batch: np.ndarray,
        reward_batch: np.ndarray,
        next_obs_batch: np.ndarray,
        done_batch: np.ndarray,
        is_first_batch: np.ndarray = None,
    ):
        """批量添加"""
        n = obs_batch.shape[0]
        if is_first_batch is None:
            is_first_batch = np.zeros((n,), dtype=np.float32)
        for i in range(n):
            self.add(
                obs_batch[i], action_batch[i], reward_batch[i],
                next_obs_batch[i], done_batch[i], bool(is_first_batch[i])
            )

    def sample(
        self,
        batch_size: int = 512,
        seq_length: int = 50,
    ) -> dict:
        """
        采样序列批次。

        Args:
            batch_size: 批大小
            seq_length: 每个序列长度
        Returns:
            dict with:
                obs: [B, T, obs_dim]
                action: [B, T, action_dim]
                reward: [B, T]
                next_obs: [B, T, obs_dim]
                done: [B, T]
                is_first: [B, T] bool (从 buffer 中读,不再推断)
        """
        max_start = self._size - seq_length - 1
        if max_start <= 0:
            raise ValueError(f"Not enough data: size={self._size}, need {seq_length+1}")

        starts = np.random.randint(0, max_start, size=batch_size)

        obs_batch = np.zeros((batch_size, seq_length, self._obs_dim), dtype=np.float32)
        action_batch = np.zeros((batch_size, seq_length, self._action_dim), dtype=np.float32)
        reward_batch = np.zeros((batch_size, seq_length), dtype=np.float32)
        next_obs_batch = np.zeros((batch_size, seq_length, self._obs_dim), dtype=np.float32)
        done_batch = np.zeros((batch_size, seq_length), dtype=np.float32)
        is_first_batch = np.zeros((batch_size, seq_length), dtype=np.bool_)

        for i, start in enumerate(starts):
            end = start + seq_length
            obs_batch[i] = self._obs[start:end]
            action_batch[i] = self._action[start:end]
            reward_batch[i] = self._reward[start:end]
            next_obs_batch[i] = self._next_obs[start:end]
            done_batch[i] = self._done[start:end]
            # 从 buffer 直接读取 is_first
            is_first_batch[i] = self._is_first[start:end].astype(np.bool_)

        # 转 tensor
        return {
            'obs': torch.from_numpy(obs_batch).to(self._device),
            'action': torch.from_numpy(action_batch).to(self._device),
            'reward': torch.from_numpy(reward_batch).to(self._device),
            'next_obs': torch.from_numpy(next_obs_batch).to(self._device),
            'done': torch.from_numpy(done_batch).to(self._device),
            'is_first': torch.from_numpy(is_first_batch).to(self._device),
        }

    def save(self, path: str):
        """保存到 .npz"""
        size = self._size
        np.savez_compressed(
            path,
            obs=self._obs[:size],
            action=self._action[:size],
            reward=self._reward[:size],
            next_obs=self._next_obs[:size],
            done=self._done[:size],
            is_first=self._is_first[:size],  # 一起保存
        )
        print(f"[ReplayBuffer] Saved {size} transitions to {path}")

    def load(self, path: str):
        """从 .npz 加载"""
        data = np.load(path)
        size = data['obs'].shape[0]
        self._obs[:size] = data['obs']
        self._action[:size] = data['action']
        self._reward[:size] = data['reward']
        self._next_obs[:size] = data['next_obs']
        self._done[:size] = data['done']
        # 兼容旧版 npz(没有 is_first 字段)
        if 'is_first' in data.files:
            self._is_first[:size] = data['is_first']
        # 否则保持默认 0(但 sample() 期望非 0,所以补一下)
        else:
            # 把每个 episode 的第一步标记为 is_first
            self._mark_episode_starts(size)
        self._size = size
        self._ptr = size
        print(f"[ReplayBuffer] Loaded {size} transitions from {path}")

    def _mark_episode_starts(self, size: int):
        """从 done 推断 episode 起点(兼容旧版 npz)"""
        # 第一个总是 first
        self._is_first[0] = 1.0
        for i in range(1, size):
            # done[t-1] == 1 表示上一个 episode 结束,t 是新 episode 起点
            self._is_first[i] = self._done[i - 1]

    @classmethod
    def from_npz(cls, path: str, obs_dim: int = 48, action_dim: int = 12, device: str = 'cpu'):
        """从 .npz 创建 buffer"""
        buffer = cls(capacity=1_000_000, obs_dim=obs_dim, action_dim=action_dim, device=device)
        buffer.load(path)
        return buffer

    def __len__(self) -> int:
        return self._size