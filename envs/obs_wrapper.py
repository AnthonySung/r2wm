"""
观测包装器:统一不同环境的观测格式。

A1AMP 的 45 维本体感知组成:
- base_ang_vel (3): 基座角速度
- projected_gravity (3): 投影到基座的重力
- commands (3): 速度指令 (vx, vy, wz)
- dof_pos (12): 关节位置(相对默认)
- dof_vel (12): 关节速度
- last_actions (12): 上一时刻动作
合计: 3 + 3 + 3 + 12 + 12 + 12 = 45
"""

import torch


PROPRIO_DIM = 45
PROPRIO_KEYS = {
    'base_ang_vel': (0, 3),
    'projected_gravity': (3, 6),
    'commands': (6, 9),
    'dof_pos': (9, 21),
    'dof_vel': (21, 33),
    'last_actions': (33, 45),
}


def extract_proprio_from_full_obs(full_obs: torch.Tensor) -> torch.Tensor:
    """
    从完整 235 维 obs 中提取 45 维 proprio。

    Args:
        full_obs: [..., 235] 完整观测
    Returns:
        proprio: [..., 45] 本体感知
    """
    # A1AMP 的 obs 结构(根据 WMP 代码):
    # [0:33] proprio
    # [33:33+12] last_actions (也可能包含在前面)
    # [33:55+187+...] privileged info + heightmap

    # 实际位置取决于 WMP 实现
    # 这里假设前 45 维就是 proprio + last_actions
    return full_obs[..., :PROPRIO_DIM]


def build_proprio_obs(
    base_ang_vel: torch.Tensor,
    projected_gravity: torch.Tensor,
    commands: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_vel: torch.Tensor,
    last_actions: torch.Tensor,
) -> torch.Tensor:
    """
    构建 45 维 proprio 观测。

    Args:
        base_ang_vel: [..., 3]
        projected_gravity: [..., 3]
        commands: [..., 3]
        dof_pos: [..., 12]
        dof_vel: [..., 12]
        last_actions: [..., 12]

    Returns:
        proprio: [..., 45]
    """
    return torch.cat([
        base_ang_vel,
        projected_gravity,
        commands,
        dof_pos,
        dof_vel,
        last_actions,
    ], dim=-1)


def decompose_proprio(proprio: torch.Tensor) -> dict:
    """
    分解 45 维 proprio 为各组件。

    Args:
        proprio: [..., 45]
    Returns:
        dict with keys: base_ang_vel, projected_gravity, etc.
    """
    return {
        key: proprio[..., start:end]
        for key, (start, end) in PROPRIO_KEYS.items()
    }


def apply_obs_scales(proprio: torch.Tensor, scales: dict) -> torch.Tensor:
    """
    应用观测缩放(对齐 WMP 的 obs_scales)。

    Args:
        proprio: [..., 45]
        scales: dict,例如 {'base_ang_vel': 0.25, 'dof_pos': 1.0, ...}
    Returns:
        scaled_proprio: [..., 45]
    """
    components = decompose_proprio(proprio)
    scaled = []
    for key in PROPRIO_KEYS.keys():
        comp = components[key]
        scale = scales.get(key, 1.0)
        scaled.append(comp * scale)
    return torch.cat(scaled, dim=-1)