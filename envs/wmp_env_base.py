"""
WMP 环境基类:包装 WMP 的 LeggedRobot 类,标准化输出。

关键设计:
1. 通过 sys.path 引入 WMP 仓库
2. 调用 LeggedRobot 创建 Isaac Gym 仿真
3. 标准化 step() 返回为 (obs, reward, done, info) 4 元组
4. 从 WMP 的 235 维 policy_obs 提取 45 维 proprio
"""

import os
import sys
import numpy as np
import torch
from typing import Tuple, Dict, Optional, Any

from .base_env import BaseEnv

# 添加 WMP 路径(优先用环境变量,否则用默认)
WMP_ROOT = os.environ.get('WMP_ROOT', 'D:/songay/sim2real/WMP')
if WMP_ROOT not in sys.path:
    sys.path.insert(0, WMP_ROOT)


class WMPEnvBase(BaseEnv):
    """
    WMP LeggedRobot 的基类包装。

    子类需要实现 _configure_domain_rand() 配置不同的物理参数。
    """

    def __init__(
        self,
        cfg,                       # WMP 的 LeggedRobotCfg
        num_envs: int = 4096,
        device: str = 'cuda',
        headless: bool = True,
        max_episode_steps: int = 1000,
        terrain_config: dict = None,
    ):
        super().__init__(
            num_envs=num_envs,
            device=device,
            headless=headless,
            max_episode_steps=max_episode_steps,
        )

        self._cfg = cfg
        self._terrain_config = terrain_config or {}

        # 延迟导入 Isaac Gym 和 WMP
        try:
            from isaacgym import gymapi, gymtorch
            from legged_gym.envs.base.legged_robot import LeggedRobot
        except ImportError as e:
            raise ImportError(
                f"无法导入 Isaac Gym 或 WMP。请确保:\n"
                f"1. Isaac Gym 已安装\n"
                f"2. WMP 仓库在 {WMP_ROOT}\n"
                f"3. rsl_rl 和 legged_gym 已安装\n"
                f"原始错误: {e}"
            )

        self._gymapi = gymapi
        self._gymtorch = gymtorch

        # 配置 env
        cfg.env.num_envs = num_envs
        cfg.env.episode_length_s = max_episode_steps * 0.02  # 1000 * 0.02 = 20s

        # 关键修复: 禁用 camera(headless + 无显示器必须关)
        # 否则会 segfault (graphics_device_id 与 sim_device_id 冲突)
        cfg.depth.use_camera = False

        # 配置地形(子类的关键控制点)
        self._configure_terrain(cfg, terrain_config)

        # 配置域随机化(子类必须实现)
        self._configure_domain_rand(cfg)

        # 创建 sim params
        sim_params = self._make_sim_params(cfg, device)

        # 创建 LeggedRobot
        self._wmp_env = LeggedRobot(
            cfg, sim_params, gymapi.SIM_PHYSX, device, headless
        )

        # 缓存
        self._phys_engine = gymapi.SIM_PHYSX

    # ============================================================
    # 子类必须实现的方法
    # ============================================================

    def _configure_domain_rand(self, cfg):
        """
        子类实现:配置域随机化。
        - InaccurateSim: 启用各种域随机化
        - PseudoReal: 全部关闭
        """
        raise NotImplementedError

    def _configure_terrain(self, cfg, terrain_config: dict):
        """
        配置地形(两个环境必须一致)。

        terrain_config 包含:
        - mesh_type: 'trimesh' 或 'plane'
        - measure_heights: True/False
        - horizontal_scale: 0.1
        - vertical_scale: 0.005
        - terrain_proportions: [...]

        注意:
        - 必须用 'trimesh',不能用 'plane'(WMP 的 reward 函数依赖 terrain)
        - 必须 measure_heights=False(简化 obs,避免 heightmap 处理)
        """
        # 默认 trimesh(关键:不能用 plane!)
        cfg.terrain.mesh_type = terrain_config.get('mesh_type', 'trimesh')
        assert cfg.terrain.mesh_type == 'trimesh', (
            "必须用 trimesh terrain!Plane 会导致 WMP reward 函数报错。"
        )

        cfg.terrain.horizontal_scale = terrain_config.get('horizontal_scale', 0.1)
        cfg.terrain.vertical_scale = terrain_config.get('vertical_scale', 0.005)
        cfg.terrain.measure_heights = terrain_config.get('measure_heights', False)

        # Trimesh 配置
        if cfg.terrain.mesh_type == 'trimesh':
            cfg.terrain.terrain_length = terrain_config.get('terrain_length', 8.0)
            cfg.terrain.terrain_width = terrain_config.get('terrain_width', 8.0)
            cfg.terrain.num_rows = terrain_config.get('num_rows', 10)
            cfg.terrain.num_cols = terrain_config.get('num_cols', 20)
            cfg.terrain.terrain_proportions = terrain_config.get(
                'terrain_proportions', [0.1, 0.1, 0.30, 0.25, 0.15, 0.1]
            )
            cfg.terrain.slope_treshold = terrain_config.get('slope_treshold', 0.75)
            cfg.terrain.curriculum = terrain_config.get('curriculum', False)

        # 摩擦
        cfg.terrain.static_friction = terrain_config.get('static_friction', 1.0)
        cfg.terrain.dynamic_friction = terrain_config.get('dynamic_friction', 1.0)
        cfg.terrain.restitution = terrain_config.get('restitution', 0.0)

    def _make_sim_params(self, cfg, device) -> Any:
        """创建 sim params"""
        from isaacgym import gymapi
        sim_params = gymapi.SimParams()
        sim_params.dt = 0.005  # 200Hz
        sim_params.num_client_threads = 0
        sim_params.use_gpu_pipeline = True if 'cuda' in device else False
        sim_params.substeps = 1
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.physx.use_gpu = True if 'cuda' in device else False
        sim_params.physx.num_threads = 4
        sim_params.physx.solver_type = 1  # TGS
        sim_params.physx.num_subscenes = 4
        return sim_params

    # ============================================================
    # 标准接口
    # ============================================================

    def reset(self) -> torch.Tensor:
        """重置所有 env,返回 45 维 proprio"""
        policy_obs, _ = self._wmp_env.reset()
        proprio = self.get_proprio_obs_from_full(policy_obs)
        self._current_obs = proprio
        self._step_count = 0
        self._episode_returns.zero_()
        self._episode_lengths.zero_()
        return proprio

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        执行一步。

        Args:
            action: [num_envs, 12] 范围 [-1, 1]
        Returns:
            obs: [num_envs, 48] 本体感知
            reward: [num_envs]
            done: [num_envs] bool
            info: dict
        """
        action = self._check_action(action).to(self._device)

        # 调用 WMP LeggedRobot.step()
        # 返回: (policy_obs, privileged_obs_buf, rew_buf, reset_buf, extras, reset_env_ids, terminal_amp_states)
        result = self._wmp_env.step(action)
        assert len(result) == 7, (
            f"WMP LeggedRobot.step() 返回 {len(result)} 元组, 期望 7. "
            f"WMP 版本可能不兼容, 请检查 D:\\songay\\sim2real\\WMP\\legged_gym\\envs\\base\\legged_robot.py"
        )
        policy_obs, _, reward, reset_buf, extras, _, _ = result

        self._step_count += 1

        # 提取 proprio
        obs = self.get_proprio_obs_from_full(policy_obs)
        self._current_obs = obs

        # 计算 done(WMP 的 reset_buf 已经包含 terminate,timeout)
        done = reset_buf.clone()

        # Episode 统计
        self._episode_returns += reward
        self._episode_lengths += 1

        # 自动 reset(detect done)
        info = {
            'episode': {
                'r': self._episode_returns.clone(),
                'l': self._episode_lengths.clone(),
            },
            'extras': extras,
        }

        # 重置已 done 的 env 的统计
        if done.any():
            done_idx = done.nonzero(as_tuple=False).squeeze(-1)
            self._episode_returns[done_idx] = 0.0
            self._episode_lengths[done_idx] = 0

        return obs, reward, done, info

    def get_proprio_obs(self) -> torch.Tensor:
        """获取当前 proprio obs"""
        return self._current_obs

    def get_proprio_obs_from_full(self, full_obs: torch.Tensor) -> torch.Tensor:
        """
        从 WMP 完整 obs(259 维或更多)提取 48 维 proprio。

        WMP policy_obs 顺序(根据 compute_observations, A1AMP include_history_steps=None):
        [0:3]   base_lin_vel * obs_scales.lin_vel        # [3]
        [3:6]   base_ang_vel * obs_scales.ang_vel        # [3]
        [6:9]   projected_gravity                         # [3]
        [9:12]  commands[:, :3] * commands_scale          # [3]
        [12:24] (dof_pos - default_dof_pos) * obs_scales.dof_pos  # [12]
        [24:36] dof_vel * obs_scales.dof_vel              # [12]
        [36:48] actions (last_actions)                    # [12]
        [48:259] heightmap + privileged + ...            # [211+]
        总前 48 维 = 本体感知(包括 commands 和 last_actions)
        """
        if full_obs.dim() == 1:
            full_obs = full_obs.unsqueeze(0)
        if full_obs.shape[-1] < 48:
            raise ValueError(
                f"WMP obs 维度不足: {full_obs.shape}, 期望 >= 48. "
                f"请检查 A1AMP 是否启用了 include_history_steps 或其他维配置。"
            )
        return full_obs[..., :48]

    def close(self):
        """清理"""
        if hasattr(self._wmp_env, 'gym') and self._wmp_env.gym is not None:
            try:
                self._wmp_env.gym.destroy_sim(self._wmp_env.sim)
            except Exception:
                pass

    def get_full_obs(self) -> torch.Tensor:
        """
        仅用于评估日志:返回完整 235 维 obs(包含 privileged)。
        Stage 1/2 训练时不应调用!
        """
        if hasattr(self._wmp_env, 'obs_buf'):
            return self._wmp_env.obs_buf
        return None


# ============================================================
# 默认地形配置(两个环境共享,确保一致!)
# ============================================================

DEFAULT_TERRAIN_CONFIG = {
    'mesh_type': 'trimesh',
    'horizontal_scale': 0.1,
    'vertical_scale': 0.005,
    'measure_heights': False,  # 默认不测量,简化观测
    'terrain_length': 8.0,
    'terrain_width': 8.0,
    'num_rows': 10,
    'num_cols': 20,
    # 地形类型比例:wave, rough slope, stairs up, stairs down, discrete, rough_flat
    'terrain_proportions': [0.1, 0.1, 0.30, 0.25, 0.15, 0.1],
    'slope_treshold': 0.75,
    'curriculum': False,        # 关闭 curriculum
    'static_friction': 1.0,
    'dynamic_friction': 1.0,
    'restitution': 0.0,
}