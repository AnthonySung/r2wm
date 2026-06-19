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


# Lazy import WMP AMP modules(避免强制依赖)
def _import_amp_modules():
    """延迟导入 WMP 的 AMPDiscriminator / AMPLoader / Normalizer"""
    from rsl_rl.algorithms.amp_discriminator import AMPDiscriminator
    from rsl_rl.datasets.motion_loader import AMPLoader
    from rsl_rl.utils.utils import Normalizer
    return AMPDiscriminator, AMPLoader, Normalizer


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
        amp_config: dict = None,
    ):
        super().__init__(
            num_envs=num_envs,
            device=device,
            headless=headless,
            max_episode_steps=max_episode_steps,
        )

        self._cfg = cfg
        self._terrain_config = terrain_config or {}
        self._amp_config = amp_config

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

        # 关键 hack: 把 rew_buf 改成 float dtype
        # WMP bug: rew_buf 初始化为 int,bool 加法后变 bool
        self._wmp_env.rew_buf = self._wmp_env.rew_buf.float()
        # 同时把 episode_sums 也转 float(避免累加 bool)
        for k in self._wmp_env.episode_sums:
            self._wmp_env.episode_sums[k] = self._wmp_env.episode_sums[k].float()

        # 缓存
        self._phys_engine = gymapi.SIM_PHYSX

        # ============================================================
        # AMP (Adversarial Motion Priors) 初始化 — B1
        # ============================================================
        self._init_amp(amp_config)

    def _init_amp(self, amp_config):
        """
        初始化 AMP 模块(Discriminator + AMPLoader + Normalizer)

        B1: 只是前向算 AMP reward,discriminator 不训(权重随机)
        B2: 加 discriminator 训练
        """
        # 默认配置
        default = {
            'reward_coef': 0.3,
            'discr_hidden_dims': [1024, 512],
            'task_reward_lerp': 0.0,
            # 用绝对路径(因为 WMP_ROOT 环境变量可能没设)
            'motion_files': [
                '/home/WMP/datasets/mocap_motions/hop1.txt',
                '/home/WMP/datasets/mocap_motions/hop2.txt',
                '/home/WMP/datasets/mocap_motions/trot1.txt',
                '/home/WMP/datasets/mocap_motions/trot2.txt',
            ],
            'time_between_frames': 0.02,
            'use_normalizer': True,
            'num_preload_transitions': 100000,
        }
        if amp_config is None:
            amp_config = default
        else:
            for k, v in default.items():
                amp_config.setdefault(k, v)

        # 如果 amp_config 是 dict 但缺字段,用 default 补
        for k, v in default.items():
            amp_config.setdefault(k, v)

        self._amp_cfg = amp_config
        self._use_amp = amp_config is not None

        if not self._use_amp:
            return

        # 延迟 import (避免 WMP 没装时报错)
        try:
            AMPDiscriminator, AMPLoader, Normalizer = _import_amp_modules()
        except ImportError as e:
            print(f"[WMPEnvBase] AMP 模块 import 失败: {e}")
            print(f"[WMPEnvBase] 降级到无 AMP 模式")
            self._use_amp = False
            return

        # 1. AMPLoader: 读 mocap motion files
        # 关键: motion_files 路径用相对于 WMP_ROOT
        motion_files_full = []
        for mf in amp_config['motion_files']:
            if os.path.isabs(mf):
                motion_files_full.append(mf)
            else:
                # 默认在 WMP_ROOT 下
                motion_files_full.append(os.path.join(WMP_ROOT, mf))

        if not motion_files_full or not all(os.path.exists(f) for f in motion_files_full):
            print(f"[WMPEnvBase] AMP motion files 缺失,降级到无 AMP")
            self._use_amp = False
            return

        try:
            self._amp_loader = AMPLoader(
                device=self._device,
                time_between_frames=amp_config['time_between_frames'],
                preload_transitions=True,
                num_preload_transitions=amp_config['num_preload_transitions'],
                motion_files=motion_files_full,
            )
            amp_obs_dim = self._amp_loader.observation_dim
            print(f"[WMPEnvBase] AMPLoader OK: obs_dim={amp_obs_dim}, motion files={len(motion_files_full)}")
        except Exception as e:
            print(f"[WMPEnvBase] AMPLoader 创建失败: {e}")
            self._use_amp = False
            return

        # 2. AMPDiscriminator
        try:
            self._amp_discriminator = AMPDiscriminator(
                input_dim=amp_obs_dim * 2,  # cat [state, next_state]
                amp_reward_coef=amp_config['reward_coef'],
                hidden_layer_sizes=amp_config['discr_hidden_dims'],
                device=self._device,
                task_reward_lerp=amp_config['task_reward_lerp'],
            )
            self._amp_obs_dim = amp_obs_dim
            print(f"[WMPEnvBase] AMPDiscriminator OK: reward_coef={amp_config['reward_coef']}, hidden={amp_config['discr_hidden_dims']}")
        except Exception as e:
            print(f"[WMPEnvBase] AMPDiscriminator 创建失败: {e}")
            self._use_amp = False
            return

        # 3. AMPNormalizer (B2 会更新它,B1 暂时不用)
        if amp_config['use_normalizer']:
            self._amp_normalizer = Normalizer(amp_obs_dim)
            # B1: 用初始 normalizer(uniform mean=0, var=1)
            # B2: 在训练过程中 update
        else:
            self._amp_normalizer = None

        # AMP obs 缓存 (current step 的 amp_obs,用于下一步的 reward 计算)
        self._current_amp_obs = None

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
        # AMP: 缓存当前 amp_obs (用于下一步的 reward 计算)
        if self._use_amp:
            self._current_amp_obs = self._wmp_env.get_amp_observations()
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

        # AMP: 缓存当前 amp_obs (predict_amp_reward 需要 [state, next_state])
        if self._use_amp:
            current_amp_obs = self._current_amp_obs

        # 调用 WMP LeggedRobot.step()
        # 返回: (policy_obs, privileged_obs_buf, rew_buf, reset_buf, extras, reset_env_ids, terminal_amp_states)
        result = self._wmp_env.step(action)
        assert len(result) == 7, (
            f"WMP LeggedRobot.step() 返回 {len(result)} 元组, 期望 7. "
            f"WMP 版本可能不兼容, 请检查 D:\\songay\\sim2real\\WMP\\legged_gym\\envs\\base\\legged_robot.py"
        )
        policy_obs, _, reward, reset_buf, extras, reset_env_ids, terminal_amp_states = result

        self._step_count += 1

        # AMP: 算 AMP reward 并加到 task reward 上
        if self._use_amp:
            # next_amp_obs = step 后状态
            next_amp_obs = self._wmp_env.get_amp_observations()
            # 用 terminal_amp_states 替换已 reset env 的 next_amp_obs
            # (WMP 标准做法,让 discriminator 知道 episode 真的结束了)
            next_amp_obs_with_term = torch.clone(next_amp_obs)
            if reset_env_ids is not None and len(reset_env_ids) > 0:
                next_amp_obs_with_term[reset_env_ids] = terminal_amp_states

            # 算 AMP reward (no_grad, eval mode)
            with torch.no_grad():
                amp_reward, _ = self._amp_discriminator.predict_amp_reward(
                    current_amp_obs, next_amp_obs_with_term, reward,
                    normalizer=self._amp_normalizer
                )
            # 合并: total reward = task + AMP
            reward = reward + amp_reward
            # 缓存 next amp_obs 给下一步用
            self._current_amp_obs = next_amp_obs

        # 提取 proprio
        obs = self.get_proprio_obs_from_full(policy_obs)
        self._current_obs = obs

        # 计算 done(WMP 的 reset_buf 已经包含 terminate,timeout)
        done = reset_buf.clone()

        # 关键: 强制转 float 避免 dtype 错误
        # WMP 的 _reward_termination 返回 bool,累加会污染 rew_buf dtype
        reward = reward.float()

        # Episode 统计
        # WMP 的 reward 可能是 bool(来自 base_terminated 项),需转 float
        self._episode_returns += reward.float()
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

        # 返回时也转 float(避免 mean() 报错)
        return obs, reward.float(), done, info

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