"""
InaccurateSimEnv: 不准确仿真环境(Stage 1 训练)

配置 WMP 的域随机化,模拟 sim-to-real gap:
- 电机扭矩: 0.65-0.75(标称 1.0)
- PD 增益: Kp 14-18(标称 20), Kd 0.3-0.5(标称 0.5)
- 摩擦: 0.3-1.5
- 质量: +1-3 kg
- 外力推: 周期性
"""

import torch
from .wmp_env_base import WMPEnvBase, DEFAULT_TERRAIN_CONFIG


class InaccurateSimEnv(WMPEnvBase):
    """
    InaccurateSim 环境。

    用于 Stage 1 训练。配置:
    - num_envs: 4096(并行训练)
    - 域随机化: 启用(模拟 sim-to-real gap)
    - 观测: 48 维 proprio(从 WMP 235 维 policy_obs 取前 48 维)
    - 地形: 与 PseudoReal 相同
    """

    def __init__(
        self,
        num_envs: int = 4096,
        device: str = 'cuda',
        headless: bool = True,
        max_episode_steps: int = 1000,
        terrain_config: dict = None,
        domain_rand_config: dict = None,
    ):
        # 加载 WMP 配置
        try:
            from legged_gym.envs.a1.a1_amp_config import A1AMPCfg
        except ImportError:
            raise ImportError("无法导入 WMP 的 a1_amp_config")

        cfg = A1AMPCfg()

        # 使用共享地形配置(与 PseudoReal 一致)
        if terrain_config is None:
            terrain_config = DEFAULT_TERRAIN_CONFIG.copy()

        # 默认域随机化配置
        if domain_rand_config is None:
            domain_rand_config = self._default_domain_rand()

        self._domain_rand_config = domain_rand_config

        super().__init__(
            cfg=cfg,
            num_envs=num_envs,
            device=device,
            headless=headless,
            max_episode_steps=max_episode_steps,
            terrain_config=terrain_config,
        )

        print(f"[InaccurateSimEnv] Created: num_envs={num_envs}, "
              f"motor_strength={domain_rand_config.get('motor_strength_range')}, "
              f"Kp_range={domain_rand_config.get('kp_range')}")

    def _configure_domain_rand(self, cfg):
        """配置域随机化(关键!)"""
        dr = self._domain_rand_config

        # 电机扭矩(关键:降 30%,ReDRAW 论文做法)
        cfg.domain_rand.randomize_motor_strength = dr.get('randomize_motor_strength', True)
        cfg.domain_rand.motor_strength_range = dr.get('motor_strength_range', [0.65, 0.75])

        # PD 增益
        cfg.domain_rand.randomize_PD_gains = dr.get('randomize_PD_gains', True)
        cfg.domain_rand.Kp_range = dr.get('kp_range', [14.0, 18.0])
        cfg.domain_rand.Kd_range = dr.get('kd_range', [0.3, 0.5])

        # 额外质量
        cfg.domain_rand.randomize_base_mass = dr.get('randomize_base_mass', True)
        cfg.domain_rand.added_mass_range = dr.get('added_mass_range', [1.0, 3.0])

        # 摩擦
        cfg.domain_rand.randomize_friction = dr.get('randomize_friction', True)
        cfg.domain_rand.friction_range = dr.get('friction_range', [0.3, 1.5])

        # 外力推
        cfg.domain_rand.push_robots = dr.get('push_robots', True)
        # WMP 期望 push_interval_s 是单个 float(不是 list)
        push_interval_s = dr.get('push_interval_s', [3.0, 8.0])
        cfg.domain_rand.push_interval_s = push_interval_s[0]  # 取第一个值
        cfg.domain_rand.push_force = dr.get('push_force', [10.0, 30.0])
        # WMP _push_robots 实际用的是 max_push_vel_xy(线速度扰动)
        cfg.domain_rand.max_push_vel_xy = dr.get('max_push_vel_xy', 1.0)

        # Action latency
        cfg.domain_rand.randomize_action_latency = dr.get('randomize_action_latency', False)
        cfg.domain_rand.latency_range = dr.get('latency_range', [0.0, 0.0])

        # 其他域随机化(补全 WMP 字段)
        cfg.domain_rand.randomize_com_pos = dr.get('randomize_com_pos', False)
        cfg.domain_rand.com_pos_range = dr.get('com_pos_range', [-0.05, 0.05])
        cfg.domain_rand.randomize_restitution = dr.get('randomize_restitution', False)
        cfg.domain_rand.restitution_range = dr.get('restitution_range', [0.0, 1.0])
        cfg.domain_rand.randomize_gains = dr.get('randomize_gains', False)
        cfg.domain_rand.damping_multiplier_range = dr.get('damping_multiplier_range', [0.5, 1.5])
        cfg.domain_rand.stiffness_multiplier_range = dr.get('stiffness_multiplier_range', [0.5, 1.5])
        cfg.domain_rand.randomize_link_mass = dr.get('randomize_link_mass', False)
        cfg.domain_rand.link_mass_range = dr.get('link_mass_range', [0.5, 1.5])

    @staticmethod
    def _default_domain_rand() -> dict:
        """默认域随机化配置"""
        return {
            'randomize_motor_strength': True,
            'motor_strength_range': [0.65, 0.75],
            'randomize_PD_gains': True,
            'kp_range': [14.0, 18.0],
            'kd_range': [0.3, 0.5],
            'randomize_base_mass': True,
            'added_mass_range': [1.0, 3.0],
            'randomize_friction': True,
            'friction_range': [0.3, 1.5],
            'push_robots': True,
            'push_interval_s': [3.0, 8.0],
            'push_force': [10.0, 30.0],
            'max_push_vel_xy': 1.0,
            'randomize_action_latency': False,
            'latency_range': [0.0, 0.0],
            # 补全的次要字段
            'randomize_com_pos': False,
            'com_pos_range': [-0.05, 0.05],
            'randomize_restitution': False,
            'restitution_range': [0.0, 1.0],
            'randomize_gains': False,
            'damping_multiplier_range': [0.5, 1.5],
            'stiffness_multiplier_range': [0.5, 1.5],
            'randomize_link_mass': False,
            'link_mass_range': [0.5, 1.5],
        }


def create_inaccurate_sim_env(num_envs=4096, device='cuda', headless=True):
    """工厂函数"""
    return InaccurateSimEnv(num_envs=num_envs, device=device, headless=headless)