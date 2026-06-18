"""
PseudoRealEnv: 伪真实环境(评估 + 数据采集)

代表"真机":
- 关闭所有域随机化(标称物理)
- num_envs: 64(模拟真机只有少量并行)
- 地形: 与 InaccurateSim 相同!
"""

import torch
from .wmp_env_base import WMPEnvBase, DEFAULT_TERRAIN_CONFIG


class PseudoRealEnv(WMPEnvBase):
    """
    PseudoReal 环境。

    用于 Stage 2 数据采集和最终评估。配置:
    - num_envs: 64(模拟真机)
    - 域随机化: 全部关闭(标称物理,代表"真机")
    - 观测: 45 维 proprio
    - 地形: **与 InaccurateSimEnv 完全一致**
    """

    def __init__(
        self,
        num_envs: int = 64,
        device: str = 'cuda',
        headless: bool = True,
        max_episode_steps: int = 1000,
        terrain_config: dict = None,
    ):
        # 加载 WMP 配置
        try:
            from legged_gym.envs.a1.a1_amp_config import A1AMPCfg
        except ImportError:
            raise ImportError("无法导入 WMP 的 a1_amp_config")

        cfg = A1AMPCfg()

        # 关键:使用共享的默认地形配置(与 InaccurateSim 一致!)
        if terrain_config is None:
            terrain_config = DEFAULT_TERRAIN_CONFIG.copy()

        super().__init__(
            cfg=cfg,
            num_envs=num_envs,
            device=device,
            headless=headless,
            max_episode_steps=max_episode_steps,
            terrain_config=terrain_config,
        )

        print(f"[PseudoRealEnv] Created: num_envs={num_envs}, "
              f"nominal physics, obs_dim = 48")

    def _configure_domain_rand(self, cfg):
        """
        关键:全部关闭域随机化,代表"真机"标称物理。
        """
        # 电机扭矩(标称)
        cfg.domain_rand.randomize_motor_strength = False
        cfg.domain_rand.motor_strength_range = [1.0, 1.0]

        # PD 增益(标称)
        cfg.domain_rand.randomize_PD_gains = False
        cfg.domain_rand.Kp_range = [20.0, 20.0]
        cfg.domain_rand.Kd_range = [0.5, 0.5]

        # 额外质量(无)
        cfg.domain_rand.randomize_base_mass = False
        cfg.domain_rand.added_mass_range = [0.0, 0.0]

        # 摩擦(标称)
        cfg.domain_rand.randomize_friction = False
        cfg.domain_rand.friction_range = [1.0, 1.0]

        # 外力推(无)
        cfg.domain_rand.push_robots = False

        # 延迟(无)
        cfg.domain_rand.randomize_action_latency = False
        cfg.domain_rand.latency_range = [0.0, 0.0]

        # 重置(无)
        cfg.domain_rand.randomize_restitution = False
        cfg.domain_rand.randomize_com_pos = False
        cfg.domain_rand.randomize_gains = False


def create_pseudo_real_env(num_envs=64, device='cuda', headless=True):
    """工厂函数"""
    return PseudoRealEnv(num_envs=num_envs, device=device, headless=headless)


# ============================================================
# 验证函数:确保两个环境的地形一致
# ============================================================

def verify_terrain_consistency():
    """
    验证 InaccurateSim 和 PseudoReal 的地形配置完全一致。
    """
    sim_env = InaccurateSimEnv(num_envs=2, device='cpu', headless=True)
    pr_env = PseudoRealEnv(num_envs=2, device='cpu', headless=True)

    # 比较关键地形参数
    sim_terrain = sim_env._wmp_env.cfg.terrain
    pr_terrain = pr_env._wmp_env.cfg.terrain

    keys_to_check = ['mesh_type', 'horizontal_scale', 'vertical_scale',
                      'measure_heights', 'terrain_length', 'terrain_width',
                      'num_rows', 'num_cols', 'terrain_proportions',
                      'slope_treshold', 'curriculum']

    all_consistent = True
    for key in keys_to_check:
        sim_val = getattr(sim_terrain, key, None)
        pr_val = getattr(pr_terrain, key, None)
        if sim_val != pr_val:
            print(f"❌ {key}: sim={sim_val} vs pr={pr_val}")
            all_consistent = False
        else:
            print(f"✅ {key}: {sim_val}")

    sim_env.close()
    pr_env.close()

    return all_consistent


if __name__ == '__main__':
    verify_terrain_consistency()