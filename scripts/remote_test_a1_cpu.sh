#!/bin/bash
# 用 CPU 模式测试 A1 环境(避免 GPU pipeline crash)

source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

export LEGGED_GYM_ROOT_DIR=/home/WMP

cd /root/r2wm

timeout 120 python3 -u -c "
import sys
sys.path.insert(0, '/home/WMP')

from legged_gym.envs.a1.a1_amp_config import A1AMPCfg
from legged_gym.envs.base.legged_robot import LeggedRobot
from isaacgym import gymapi

print('=== A1 环境测试 (CPU pipeline) ===')

cfg = A1AMPCfg()
cfg.env.num_envs = 4
cfg.terrain.measure_heights = False

# 关键:用 CPU pipeline(更稳定)
sim_params = gymapi.SimParams()
sim_params.dt = 0.005
sim_params.use_gpu_pipeline = False  # ← 改成 False
sim_params.physx.use_gpu = False    # ← 改成 False
sim_params.substeps = 1
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

try:
    env = LeggedRobot(cfg, sim_params, gymapi.SIM_PHYSX, 'cuda:0', headless=True)
    print('✅ LeggedRobot 创建成功 (CPU pipeline)')

    obs, priv_obs = env.reset()
    print(f'reset obs.shape: {obs.shape}')
    print(f'reset priv_obs.shape: {priv_obs.shape}')

    import torch
    action = torch.zeros(4, 12, device='cuda:0')
    result = env.step(action)
    print(f'step 返回 {len(result)} 元组')
    print(f'  obs shape: {result[0].shape}')
    print(f'  reward: {result[2].mean().item():.3f}')
    print(f'  done: {result[3].sum().item()}')
    print('✅ A1 环境 step 工作!')

except Exception as e:
    print(f'❌ 失败: {e}')
    import traceback
    traceback.print_exc()
" 2>&1 | tail -40
echo "exit: $?"