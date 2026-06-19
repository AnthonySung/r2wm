#!/bin/bash
# 用 CPU pipeline 创建 A1 环境(GPU pipeline 有 bug)

source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

export LEGGED_GYM_ROOT_DIR=/home/WMP

cd /root/r2wm

echo "=== A1 环境测试 (CPU pipeline) ==="
timeout 90 python3 -u -c "
import sys
sys.path.insert(0, '/home/WMP')

from legged_gym.envs.a1.a1_amp_config import A1AMPCfg
from legged_gym.envs.base.legged_robot import LeggedRobot
from isaacgym import gymapi
import torch

cfg = A1AMPCfg()
cfg.env.num_envs = 4
cfg.terrain.mesh_type = 'plane'
cfg.terrain.measure_heights = False

# CPU pipeline(更稳定)
sim_params = gymapi.SimParams()
sim_params.dt = 0.005
sim_params.use_gpu_pipeline = False
sim_params.physx.use_gpu = False
sim_params.substeps = 1
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

try:
    env = LeggedRobot(cfg, sim_params, gymapi.SIM_PHYSX, 'cuda:0', headless=True)
    print('✅ LeggedRobot 创建成功')

    obs, priv_obs = env.reset()
    print(f'obs.shape: {obs.shape}')
    print(f'obs[0, :8]: {obs[0, :8]}')

    # Step
    action = torch.zeros(4, 12)
    result = env.step(action)
    print(f'step 返回 {len(result)} 元组')
    print(f'obs.shape after step: {result[0].shape}')
    print(f'reward.mean: {result[2].mean().item():.3f}')
    print(f'done.sum: {result[3].sum().item()}')

    print('✅ A1 环境 step 工作!')
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f'❌ 失败: {e}')
" 2>&1 | tail -30
echo "exit: $?"