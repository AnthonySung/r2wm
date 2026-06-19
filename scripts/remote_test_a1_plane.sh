#!/bin/bash
# 用 plane terrain 测试(避免 trimesh 卡住)

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

print('=== A1 环境测试 (plane terrain) ===')

cfg = A1AMPCfg()
cfg.env.num_envs = 4
cfg.terrain.mesh_type = 'plane'      # ← 改成 plane(简单)
cfg.terrain.measure_heights = False

sim_params = gymapi.SimParams()
sim_params.dt = 0.005
sim_params.use_gpu_pipeline = True
sim_params.physx.use_gpu = True
sim_params.substeps = 1
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

try:
    env = LeggedRobot(cfg, sim_params, gymapi.SIM_PHYSX, 'cuda:0', headless=True)
    print('✅ LeggedRobot 创建成功 (plane + GPU pipeline)')

    obs, priv_obs = env.reset()
    print(f'reset obs.shape: {obs.shape}')

    import torch
    action = torch.zeros(4, 12, device='cuda:0')
    result = env.step(action)
    print(f'step obs.shape: {result[0].shape}')
    print(f'step reward mean: {result[2].mean().item():.3f}')
    print('✅ A1 环境完全工作!')

except Exception as e:
    print(f'❌ 失败: {e}')
    import traceback
    traceback.print_exc()
" 2>&1 | tail -40
echo "exit: $?"