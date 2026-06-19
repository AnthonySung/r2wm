#!/bin/bash
# 测试创建真实 A1 环境

source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

cd /root/r2wm

python3 -c "
import sys
sys.path.insert(0, '/home/WMP')

from legged_gym.envs.a1.a1_amp_config import A1AMPCfg
from legged_gym.envs.base.legged_robot import LeggedRobot
from isaacgym import gymapi

import torch

cfg = A1AMPCfg()
cfg.env.num_envs = 4  # 小一点先测试

sim_params = gymapi.SimParams()
sim_params.dt = 0.005
sim_params.use_gpu_pipeline = True
sim_params.physx.use_gpu = True
sim_params.substeps = 1
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

print('尝试创建 LeggedRobot...')
env = LeggedRobot(cfg, sim_params, gymapi.SIM_PHYSX, 'cuda:0', headless=True)
print('LeggedRobot 创建成功')
obs, priv_obs = env.reset()
print(f'reset obs shape: {obs.shape}')
print(f'reset priv_obs shape: {priv_obs.shape}')
print(f'obs[0, :5]: {obs[0, :5]}')
" 2>&1 | tail -20