#!/bin/bash
# 测试 A1 环境 + 设置 LEGGED_GYM_ROOT_DIR

source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

export LEGGED_GYM_ROOT_DIR=/home/WMP

cd /root/r2wm

timeout 90 python3 -u -c "
import sys
sys.path.insert(0, '/home/WMP')

from legged_gym.envs.a1.a1_amp_config import A1AMPCfg
from legged_gym.envs.base.legged_robot import LeggedRobot
from isaacgym import gymapi

import torch

print('=== Step 1: 创建 LeggedRobot ===')
cfg = A1AMPCfg()
cfg.env.num_envs = 4
cfg.terrain.measure_heights = False  # 简化

sim_params = gymapi.SimParams()
sim_params.dt = 0.005
sim_params.use_gpu_pipeline = True
sim_params.physx.use_gpu = True
sim_params.substeps = 1
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

env = LeggedRobot(cfg, sim_params, gymapi.SIM_PHYSX, 'cuda:0', headless=True)
print('  LeggedRobot 创建成功!')

print('=== Step 2: reset ===')
obs, priv_obs = env.reset()
print(f'  obs shape: {obs.shape}')
print(f'  priv_obs shape: {priv_obs.shape}')
print(f'  obs[0, :8]: {obs[0, :8]}')

print('=== Step 3: step ===')
import torch
action = torch.zeros(4, 12, device='cuda:0')
result = env.step(action)
print(f'  step 返回 {len(result)} 元组')
print(f'  obs shape: {result[0].shape}')
print(f'  reward shape: {result[2].shape}')
print(f'  done shape: {result[3].shape}')
print('✅ A1 环境完全工作!')
" 2>&1 | tail -40
echo "exit: $?"