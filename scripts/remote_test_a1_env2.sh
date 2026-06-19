#!/bin/bash
# 简化版测试 - 只看创建是否成功

source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

cd /root/r2wm

# 用 timeout 保护
timeout 120 python3 -u -c "
import sys
sys.path.insert(0, '/home/WMP')

print('Step 1: import A1AMPCfg')
from legged_gym.envs.a1.a1_amp_config import A1AMPCfg
cfg = A1AMPCfg()
print(f'  A1AMP: num_envs={cfg.env.num_envs}')

print('Step 2: import LeggedRobot')
from legged_gym.envs.base.legged_robot import LeggedRobot
print('  OK')

print('Step 3: import isaacgym')
from isaacgym import gymapi
print('  OK')

print('Step 4: Create LeggedRobot')
cfg.env.num_envs = 4
sim_params = gymapi.SimParams()
sim_params.dt = 0.005
sim_params.use_gpu_pipeline = True
sim_params.physx.use_gpu = True
sim_params.substeps = 1
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

env = LeggedRobot(cfg, sim_params, gymapi.SIM_PHYSX, 'cuda:0', headless=True)
print('  LeggedRobot created!')

print('Step 5: reset')
obs, priv_obs = env.reset()
print(f'  obs.shape: {obs.shape}')
print(f'  priv_obs.shape: {priv_obs.shape}')
print(f'  obs[0, :8]: {obs[0, :8].cpu().numpy()}')
" 2>&1 | tail -40
echo "exit: $?"