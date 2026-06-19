#!/bin/bash
# 最小测试 - 看 Isaac Gym 能否运行任何东西

source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

export LEGGED_GYM_ROOT_DIR=/home/WMP

cd /root/r2wm

echo "=== 测试 1: Isaac Gym 最小 demo ==="
timeout 60 python3 -u -c "
from isaacgym import gymapi, gymutil, gymtorch
import torch

# 创建最小 sim
gym = gymapi.acquire_gym()

# 创建 sim params
sim_params = gymapi.SimParams()
sim_params.dt = 0.01
sim_params.use_gpu_pipeline = False  # CPU pipeline
sim_params.physx.use_gpu = False
sim_params.substeps = 1
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

try:
    sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, sim_params)
    print(f'✅ Sim 创建: {sim}')
    print('✅ Isaac Gym 基础功能 OK')
except Exception as e:
    print(f'❌ Sim 创建失败: {e}')
" 2>&1 | tail -15
echo "exit: $?"