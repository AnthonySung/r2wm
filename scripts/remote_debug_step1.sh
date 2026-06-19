#!/bin/bash
# 逐步调试:加载 URDF 看是否崩溃

source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

export LEGGED_GYM_ROOT_DIR=/home/WMP

cd /root/r2wm

echo "=== 步骤 1: 加载 URDF ==="
timeout 30 python3 -u -c "
from isaacgym import gymapi, gymutil, gymtorch

gym = gymapi.acquire_gym()

# Sim params (CPU)
sim_params = gymapi.SimParams()
sim_params.dt = 0.005
sim_params.use_gpu_pipeline = False
sim_params.physx.use_gpu = False
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

sim = gym.create_sim(0, -1, gymapi.SIM_PHYSX, sim_params)
print(f'  Sim: {sim}')

# 加载 A1 URDF
asset_root = '/home/WMP/resources/robots/a1'
urdf_file = 'urdf/a1.urdf'
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False
asset_options.flip_visual_attachments = False
asset_options.collapse_fixed_joints = True

print(f'  加载 URDF: {asset_root}/{urdf_file}')
asset = gym.load_asset(sim, asset_root, urdf_file, asset_options)
print(f'  ✅ Asset 加载: {asset}')
" 2>&1 | tail -15
echo "exit: $?"