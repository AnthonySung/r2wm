#!/bin/bash
# 验证 /root/miniconda3 的完整环境

source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

echo "=== Python ==="
python --version
which python

echo ""
echo "=== Isaac Gym ==="
python -c "import isaacgym; print('isaacgym:', isaacgym.__file__)"

echo ""
echo "=== PyTorch ==="
python -c "
import torch
print('torch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
"

echo ""
echo "=== WMP ==="
python -c "
from legged_gym.envs.a1.a1_amp_config import A1AMPCfg
cfg = A1AMPCfg()
print('A1AMPCfg: num_envs=', cfg.env.num_envs)
print('A1AMPCfg: num_observations=', cfg.env.num_observations)
"

echo ""
echo "=== LeggedRobot 导入 ==="
python -c "from legged_gym.envs.base.legged_robot import LeggedRobot; print('LeggedRobot OK')"