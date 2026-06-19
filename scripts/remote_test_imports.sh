#!/bin/bash
# 测试 legged_gym / rsl_rl / isaacgym

source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

echo "===测试 legged_gym==="
python3 -c "
import sys
from legged_gym.envs.a1.a1_amp_config import A1AMPCfg
cfg = A1AMPCfg()
print('legged_gym: OK')
print('  num_envs:', cfg.env.num_envs)
print('  num_observations:', cfg.env.num_observations)
" 2>&1 | head -10

echo ""
echo "===测试 rsl_rl==="
python3 -c "
import rsl_rl
print('rsl_rl: OK')
print('  path:', rsl_rl.__file__)
" 2>&1 | head -5

echo ""
echo "===测试 Isaac Gym==="
python3 -c "
import isaacgym
print('isaacgym: OK')
print('  path:', isaacgym.__file__)
" 2>&1 | head -5

echo ""
echo "===测试 LeggedRobot==="
python3 -c "
from legged_gym.envs.base.legged_robot import LeggedRobot
print('LeggedRobot: OK')
print('  path:', LeggedRobot.__module__)
" 2>&1 | head -5