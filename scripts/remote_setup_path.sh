#!/bin/bash
# 用 PYTHONPATH 安装 legged_gym 和 rsl_rl(更直接)

source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

# 创建 sitecustomize.py 让 Python 自动加载
mkdir -p /root/miniconda3/lib/python3.8/site-packages

# 添加 .pth 文件(Python 自动加载)
cat > /root/miniconda3/lib/python3.8/site-packages/_r2wmp.pth << 'EOF'
/home/WMP
/home/WMP/legged_gym
/home/WMP/rsl_rl
EOF

echo "===验证==="
python3 -c "import sys; print('PYTHONPATH:'); [print(' ', p) for p in sys.path if 'home' in p or 'WMP' in p]"

echo ""
echo "===测试 legged_gym==="
python3 -c "
import sys
sys.path.insert(0, '/home/WMP')
from legged_gym.envs.a1.a1_amp_config import A1AMPCfg
cfg = A1AMPCfg()
print('legged_gym: OK')
print('  num_envs:', cfg.env.num_envs)
print('  num_observations:', cfg.env.num_observations)
" 2>&1 | head -20

echo ""
echo "===测试 rsl_rl==="
python3 -c "
import sys
sys.path.insert(0, '/home/WMP')
import rsl_rl
print('rsl_rl: OK')
" 2>&1 | head -5

echo ""
echo "===测试 Isaac Gym==="
python3 -c "
import isaacgym
print('isaacgym: OK')
" 2>&1 | head -5