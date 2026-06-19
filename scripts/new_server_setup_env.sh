#!/bin/bash
# 在新服务器完整搭建 r2wm 环境(复用 /root/miniconda3)

set -e

# 1. 设置 PYTHONPATH 环境变量(/etc/profile.d 持久化)
if [ ! -f /etc/profile.d/r2wmp.sh ]; then
    cat > /etc/profile.d/r2wmp.sh << 'EOF'
# r2wmp 项目环境
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

# WMP 路径
export WMP_ROOT=/home/WMP
export PYTHONPATH=/home/WMP:/home/WMP/legged_gym:/home/WMP/rsl_rl:/home/WMP/isaacgym/python:$PYTHONPATH

# LEGGED_GYM_ROOT_DIR (WMP 找 URDF)
export LEGGED_GYM_ROOT_DIR=/home/WMP
EOF
    chmod +x /etc/profile.d/r2wmp.sh
    echo "✅ /etc/profile.d/r2wmp.sh 创建"
fi

source /etc/profile.d/r2wmp.sh
echo "=== 当前环境 ==="
echo "Python: $(which python)"
echo "PYTHONPATH: $PYTHONPATH" | tr ':' '\n' | head -5

# 2. 测试 WMP 导入
echo ""
echo "=== 测试 WMP 导入 ==="
python -c "
from legged_gym.envs.a1.a1_amp_config import A1AMPCfg
from legged_gym.envs.base.legged_robot import LeggedRobot
print('✅ WMP 全部 OK')
"

# 3. 创建工作目录
mkdir -p /root/r2wm