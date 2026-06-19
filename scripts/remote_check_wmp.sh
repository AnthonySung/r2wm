#!/bin/bash
# 检查 WMP 实际状态

echo "===查找 WMP==="
find / -name "legged_gym" -type d 2>/dev/null | head -5
find / -name "rsl_rl" -type d 2>/dev/null | head -5
find / -name "a1_amp_config.py" 2>/dev/null | head -3
find / -name "LeggedRobot.py" -o -name "legged_robot.py" 2>/dev/null | head -3

echo ""
echo "===检查 /home==="
ls /home/ 2>&1
echo ""
echo "===检查 /home/WMP==="
ls /home/WMP/ 2>&1 | head -20

echo ""
echo "===检查 /home/WMP 是否是完整仓库==="
ls /home/WMP/legged_gym/ 2>&1 | head -10
ls /home/WMP/rsl_rl/ 2>&1 | head -10

echo ""
echo "===pip list 看是否安装==="
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
pip list 2>&1 | grep -iE "legged|rsl|gym" | head -10