#!/bin/bash
# 调试:看 asset 路径

source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

cd /root/r2wm

# 检查 URDF 文件
ls /home/WMP/resources/robots/a1/ 2>&1 | head -10
echo "---"

# 看 WMP 怎么加载 asset
grep -rn "asset.file" /home/WMP/legged_gym/envs/a1/a1_amp_config.py 2>&1 | head -5
echo "---"

# A1AMP 的 asset 配置
sed -n '60,80p' /home/WMP/legged_gym/envs/a1/a1_amp_config.py
echo "---"

# 检查 URDF 完整路径
ls /home/WMP/resources/robots/a1/urdf/ 2>&1 | head -5