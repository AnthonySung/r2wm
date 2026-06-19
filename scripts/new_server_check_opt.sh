#!/bin/bash
echo "=== /opt/miniconda3_r2wmp 内容 ==="
ls -la /opt/miniconda3_r2wmp/ 2>&1 | head -10
echo "---"
echo "=== python version ==="
/opt/miniconda3_r2wmp/bin/python --version 2>&1
echo "---"
echo "=== pip list ==="
/opt/miniconda3_r2wmp/bin/pip list 2>/dev/null | head -25
echo "---"
echo "=== 检查 rlgpu conda env ==="
cat /home/WMP/rlgpu_conda_env.yml 2>&1 | head -30