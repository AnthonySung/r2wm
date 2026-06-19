#!/bin/bash
# 检查新服务器现有环境(不动它,只读)

echo "=== 当前 root 目录 ==="
ls -la /root/ | head -20

echo ""
echo "=== 现有 Python 环境 ==="
which python python3
python3 --version 2>&1

echo ""
echo "=== 现有 conda ==="
ls /root/miniconda3/ 2>/dev/null | head -5
ls /opt/conda/ 2>/dev/null | head -5

echo ""
echo "=== 现有 WMP / Isaac Gym ==="
ls /home/WMP/ 2>/dev/null | head -10
ls /home/ 2>/dev/null

echo ""
echo "=== GPU 状态 ==="
nvidia-smi 2>&1 | head -15

echo ""
echo "=== 已安装包 ==="
pip list 2>&1 | head -20 || echo "pip not available"

echo ""
echo "=== 现有 conda envs ==="
conda env list 2>&1 || echo "no conda"