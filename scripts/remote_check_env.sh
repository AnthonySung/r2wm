#!/bin/bash
# 检查 Python 环境

echo "===Python==="
which python3
python3 --version

echo ""
echo "===pip==="
which pip
pip --version 2>&1

echo ""
echo "===conda==="
which conda 2>/dev/null && conda --version 2>/dev/null || echo "no conda"
ls /opt/conda/bin/conda 2>/dev/null && echo "找到 /opt/conda"

echo ""
echo "===CUDA==="
nvcc --version 2>&1 | tail -3 || echo "no nvcc"

echo ""
echo "===搜索已安装 Python 包==="
echo "torch 位置:"
find / -name "torch" -type d 2>/dev/null | head -3 || echo "未找到"

echo ""
echo "numpy 位置:"
find / -name "numpy" -type d 2>/dev/null | head -3 || echo "未找到"

echo ""
echo "===pip list 顶层==="
pip list 2>&1 | head -20