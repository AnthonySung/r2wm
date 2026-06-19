#!/bin/bash
# 详细检查 WMP 安装方式

echo "===rsl_rl 目录==="
ls /home/WMP/rsl_rl/ | head -10
echo ""
echo "===legged_gym 目录==="
ls /home/WMP/legged_gym/ | head -10

echo ""
echo "===找 setup.py==="
find /home/WMP -name "setup.py" -maxdepth 3 2>/dev/null
find /home/WMP -name "pyproject.toml" -maxdepth 3 2>/dev/null

echo ""
echo "===查看 README==="
head -50 /home/WMP/README.md 2>/dev/null

echo ""
echo "===requirements.txt==="
cat /home/WMP/requirements.txt 2>/dev/null | head -20