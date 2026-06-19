#!/bin/bash
echo "=== 查找 conda ==="
ls /root/miniconda3/bin/python 2>/dev/null && echo "找到 /root/miniconda3/bin/python"
ls /opt/miniconda3/bin/python 2>/dev/null && echo "找到 /opt/miniconda3/bin/python"
find / -maxdepth 5 -name "conda" -type f 2>/dev/null | head -5

echo ""
echo "=== /home/WMP 确认 ==="
ls -la /home/WMP/ 2>&1 | head -5
ls /home/WMP/isaacgym/python/ 2>&1 | head -5
ls /home/WMP/isaacgym4/python/ 2>&1 | head -5

echo ""
echo "=== 检查 conda envs ==="
[ -f /root/miniconda3/bin/conda ] && /root/miniconda3/bin/conda env list 2>&1 || echo "no conda found"