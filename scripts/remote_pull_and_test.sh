#!/bin/bash
# 同步修复后的代码到远程,跑测试

cd /root/r2wm

echo "===同步修复==="
# 从 GitHub pull 最新代码
git pull origin main 2>&1 | tail -3

echo ""
echo "===跑 mock 测试==="
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
python3 tests/test_mock.py 2>&1 | tail -60