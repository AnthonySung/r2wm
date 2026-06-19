#!/bin/bash
# 跑 mock 测试

source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

cd /root/r2wm

echo "===运行 mock 测试==="
python3 tests/test_mock.py 2>&1 | tail -60