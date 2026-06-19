#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/r2wm

# 直接调用 test 5
python3 -c "
import sys
sys.path.insert(0, 'tests')
import test_mock
test_mock.test_replay_buffer()
print('Test 5 单独通过!')
" 2>&1 | tail -10