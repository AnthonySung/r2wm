#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

python3 -c "
import numpy as np
import torch
arr = np.array([[True, False], [True, True]])
print('numpy:', arr, arr.dtype)
t = torch.from_numpy(arr)
print('torch:', t, t.dtype)
print('torch any:', t.any().item())

# 测试 is_first 测试场景
import numpy as np
from training.replay_buffer import ReplayBuffer
b = ReplayBuffer(capacity=1000, obs_dim=48, action_dim=12)
for i in range(200):
    is_start_of_ep = (i % 100 == 0)
    b.add(obs=np.zeros(48, dtype=np.float32),
          action=np.zeros(12, dtype=np.float32),
          reward=0.0,
          next_obs=np.zeros(48, dtype=np.float32),
          done=False,
          is_first=is_start_of_ep)
print('sum is_first:', b._is_first.sum())
batch = b.sample(batch_size=4, seq_length=10)
print('batch is_first:', batch['is_first'])
print('sum:', batch['is_first'].sum().item())
"