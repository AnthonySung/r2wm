#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/r2wm

python3 -c "
import torch
from training.ac_loss import compute_lambda_return
B, T = 4, 5
rewards = torch.randn(B, T)
values = torch.randn(B, T + 1)
continues = torch.ones(B, T)

print('rewards shape:', rewards.shape)
print('values shape:', values.shape)
print('continues shape:', continues.shape)

# 调函数前先看 shapes
print('values[:, -1]:', values[:, -1].shape)
print('values[:, 4]:', values[:, 4].shape)
print('values[:, 5]:', values[:, 5].shape)

try:
    result = compute_lambda_return(rewards, values, continues, 0.95, 0.997)
    print('OK:', result.shape)
except Exception as e:
    print('Error:', e)
"