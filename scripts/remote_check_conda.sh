#!/bin/bash
# 激活 conda 环境

echo "===conda 位置==="
ls /root/miniconda3/bin/ 2>&1 | head -10
echo ""

echo "===conda 信息==="
/root/miniconda3/bin/conda info 2>&1 | head -20
echo ""

echo "===conda envs==="
/root/miniconda3/bin/conda env list 2>&1
echo ""

echo "===激活 base 环境==="
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
echo "Python: $(which python3)"
echo "Version: $(python3 --version)"
echo ""

echo "===验证 torch==="
python3 -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo ""
echo "===验证其他包==="
python3 -c "import numpy; print('numpy:', numpy.__version__)"
python3 -c "import yaml; print('yaml:', yaml.__version__)" 2>&1

echo ""
echo "===Isaac Gym==="
python3 -c "import isaacgym; print('isaacgym found at:', isaacgym.__file__)" 2>&1 | head -3
echo ""

echo "===WMP / legged_gym==="
python3 -c "import sys; sys.path.append('/root/WMP'); from legged_gym.envs.a1.a1_amp_config import A1AMPCfg; cfg = A1AMPCfg(); print('A1AMP: OK, num_envs=', cfg.env.num_envs)" 2>&1 | head -5