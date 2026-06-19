#!/bin/bash
# 远程环境验证

echo "===Python==="
python3 --version
which python3

echo ""
echo "===PyTorch==="
python3 -c "import torch; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo ""
echo "===NumPy==="
python3 -c "import numpy; print('numpy:', numpy.__version__)"

echo ""
echo "===PyYAML==="
python3 -c "import yaml; print('yaml:', yaml.__version__) 2>/dev/null || echo 'PyYAML NOT INSTALLED'"

echo ""
echo "===进入项目==="
cd /root/r2wm 2>/dev/null && pwd && ls

echo ""
echo "===运行 mock 测试==="
python3 tests/test_mock.py 2>&1 | tail -40