# r2wmp 安装指南

> 本文档说明如何安装 r2wmp 项目的所有依赖。

---

## 1. 系统要求

| 组件 | 最低 | 推荐 |
|------|------|------|
| **操作系统** | Windows 10/11 (需 WSL2) | Linux (Ubuntu 20.04+) |
| **GPU** | NVIDIA RTX 3060 (12GB) | NVIDIA A100 (40GB) |
| **CUDA** | 11.7 | 11.8 |
| **Python** | 3.8 | 3.9 |
| **显存** | 12GB (4096 env 训练) | 24GB+ |

⚠️ **Isaac Gym 不支持 Windows 原生**,必须在 **WSL2** 或 **Linux** 下运行。

---

## 2. 安装步骤(按顺序)

### 步骤 1: 安装 CUDA + PyTorch

```bash
# 验证 CUDA
nvidia-smi  # 应该看到 CUDA 版本 >= 11.7

# 安装 PyTorch(以 CUDA 11.8 为例)
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118
```

### 步骤 2: 安装 Isaac Gym Preview 4

```bash
# 下载 Isaac Gym
# https://developer.nvidia.com/isaac-gym

# 解压
tar -xvf isaacgym_preview4.tar.gz
cd isaacgym_preview4/python

# 安装
pip install -e .

# 验证
cd ../examples
python joint_monkey.py  # 应该能看到猴子
```

### 步骤 3: 安装 WMP 依赖

```bash
# 进入 WMP 目录
cd D:/songay/sim2real/WMP  # 或在 WSL2 中的对应路径

# 安装 rsl_rl
cd rsl_rl
pip install -e .
cd ..

# 安装 legged_gym
cd legged_gym
pip install -e .
cd ..
```

**注意**:WMP 的 `setup.py` 可能不完整,如果报错,手动安装依赖:
```bash
pip install numpy torch torchvision pyyaml isaacgym
```

### 步骤 4: 安装 r2wmp 依赖

```bash
cd D:/songay/sim2real/r2wmp

# 基础依赖
pip install numpy pyyaml matplotlib tensorboard

# 可选(用于单元测试)
pip install pytest
```

---

## 3. 验证安装

### 验证 Isaac Gym

```bash
python -c "
from isaacgym import gymapi
print('Isaac Gym imported successfully')
"
```

### 验证 WMP

```bash
python -c "
import sys
sys.path.append('D:/songay/sim2real/WMP')
from legged_gym.envs.base.legged_robot import LeggedRobot
print('WMP LeggedRobot imported successfully')
"
```

### 验证 A1AMP

```bash
python -c "
import sys
sys.path.append('D:/songay/sim2real/WMP')
from legged_gym.envs.a1.a1_amp_config import A1AMPCfg
cfg = A1AMPCfg()
print(f'A1AMPCfg loaded: num_envs={cfg.env.num_envs}, num_obs={cfg.env.num_observations}')
"
```

### 验证 r2wmp

```bash
python tests/test_mock.py
```

期望输出:**全部测试通过** ✅

---

## 4. WSL2 配置(Windows 用户)

### 安装 WSL2

```powershell
# PowerShell(管理员)
wsl --install
wsl --set-default-version 2

# 安装 Ubuntu
wsl --install -d Ubuntu-22.04
```

### 在 WSL2 中配置 CUDA

```bash
# 参考 https://docs.nvidia.com/cuda/wsl-user-guide/index.html
# 在 WSL2 中安装 NVIDIA 驱动 + CUDA toolkit

# 验证
nvidia-smi
```

### 路径映射

WSL2 中访问 Windows 盘符:
```bash
/mnt/d/songay/sim2real/r2wmp
/mnt/d/songay/sim2real/WMP
```

---

## 5. 常见安装问题

### Q1: `libpython3.8.so.1.0` 找不到

```bash
# Ubuntu
sudo apt install libpython3.8

# Arch
sudo pacman -S python
```

### Q2: `torch.cuda.is_available()` 返回 False

```bash
# 检查 PyTorch CUDA 版本匹配
python -c "import torch; print(torch.version.cuda)"

# 重新安装匹配版本
pip uninstall torch torchvision
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Q3: Isaac Gym `pyglet.gl` 错误

```bash
# 这是因为 pyglet 版本不兼容
pip install pyglet==1.5.27
```

### Q4: 4096 env OOM(显存不够)

```bash
# 修改 train_stage1.py
python scripts/train_stage1.py --num_envs 2048  # 降低并行数
```

---

## 6. 检查清单

- [ ] `nvidia-smi` 显示 GPU
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` 返回 True
- [ ] Isaac Gym `joint_monkey.py` 能跑
- [ ] WMP `LeggedRobot` 能 import
- [ ] r2wmp `tests/test_mock.py` 全部通过
- [ ] GPU 显存 ≥ 12GB

---

## 7. 下一步

环境装好后:
1. 跑 `tests/test_mock.py` 验证 r2wmp 逻辑
2. 跑 `scripts/train_stage1.py --total_steps 1000` 验证训练流程
3. 查看 `docs/troubleshooting.md` 解决可能的问题

---

**文档版本**: 1.0  
**更新日期**: 2026-06-18  
**适用项目**: r2wmp v0.1