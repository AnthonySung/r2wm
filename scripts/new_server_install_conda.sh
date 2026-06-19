#!/bin/bash
# 在新服务器安装独立 miniconda(不动现有环境)
# 安装位置: /opt/miniconda3_r2wmp(避开 /root/miniconda3)

set -e

INSTALL_DIR=/opt/miniconda3_r2wmp
MINICONDA_INSTALLER=/tmp/miniconda.sh

echo "=== 检查现有环境(不动)==="
ls /root/miniconda3/ 2>/dev/null | head -3 && echo "⚠️ 现有 /root/miniconda3"
ls /opt/miniconda3_r2wmp/ 2>/dev/null && echo "✅ r2wmp conda 已存在"

echo ""
echo "=== 下载 Miniconda ==="
if [ ! -f "$MINICONDA_INSTALLER" ]; then
    # 用 Python 3.8 兼容的 miniconda
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-py38_23.11.0-2-Linux-x86_64.sh -O $MINICONDA_INSTALLER
fi
ls -la $MINICONDA_INSTALLER

echo ""
echo "=== 安装到 $INSTALL_DIR ==="
mkdir -p /opt
bash $MINICONDA_INSTALLER -b -p $INSTALL_DIR -u

echo ""
echo "=== 验证 ==="
$INSTALL_DIR/bin/conda --version
$INSTALL_DIR/bin/python --version

echo ""
echo "=== 设置环境变量脚本(永久生效)==="
cat > /etc/profile.d/r2wmp_conda.sh << 'EOF'
# r2wmp 项目 conda 环境
export R2WMP_CONDA=/opt/miniconda3_r2wmp
export PATH=$R2WMP_CONDA/bin:$PATH
# Isaac Gym 路径(已存在,直接复用)
export PYTHONPATH=/home/WMP:/home/WMP/legged_gym:/home/WMP/rsl_rl:$PYTHONPATH
# LEGGED_GYM_ROOT_DIR(WMP 找 URDF 用)
export LEGGED_GYM_ROOT_DIR=/home/WMP
EOF
chmod +x /etc/profile.d/r2wmp_conda.sh
echo "✅ /etc/profile.d/r2wmp_conda.sh 创建完成"

echo ""
echo "=== 创建 r2wmp conda 环境 ==="
source /etc/profile.d/r2wmp_conda.sh

# 创建新环境(r2wmp_env),Python 3.8
conda create -n r2wmp_env python=3.8 -y
conda activate r2wmp_env

# 验证
python --version
which python
which pip