#!/bin/bash
# 拉 r2wm 代码 + 跑 mock 测试

source /etc/profile.d/r2wmp.sh

cd /root

# 1. 克隆(用 GITHUB_TOKEN 或 ssh)
echo "=== 拉取 r2wm ==="
if [ -d "r2wm" ]; then
    cd r2wm
    git pull 2>&1 | tail -5
else
    if [ -n "$GITHUB_TOKEN" ]; then
        git clone https://AnthonySung:${GITHUB_TOKEN}@github.com/AnthonySung/r2wm.git 2>&1 | tail -5
    else
        git clone https://github.com/AnthonySung/r2wm.git 2>&1 | tail -5
    fi
    cd r2wm
fi

echo ""
echo "=== Git 状态 ==="
git log --oneline | head -5

# 2. 跑 mock 测试
echo ""
echo "=== 跑 mock 测试 ==="
python tests/test_mock.py 2>&1 | tail -60