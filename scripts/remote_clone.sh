#!/bin/bash
# 在远程服务器上 git clone r2wm
# 用法: ssh user@server "GITHUB_TOKEN=ghp_xxx bash remote_clone.sh"
# 注意: 不要把含 token 的脚本上传到 git!

cd /home 2>/dev/null || cd /root

if [ -z "$GITHUB_TOKEN" ]; then
    echo "错误: 请设置 GITHUB_TOKEN 环境变量"
    exit 1
fi

git clone https://AnthonySung:${GITHUB_TOKEN}@github.com/AnthonySung/r2wm.git 2>&1 | tail -5
cd r2wm && git pull 2>&1 | tail -3
