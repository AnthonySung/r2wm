#!/bin/bash
# 在本地 Windows 机器上跑,把文件 base64 编码后通过 ssh 上传到远程
# 用法: ssh user@server "bash remote_scp_fix.sh"
# 注意: 这个脚本不包含 token,但需要调用方提供 base64 内容

if [ -z "$FILE_CONTENT" ]; then
    echo "用法: FILE_CONTENT=\"$(base64 -w 0 file)\" ssh user@server \"bash remote_scp_fix.sh\""
    exit 1
fi

REMOTE_PATH=${REMOTE_PATH:-/root/r2wm/}
echo "$FILE_CONTENT" | base64 -d > "$REMOTE_PATH"
echo "✅ 文件已写入 $REMOTE_PATH"
