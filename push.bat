@echo off
REM ===============================================
REM  r2wm 日常 push 脚本(Windows)
REM ===============================================
REM  用法:
REM  1. 在编辑器里改完代码
REM  2. 双击运行 push.bat
REM  3. 输入 commit message
REM  4. 自动 git add + commit + push
REM ===============================================

cd /d %~dp0

echo === git status ===
git status --short
echo.

REM 如果没有改动,直接退出
git diff --cached --quiet 2>nul
if not errorlevel 1 (
    echo No staged changes. Run 'git add' first.
    pause
    exit /b 0
)

REM 让用户输入 commit message
set /p MSG="*** Commit message: "

if "%MSG%"=="" (
    echo Error: commit message cannot be empty
    pause
    exit /b 1
)

echo.
echo === git add ===
git add .

echo === git commit ===
git commit -m "%MSG%"

if errorlevel 1 (
    echo.
    echo Commit failed. No changes to commit?
    pause
    exit /b 1
)

echo.
echo === git push ===
git push

if errorlevel 1 (
    echo.
    echo Push failed. 可能原因:
    echo 1. Token 失效 → 撤销后重新生成,需要 git credential-manager 重新认证
    echo 2. 网络问题
    echo 3. 远程 main 分支有冲突 → git pull --rebase 后再 push
) else (
    echo.
    echo === Push successful! ===
)

pause