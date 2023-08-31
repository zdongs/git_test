#!/bin/bash

# 进入本地仓库目录
cd /workspace/git_test

# 添加所有更改
git add .

# 提交更改
git commit -m "Auto sync $(date)"

# 推送到远程仓库
git push origin main

echo "Auto sync completed at $(date)"

