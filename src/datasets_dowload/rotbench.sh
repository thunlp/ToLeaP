#!/bin/bash

# 克隆仓库
git clone https://github.com/Junjie-Ye/RoTBench.git

# 进入仓库目录
cd RoTBench || { echo "Failed to enter RoTBench directory"; exit 1; }

# 删除除了 data 文件夹以外的所有内容
find . -mindepth 1 -maxdepth 1 ! -name 'Data' -exec rm -rf {} \;


# 移动 Data 目录下的所有文件夹到当前目录
mv Data/* .

# 删除空的 Data 文件夹
rmdir Data

echo "RoTBench Data Done."

cd ..
