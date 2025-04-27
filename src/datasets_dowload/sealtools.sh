#!/bin/bash

# 克隆仓库
git clone https://github.com/fairyshine/Seal-Tools.git

# 进入仓库目录
cd Seal-Tools || { echo "Failed to enter Seal-Tools directory"; exit 1; }

# 创建一个临时目录保存 dataset_for_finetune 目录
mkdir -p ../temp_dir

# 复制 dataset_for_finetune 到临时目录
cp -r Seal-Tools_Dataset/dataset_for_finetune/* ../temp_dir/

# 返回上一级目录
cd ..

# 删除整个仓库
rm -rf Seal-Tools

# 创建 Seal-Tools 文件夹
mkdir Seal-Tools

# 移动 dataset_for_finetune 目录下的所有文件到 Seal-Tools 文件夹
mv temp_dir/* Seal-Tools/

# 删除临时目录
rm -rf temp_dir

echo "Done: data is saved under 'Seal-Tools/'"
echo "Seal-Tools Data Done."