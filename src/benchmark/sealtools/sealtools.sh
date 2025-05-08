#!/bin/bash

git clone https://github.com/fairyshine/Seal-Tools.git

cd Seal-Tools || { echo "Failed to enter Seal-Tools directory"; exit 1; }

# Create a temporary directory to store the `dataset_for_finetune` directory
mkdir -p ../temp_dir

cp -r Seal-Tools_Dataset/dataset_for_finetune/* ../temp_dir/

cd ..

rm -rf Seal-Tools

mv temp_dir/* ../../../data/sealtools

rm -rf temp_dir

echo "Seal-Tools Data Done."