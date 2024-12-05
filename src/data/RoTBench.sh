
#dowload RoTBench
git clone https://github.com/Junjie-Ye/RoTBench.git
mkdir -p sft_data
mkdir -p sft_data/RoTBench
cd RoTBench 
#Retain Data and Delete others
shopt -s extglob
rm -rf !(Data)
cd Data

#Run Python file
python ../../RoTBench_sharegpt.py

#Delete RoTBench
cd ..
cd ..
rm -rf RoTBench


