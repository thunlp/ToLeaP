
#dowload RoTBench
git clone https://github.com/Junjie-Ye/RoTBench.git

mkdir RoTBenchData

cd RoTBench 

#Retain Data and Delete others
shopt -s extglob
rm -rf !(Data)

cd Data

#Run Python file
python ../../RoTBench_sharegpt.py


#Delete RoTBench
cd ../../
rm -rf RoTBench
