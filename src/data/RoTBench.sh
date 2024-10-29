
#dowload RoTBench
git clone https://github.com/Junjie-Ye/RoTBench.git
cd RoTBench 

#Retain Data and Delete others
shopt -s extglob
rm -rf !(Data)

#Move Data
mv Data ../data/RoTBench/

#Delete RoTBench
cd ..
rm -rf RoTBench
