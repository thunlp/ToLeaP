
#dowload RoTBench
mkdir -p sft_data
git clone https://huggingface.co/datasets/liminghao1630/API-Bank

cd API-Bank


#Retain Data and Delete others
shopt -s extglob

#Run Python file
echo "Start to transfer API-BANK into Sharegpt format"
python ../APIBANK_sharegpt.py

#Delete RoTBench
cd ../
rm -rf API-Bank

#move data to sft_data
# cp -r RoTBenchData sft_data
# rm -rf RoTBenchData

