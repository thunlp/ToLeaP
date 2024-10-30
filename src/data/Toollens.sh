
#dowload toollens
git clone https://github.com/quchangle1/COLT.git

mkdir ToolLensData

cd COLT 

#Retain Data and Delete others
shopt -s extglob
rm -rf !(datasets)

cd datasets
echo "Start transfer into sharegpt format!!! COLT datasets has a paired process, thus needs to take a while, please wait for a minute!!!"

#Run Python file
python ../../Toollens_sharegpt.py


#Delete RoTBench
cd ../../
rm -rf COLT

#move data to sft_data
mkdir -p sft_data
cp -r ToolLensData RoTBench
rm -rf ToolLensData
