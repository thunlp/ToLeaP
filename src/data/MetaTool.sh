sed -i 's/\r$//' MetaTool.sh
git clone https://github.com/HowieHwong/MetaTool.git
cd MetaTool

shopt -s extglob
rm -rf !(dataset)

mv dataset ../../data/MetaTool_data

cd ..
rm -rf MetaTool
