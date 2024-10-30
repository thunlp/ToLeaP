sed -i 's/\r$//' RestGPT.sh
git clone https://github.com/Yifan-Song793/RestGPT.git
cd RestGPT

shopt -s extglob
rm -rf !(datasets)

mv datasets ../../data/RestGPT_data

cd ..
rm -rf RestGPT
