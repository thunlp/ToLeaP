git clone https://https://github.com/Yifan-Song793/RestGPT.git
cd RestGPT

shopt -s extglob
rm -rf !(datasets)

mv data ../../BodhiAgent/src/RestGPT

cd ..
rm -rf RestGPT
