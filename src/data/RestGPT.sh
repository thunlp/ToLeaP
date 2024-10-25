git clone https://github.com/HowieHwong/RestGPT.git
cd RestGPT

shopt -s extglob
rm -rf !(datasets)

mv data ../../BodhiAgent/src/RestGPT

cd ..
rm -rf RestGPT