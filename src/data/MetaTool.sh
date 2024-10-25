git clone https://github.com/HowieHwong/MetaTool.git
cd MetaTool

shopt -s extglob
rm -rf !(dataset)

mv data ../../BodhiAgent/src/MetaTool

cd ..
rm -rf MetaTool
