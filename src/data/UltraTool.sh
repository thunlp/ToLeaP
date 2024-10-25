git clone https://github.com/JoeYing1019/UltraTool.git
cd UltraTool

shopt -s extglob
rm -rf !(data)

mv data ../../BodhiAgent/src/UltraTool

cd ..
rm -rf UltraTool