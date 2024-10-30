git clone https://github.com/ShishirPatil/gorilla.git
cd gorilla || exit

shopt -s extglob
rm -rf !(berkeley-function-call-leaderboard)
cd berkeley-function-call-leaderboard || exit
rm -rf !(data)

mv data ../../../data/bfcl_data

cd ../..
rm -rf gorilla
