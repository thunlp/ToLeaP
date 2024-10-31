git clone https://github.com/ShishirPatil/gorilla.git
cd gorilla || exit

shopt -s extglob
rm -rf !(berkeley-function-call-leaderboard)
cd berkeley-function-call-leaderboard || exit
rm -rf !(data)

mv data ../../bfcl


cd ..
cd ..

rm -rf gorilla

mkdir -p sft_data

echo "The bfcl has been downloaded..."

python bfcl_sharegpt.py all

echo "The bfcl has been transformed into sharegpt format..."
