git clone https://huggingface.co/datasets/lovesnowbest/T-Eval
cd T-Eval || exit

shopt -s extglob


rm -rf !(data)

mv data ../../data/teval

cd ..

rm -rf T-Eval
