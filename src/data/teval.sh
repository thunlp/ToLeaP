git clone https://huggingface.co/datasets/lovesnowbest/T-Eval
cd T-Eval || exit

shopt -s extglob


rm -rf !(data)

mv data ../../BodhiAgent/src/teval

cd ..

rm -rf T-Eval
