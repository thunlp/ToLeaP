git clone https://huggingface.co/datasets/lovesnowbest/T-Eval
cd T-Eval || exit

shopt -s extglob


rm -rf !(data)

mv data ../teval

cd ..

rm -rf T-Eval

mkdir -p sft_data

echo "The T-eval has been downloaded..."

python teval_sharegpt.py all

echo "The Teval has been transformed into sharegpt format..."