git clone https://github.com/HowieHwong/MetaTool.git
mkdir MetaTool_data
cd MetaTool

shopt -s extglob
cd dataset
cd data

mv multi_tool_query_golden.json ../../../../data/MetaTool_data

cd ../../..
rm -rf MetaTool

echo "The MetaTool_data has been downloaded..."

python MetaTool_sharegpt.py

echo "The MetaTool_data has been transformed into sharegpt format..."
