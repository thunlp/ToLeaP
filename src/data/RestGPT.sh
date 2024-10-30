git clone https://github.com/Yifan-Song793/RestGPT.git
cd RestGPT

shopt -s extglob
rm -rf !(datasets)

mv datasets ../../data/RestGPT_data

cd ..
rm -rf RestGPT

echo "The RestGPT_data has been downloaded..."

python RestGPT_sharegpt.py

echo "The RestGPT_data has been transformed into sharegpt format..."