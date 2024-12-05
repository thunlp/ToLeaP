# pip install -q gdown
# gdown "https://drive.google.com/uc?export=download&id=1XFjDxVZdUY7TXYF2yvzx3pJlS2fy78jk"
# unzip -o data.zip -d toolllm_data

cd toolllm_data
shopt -s extglob
rm -rf !(data)
cd data
mv toolllama_G123_dfs_train.json ../../toolllm_data
cd ..
rm -rf data

cd ..
python Toolllm_sharegpt.py
