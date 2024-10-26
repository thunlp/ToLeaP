wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1XFjDxVZdUY7TXYF2yvzx3pJlS2fy78jk&confirm=yes' -O data.zip
unzip data.zip
rm data.zip
mv data toolllm_data
cd toolllm_data

# Toolllama_G123_defs_train.json is the main training data for ToolLLaMA
# It does not conform to the data format of Sharegpt Format Supervised Fine Tuning Dataset
# Running raw_to-factory.py can convert it to the data format of Sharegpt Format - Supervised Fine Tuning Dataset
python ../Toolllm_sharegpt.py
