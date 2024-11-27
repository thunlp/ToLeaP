## 🛠️ Preparations of dataset
1. **APIBANK**

   Please run `bash APIBANK.sh` to download APIBANK datasets and transform them into sharegpt format. There will be 11 sharegpt format datasets stored under the folder `data/sft_data/APIBANK`.

2. **RoTBench**

   Please run `bash RoTBench.sh` to download RoTBench datasets and transform them into sharegpt format. There will be 10 sharegpt format datasets stored under the folder `data/sft_data/RoTbench`.

3. **Toolllm**

   Please run `bash toolllm.sh` to download the Toolllm dataset and transform it into sharegpt format. The sharegpt format dataset `toolllm_processed.json` will be found under the folder `data/sft_data`.

4. **bfcl**

   Please run `bash bfcl.sh` to download the bfcl datasets and transform them into sharegpt format. There will be 9 sharegpt format datasets named `sft_bfcl*.json` stored under the folder `data/sft_data`.

5. **taskbench**

   Please run `bash taskbench.sh` to download the taskbench dataset and transform it into sharegpt format. The sharegpt format dataset `taskbench_data.json` will be found under the folder `data/sft_data`.

6. **t-eval** 

   Please run `bash teval.sh` to download the teval datasets and transform them into sharegpt format. There will be ? sharegpt format datasets named `teval*.json` stored under the folder `data/sft_data`.

 

To download **Toollens** and deal with Datasets
```
cd ./BodhiAgent/src/data
./Toollens.sh
```



