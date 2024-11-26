## 🛠️ Preparations of dataset

To download **gorllia/bfcl** dataset

```
cd BodhiAgent/src/data
./bfcl.sh
```

To download **t-eval** dataset

```
cd BodhiAgent/src/data
./teval.sh
```
**RoTBench**
Please run `bash RoTBench.sh` to download RoTBench datasets and transform them into sharegpt format. There will be 10 sharegpt format datasets stored under the folder `data/sft_data/RoTbench`. 

To download **Toollens** and deal with Datasets
```
cd ./BodhiAgent/src/data
./Toollens.sh
```

**Toolllm**
Please run `bash toolllm.sh` to download the Toolllm dataset and transform it into sharegpt format. The sharegpt format dataset `toolllm_processed.json` will be found under the folder `data/sft_data`.
