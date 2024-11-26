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

To download **RoTBench** datasets and transform them into sharegpt format:
```
cd BodhiAgent/src/data
bash RoTBench.sh
```
There will be 10 sharegpt format datasets stored under teh folder `data/sft_data/RoTbench`. 

To download **Toollens** and deal with Datasets
```
cd ./BodhiAgent/src/data
./Toollens.sh
```

To download the **MetaTool** dataset and transform it into sharegpt format:
```
bash MetaTool.sh
```
The sharegpt format dataset `MetaTool_processed.json` will be found under the folder `data/sft_data`.

To download **RestGPT** datasets and transform them into sharegpt format:
```
bash RestGPT.sh
```
The sharegpt format datasets `spotify_processed.json` and `tmdb_processed.json` will be found under the folder `data/sft_data`.

To download the **Toolllm** dataset and transform it into sharegpt format:
```
bash toolllm.sh
```
The sharegpt format dataset `toolllm_processed.json` will be found under the folder `data/sft_data`.
