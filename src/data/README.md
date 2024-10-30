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

To download **RoTBenc** and deal with Datasets
```
cd ./BodhiAgent/src/data
./RoTBench.sh
```

To download **Toollens** and deal with Datasets
```
cd ./BodhiAgent/src/data
./Toollens.sh
```

To download the **MetaTool** dataset and transform it into sharegpt format:
```
bash MetaTool.sh
```
The sharegpt format dataset `multi_tool_query_golden.json` will be found under the folder `data/sft_data`.

To download **RestGPT** datasets and transform them into sharegpt format:
```
bash MetaTool.sh
```
The sharegpt format datasets `spotify_processed.json` and `tmdb_processed.json` will be found under the folder `data/sft_data`.

To download the **Toolllm** dataset and transform it into sharegpt format:
```
bash toolllm.sh
```
The sharegpt format dataset `toolllm_processed.json` will be found under the folder `data/sft_data`.
