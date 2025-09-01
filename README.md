<div align= "center">
    <h1> 🛠️ToLeaP：A Tool Learning Platform🛠️</h1>
</div>

<p align="center">
  <a href="#set-up-the-environment">Enviroment</a> •
  <a href="#download-the-data">Data</a> •
  <a href="#evaluate">Evaluation</a>
</p>

We investigate the tool learning ability of 41 prevalent LLMs by reproducing 33 benchmarks and enabling one-click evaluation for seven of them, forming a Tool Learning Platform named ToLeaP. Our motivation is to deepen insights into current work and thus facilitate foresight into future directions in the tool learning domain. ToLeaP comprises 8 out of 33 benchmarks and possesses the functionality that takes an LLM as input and outputs the values of all 64 evaluation metrics proposed by the benchmarks.

## ⚙️Set up the environment
```
conda create -n toleap python=3.10 -y && conda activate toleap
git clone https://github.com/Hytn/ToLeaP.git && cd ToLeap
pip install -e .

cd scripts
pip install vllm==0.6.5
pip install rouge_score # taskbench
pip install mmengine # teval
pip install nltk accelerate # injecagent
bash ../src/benchmark/bfcl/bfcl_setup.sh
```
**Note:** Please use **vllm==0.6.5**. If you want to test new models such as the **Qwen3** series, it is recommended to use the **Transformers** library instead of **vllm**. The latest version of **vllm** has some conflicts with our code implementation, which may affect the results of benchmarks such as **RoTBench**.


## ⬇️Download the data

First, run 
```
mkdir data && cd data
mkdir rotbench sealtools taskbench injecagent glaive stabletoolbench apibank
cd ..
```

### RoTBench
```
cd src/benchmark/rotbench
bash rotbench.sh
```

### Seal-Tools
```
cd src/benchmark/sealtools
bash sealtools.sh
```

### task-bench
```
cd src/benchmark/taskbench
python taskbench.py
```

### Glaive
```
cd src/benchmark/glaive
python glaive2sharegpt.py
```

### T-Eval
```
cd data
unzip teval.zip
rm teval.zip
```

### InjecAgent
```
cd src/benchmark/injecagent
bash injecagent.sh
```

### StableToolBench
Download the data from [this link](https://cloud.tsinghua.edu.cn/f/07ee752ad20b43ed9b0d/?dl=1), and place the files in the `data/stabletoolbench` folder.

After downloading the data, the directory structure should look like this:
```
├── /data/
│  ├── /glaive/
│  │  ├── 
│  │  └── ...
│  ├── /injecagent/
│  │  ├── attacker_cases_dh.jsonl
│  │  └── ...
├── /scripts/
│  ├── /gorilla/
│  ├── bfcl_standard.py
│  ├── ...
├── /src/
│  ├── /benchmark/
│  │  └── ...
│  ├── /cfg/
│  │  └── ...
│  ├── /utils/
│  │  └── ...
```

## 📊Evaluate
First, run:
```
mkdir results
cd results
mkdir rotbench sealtools taskbench teval injecagent glaive stabletoolbench
cd ..
```

If you want to perform one-click evaluation, run:
```
cd scripts
# bash one-click-evaluation.sh model_path is_api gpu_num batch_size input output display_name
bash one-click-evaluation.sh meta-llama/Llama-3.1-8B-Instruct false 1 256 4096 512 llama3.1

```
If you prefer to evaluate each benchmark separately, follow the instructions below. Once `{model_name}_results.json` is generated, you can easily convert the `.json` results into `.csv` format using `json2csv.py`, making it convenient for you to fill out the form.

In addition, if you want to use the debug mode, you can add the `--debug` parameter in the script. To evaluate the inference model, you need to add the `--think_mode` and `--think_special_tokens` parameters.

### RoTBench
```
cd scripts
python rotbench_eval.py --model meta-llama/Llama-3.1-8B-Instruct
```
To evaluate API models, add `--is_api` True.

### Seal-Tools
```
cd scripts
python sealtools_eval.py --model meta-llama/Llama-3.1-8B-Instruct
```
To evaluate API models, add `--is_api` True.

### task-bench
```
cd scripts
python taskbench_eval.py --model meta-llama/Llama-3.1-8B-Instruct
```
To evaluate API models, add `--is_api` True.

### BFCL

WARNING: As the official BFCL codebase changes frequently, if the following instructions do not work, please refer to [the latest official repository](https://github.com/ShishirPatil/gorilla/tree/main).

Before using BFCL for evaluation, some preparation steps are required:

1. Ensure that the model you want to evaluate is included in the handler mapping file:
`scripts/gorilla/berkeley-function-call-leaderboard/bfcl/constants/model_config.py`.
If you want to evaluate API models, set the API key:

    ```
    export OPENAI_API_KEY="your-api-key"
    ```
    To use an unofficial base URL, modify the following code in
`scripts/gorilla/berkeley-function-call-leaderboard/bfcl/model_handler/api_inference/openai.py`:
    ```
    self.client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE")
    )
    ```
    Then
    ```
    export OPENAI_API_KEY="your-api-key"
    export OPENAI_API_BASE="your-base-url"
    ```

2. To add the `--max-model-len` or `--tensor-parallel-size` parameters, modify the code around line 130 in:
`scripts/gorilla/berkeley-function-call-leaderboard/bfcl/model_handler/local_inference/base_oss_handler.py`.

3. To run the evaluation in parallel, change the `VLLM_PORT` in:
`scripts/gorilla/berkeley-function-call-leaderboard/bfcl/constants/eval_config.py`.

4. If you want to use a locally trained model, ensure the model path does not contain underscores.
Otherwise, to avoid conflicts, manually add the following code after
`model_name_escaped = model_name.replace("_", "/")`:
- In the `generate_leaderboard_csv` function in `scripts/gorilla/berkeley-function-call-leaderboard/bfcl/eval_checker/eval_runner_helper.py`.
- And also in the `runner` function in `scripts/gorilla/berkeley-function-call-leaderboard/bfcl/eval_checker/eval_runner.py`.

    ```python
    if model_name == "sft_model_merged_lora_checkpoint-20000":
        model_name_escaped = "/sft_model/merged_lora/checkpoint-20000"
    ```

5. To ensure the evaluation results are properly recorded, add the model path to:
`scripts/gorilla/berkeley-function-call-leaderboard/bfcl/constants/model_metadata.py`.
Example:
    ```python
    MODEL_METADATA_MAPPING = {
        "/path/to/sft_model/merged_lora/checkpoint-60000": [
            "",
            "",
            "",
            "",
        ],
        ...
    }
    ```

Finally, run
```
bfcl generate \
	--model meta-llama/Llama-3.1-8B-Instruct \
	--test-category parallel,multiple,simple,parallel_multiple,java,javascript,irrelevance,live,multi_turn \
    --num-threads 1
bfcl evaluate --model meta-llama/Llama-3.1-8B-Instruct
```

### Glaive
```
cd scripts
python glaive_eval.py --model meta-llama/Llama-3.1-8B-Instruct
```
To evaluate API models, add `--is_api` True.

### T-Eval
```
cd scripts
bash teval_eval.sh meta-llama/Llama-3.1-8B-Instruct Llama-3.1-8B-Instruct False 4
```
Then run 
```
python standard_teval.py ../results/teval/Llama-3.1-8B-Instruct/Llama-3.1-8B-Instruct_-1_.json
``` 
to obtain the clean results.

To evaluate API models, run:
```
bash teval_eval.sh gpt-3.5-turbo gpt-3.5-turbo True
```

### InjecAgent
```
cd scripts
export OPENAI_API_KEY="your-open-api-kei"
python injecagent_eval.py \
    --model_name meta-llama/Llama-3.1-8B-Instruct \
    --use_cach
```

### APIBank
```
cd scripts
python apibank_eval.py --model_name meta-llama/Llama-3.1-8B-Instruct
```
To evaluate API models, add `--is_api` True.

## Other Benchmarks

### SkyThought
First, install the environment dependencies with `pip install skythought`. As recommended by the original authors, use `datasets==2.21.0`; otherwise, some benchmarks may fail. Then run `bash one-click-sky.sh` to evaluate all tasks. You can customize the models and tasks in the script.

### BoolQ
Download the dataset from [GitHub](https://github.com/google-research-datasets/boolean-questions) and place `train.jsonl` in the `data/boolq` directory. Then run `boolq_eval.py` to start the evaluation.

### BIG-Bench-Hard
Download the dataset from [GitHub](https://github.com/suzgunmirac/BIG-Bench-Hard/tree/main/bbh)) and place the json files in the `data/bbh` directory. Then run `bbh_eval.py` to start the evaluation.
