## 🛠️ Evaluation Method

### Configuring the Evaluation Method
- For **offline vllm batch inference**, set the default values in `cfg/config.yml`.
- For **API models**, set the `port`, `host`, `use_chat`, `api_key`, and `api_base` values.

To perform one-click evaluation, you need to configure a unified environment according to the instructions below, and then run 
```
nohup bash one-click-evaluation.sh /home/test/test12/models/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd false 4 128 5000 512 > bench-Qwen2.5-32B-Instruct.log 2>&1 &
```
All evaluation results will be returned in JSON format named `$MODEL_results.json` under `src/scripts` path. If you prefer to evaluate separately, please continue reading and refer to the following separate instructions.

#### Set up the unified environment
```
cd src/scripts
conda create -n benchmark python=3.10 -y && conda activate benchmark
bash bfcl_setup.sh
# taskbench
pip install rouge_score
# teval
pip install mmengine
# injecagent
pip install nltk 
pip install accelerate==0.26.0
```

### Glaive Evaluation
First, get the data by
```
cd src/scripts
unzip glaive_data.zip
```

To evaluate with open-source models:
```
python glaive_eval.py --model xxx
```
To evaluate with closed-source models:
```
python glaive_eval.py --model xxx --is_api True
```
The scores will be output in the terminal, and the original inference results along with bad cases will be saved under the path `src/scripts/benchmark_results/glaive`.

### RoTBench Evaluation
RoTBench uses three metrics to evaluate function calling: 
- **Tool Selection (TS)**: Assesses whether the agent selects the correct function.
- **Parameter Identification (PI)**: Evaluates whether the agent identifies the correct parameter names for the function.
- **Content Filling (CF)**: Checks if the agent fills the correct content into the corresponding parameters.

To evaluate with open-source models:
```
cd src/scripts
python rotbench_eval.py --model xxx
```
To evaluate with closed-source models:
```
cd src/scripts
python rotbench_eval.py --model xxx --is_api True
```
The scores will be output in the terminal, and the original inference results along with bad cases will be saved under the path `src/scripts/benchmark_results/rotbench`.

### SealTools Evaluation
In SealTools, **Format ACC** assesses the correctness of the model's output structure, while **Tool P/R/F1** evaluates the model's ability to choose the correct tool. **Parameter P/R/F1**, on the other hand, measures the model’s capability in accurately filling in tool parameters.  

To evaluate with open-source models:
```
cd src/scripts
python sealtools_eval.py --model xxx
```

To evaluate with closed-source models:
```bash
cd src/scripts
python sealtools_eval.py --model xxx --is_api True
```
The scores and the original inference results along with bad cases will be saved under the path `src/data/eval_result/Seal-Tools`.

### TaskBench Evaluation
In TaskBench, **ROUGE-1** examines whether the model can correctly capture and generate individual word matches, reflecting the surface-level accuracy of task decomposition, while **ROUGE-2** extends this by evaluating adjacent word pair matches to provide a more precise assessment of task structuring. **Node F1** assesses the model’s accuracy in selecting the appropriate tool for each subtask, and **Edge F1** evaluates its understanding of dependencies between tools, ensuring correct connections in complex workflows. **Parameter Name F1** measures whether the model correctly identifies required parameters, whereas **Parameter Name & Value F1** further ensures that, in addition to recognizing parameters, the model assigns the correct values, thereby validating the completeness and accuracy of tool configuration.

Unzip `src/data/sft_data/Taskbench_data` to get the data.

To evaluate with open-source models:
```
cd src/scripts
python taskbench_eval.py --model xxx --data_path ../data/sft_data/TaskBench/taskbench_data_dailylifeapis.json
python taskbench_eval.py --model xxx --data_path ../data/sft_data/TaskBench/taskbench_data_huggingface.json
python taskbench_eval.py --model xxx --data_path ../data/sft_data/TaskBench/taskbench_data_multimedia.json
```

To evaluate with closed-source models:
```bash
cd src/scripts
python taskbench_eval.py --model xxx --is_api True --data_path ../data/sft_data/TaskBench/taskbench_data_dailylifeapis.json
python taskbench_eval.py --model xxx --is_api True --data_path ../data/sft_data/TaskBench/taskbench_data_huggingface.json
python taskbench_eval.py --model xxx --is_api True --data_path ../data/sft_data/TaskBench/taskbench_data_multimedia.json
```
The original inference results along with bad cases will be saved under the path `src/scripts`.

### BFCL
The evaluation framework for BFCL focuses on **accuracy** as the primary metric, assessing the model’s correctness in function invocation across various task scenarios, including Non-Live, Live, and multi-turn tasks.

Set Up the Environment
```
conda create -n BFCL python=3.10 -y && conda activate BFCL
bash bfcl_setup.sh
```

For locally downloaded models, you need to add the corresponding processor in the handler mapping file `src/scripts/gorilla/berkeley-function-call-leaderboard/bfcl/model_handler/handler_map.py`. If you want to add the `--max-model-len` parameter, you can add it around line 108 in the file `src/scripts/gorilla/berkeley-function-call-leaderboard/bfcl/model_handler/local_inference/base_oss_handler.py`. If you want to run the program in parallel, you can Modify the port in the file `src/scripts/gorilla/berkeley-function-call-leaderboard/bfcl/model_handler/local_inference/constant.py`

If you want to use your locally trained model, make sure that the model path name does not contain underscores ("_"). Otherwise, you will need to add code similar to the following around line 335 in `src/scripts/gorilla/berkeley-function-call-leaderboard/bfcl/eval_checker/eval_runner(_helper)/py` to ensure that BFCL's processing does not cause conflicts:
```python
elif model_name == "sft_model_merged_lora_checkpoint-20000":
    model_name_escaped = "/sft_model/merged_lora/checkpoint-20000"
```

Besides, to get the multi-turn outcomes, you need to add model path in `src/scripts/gorilla/berkeley-function-call-leaderboard/bfcl/constants/model_metadata.py`. For example:
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

To evaluate with closed-resource models

- **Inference**:
  You can set the `api_key` and `base_url` in the file `src/scripts/gorilla/berkeley-function-call-leaderboard/bfcl/model_handler/api_inference`.
  ```
  bfcl generate --model MODEL_NAME --test-category TEST_CATEGORY --num-threads 1
  # Example:
  bfcl generate --model gpt-3.5-turbo-0125 --test-category parallel,multiple,simple,parallel_multiple,java,javascript,irrelevance,multi_turn --num-threads 1
  ```

- **Evaluation**:
  ```
  bfcl evaluate --model gpt-3.5-turbo-0125
  ```

The original inference results along with bad cases will be saved under the path `src/scripts/gorilla/berkeley-function-call-leaderboard/result`.
  
### T-Eval
T-Eval uses accuracy as the primary evaluation metric, measuring the model’s **correctness** across six task scenarios: planning, reasoning, retrieval, understanding, instruction, and review. Each task except review is assessed in two formats: JSON, which requires structured outputs containing tool names and parameters, and string (str), which allows more flexible textual responses.

Set Up the Environment
```bash
conda create -n teval python=3.10 -y && conda activate teval
unzip teval_data
bash teval_setup.sh
```
Move the files related to teval to the folder `T-Eval`
```
mv T-Eval_evaluation/* T-Eval/
cd T-Eval
```

To evaluate with closed-resource models
```bash
bash test_all_teval.sh api model_name display_name True
# Example:
bash test_all_teval.sh api claude-3-5-sonnet-20240620 claude-3-5-sonnet-20240620 True
```

To evaluate with open-resource models
```bash
# Inference (model_path, display_name, is_api)
bash test_all_teval.sh model_path display_name False [tensor_parallel_size] [gpu_utilization] 
# Evaluate (model_name, display_name, is_api, nums of gpu)
bash test_all_teval.sh /models/Llama-3.1-8B-Instruct llama3 False 2
```
The results will be found in `src/scripts/T-Eval/work_dirs`.

### InjecAgent Evaluation
InjecAgent primarily assesses the model’s resilience under adversarial conditions, focusing on the validity of responses and the success rate of attacks. **Valid rate** measures the proportion of responses that are both non-empty and correctly formatted under attack scenarios. Attack success rate (ASR-valid) specifically quantifies the proportion of successful attacks within valid responses, offering a finer-grained evaluation of model vulnerability. **ASR-valid** is further categorized into specific attack types: **ASR-valid (Direct Harm)** evaluates the model’s susceptibility to direct harm attacks, where it executes malicious tool-based instructions; **ASR-valid (S1)** and **ASR-valid (S2)** respectively assess the success rates of the first and second stages of data-stealing attacks, corresponding to data extraction and data transmission. **ASR-valid (Data Stealing)** aggregates the results of S1 and S2 to provide a comprehensive measure of vulnerability to data theft, while **ASR-valid (Total)** encapsulates the overall attack success rate across all tested adversarial scenarios.  

To evaluate with open-resource models
```
python injecagent_eval.py --model_type OpenModel --model_name xxx --use_cach
```

To evaluate with close-resource models, replace `model_type` with `GPT` or `Claude`.

The results will be found in `src/scripts/InjecAgent_results`.

### SkyThought
Use `pip install skythought` to set up the environment. Then run `one-click-sky.sh /your/model` to get the result.
Modify the `tasks` in `one-click-sky.sh` to use your expected datasets. It contains math500, numina, numina_amc_aime, numina_math, numina_olympiads, taco, olympiadbench_math_en, aime24_sky, aime24, aime25, amc23, livecodebench_medium, livecodebench, livecodebench_easy, livecodebench_hard, arc_c, apps, mmlu, mmlu_pro, gsm8k, minervamath, gpqa_diamond.

详细内容：
## 

1. Set up the environment

```bash
conda create -n skybench python=3.10 -y && conda activate skybench
git clone https://github.com/NovaSky-AI/SkyThought.git && cd SkyThought

pip install skythought -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

1. Evaluate the model on different tasks

```bash
conda activate skybench && cd SkyThought
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=5
skythought evaluate \
	--model /data3/models/models--Qwen--Qwen2.5-Coder-3B-Instruct/snapshots/488639f1ff808d1d3d0ba301aef8c11461451ec5 \
	--task math500,aime24,aime24_sky
	
nohup skythought evaluate \
	--model /data3/models/models--Qwen--Qwen2.5-Coder-3B-Instruct/snapshots/488639f1ff808d1d3d0ba301aef8c11461451ec5 \
	--task numina \
	> skythought-numina-Qwen2.5-Coder-3B-Instruct.log 2>&1 &
```

If you wish to add parameters, please modify line 188 in `skythought/evals/inference_and_check.py`, as follows:

```python
llm = LLM(**engine_kwargs, trust_remote_code=True, max_model_len=4096)
```

`--task` contains: `math500`, `numina`, `numina_amc_aime`, `numina_math`, `numina_olympiads`, `taco`, `olympiadbench_math_en`, `aime24_sky`, `aime24`, `aime25`, `amc23`, `livecodebench_medium`, `livecodebench`, `livecodebench_easy`, `livecodebench_hard`, `arc_c`, `apps`, `mmlu`, `mmlu_pro`, `gsm8k`, `minervamath`, `gpqa_diamond`.

备注：`numina` 耗时极长（数天），`taco` 貌似用不了，`livecodebench_medium` 貌似用不了，`aime25`分为['AIME2025-I', 'AIME2025-II']。

### StableToolBench
1. Set up the environment
```shell
conda create -n stb python=3.10 -y && conda activate stb
git clone https://github.com/THUNLP-MT/StableToolBench.git && cd StableToolBench

pip install -r requirements.txt
pip install --upgrade transformers # Ensure to update to the latest version for compatibility with new models
```
**Note:** After updating transformers, the retriever may not function properly. However, since the retriever is set to None by default during runtime, you need to make the following modifications in toolbench/inference/Downstream_tasks/rapidapi_multithread.py:
- Remove the import line for the retriever function at the beginning.
- Modify the if-else statement at line 573 to retriever = None.

2. Pull the port
Modify the settings in `server/config.yml`, then run the following command:
```shell
cd server
python main.py
```

3. Inference
By default, the original repository only provides scripts for inference with closed-source models. You will need to write your own script for open-source models. Here’s an example script:
```shell
export TOOLBENCH_KEY="Vhqr..."
export PYTHONPATH=./
export SERVICE_URL="http://0.0.0.0:13115/virtual"
export OUTPUT_DIR="data/answer/Llama-3.1-8B-Instruct_CoT"
export CUDA_VISIBLE_DEVICES=0,1,2,3
group=G1_instruction
mkdir -p $OUTPUT_DIR; mkdir -p $OUTPUT_DIR/$group
python toolbench/inference/qa_pipeline_multithread.py \
    --tool_root_dir server/tools/ \
    --backbone_model toolllama \
    --model_path /bjzhyai03/workhome/chenhaotian/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
    --max_observation_length 1024 \
    --method CoT@1 \
    --input_query_file solvable_queries/test_instruction/${group}.json \
    --output_answer_file $OUTPUT_DIR/$group \
    --toolbench_key $TOOLBENCH_KEY \
    --num_thread 1
```
Customizations:
- You must apply for a `TOOLBENCH_KEY` to use this project.
- Ensure that `SERVICE_URL` matches the configuration in `config.yml`.
- Adjust export OUTPUT_DIR based on the model you are using. It is recommended to create separate folders for `CoT` and `DFS`.
- The `group` dataset has several options：`G1_category`, `G1_instruction`, `G1_tool`, `G2_category`, `G2_instruction`, and `G3_instruction`.
- The --tool_root_dir needs to be downloaded as per the repository's instructions. Click the url [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/07ee752ad20b43ed9b0d/?dl=1) link to download two folders named `tools` and `tool_response_cache`, and place them under the `server` folder.
- For open-source models, using `toolllama` as the `--backbone_model` is sufficient. If you want to modify this option, you must add new Python code and adjust the model initialization logic.
- The `--method` can be either CoT or DFS. Use `CoT@1` or `DFS_woFilter_w2`.
- The `--num_thread` can be increased for parallel processing.
- The `--overwrite` flag is optional and not used here.

The results are saved in `OUTPUT_DIR`. To convert the data format, run the following:
```
cd toolbench/tooleval
export RAW_ANSWER_PATH=../../data/answer
export CONVERTED_ANSWER_PATH=../../data_example/model_predictions_converted
export MODEL_NAME=Llama-3.1-8B-Instruct_CoT
export test_set=G1_instruction

mkdir -p ${CONVERTED_ANSWER_PATH}/${MODEL_NAME}
answer_dir=${RAW_ANSWER_PATH}/${MODEL_NAME}/${test_set}
output_file=${CONVERTED_ANSWER_PATH}/${MODEL_NAME}/${test_set}.json

python convert_to_answer_format.py \
    --answer_dir ${answer_dir} \
    --method CoT@1 \
    --output ${output_file}
```
Customizations:
- `MODEL_NAME` should match the `OUTPUT_DIR` you set earlier.
- The `--output` parameter may not always be read correctly, but the code has a default path. A `converted_answers.json` file will be generated in the current directory.

Next, you can calculate the Solvable Pass Rate. Before running the process, you need to specify your evaluation OpenAI key in openai_key.json as follows:
```
[
    {
        "api_key": "sk-qIxQ...",
        "api_base": "https://toollearning.cn/v1"
    },
    ...
]
```

Then calculate SoPR with
```
cd  toolbench/tooleval
export API_POOL_FILE=../../openai_key.json
export CONVERTED_ANSWER_PATH=../../data_example/model_predictions_converted
export SAVE_PATH=../../data_example/pass_rate_results
mkdir -p ${SAVE_PATH}
export CANDIDATE_MODEL=Llama-3.1-8B-Instruct_CoT
export EVAL_MODEL=gpt-3.5-turbo
mkdir -p ${SAVE_PATH}/${CANDIDATE_MODEL}

python eval_pass_rate.py \
    --converted_answer_path ${CONVERTED_ANSWER_PATH} \
    --save_path ${SAVE_PATH}/${CANDIDATE_MODEL} \
    --reference_model ${CANDIDATE_MODEL} \
    --test_ids ../../solvable_queries_example/test_query_ids \
    --max_eval_threads 35 \
    --evaluate_times 3 \
    --test_set G1_instruction 
```
