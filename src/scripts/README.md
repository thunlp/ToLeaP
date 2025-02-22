## 🛠️ Evaluation Method

### Configuring the Evaluation Method
- For **hf batch inference**, set the default values in `cfg/config.yml`.
- For **vllm batch inference**, configure the `port`, `host`, and `use_chat` values.
- For **API models**, set the `port`, `host`, `use_chat`, `api_key`, and `api_base` values.

To perform one-click evaluation, you need to configure a unified environment according to the instructions below, and then run 
```
bash one-click-evaluation.sh /path/to/your/model $IS_API
# Example
bash one-click-evaluation.sh /home/test/test03/models/Meta-Llama-3.1-8B-Instruct False
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
The evaluation framework for BFCL focuses on **accuracy** as the primary metric, assessing the model’s correctness in function invocation across various task scenarios, including simple, multiple, parallel, multiple-parallel, irrelevance, and multi-turn tasks.

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

Besides, to get the multi-turn outcomes, you need to add model path in `src/scripts/gorilla/berkeley-function-call-leaderboard/bfcl/eval_checker/model_metadata.py`. For example:
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
bash test_all_teval.sh vllm qwen_PATH qwen2.5 False  
# Evaluate (model_name, display_name, is_api)
bash eval_all.sh mistral8b mistral8b False  
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

### StableToolBench（测试版）
1. 配环境
```
conda create -n stb python=3.10 -y && conda activate stb
git clone https://github.com/THUNLP-MT/StableToolBench.git && cd StableToolBench

pip install -r requirements.txt
pip install --upgrade transformers # 务必更新到最新，要不然用不了新模型
# 其他的包再说
```
但是更新完transformers后用不了retriever了，不过这东西本身运行时设的也是None，所以`toolbench/inference/Downstream_tasks/rapidapi_multithread/py`里面，一个要在开头把导入retriever函数那一行删掉，然后第573行的if-else也直接改为`retriever = None`。

2. 推理
默认情况下，这个仓库里只提供了闭源模型做推理的脚本，所以得自己写一个
```
export TOOLBENCH_KEY="VhqrQaPvY9g7OA507b2N8JTXzW4Gneqxi9TAYZooqvEX7iR8fD"
export PYTHONPATH=./
export SERVICE_URL="http://0.0.0.0:13115/virtual"
export OUTPUT_DIR="data/answer/llama2-7b"
export CUDA_VISIBLE_DEVICES=6
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
这个脚本运行时有几个地方要改，一个是`export OUTPUT_DIR`视模型而定，然后数据集`group`有好几个，`--tool_root_dir`那个需要依据代码仓库的指示下载，`backbone_model`开源的就用toolllama就行，目前没遇到任何问题，用别的得自己再写文件，`method`有CoT和DFS，默认就用CoT了，`num_thread`可以改，大于1的话`rapidapi_multithread/py`里面就走别的分支，问题不大，还有一个`--overwrite`，不过就不覆盖了。

这个推理速度还是很慢的，一个数据集可能就得半天，要是数据集都跑一遍，然后CoT和DFS再跑一遍，不知道要到什么时候。

结果就在`OUTPUT_DIR="data/answer/llama2-7b"`这里保存了，首先需要转换数据格式，直接跑也行，写个脚本也行
```
cd toolbench/tooleval
export RAW_ANSWER_PATH=../../data/answer
export CONVERTED_ANSWER_PATH=../../data_example/model_predictions_converted
export MODEL_NAME=toolllama2
export test_set=G1_instruction

mkdir -p ${CONVERTED_ANSWER_PATH}/${MODEL_NAME}
answer_dir=${RAW_ANSWER_PATH}/${MODEL_NAME}/${test_set}
output_file=${CONVERTED_ANSWER_PATH}/${MODEL_NAME}/${test_set}.json

python convert_to_answer_format.py \
    --answer_dir ${answer_dir} \
    --method CoT@1 # DFS_woFilter_w2 for DFS \
    --output ${output_file}
```
这个里面`MODEL_NAME`就看你刚才设置的保存路径是什么了，`--output`参数跑的时候不知道有什么毛病不读，不过代码里面也有默认路径，不是必选参数，注意当前路径下生成了一个`converted_answers.json`文件就ok。

然后得设置api
Next, you can calculate the Solvable Pass Rate. Before running the process, you need to specify your evaluation OpenAI key in openai_key.json as follows:
```
[
    {
        "api_key": "sk-qIxQ0Q30C9HQICt754Ae55168eDb4aD5890b039522A7243b",
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
export CANDIDATE_MODEL=virtual_chatgpt_cot
export EVAL_MODEL=gpt-3.5-turbo # 先用便宜的测了
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
