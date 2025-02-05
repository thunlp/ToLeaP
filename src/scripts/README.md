## 🛠️ Evaluation Method

### Configuring the Evaluation Method
- For **hf batch inference**, set the default values in `cfg/config.yml`.
- For **vllm batch inference**, configure the `port`, `host`, and `use_chat` values.
- For **API models**, set the `port`, `host`, `use_chat`, `api_key`, and `api_base` values.

To perform one-click evaluation, simply run `bash one-click-evaluation.sh /path/to/your/model` in the BFCL environment. If you prefer to evaluate separately, follow the steps below.

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

For locally downloaded models, you need to add the corresponding processor in the handler mapping file `gorilla/berkeley-function-call-leaderboard/bfcl/model_handler/handler_map.py`. If you want to add the `--max-model-len` parameter, you can add it around line 108 in the file `src/scripts/gorilla/berkeley-function-call-leaderboard/bfcl/model_handler/local_inference/base_oss_handler.py`.

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
conda create -n teval python=3.10 && conda activate teval
bash teval_setup.sh
```
Move the files related to teval to the folder `T-Eval`
```
mv T-Eval_evaluation/* T-Eval/
cd T-Eval
```

To evaluate with closed-resource models
```bash
sh teval-eval.sh api model_name display_name
# Example:
sh teval-eval.sh api gpt-3.5-turbo-0125 gpt3.5-turbo
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
