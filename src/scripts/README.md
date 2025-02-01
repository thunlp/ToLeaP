## 🛠️ Evaluation Method

### Configuring the Evaluation Method
- For **hf batch inference**, set the default values in `cfg/config.yml`.
- For **vllm batch inference**, configure the `port`, `host`, and `use_chat` values.
- For **API models**, set the `port`, `host`, `use_chat`, `api_key`, and `api_base` values.

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

### TaskBench Evaluation

TaskBench measures task step correctness, action correctness and action input correctness separately.  
Task Step Evaluation:
- **ROUGE**: Measures the similarity between the predicted task steps and the ground truth.
- **BERTScore**: Similar to ROUGE, but uses BERT embeddings to measure similarity.

Action Evaluation:
- **Node F1**: Measures f1 of the predicted sequence of actions against the ground truth. (Order insensitive)  
- **Edge F1**: Concatenate consecutive actions and compare against the ground truth. (Order sensitive)

Action Input Evaluation:
Assume action input is a json object with key-value pairs. Keys are the parameter names and values are the parameter values.
- **Name F1**: Measures f1 of the predicted parameter names against the ground truth. (Order insensitive)  
- **Value F1**: Measures f1 of the predicted parameter values against the ground truth. (Order insensitive)

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

### BFCL
Set Up the Environment
```
conda create -n BFCL python=3.10 && conda activate BFCL
bash bfcl_setup.sh
```

For locally downloaded models, you need to add the corresponding processor in the handler mapping file: `gorilla/berkeley-function-call-leaderboard/bfcl/model_handler/handler_map.py`.

To evaluate with closed-resource models

- **Inference**:
  ```bash
  bfcl generate --model MODEL_NAME --test-category TEST_CATEGORY --num-threads 1
  # Example:
  bfcl generate --model gpt-3.5-turbo-0125 --test-category parallel,multiple,simple,parallel_multiple,java,javascript,irrelevance,multi_turn --num-threads 1
  ```

- **Evaluation**:
  ```bash
  bfcl evaluate --model gpt-3.5-turbo-0125
  ```
  
### T-Eval
Set Up the Environment
```bash
conda create -n teval python=3.10 && conda activate teval
bash teval_setup.sh
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

### InjecAgent Evaluation
To evaluate with open-resource models
```
python injecagent_eval.py --model_type OpenModel --model_name xxx --use_cach
```

To evaluate with close-resource models, replace `model_type` with `GPT` or `Claude`.
