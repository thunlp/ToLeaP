## 🛠️ Evaluation Method
以下是润色后的版本，语言更加简洁清晰，符合GitHub文档的规范：

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
```bash
cd src/scripts
python rotbench_eval.py --model xxx
```

To evaluate with closed-source models:
```bash
cd src/scripts
python rotbench_eval.py --model xxx --is_api True
```

### SealTools Evaluation

Run RoTBench Evaluation:
```
cd src/scripts
python sealtools_eval.py --model xxx
```

# BFCL 和 T-Eval 环境配置与使用指南

## BFCL 环境准备

1. 创建并激活 Conda 环境
```bash
conda create -n BFCL python=3.10
conda activate BFCL
```

2. 安装 BFCL
```bash
./bfcl_setup.sh
```

## BFCL 注意事项

对于下载到本地的模型，需要在处理器映射文件中添加对应的处理器。配置文件位置：
```bash
BodhiAgent/src/benchmark/gorilla/berkeley-function-call-leaderboard/bfcl/model_handler/handler_map.py
```

添加方式示例：
```python
"/model_path": LlamaHandler,
```

## BFCL 使用方法

### API 模型测试
使用以下命令进行测试：
```bash
bfcl generate --model MODEL_NAME --test-category TEST_CATEGORY --num-threads 1
```

示例（测试 GPT-3.5-turbo 在 benchmark 上的表现）：
```bash
bfcl generate --model gpt-3.5-turbo-0125 --test-category parallel,multiple,simple,parallel_multiple,java,javascript,irrelevance,multi_turn --num-threads 1
```

生成结果评估：
```bash
bfcl evaluate --model gpt-3.5-turbo-0125
```

以下是润色后的版本：

---

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

### TaskBench Evaluation

TaskBench measures task step correctness, action correctness and action input correctness separately.  
*Task Step Evaluation*:
- **ROUGE**: Measures the similarity between the predicted task steps and the ground truth.
- **BERTScore**: Similar to ROUGE, but uses BERT embeddings to measure similarity.

*Action Evaluation*:
- **Node F1**: Measures f1 of the predicted sequence of actions against the ground truth. (Order insensitive)  
- **Edge F1**: Concatenate consecutive actions and compare against the ground truth. (Order sensitive)

*Action Input Evaluation*:
Assume action input is a json object with key-value pairs. Keys are the parameter names and values are the parameter values.
- **Name F1**: Measures f1 of the predicted parameter names against the ground truth. (Order insensitive)  
- **Value F1**: Measures f1 of the predicted parameter values against the ground truth. (Order insensitive)
