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

## T-Eval 环境准备 (数据集比较难下就单独下好放到了当前目录的data里面)

1. 创建并激活 Conda 环境
```bash
conda create -n teval python=3.10
conda activate teval
```

2. 安装依赖
```bash
./teval_setup.sh
```

3. 配置 API 密钥
```bash
export OPENAI_API_KEY=xxxxxxxxx
```

## T-Eval 注意事项

对于原文中没有涉及到的模型，需要自己写对应的meta_template，具体位置在
```bash
BodhiAgent/src/benchmark/T-Eval/teval/utils/meta_template.py
```

## T-Eval 使用方法

### API 模型测试
```bash
sh test_all_en.sh api model_name display_name
# 示例
sh test_all_en.sh api gpt-3.5-turbo-0125 gpt3.5-turbo
```

### Local Host 模型测试
```bash
sh test_all_en.sh hf $HF_PATH $HF_MODEL_NAME $META_TEMPLATE
```

### 结果转换
```bash
python teval/utils/convert_results.py --result_path data/$model_display_name/$model_display_name_-1.json
# 示例
python teval/utils/convert_results.py --result_path data/gpt3.5-turbo/gpt3.5-turbo-1.json
```



