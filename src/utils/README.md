# sharegpt_inference.LLM 使用说明

## 1. 环境变量

目前暂设5个环境变量，分别是：

- `PORT`：vllm 服务端口
- `HOST`：vllm 服务地址
- `API_KEY`：服务api key
- `USE_HF`：是否使用hf模型（**特别注意，dotenv会视所有输入为字符串，所以如果不使用hf模型，需要放空USE_HF**）
- `API_BASE`：服务地址

其中API_BASE只在使用非本地模型时需要设置，例如调openai借口，将其设为base_url。
如果使用hf模型，则无需设置服务地址和端口。

## 2. LLM初始化

初始化传入以下参数：

- `model`：模型名称
- `dtype`：模型精度（默认不设置， 使用模型config默认值）
- `gpu_memory_utilization`：gpu使用率
- `tensor_parallel_size`：用几张gpu
- `use_api_model`：是否使用openai接口
- `use_sharegpt_format`：是否使用sharegpt格式
- `max_past_message_include`：历史记录看多少，-1代表全部（只在single_generate中使用）

如果使用vllm在线，client的base_url会由环境变量设置的端口和地址拼接，否则使用openai的base_url。

## 3. 调用生成

提供两个方法，针对不同场景：

- `single_generate`：单轮对话生成
    - user_prompt：string, 用户输入
    - system_prompt：string, 系统提示（默认为空）
    - former_messages：list[dict], 历史记录（默认为空）
    - shrink_multiple_break：bool, 是否缩减多轮对话的break（默认为False）
    - temperature：float, 温度（默认为0）
    - 返回：string, 生成结果
- `batch_generate`：多轮对话生成
    - messages_batch：list[list[dict]], 所有测试数据，默认已处理好sharegpt或openai格式
    - max_concurrent_calls：int, 最大并发数（默认为2，根据模型大小适当调整）
    - temperature：float, 温度（默认为0）
    - 返回：list[string], 生成结果

额外提供vllm在线服务初始化，若使用该方法，确保没有其他vllm服务在运行。   
用法：
```
llm = LLM(...)
with llm.start_server():
    # 在with块中运行生成
print("脱离with块后，服务自动关闭")
```



