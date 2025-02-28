import sys
import os
import click
import json
import re
from typing import List, Dict, Union

current_dir = os.path.dirname(os.path.abspath(__file__)) 
utils_dir = os.path.join(current_dir, '..')
sys.path.append(utils_dir)

from cfg.config import Config
from utils.llm import LLM

def create_messages(conversation_data: List[Dict]) -> List[List[Dict]]:
    messages = []
    for conv in conversation_data:
        message = []
        for prompt in conv["conversations"]:
            if prompt["from"] == "system":
                message.append({
                    "role": "system",
                    "content": prompt["value"]
                })
            elif prompt["from"] == "user":
                message.append({
                    "role": "user",
                    "content": prompt["value"]
                })
        messages.append(message)
    return messages

conf = Config()

def initialize_llm(model: str, is_api: bool, conf: Config, tensor_parallel_size: int,
                   max_model_len: int, gpu_memory_utilization: float, batch_size: int) -> LLM:
    if not is_api:
        llm = LLM(
                model=model,
                tensor_parallel_size=tensor_parallel_size,
                use_sharegpt_format=False,
                max_input_tokens=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                batch_size=batch_size,
                max_output_tokens=512
        )
    else:
        llm = LLM(model=model, is_api=is_api)
    return llm

def extract_first_json(text):
    """从文本中提取第一个完整JSON对象（容错版本）"""
    try:
        # 查找第一个左花括号
        start = text.index("{")
    except ValueError:
        return None
    
    # 从起始位置开始查找匹配的右花括号
    stack = []
    end = start
    for i, c in enumerate(text[start:]):
        if c == "{":
            stack.append(i)
        elif c == "}":
            if stack:
                stack.pop()
                if not stack:  # 找到最外层匹配的右花括号
                    end = start + i + 1
                    break
    try:
        json_str = text[start:end]
        # 预处理非法JSON格式（处理键名缺少引号的情况）
        json_str = re.sub(r'(\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*):', r'\1"\2"\3:', json_str)
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return None

def extract_function_call(input_str):
    pattern = r"<functioncall> (\{.*?\})"
    match = re.search(pattern, input_str)
    if match:
        # 提取匹配到的字符串
        function_call_str = match.group(1)
        
        # 进一步处理：分割出函数的名称和参数
        try:
            function_call_json = json.loads(function_call_str.replace("'", '"'))  # 处理引号替换
            return function_call_json
        except json.JSONDecodeError:
            try:
                # 如果解析失败，尝试加一个 } 再解析
                function_call_str = function_call_str + "}"
                function_call_json = json.loads(json.dumps(function_call_str))
                return function_call_json
            except json.JSONDecodeError:
                return None
    return None

def validate_schema(data: dict) -> bool:
    """验证是否为有效的函数调用结构"""
    return isinstance(data, dict) and "name" in data and "arguments" in data

def normalize_value(value) -> object:
    """统一参数值类型"""
    if isinstance(value, str):
        # 尝试转换为数字类型
        try:
            return float(value) if '.' in value else int(value)
        except ValueError:
            # 处理布尔字符串
            if value.lower() in ["true", "false"]:
                return value.lower() == "true"
            # 统一小写处理字符串
            return value.lower().strip()
    return value

def evaluate_function_calls(predictions: List[str], ground_truths: List[str], output_dir: str) -> Dict:
    stats = {
        "total": 0,
        "function_correct": 0,
        "argument_name_correct": 0,
        "argument_value_correct": 0,
        "invalid_gt": 0,
        "invalid_pred": 0
    }

    # 记录每个错误类型的原始数据
    function_accuracy_errors = []
    argument_name_accuracy_errors = []
    argument_value_accuracy_errors = []

    # 记录无效的ground truth和预测
    os.makedirs(output_dir, exist_ok=True)

    invalid_gt_path = os.path.join(output_dir, "invalid_gt.json")
    invalid_pred_path = os.path.join(output_dir, "invalid_pred.json")

    invalid_gt_data = []
    invalid_pred_data = []
    for pred_str, gt_str in zip(predictions, ground_truths):
        gt_data = extract_function_call(gt_str)
        if gt_data is None:
            continue
        try:
            gt_data = json.loads(gt_data)
        except:
            continue
        if not validate_ground_truth(gt_data):
            stats["invalid_gt"] += 1
            invalid_gt_data.append(
                {"gt_str": gt_str, "gt_data": gt_data}
            )
            continue

        pred_data = extract_first_json(pred_str)
        if not validate_prediction(pred_data):
            stats["invalid_pred"] += 1
            stats["total"] += 1
            invalid_pred_data.append(
                {"pred_str": pred_str, "pred_data": pred_data}
            )
            continue

        stats["total"] += 1
        
        if pred_data["name"] == gt_data["name"]:
            stats["function_correct"] += 1
            
            pred_args = pred_data.get("arguments", {})
            gt_args = gt_data.get("arguments", {})
            
            # 比较参数名称集合
            if set(pred_args.keys()) == set(gt_args.keys()):
                stats["argument_name_correct"] += 1

                # 比较参数值（逐个参数比较）
                all_values_match = True
                for key in gt_args.keys():
                    # 统一参数值类型
                    normalized_pred = normalize_value(pred_args[key])
                    normalized_gt = normalize_value(gt_args[key])
                    
                    if normalized_pred != normalized_gt:
                        all_values_match = False
                        argument_value_accuracy_errors.append({"prediction": pred_data, "ground_truth": gt_data})
                        break
                
                if all_values_match:
                    stats["argument_value_correct"] += 1
            else:
                argument_name_accuracy_errors.append({"prediction": pred_data, "ground_truth": gt_data})
        else:
            function_accuracy_errors.append({"prediction": repr(pred_data["name"]), "ground_truth": repr(gt_data["name"])})
                
    # 保存每种错误类型
    with open(os.path.join(output_dir, "function_accuracy_errors.json"), "w", encoding="utf-8") as f:
        json.dump(function_accuracy_errors, f, indent=4)

    with open(os.path.join(output_dir, "argument_name_accuracy_errors.json"), "w", encoding="utf-8") as f:
        json.dump(argument_name_accuracy_errors, f, indent=4)

    with open(os.path.join(output_dir, "argument_value_accuracy_errors.json"), "w", encoding="utf-8") as f:
        json.dump(argument_value_accuracy_errors, f, indent=4)

    # 保存无效的ground truth和预测
    with open(invalid_gt_path, "w", encoding="utf-8") as f:
        json.dump(invalid_gt_data, f, indent=4)

    with open(invalid_pred_path, "w", encoding="utf-8") as f:
        json.dump(invalid_pred_data, f, indent=4)

    print(stats["total"])
    print(stats["function_correct"])
    print(stats["argument_name_correct"])

    # 计算各项指标
    return {
        "function_accuracy": round(stats["function_correct"] / stats["total"], 4) if stats["total"] > 0 else 0.0,
        "argument_name_accuracy": round(stats["argument_name_correct"] / stats["function_correct"], 4) if stats["function_correct"] > 0 else 0.0,
        "argument_value_accuracy": round(stats["argument_value_correct"] / stats["argument_name_correct"], 4) if stats["argument_name_correct"] > 0 else 0.0,
        "total_samples": stats["total"],
        "function_correct": stats["function_correct"],
        "argument_name_correct": stats["argument_name_correct"],
        "argument_value_correct": stats["argument_value_correct"],
        "invalid_ground_truth": stats["invalid_gt"],
        "invalid_predictions": stats["invalid_pred"]
    }

def validate_ground_truth(gt_json):
    """ 强化ground truth格式验证 """
    return (
        gt_json is not None and 
        isinstance(gt_json, dict) and 
        ("name" in gt_json and 
        "arguments" in gt_json and 
        isinstance(gt_json["arguments"], dict)) 
    )

def validate_prediction(pred_json):
    """ 强化预测结果格式验证 """
    return (
        pred_json is not None and 
        isinstance(pred_json, dict) and 
        (("name" in pred_json and 
        "arguments" in pred_json and 
        isinstance(pred_json["arguments"], dict)) or
        ("name" in pred_json and 
        "parameters" in pred_json and 
        isinstance(pred_json["parameters"], dict))) 
    )

def get_prompt(data_entry):
    sample_str = data_entry

    functioncall_index = sample_str.find("<functioncall>")
    if functioncall_index == -1:
        return "This task has no <functioncall>. Evaluation code will skip this query."
    prompt_end = functioncall_index + len("<functioncall>")
    prompt = sample_str[:prompt_end]
    
    return prompt

@click.command()
@click.option("--model", type=str, default="/home/test/test03/models/Meta-Llama-3.1-8B-Instruct")
@click.option("--is_api", type=bool, default=False)
@click.option("--tensor_parallel_size", type=int, default=1)
@click.option("--batch_size", type=int, default=16)
@click.option("--gpu_memory_utilization", type=float, default=0.9)
@click.option("--max_model_len", type=int, default=4096)
def main(
    model: str, 
    is_api: bool, 
    tensor_parallel_size: int, 
    batch_size: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    ):
    ### Setup
    model_name = os.path.basename(model)

    data_results = {}
    raw_data_path = f"glaive-function-calling.json"
    with open(raw_data_path, "r", encoding='utf-8') as f:
        eval_data = json.load(f)

    ### Run inference
    output_path = f"benchmark_results/glaive/{model_name}/{model_name}_glaive_results.json"
    if not os.path.exists(f"benchmark_results/glaive/{model_name}"):
        os.makedirs(f"benchmark_results/glaive/{model_name}")

    if not os.path.exists(output_path):
        llm = initialize_llm(model, is_api, conf, tensor_parallel_size, max_model_len, gpu_memory_utilization, batch_size)

    PROMPT = "Do not say any nonsense, only reply in the strict JSON format of {\"name\":\"specific function name\", \"arguments\": {\"specific parameter name\": \"specific parameter value\"}}."
    
    def run_inference() -> List:
        if os.path.exists(output_path): # if exists
            with open(output_path, "r") as f:
                results = json.load(f)
        else: # if not 
            if not conf.use_chat: # hf batch generate
                # print(type(eval_data["sample"]))
                # for ed in eval_data["sample"]:
                #     print(get_prompt(ed))
                #     assert False
                results = llm.batch_generate_complete(
                    [str(PROMPT + get_prompt(ed)) for ed in eval_data["sample"]],
                    temperature=0
                )
            else:  # vllm batch generate
                messages = create_messages(eval_data)
                if not is_api:
                    with llm.start_server():
                        results = llm.batch_generate_chat(messages)
                else:
                    # print("You are using batch_generate_chat to execute inference")
                    results = llm.batch_generate_chat(messages)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=4)
        return results
    
    results = run_inference()

    ### Evaluation
    with open(output_path, encoding="utf-8") as f:
        test_data = json.load(f)
    
    print("*"*10 + "OUTCOME" + "*"*10)
    print(model)
    output_dir = f"benchmark_results/glaive/{model_name}"
    print(evaluate_function_calls(test_data, eval_data["sample"], output_dir))

if __name__ == "__main__":
    main()
