# This file evaluates the tool-calling ability of the LLM based on the google/boolq data.
# Author: Zijun Song
# Date: 2025-05
# Copyright (c) THUNLP, Tsinghua University. All rights reserved.
# See LICENSE file in the project root for license information.

import os
import click
import json
import re
from typing import List, Dict, Union, Tuple
from cfg.config import Config
from utils.llm import LLM

conf = Config()

@click.command()
@click.option("--model", type=str, default="/home/test/test03/models/Meta-Llama-3.1-8B-Instruct")
@click.option("--is_api", type=bool, default=False)
@click.option("--tensor_parallel_size", type=int, default=4)
@click.option("--batch_size", type=int, default=1024)
@click.option("--gpu_memory_utilization", type=float, default=0.9)
@click.option("--max_model_len", type=int, default=4096)
@click.option("--max_output_tokens", type=int, default=512)
def main(
    model: str,
    is_api: bool,
    tensor_parallel_size: int,
    batch_size: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_output_tokens: int,
):
    """
    主流程：
    1. 读取 .jsonl 数据；
    2. 调用 LLM 生成 raw_results，保存原始输出；
    3. 提取 JSON 片段解析为 parsed_results，保存解析后结果；
    4. 根据 parsed_results 与 eval_data 对齐，计算准确率，并保存 bad_cases。
    """
    model_name = os.path.basename(model)
    raw_data_path = os.path.join("..", "data", "boolq", "train.jsonl")

    # 读取 eval_data
    eval_data: List[Dict] = []
    with open(raw_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                eval_data.append(json.loads(line))

    # 准备输出目录和文件路径
    output_dir = os.path.join("..", "results", "boolq", model_name)
    os.makedirs(output_dir, exist_ok=True)
    raw_path = os.path.join(output_dir, "boolq_raw.json")
    parsed_path = os.path.join(output_dir, "boolq_parsed.json")
    bad_path = os.path.join(output_dir, "boolq_bad_cases.json")

    # 调用 LLM 或加载 raw_results
    if os.path.exists(raw_path):
        with open(raw_path, "r", encoding="utf-8") as f:
            raw_results = json.load(f)
    else:
        llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            is_api=is_api,
            use_sharegpt_format=False,
            max_input_tokens=max_model_len,
            batch_size=batch_size,
            max_output_tokens=max_output_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        SYSTEM = (
"""
Your task is to:
1. Understand the meaning of question and answer the question;
2. Provide a brief, clear explanation;
3. Output strictly in the following JSON structure, without additional text:
```json
{
  "valid_answer": <true|false>,
  "explanation": "<a concise and easy-to-understand explanation>"
}
```"""
        )
        prompts = [SYSTEM + "\n" + ed["question"] for ed in eval_data]
        raw_results = llm.batch_generate_complete(prompts)
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(raw_results, f, indent=4, ensure_ascii=False)

    # 解析 raw_results
    parsed_results: List[Dict] = []
    for item in raw_results:
        text = item if isinstance(item, str) else json.dumps(item)
        match = re.search(r"\{[\s\S]*?\}", text)
        if match:
            try:
                parsed = json.loads(match.group())
            except json.JSONDecodeError:
                parsed = {"valid_answer": None, "explanation": None}
        else:
            parsed = {"valid_answer": None, "explanation": None}
        parsed_results.append(parsed)
    with open(parsed_path, "w", encoding="utf-8") as f:
        json.dump(parsed_results, f, indent=4, ensure_ascii=False)

    # 评估并保存 bad_cases
    print("********** EVALUATION **********")
    metrics, bad_cases = evaluate_method(parsed_results, eval_data)
    print(f"Model: {model_name}")
    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['total']})")
    if bad_cases:
        print(f"Saved {len(bad_cases)} bad cases to {bad_path}")
        with open(bad_path, "w", encoding="utf-8") as f:
            json.dump(bad_cases, f, indent=4, ensure_ascii=False)


def evaluate_method(
    parsed: List[Dict],
    eval_data: List[Dict]
) -> Tuple[Dict[str, Union[int, float]], List[Dict]]:
    """
    对齐 parsed 与 eval_data，计算准确率并收集 bad_cases。
    返回 (metrics, bad_cases)
    metrics 包含 accuracy, correct, total (解析成功及类型匹配数量)
    bad_cases 列表包含每个不符合期望的案例：{{"index", "question", "ground_truth", "parsed_answer"}}
    """
    ground_truths: List[bool] = []
    preds: List[bool] = []
    bad_cases: List[Dict] = []
    for idx, (ed, pr) in enumerate(zip(eval_data, parsed)):
        gt = bool(ed.get("answer", False))
        va = pr.get("valid_answer")
        # 支持字符串 "true"/"false"
        if isinstance(va, str):
            if va.lower() == 'true':
                va_bool = True
            elif va.lower() == 'false':
                va_bool = False
            else:
                va_bool = None
        elif isinstance(va, bool):
            va_bool = va
        else:
            va_bool = None
        # 判定
        if va_bool is None:
            bad_cases.append({
                "index": idx,
                "question": ed.get("question"),
                "ground_truth": gt,
                "parsed_answer": pr
            })
        else:
            ground_truths.append(gt)
            preds.append(va_bool)
            if va_bool != gt:
                bad_cases.append({
                    "index": idx,
                    "question": ed.get("question"),
                    "ground_truth": gt,
                    "parsed_answer": va_bool
                })
    total = len(ground_truths)
    correct = sum(1 for gt, pr in zip(ground_truths, preds) if gt == pr)
    accuracy = correct / total if total > 0 else 0.0
    return {"accuracy": accuracy, "correct": correct, "total": total}, bad_cases

if __name__ == "__main__":
    main()
