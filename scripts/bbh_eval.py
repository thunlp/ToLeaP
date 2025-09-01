# This file evaluates the tool-calling ability of the LLM based on the google/boolq data.
# Author: Zijun Song
# Date: 2025-05
# Copyright (c) THUNLP, Tsinghua University. All rights reserved.
# See LICENSE file in the project root for license information.

import os
import click
import json
import re
import yaml
from typing import List, Dict, Union, Tuple
from cfg.config import Config
from utils.llm import LLM

conf = Config()

yaml_path = os.path.join("..", "src", "benchmark", "bbh", "prompts.yaml")
with open(yaml_path, "r", encoding="utf-8") as f:
    PROMPTS: Dict[str, Dict[str, str]] = yaml.safe_load(f)

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
    datas = ["boolean_expressions", "causal_judgement", "date_understanding", "disambiguation_qa", "dyck_languages", "formal_fallacies", 
             "geometric_shapes", "hyperbaton", "logical_deduction_five_objects", "logical_deduction_seven_objects", "logical_deduction_three_objects", 
             "movie_recommendation", "multistep_arithmetic_two", "navigate", "object_counting", "penguins_in_a_table", "reasoning_about_colored_objects", 
             "ruin_names", "salient_translation_error_detection", "snarks", "sports_understanding", "temporal_sequences", 
             "tracking_shuffled_objects_five_objects", "tracking_shuffled_objects_seven_objects", "tracking_shuffled_objects_three_objects", 
             "web_of_lies", "word_sorting"]
    
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

    for data in datas:
        model_name = os.path.basename(model)
        raw_data_path = os.path.join("..", "data", "bbh", f"{data}.json")

        with open(raw_data_path, "r", encoding="utf-8") as f:
            eval_data = json.load(f)["examples"]

        # 准备输出目录和文件路径
        output_dir = os.path.join("..", "results", "bbh", model_name, data)
        os.makedirs(output_dir, exist_ok=True)
        raw_path = os.path.join(output_dir, "bbh_raw.json")
        parsed_path = os.path.join(output_dir, "bbh_parsed.json")
        bad_path = os.path.join(output_dir, "bbh_bad_cases.json")

        # 调用 LLM 或加载 raw_results
        if os.path.exists(raw_path):
            with open(raw_path, "r", encoding="utf-8") as f:
                raw_results = json.load(f)
        else:
            SYSTEM = PROMPTS[data]["system"]
            # for ed in eval_data:
            #     print(str(SYSTEM + "\n" + ed["input"]))
            #     assert False
            prompts = [SYSTEM + "\nQuestion: " + ed["input"] for ed in eval_data]
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
                    parsed = {"answer": None, "explanation": None}
            else:
                parsed = {"answer": None, "explanation": None}
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
    parsed: List[Dict[str, Union[str, None]]],
    eval_data: List[Dict[str, Union[str, Dict]]],
) -> Tuple[Dict[str, Union[int, float]], List[Dict]]:
    """
    比较 parsed_results 中每个 item['answer'] 和原数据 eval_data 中对应 example['target']，
    统计准确率，并返回所有错误样本。
    """
    total = len(eval_data)
    correct = 0
    bad_cases = []

    for idx, (pred, example) in enumerate(zip(parsed, eval_data)):
        # 原始 target 在 example 里，与 input 并列
        true_label = example.get("target")
        pred_label = pred.get("answer")
        # 这里可以根据实际格式需求做一些预处理，比如统一大小写、去空格
        if isinstance(pred_label, str):
            pred_norm = pred_label.strip().lower()
        else:
            pred_norm = pred_label

        if isinstance(true_label, str):
            true_norm = true_label.strip().lower()
        else:
            true_norm = true_label

        if pred_norm == true_norm:
            correct += 1
        else:
            bad_cases.append({
                "index": idx,
                "input": example.get("input"),
                "target": true_label,
                "prediction": pred_label,
                "explanation": pred.get("explanation"),
            })

    accuracy = correct / total if total > 0 else 0.0
    metrics = {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
    }
    return metrics, bad_cases

    

if __name__ == "__main__":
    main()
