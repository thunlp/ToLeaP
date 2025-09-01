# This file evaluates the tool-calling ability of the LLM based on the processed simplified version of the sharegpt format glaiveai/glaive-function-calling data.
# Author: Zijun Song
# Date: 2025-04
# Copyright (c) THUNLP, Tsinghua University. All rights reserved.
# See LICENSE file in the project root for license information.

import os
import click
import json
from typing import List, Dict
from cfg.config import Config
from utils.llm import LLM

conf = Config()

def extract_first_json(text):
    # Find the first occurrence of { or [
    start_idx = -1
    start_char = None
    
    for i, char in enumerate(text):
        if char in '{[':
            start_idx = i
            start_char = char
            break
    
    if start_idx == -1:
        return None
        
    # Define the matching closing bracket
    end_char = '}' if start_char == '{' else ']'
    
    # Initialize counter for nested brackets
    bracket_count = 1
    current_idx = start_idx + 1
    
    # Process string until we find matching closing bracket
    while current_idx < len(text) and bracket_count > 0:
        current_char = text[current_idx]
        
        # Handle string literals to avoid counting brackets inside quotes
        if current_char == '"':
            current_idx += 1
            # Skip through the string
            while current_idx < len(text) and text[current_idx] != '"':
                if text[current_idx] == '\\':  # Handle escaped characters
                    current_idx += 2
                else:
                    current_idx += 1
            if current_idx >= len(text):
                return None
        
        # Count brackets
        elif current_char == start_char:
            bracket_count += 1
        elif current_char == end_char:
            bracket_count -= 1
            
        current_idx += 1
    
    # If we found a complete object, return it
    if bracket_count == 0:
        return text[start_idx:current_idx]
    
    return None

def normalize_value(value) -> object:
    if isinstance(value, str):
        try:
            return float(value) if '.' in value else int(value)
        except ValueError:
            if value.lower() in ["true", "false"]:
                return value.lower() == "true"
            return value.lower().strip()
    return value

def evaluate_function_calls(test_data: List[str], eval_data: List[dict], output_dir: str) -> Dict:
    stats = {
        "total": 0,
        "function_correct": 0,
        "argument_name_correct": 0,
        "argument_value_correct": 0,
        "invalid_gt": 0,
        "invalid_pred": 0
    }

    function_accuracy_errors = []
    argument_name_accuracy_errors = []
    argument_value_accuracy_errors = []

    os.makedirs(output_dir, exist_ok=True)

    invalid_gt_path = os.path.join(output_dir, "invalid_gt.json")
    invalid_pred_path = os.path.join(output_dir, "invalid_pred.json")

    invalid_gt_data = []
    invalid_pred_data = []
    
    for i in range(len(test_data)):
        stats["total"] += 1
        print(f"{i}-th case")
        pred_str = test_data[i]
        try:
            gt_str = eval_data[i]["conversations"][1]["value"]
        except:
            stats["invalid_gt"] += 1
            continue
        
        try:
            gt_data = json.loads(gt_str)
        except:
            stats["invalid_gt"] += 1
            continue
         
        pred_data = extract_first_json(pred_str)
        try:
            pred_data = json.loads(pred_data) 
        except:
            stats["invalid_pred"] += 1
            continue
        if not validate_prediction(pred_data): 
            stats["invalid_pred"] += 1
            stats["total"] += 1
            invalid_pred_data.append(
                {"pred_str": pred_str, "pred_data": pred_data}
            )
            continue

        # print("ground-truth:", gt_data)
        # print("prediction:", pred_data)
        
        if pred_data["name"] == gt_data["name"]:
            stats["function_correct"] += 1
            
            pred_args = pred_data.get("arguments", {})
            gt_args = gt_data.get("arguments", {})
            
            try:
                if set(pred_args.keys()) == set(gt_args.keys()):
                    stats["argument_name_correct"] += 1

                    all_values_match = True
                    for key in gt_args.keys():
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
            except:
                stats["total"] -= 1
                continue
        else:
            function_accuracy_errors.append({"prediction": repr(pred_data["name"]), "ground_truth": repr(gt_data["name"])})

    with open(os.path.join(output_dir, "function_accuracy_errors.json"), "w", encoding="utf-8") as f:
        json.dump(function_accuracy_errors, f, indent=4)

    with open(os.path.join(output_dir, "argument_name_accuracy_errors.json"), "w", encoding="utf-8") as f:
        json.dump(argument_name_accuracy_errors, f, indent=4)

    with open(os.path.join(output_dir, "argument_value_accuracy_errors.json"), "w", encoding="utf-8") as f:
        json.dump(argument_value_accuracy_errors, f, indent=4)

    with open(invalid_gt_path, "w", encoding="utf-8") as f:
        json.dump(invalid_gt_data, f, indent=4)

    with open(invalid_pred_path, "w", encoding="utf-8") as f:
        json.dump(invalid_pred_data, f, indent=4)

    total = stats["total"]
    fn_corr  = stats["function_correct"]
    an_corr  = stats["argument_name_correct"]
    av_corr  = stats["argument_value_correct"]

    # percentages with two decimals
    def pct(num, denom):
        return round((num / denom) * 100, 2) if denom > 0 else 0.0

    return {
        "function_accuracy":       pct(fn_corr, total),
        "argument_name_accuracy":  pct(an_corr, total),
        "argument_value_accuracy": pct(av_corr, total),
        # "total_samples":           total,
        # "function_correct":        fn_corr,
        # "argument_name_correct":   an_corr,
        # "argument_value_correct":  av_corr,
        # "invalid_ground_truth":    stats["invalid_gt"],
        # "invalid_predictions":     stats["invalid_pred"]
    }

def validate_ground_truth(gt_json):
    return (
        gt_json is not None and 
        isinstance(gt_json, dict) and 
        ("name" in gt_json and 
        "arguments" in gt_json and 
        isinstance(gt_json["arguments"], dict)) 
    )

def validate_prediction(pred_json):
    return (
        pred_json is not None and 
        isinstance(pred_json, dict) and
        set(pred_json.keys()) == {"name", "arguments"} and 
        isinstance(pred_json["arguments"], dict)
    )


@click.command()
@click.option("--model", type=str, default="/home/test/test03/models/Meta-Llama-3.1-8B-Instruct")
@click.option("--is_api", type=bool, default=False)
@click.option("--tensor_parallel_size", type=int, default=4)
@click.option("--batch_size", type=int, default=1024)
@click.option("--gpu_memory_utilization", type=float, default=0.9)
@click.option("--max_model_len", type=int, default=4096)
@click.option("--max_output_tokens", type=int, default=512)
@click.option("--debug", "debug_mode", is_flag=True, default=False, help="Run in debug mode with only one data sample.")
@click.option("--think_mode", "think_mode", is_flag=True, default=False)
@click.option("--think_special_tokens", "think_special_tokens", type=str, default="think")
def main(
    model: str, 
    is_api: bool, 
    tensor_parallel_size: int, 
    batch_size: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_output_tokens: int,
    debug_mode: bool,
    think_mode: bool,
    think_special_tokens: str
    ):
    ### Setup
    model_name = os.path.basename(model)

    raw_data_path = f"../data/glaive/glaive-function-calling-sharegpt.json"
    with open(raw_data_path, "r", encoding='utf-8') as f:
        eval_data = json.load(f)

    ### Run inference
    output_path = f"../results/glaive/{model_name}/glaive_results.json"
    if not os.path.exists(f"../results/glaive/{model_name}"):
        os.makedirs(f"../results/glaive/{model_name}")

    if debug_mode:
        eval_data = eval_data[:1]
        print("[Debug] - in rotbench_eval.py - The first query sample is: ")
        print(eval_data[0]["conversations"][0]["value"])

    if not os.path.exists(output_path):
        llm = LLM(
            model=model, 
            tensor_parallel_size=tensor_parallel_size,
            is_api=is_api,
            use_sharegpt_format=False,
            max_input_tokens=max_model_len,
            batch_size=batch_size, 
            max_output_tokens=max_output_tokens,
            think_mode=think_mode,
            think_special_tokens=think_special_tokens,
        )

    def run_inference() -> List:
        if os.path.exists(output_path): # if exists
            with open(output_path, "r") as f:
                results = json.load(f)
        else: 
            results = llm.batch_generate_complete(
                [str(ed["system"] + "\n" + ed["conversations"][0]["value"]) for ed in eval_data]
            )
            with open(output_path, "w") as f:
                json.dump(results, f, indent=4)
    
    print("*"*10 + "INFERENCE" + "*"*10)
    test_data = run_inference()

    if debug_mode:
        print("[Debug] - in rotbench_eval.py - The answer is: ")
        print(test_data[0]) 
        assert False

    ### Evaluation
    with open(output_path, encoding="utf-8") as f:
        test_data = json.load(f)
    
    print("*"*10 + "EVALUATION" + "*"*10)
    print(model)
    output_dir = f"../results/glaive/{model_name}"
    print(evaluate_function_calls(test_data, eval_data, output_dir))

if __name__ == "__main__":
    main()
