# This file is to run stabletoolbench evaluation.
# Author: Zijun Song
# Date: 2025-04
# Copyright (c) THUNLP, Tsinghua University. All rights reserved.
# See LICENSE file in the project root for license information.

import json
import re
import os
import argparse
from openai import OpenAI
from tqdm import tqdm

# Initialize the API client
client = OpenAI(
    api_key='sk-', # Replace with your API key
    base_url="https:",
)

def judge_solution(result):
    prompt = f"""
Please evaluate whether the following query is solved based on the provided information.
Query: {result["query"]}
Thought: {result["thought"]}
Action: {result["action"]}
Action Input: {result["action_input"]}

Guidelines:
If you think the "Action" field and "Action Input" field are filled out correctly, return "Solved", otherwise return "Unsolved".
   
Please provide your result strictly in the following format:
<query>[Query]</query>
<answer_status>[Solved/Unsolved]</answer_status>
<reason>[Your explanation]</reason>
    """.strip()

# Guidelines:
# 1. If the answer contains an apology or is not a positive/straightforward response for the given query, return "Unsolved".
# 2. If the answer is positive/straightforward for the given query, then:
#    2.1. If the answer is insufficient to determine whether the query is solved, return "Unsure".
#    2.2. If you are confident that the answer is sufficient, return "Solved" or "Unsolved" accordingly.
   
# Please provide your result strictly in the following format:
# <query>[Query]</query>
# <answer_status>[Solved/Unsure/Unsolved]</answer_status>
# <reason>[Your explanation]</reason>

    response = client.chat.completions.create(
        model="deepseek-r1",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def extract_fields(api_response):
    """
    Extracts the content inside <query>, <answer_status> and <reason> tags
    from the API response, and returns a dictionary.
    """
    pattern = r"<query>(.*?)</query>\s*<answer_status>(.*?)</answer_status>\s*<reason>(.*?)</reason>"
    match = re.search(pattern, api_response, re.DOTALL)
    if match:
        return {
            "query": match.group(1).strip(),
            "answer_status": match.group(2).strip(),
            "reason": match.group(3).strip()
        }
    else:
        # If the format doesn't match, return a dict with raw response.
        return {
            "query": "",
            "answer_status": "Unknown",
            "reason": f"Failed to extract fields. Raw response: {api_response}"
        }

def main(inference_result, evaluation_result, eval_model):
    if os.path.exists(evaluation_result):
        with open(evaluation_result, 'r', encoding='utf-8') as f:
            evaluation_results = json.load(f)
        total_tasks = len(evaluation_results)
        solved_count = sum(
            1 for ev in evaluation_results.values()
            if ev.get("answer_status") == "Solved"
        )
        solved_percentage = (solved_count / total_tasks) * 100 if total_tasks else 0
        print(f"Found existing results: Solved {solved_count} / {total_tasks} "
              f"tasks ({solved_percentage:.1f}%).")
        return

    # Load JSON data file, assuming the file path is 'inference_results.json'
    with open(inference_result, 'r', encoding='utf-8') as f:
        data = json.load(f)

    evaluation_results = {}
    solved_count = 0
    total_tasks = len(data)

    for idx, result in tqdm(enumerate(data), total=len(data)):
        api_response = judge_solution(result)
        evaluation = extract_fields(api_response)
        evaluation_results[idx] = evaluation

        if evaluation.get("answer_status") == "Solved":
            solved_count += 1
        print(f"\nRecord {idx}: {evaluation}")

    solved_percentage = (solved_count / total_tasks) * 100 if total_tasks else 0
    print(f"----- {args.model_name} on {args.group} dataset -----")
    print(f"\nSolved: {solved_count} out of {total_tasks} tasks ({solved_percentage:.1f}%).")

    with open(evaluation_result, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate toolbench inference results via LLM.")
    parser.add_argument("--model_name", required=True, help="")
    parser.add_argument("--group", required=True, help="")
    parser.add_argument("--eval_model", required=True, help="")
    args = parser.parse_args()

    input_path = f"../results/stabletoolbench/{args.model_name}/{args.group}/inference_results.json"
    output_path = f"../results/stabletoolbench/{args.model_name}/{args.group}/evaluation_results_{args.eval_model}.json"

    main(input_path, output_path, args.eval_model)
