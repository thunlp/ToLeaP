# Copyright 2024 fairyshine/Seal-Tools
# Modifications Copyright 2024 BodhiAgent
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import click
import json
from typing import List, Dict

from benchmark.sealtools.vllm_SealTools_eval import (
    raw_to_pred,
    calculate_score_ToolLearning,
    write_jsonl,
    write_json,
)

from cfg.config import Config
from utils.llm import LLM

def create_messages(conversation_data: List[Dict]) -> List[List[Dict]]:
    messages = []
    for cov in conversation_data: # Dict
        message = []
        for prompt in cov.get("conversations", []): # List
            if prompt.get("from") == "human":
                message.append({"role": "user", "content": prompt["value"]})
        messages.append(message)
    return messages

conf = Config()

def create_directories(eval_data_path: str, eval_result_path: str, model_name: str):
    paths = [
        os.path.join(eval_data_path, model_name),
        os.path.join(eval_result_path, model_name)
    ]
    for path in paths:
        os.makedirs(path, exist_ok=True)
    # print(f"Mkdir '{os.path.abspath(eval_data_path)}' and '{os.path.abspath(eval_result_path)}'...")

def load_eval_data(input_data_path: str) -> List[Dict]:
    print(f"Getting data from {os.path.abspath(input_data_path)}...")
    with open(input_data_path, "r", encoding='utf-8') as f:
        eval_data = json.load(f)
    return eval_data

@click.command()
@click.option("--model", type=str, default="/bjzhyai03/workhome/chenhaotian/.cache/huggingface/hub/models--Team-ACE--ToolACE-8B/snapshots/d1893ac3ada07430e67e15005c022bcf68a86f0c")
@click.option("--dataset_name_list", type=list[str], default= ["test_out_domain", "dev", "test_in_domain"])
@click.option("--input_path", type=str, default= "../data/sealtools")
@click.option("--raw_data_path", type=str, default= "../results/sealtools/raw_pred_data")
@click.option("--eval_data_path", type=str, default= '../results/sealtools/pred_data')
@click.option("--eval_result_path", type=str, default= '../results/sealtools/eval_result')
@click.option("--is_api", type=bool, default=False)
@click.option("--tensor_parallel_size", type=int, default=4)
@click.option("--batch_size", type=int, default=128)
@click.option("--gpu_memory_utilization", type=float, default=0.8)
@click.option("--max_model_len", type=int, default=4096)
@click.option("--max_output_tokens", type=int, default=512)
@click.option("--model_name", type=str)
def main(
    model: str, 
    dataset_name_list: list[str], 
    input_path : str,  # input datasets file path (folder)
    raw_data_path: str, # raw prediction data path (folder)
    eval_data_path: str, # processed prediction data path (folder)
    eval_result_path: str, # evaluation result path (folder)
    is_api: bool, 
    tensor_parallel_size: int, 
    batch_size: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_output_tokens: int,
    model_name: str,
    ):
    # model_name = os.path.basename(model)
    create_directories(eval_data_path, eval_result_path, model_name)
    llm = LLM(
        model=model, 
        tensor_parallel_size=tensor_parallel_size,
        is_api=is_api,
        use_sharegpt_format=False,
        max_input_tokens=max_model_len,
        batch_size=batch_size, 
        max_output_tokens=max_output_tokens
    )
    data_results = {}
    for dataset in dataset_name_list:
        input_data_path = os.path.join(input_path, f"{dataset}.json") 
        eval_data = load_eval_data(input_data_path)

        ### Run inference
        model_output_dir = os.path.join(raw_data_path, model_name)
        if not os.path.exists(model_output_dir):
            os.makedirs(model_output_dir)

        output_path = os.path.join(model_output_dir, f"{dataset}.json")
        print(f"The raw result will be saved to {os.path.abspath(output_path)}...")
    
        def run_inference() -> List:
            if os.path.exists(output_path): # if exist
                with open(output_path, "r") as f:
                    results = json.load(f)
            else: 
                if not is_api:
                    results = llm.batch_generate_complete(
                        [ed["conversations"][0]["value"] for ed in eval_data]
                    )
                else: # vllm batch generate
                    messages = create_messages(eval_data) # List[List[Dict]]
                    results = llm.batch_generate_chat(messages)
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=4)
            
        run_inference()

        eval_data = raw_to_pred(output_path, input_data_path)
        eval_data_filename = os.path.join(eval_data_path, model_name, dataset + ".json")
        write_jsonl(eval_data_filename, eval_data) 
        result, badcases = calculate_score_ToolLearning(eval_data_filename) 
        data_results[f"{dataset}"] = result
        result_data_filename = os.path.join(eval_result_path, model_name, dataset + ".json")
        badcases_filename = os.path.join(eval_result_path, model_name, f"{model_name}-seal-{dataset}.json")
        write_json(result_data_filename, result, indent=4)
        write_json(badcases_filename, badcases, indent=4)
    print(json.dumps(data_results))

if __name__ == "__main__":
    main()

