import sys
import os
import click
import json
from typing import List, Dict

current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(current_dir, '..')
sys.path.append(utils_dir)

from vllm_SealTools_eval import (
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

def load_eval_data(input_data_path: str) -> List[Dict]:
    # print(f"Getting data from {os.path.abspath(input_data_path)}...")
    with open(input_data_path, "r", encoding='utf-8') as f:
        eval_data = json.load(f)
    return eval_data

@click.command()
@click.option("--model", type=str, default="/bjzhyai03/workhome/chenhaotian/.cache/huggingface/hub/models--Team-ACE--ToolACE-8B/snapshots/d1893ac3ada07430e67e15005c022bcf68a86f0c")
@click.option("--dataset_name_list", type=list[str], default= ["test_out_domain", "dev", "test_in_domain"])
# @click.option("--dataset_name_list", type=list[str], default= ["test_out_domain"])
@click.option("--input_path", type=str, default= "../../src/data/input_data/Seal-Tools")
@click.option("--raw_data_path", type=str, default= "../../src/data/raw_pred_data/Seal-Tools")
@click.option("--eval_data_path", type=str, default= '../../src/data/pred_data/Seal-Tools')
@click.option("--eval_result_path", type=str, default= '../../src/data/eval_result/Seal-Tools')
@click.option("--is_api", type=bool, default=False)
@click.option("--tensor_parallel_size", type=int, default=1)
@click.option("--batch_size", type=int, default=32)
@click.option("--gpu_memory_utilization", type=float, default=0.8)
@click.option("--max_model_len", type=int, default=4096)
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
    ):
    model_name = os.path.basename(model)
    create_directories(eval_data_path, eval_result_path, model_name)
    llm = initialize_llm(model, is_api, conf, tensor_parallel_size, max_model_len, gpu_memory_utilization, batch_size)

    data_results = {}
    for dataset in dataset_name_list:
        input_data_path = os.path.join(input_path, f"{dataset}.json") 
        eval_data = load_eval_data(input_data_path)

        ### Run inference
        if not os.path.exists(raw_data_path):
            os.makedirs(raw_data_path)
        
        if not conf.use_chat:
            output_path = os.path.join(raw_data_path, f"hf_{dataset}_{model_name}.json")  
        else:
            if is_api:
                output_path =  os.path.join(raw_data_path, f"api_{dataset}_{model_name}.json") 
            else: 
                output_path = os.path.join(raw_data_path, f"vllm_{dataset}_{model_name}.json")  
        # print(f"The raw result will be saved to {os.path.abspath(output_path)}...")

        def run_inference() -> List:
            if os.path.exists(output_path): # if exist
                with open(output_path, "r") as f:
                    results = json.load(f)
            else: # if not exist
                if not conf.use_chat: # hf batch generate
                    results = llm.batch_generate_complete(
                        [ed["conversations"][0]["value"] for ed in eval_data],
                        temperature=0
                    )
                else: # vllm batch generate
                    messages = create_messages(eval_data) # List[List[Dict]]
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

        eval_data = raw_to_pred(output_path, input_data_path)
        eval_data_filename = os.path.join(eval_data_path, model_name, dataset + ".json")
        write_jsonl(eval_data_filename, eval_data) 
        result, badcases = calculate_score_ToolLearning(eval_data_filename) 
        data_results[f"{dataset}"] = result
        result_data_filename = os.path.join(eval_result_path, model_name, dataset + ".json")
        badcases_filename = os.path.join(eval_result_path, model_name, f"{model_name}-seal-{dataset}.json")
        write_json(result_data_filename, result, indent=4)
        write_json(badcases_filename, badcases, indent=4)
    print(data_results)

if __name__ == "__main__":
    main()


