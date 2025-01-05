import sys
import os
import click
import json
from tqdm import tqdm
from typing import List, Dict

current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(current_dir, '..')
sys.path.append(utils_dir)
# ZK: This part should be concluded into a utils python file in the final version
from vllm_SealTools_eval import (
    transform_output_format,
    raw_to_pred,
    calculate_score_ToolLearning,
    write_jsonl,
    write_json,
    read_json,
    read_jsonl
)

from cfg.config import Config
from utils.llm import LLM

class SealToolsLLM(LLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create_messages_from_sharegpt(self, conversation_data: List[Dict]) -> List[List[Dict]]:
        messages = []
        for cov in conversation_data: # Dict
            message = []
            for prompt in cov.get("conversations", []): # List
                if prompt.get("from") == "human":
                    message.append({"role": "user", "content": prompt["value"]})
            messages.append(message)
        return messages

conf = Config()
print(conf.use_chat)

def parse_model_name(model_path: str) -> str:
    model_mapping = {
        "bb46c15ee4bb56c5b63245ef50fd7637234d6f75": "Qwen2.5-7B-Instruct",
        "a2cb7a712bb6e5e736ca7f8cd98167f81a0b5bd8": "Llama-2-13b-chat-hf",
        "f5db02db724555f92da89c216ac04704f23d4590": "Llama-2-7b-chat-hf",
        "f66993d6c40a644a7d7885d4c029943861e06113": "ToolLLaMA-2-7b-v2",
        "checkpoint-10000": "afm10000",
        "checkpoint-20000": "afm20000",
        "checkpoint-30000": "afm30000",
        "checkpoint-40000": "afm40000",
        "checkpoint-50000": "afm50000",
    }
    model_split = os.path.basename(model_path)
    return model_mapping.get(model_split, model_split)

def create_directories(eval_data_path: str, eval_result_path: str, model_name: str):
    paths = [
        os.path.join(eval_data_path, model_name),
        os.path.join(eval_result_path, model_name)
    ]
    for path in paths:
        os.makedirs(path, exist_ok=True)
    print(f"Mkdir '{os.path.abspath(eval_data_path)}' and '{os.path.abspath(eval_result_path)}'...")

def initialize_llm(model: str, is_api: bool, conf: Config, tensor_parallel_size: int,
                  max_model_len: int, gpu_memory_utilization: float, batch_size: int) -> LLM:
    if not is_api:
        if not conf.use_chat:
            print("Initializing LLM (hf batch path)...")
            llm = LLM(
                model=model,
                tensor_parallel_size=tensor_parallel_size,
                use_sharegpt_format=False,
                max_input_tokens=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                batch_size=batch_size
            )
        else:
            print("Initializing SealToolsLLM (vllm batch path)...")
            llm = SealToolsLLM(
                model=model,
                tensor_parallel_size=tensor_parallel_size,
                use_sharegpt_format=False,
                max_input_tokens=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                batch_size=batch_size
            )
    else:
        print("Initializing API model...")
        llm = LLM(model=model, is_api=is_api)
    return llm

def load_eval_data(input_data_path: str) -> List[Dict]:
    print(f"Getting data from {os.path.abspath(input_data_path)}...")
    with open(input_data_path, "r", encoding='utf-8') as f:
        eval_data = json.load(f)
    return eval_data

@click.command()
@click.option("--model", type=str, default="/bjzhyai03/workhome/chenhaotian/.cache/huggingface/hub/models--ToolBench--ToolLLaMA-2-7b-v2/snapshots/f66993d6c40a644a7d7885d4c029943861e06113")
@click.option("--dataset_name_list", type=list[str], default= ["test_out_domain", "dev", "test_in_domain"])
@click.option("--input_path", type=str, default= "../../src/data/input_data/Seal-Tools")
@click.option("--raw_data_path", type=str, default= "../../src/data/raw_pred_data/Seal-Tools")
@click.option("--eval_data_path", type=str, default= '../../src/data/pred_data/Seal-Tools')
@click.option("--eval_result_path", type=str, default= '../../src/data/eval_result/Seal-Tools')
@click.option("--is_api", type=bool, default=False)
@click.option("--tensor_parallel_size", type=int, default=1)
@click.option("--batch_size", type=int, default=8)
@click.option("--gpu_memory_utilization", type=float, default=0.9)
@click.option("--max_model_len", type=int, default=8192)
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
    print("[DEBUG] - Normally running...")

    model_name = parse_model_name(model)
    create_directories(eval_data_path, eval_result_path, model_name)

    ### Init LLM
    llm = initialize_llm(model, is_api, conf, tensor_parallel_size, max_model_len, gpu_memory_utilization, batch_size)

    for dataset in dataset_name_list[:1]:
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
        print(f"The raw result will be saved to {os.path.abspath(output_path)}...")

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
                    messages = llm._create_messages(eval_data) # List[List[Dict]]
                    if not is_api:
                        with llm.start_server():
                            results = llm.batch_generate_chat(messages)
                    else:
                        results = llm.batch_generate_chat(messages)
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=4)
            return results
            
        results = run_inference()

        # ZK:  开始测评，先将 raw 数据转成可以 eval的数据格式
        eval_data = raw_to_pred(output_path, input_data_path) # ZK: 将 raw data 转换成可以用于评估的 eval data
        eval_data_filename = os.path.join(eval_data_path, model_name, dataset + ".json")
        write_jsonl(eval_data_filename, eval_data) 
        # ZK:  读取可以 eval的数据格式并开始测评
        result = calculate_score_ToolLearning(eval_data_filename) # ZK:  测评函数
        result_data_filename = os.path.join(eval_result_path, model_name, dataset + ".json")
        write_json(result_data_filename, result, indent=4)

if __name__ == "__main__":
    main()
