import sys
import os
import click
import json
import requests
from typing import List, Dict
from sklearn.metrics import f1_score
from rouge_score import rouge_scorer

# import torch
# torch.cuda.set_device(0)

current_dir = os.path.dirname(os.path.abspath(__file__)) 
utils_dir = os.path.join(current_dir, '..')
sys.path.append(utils_dir)

from utils.llm import LLM
from tqdm import tqdm

@click.command()
@click.option("--model", type=str, default="/bjzhyai03/workhome/songzijun/LLaMA-Factory-main/models/llama3_lora_sft/afm_checkpoint_10000")
@click.option("--data_path", type=str, default="/bjzhyai03/workhome/songzijun/BodhiAgent-main/src/data/eval_data/RoTBench/First_turn_RC/slight.json")
@click.option("--is_api", type=bool, default=False)
@click.option("--host", type=str, default="localhost")
@click.option("--port", type=int, default=13427)
@click.option("--tensor_parallel_size", type=int, default=2)
@click.option("--batch_size", type=int, default=20)
@click.option("--gpu_memory_utilization", type=float, default=0.8)
@click.option("--max_model_len", type=int, default=65535)
def main(
    model: str, 
    data_path: str, 
    is_api: bool, 
    host: str, 
    port: int, 
    tensor_parallel_size: int, 
    batch_size: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    ):
    print("[DEBUG] - main checkpoint 1...")

    with open(data_path, "r", encoding='utf-8') as f:
        eval_data = json.load(f)

    print("[DEBUG] - main checkpoint 2...")
    if not is_api:
        llm = LLM(
            model=model, 
            tensor_parallel_size=tensor_parallel_size, 
            use_sharegpt_format=False,
        )
    else:
        llm = LLM(model=model)

    print("[DEBUG] - main checkpoint 3...")
    # Run inference
    data_filename = os.path.splitext(os.path.basename(data_path))[0]
    model_split = model.split('/')[-1]
    model_real_name = None
    if model_split == "bb46c15ee4bb56c5b63245ef50fd7637234d6f75":
        model_real_name = "Qwen2.5-7B-Instruct"
    elif model_split == "a2cb7a712bb6e5e736ca7f8cd98167f81a0b5bd8":
        model_real_name = "Llama-2-13b-chat-hf"
    elif model_split == "f5db02db724555f92da89c216ac04704f23d4590":
        model_real_name = "Llama-2-7b-chat-hf"
    elif model_split == "f66993d6c40a644a7d7885d4c029943861e06113":
        model_real_name = "ToolLLaMA-2-7b-v2"
    else:
        model_real_name = model_split
    output_path = f"benchmark_results/{model_real_name}_rotbench_{data_filename}_results.json"
                  
    def run_inference():
        print("[DEBUG] - run_inference checkpoint 1...")
        if os.path.exists(output_path):
            results = json.load(open(output_path, "r"))
        else:
            print("[DEBUG] - run_inference checkpoint 3...")
            results_partial = []
            for ed in tqdm(eval_data, desc="Processing", unit="sample"):
                user_prompt = ed["content"]
                assistant_reply = llm.single_generate(
                    user_prompt=user_prompt
                )
                assistant_reply_partial = assistant_reply[len(user_prompt):]
                results_partial.append(assistant_reply_partial)
            json.dump(results_partial, open(output_path, "w"), indent=4)
        print("[DEBUG] - run_inference checkpoint 2...")
        return results_partial
        
    print("[DEBUG] - main checkpoint 4...")
    if not os.path.exists(output_path):
        results = run_inference()
    else:
        print("[DEBUG] - main branch checkpoint 4...")
        results = json.load(open(output_path, "r"))

    print("[DEBUG] - main checkpoint 5...")

    print("[DEBUG] - main checkpoint 6...")

if __name__ == "__main__":
    main()
