import sys
import os
import click
import json
import requests
from typing import List, Dict
from sklearn.metrics import f1_score
from rouge_score import rouge_scorer

current_dir = os.path.dirname(os.path.abspath(__file__)) 
utils_dir = os.path.join(current_dir, '..')
sys.path.append(utils_dir)

from utils.sharegpt_inference import LLM

class SealToolsLLM(LLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create_messages(self, conversation_data: Dict) -> List[Dict]:
        """Create messages list from conversation data"""
        messages = []
        user_content = conversation_data["raw"][0] if conversation_data["raw"] else ""
        assistant_content = conversation_data["text"][0] if conversation_data["text"] else ""
        messages.append({
            "role": "user",
            "content": user_content
        })
        messages.append({
            "role": "assistant",
            "content": assistant_content
        })
        return messages
    
    def batch_generate(self, test_cases: List[Dict], max_concurrent_calls: int = 2, temperature: float = 0) -> List[Dict]:
        """Process test cases using the custom _create_messages method"""
        all_messages = [self._create_messages(case) for case in test_cases]
        return self._batch_inference(all_messages, max_concurrent_calls, temperature)

@click.command()
@click.option("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
@click.option("--data_path", type=str, default="../data/sft_data/taskbench_data_dailylifeapis.json")
@click.option("--is_api", type=bool, default=False)
@click.option("--host", type=str, default="localhost")
@click.option("--port", type=int, default=13427)
@click.option("--tensor_parallel_size", type=int, default=1)
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

    eval_data = []
    with open(data_path, "r", encoding='utf-8') as f:
        for line in f:
            d = json.loads(line.strip())
            eval_data.append(d)
    # print(eval_data[0])
    labels = []
    for d in eval_data:
        label_str = "\n".join(d["text"])
        labels.append(label_str)
    print(labels[0])

    print("[DEBUG] - main checkpoint 2...")
    if not is_api:
        llm = SealToolsLLM(
            model=model, 
            tensor_parallel_size=tensor_parallel_size, 
            use_sharegpt_format=False,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )
    else:
        llm = SealToolsLLM(model=model)

    print("[DEBUG] - main checkpoint 3...")
    # Run inference
    data_filename = os.path.splitext(os.path.basename(data_path))[0]
    output_path = f"benchmark_results/{model.split('/')[-1]}_sealtools_{data_filename}_results.json"
    parsed_output_path = f"{model.split('/')[-1]}_sealtools_{data_filename}_parsed_results.json"

    def run_inference():
        if os.path.exists(output_path):
            results = json.load(open(output_path, "r"))
        else:
            results = llm.batch_generate(eval_data, max_concurrent_calls=batch_size)
            json.dump(results, open(output_path, "w"), indent=4)
        # parsed_results = parse_json(llm, results, data_split) # TODO
        # json.dump(parsed_results, open(parsed_output_path, "w"), indent=4)
        return results
        
    print("[DEBUG] - main checkpoint 4...")
    if not os.path.exists(parsed_output_path):
        if not is_api:
            with llm.start_server():
                parsed_results = run_inference()
        else:
            parsed_results = run_inference()
    else:
        parsed_results = json.load(open(parsed_output_path, "r"))

    print("[DEBUG] - main checkpoint 5...")
    # evaluate(parsed_results, labels, data_split, tool_desc)
    
    print("[DEBUG] - main checkpoint 6...")

if __name__ == "__main__":
    main()

