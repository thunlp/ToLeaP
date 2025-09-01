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
import concurrent.futures

from benchmark.sealtools.vllm_SealTools_eval import (
    raw_to_pred,
    calculate_score_ToolLearning,
    write_jsonl,
    write_json,
)

from cfg.config import Config
from utils.llm import LLM

def create_messages(conversation_data: List[Dict]) -> List[List[Dict]]:
    """
    Creates a list of messages from conversation data.
    """
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
    """
    Creates the necessary directories for the evaluation.
    """
    paths = [
        os.path.join(eval_data_path, model_name),
        os.path.join(eval_result_path, model_name)
    ]
    for path in paths:
        os.makedirs(path, exist_ok=True)

def load_eval_data(input_data_path: str) -> List[Dict]:
    """
    Loads evaluation data from the specified path.
    """
    print(f"Getting data from {os.path.abspath(input_data_path)}...")
    with open(input_data_path, "r", encoding='utf-8') as f:
        eval_data = json.load(f)
    return eval_data

def process_dataset(
    dataset: str, 
    llm: LLM, 
    model_name: str, 
    input_path: str, 
    raw_data_path: str, 
    eval_data_path: str, 
    eval_result_path: str, 
    is_api: bool, 
    debug_mode: bool
):
    """
    Processes the entire evaluation pipeline for a single dataset, including
    inference, post-processing, and scoring. This function is designed
    to be called independently by a thread.
    """
    print(f"Starting processing for dataset: {dataset}")
    input_data_path = os.path.join(input_path, f"{dataset}.json") 
    eval_data = load_eval_data(input_data_path)

    # Create model output directory
    model_output_dir = os.path.join(raw_data_path, model_name)
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir, exist_ok=True)

    output_path = os.path.join(model_output_dir, f"{dataset}.json")
    print(f"The raw result for {dataset} will be saved to {os.path.abspath(output_path)}...")

    # Use only one sample in debug mode
    if debug_mode:
        eval_data = eval_data[:1]
        print("[Debug] - The first query sample is: ")
        print(eval_data[0]["conversations"][0]["value"])

    ADD_QUERY = "Do not output code; output only the JSON array as shown in the format.\nOutput:\n"
    
    def run_inference() -> List:
        """
        Runs inference. If the result file already exists, it loads it directly.
        Otherwise, it calls the LLM to generate a response.
        """
        if os.path.exists(output_path):
            print(f"Loading existing results from {output_path}")
            with open(output_path, "r") as f:
                results = json.load(f)
            return results
        else: 
            print(f"Running inference for {dataset}...")
            if not is_api:
                results = llm.batch_generate_complete(
                    [{"role": "user", "content": str(ed["conversations"][0]["value"] + ADD_QUERY)} for ed in eval_data]
                )
            else:
                messages = create_messages(eval_data)
                results = llm.batch_generate_chat(messages)
            
            # Save raw inference results
            with open(output_path, "w") as f:
                json.dump(results, f, indent=4)
            return results
        
    test_data = run_inference()

    # Print result and exit in debug mode
    if debug_mode:
        print("[Debug] - The answer is: ")
        print(test_data[0]) 
        if len(test_data) > 0:
            raise SystemExit("[Debug] Halting after one sample in debug mode.")

    # Post-processing and scoring
    eval_data_processed = raw_to_pred(output_path, input_data_path)
    eval_data_filename = os.path.join(eval_data_path, model_name, dataset + ".json")
    write_jsonl(eval_data_filename, eval_data_processed) 
    
    result, badcases = calculate_score_ToolLearning(eval_data_filename) 
    
    # Save final scores and bad cases
    result_data_filename = os.path.join(eval_result_path, model_name, dataset + ".json")
    badcases_filename = os.path.join(eval_result_path, model_name, f"{model_name}-seal-{dataset}.json")
    write_json(result_data_filename, result, indent=4)
    write_json(badcases_filename, badcases, indent=4)
    
    print(f"Finished processing for dataset: {dataset}")
    return {dataset: result}

@click.command()
@click.option("--model", type=str, default="/hy-tmp/Seal-8B-v1.1", help="Path or name of the model.")
@click.option("--dataset_name_list", type=str, default="test_out_domain,dev,test_in_domain", help="Comma-separated list of datasets to evaluate.")
@click.option("--input_path", type=str, default="../data/sealtools", help="Path to the input datasets folder.")
@click.option("--raw_data_path", type=str, default="../results/sealtools/raw_pred_data", help="Path to the folder for raw model predictions.")
@click.option("--eval_data_path", type=str, default='../results/sealtools/pred_data', help="Path to the folder for processed prediction data.")
@click.option("--eval_result_path", type=str, default='../results/sealtools/eval_result', help="Path to the folder for final evaluation results.")
@click.option("--is_api", type=bool, default=False, help="Whether to use an API for inference.")
@click.option("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size for vLLM.")
@click.option("--batch_size", type=int, default=128, help="Batch size for inference.")
@click.option("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization.")
@click.option("--max_model_len", type=int, default=8192, help="Maximum model length.")
@click.option("--max_output_tokens", type=int, default=1024, help="Maximum number of output tokens.")
@click.option("--model_name", type=str, required=True, help="Name of the model, used for creating result directories.")
@click.option("--debug", "debug_mode", is_flag=True, default=False, help="Debug mode, runs only one data sample.")
@click.option("--think_mode", "think_mode", is_flag=True, default=False, help="Enable chain-of-thought mode.")
@click.option("--think_special_tokens", "think_special_tokens", type=str, default="think", help="Special token used in chain-of-thought mode.")
@click.option("--multithread", type=int, default=3, help="Number of threads for parallel dataset evaluation. Set to 1 for sequential execution.")
def main(
    model: str, 
    dataset_name_list: str, 
    input_path : str,
    raw_data_path: str,
    eval_data_path: str,
    eval_result_path: str,
    is_api: bool, 
    tensor_parallel_size: int, 
    batch_size: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_output_tokens: int,
    model_name: str,
    debug_mode: bool,
    think_mode: bool,
    think_special_tokens: str,
    multithread: int
    ):
    """
    Main function for the Seal-Tools evaluation script.
    """
    # Parse dataset list
    datasets = [name.strip() for name in dataset_name_list.split(',')]
    
    # Create result directories
    create_directories(eval_data_path, eval_result_path, model_name)
    
    # Initialize LLM
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
        gpu_memory_utilization=gpu_memory_utilization
    )
    
    all_results = {}
    
    # Choose execution mode based on the multithread parameter
    if multithread > 1:
        print(f"Multithreading enabled. Processing datasets in parallel with {multithread} workers...")
        # Execute tasks in parallel using a thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=multithread) as executor:
            future_to_dataset = {
                executor.submit(
                    process_dataset, 
                    dataset, llm, model_name, input_path, raw_data_path, 
                    eval_data_path, eval_result_path, is_api, debug_mode
                ): dataset 
                for dataset in datasets
            }
            
            # Collect results as tasks complete
            for future in concurrent.futures.as_completed(future_to_dataset):
                dataset_name = future_to_dataset[future]
                try:
                    result_dict = future.result()
                    all_results.update(result_dict)
                except Exception as exc:
                    print(f"'{dataset_name}' generated an exception: {exc}")
                    
    else:
        print("Processing datasets sequentially...")
        # Sequential execution logic
        for dataset in datasets:
            result_dict = process_dataset(
                dataset, llm, model_name, input_path, raw_data_path, 
                eval_data_path, eval_result_path, is_api, debug_mode
            )
            all_results.update(result_dict)

    print("\n--- Final Evaluation Results ---")
    print(json.dumps(all_results, indent=4))

if __name__ == "__main__":
    main()
