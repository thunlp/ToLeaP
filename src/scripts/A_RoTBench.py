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

class RoTBench(LLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create_messages(self, conversation_data: Dict) -> List[Dict]:
        """Create messages list from conversation data"""
        print("[DEBUG] - _create_messages checkpoint 1...")
        messages = []
        conversations = conversation_data["conversations"][:-1]  # the last as label
        for conv in conversations:
            if conv["from"] == "system":
                messages.append({
                    "role": "system",
                    "content": conv["value"]
                })
            elif conv["from"] == "user":
                messages.append({
                    "role": "user",
                    "content": conv["value"]
                })
            elif conv["from"] == "assistant":
                if isinstance(conv["value"], list):
                    conv["value"] = "\n".join(conv["value"])
                messages.append({
                    "role": "assistant",
                    "content": conv["value"]
                })
        return messages

    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        prompt = ""
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                prompt += f"[System]\n{content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        prompt += "Assistant:"
        return prompt

    def _batch_inference(self, messages_batch: List[List[Dict]], max_concurrent_calls: int = 2, temperature: float = 0) -> List[str]:
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm

        print("[DEBUG] Using modified _batch_inference...")

        responses = [None] * len(messages_batch)

        if not self.use_hf:
            print("[DEBUG] - Using OpenAI API style inference ...")

            def process_single_message(index, messages):
                try:
                    chat_output = self.client.chat.completions.create(
                        model=self.model_path_or_name,
                        messages=messages,
                        temperature=temperature,
                    )
                    return index, chat_output.choices[0].message.content
                except Exception as e:
                    print(f"Error: {e}")
                    return index, None

            with ThreadPoolExecutor(max_workers=max_concurrent_calls) as executor:
                futures = {
                    executor.submit(process_single_message, idx, msgs): idx
                    for idx, msgs in enumerate(messages_batch)
                }
                for future in tqdm(as_completed(futures), total=len(messages_batch), desc="Processing concurrent calls"):
                    index, result = future.result()
                    responses[index] = result

        else:
            # 如果是使用 HF pipeline，则先把 messages -> prompt_text，然后再调用 pipeline
            print("[DEBUG] - Using HF pipeline inference ...")

            def process_single_prompt(index, msgs):
                prompt_text = self._messages_to_prompt(msgs)
                outputs = self.pipeline(prompt_text, max_length=1024, temperature=temperature)
                gen_text = outputs[0]["generated_text"]
                if gen_text.startswith(prompt_text):
                    gen_text = gen_text[len(prompt_text):]
                return index, gen_text

            with ThreadPoolExecutor(max_workers=max_concurrent_calls) as executor:
                futures = {
                    executor.submit(process_single_prompt, idx, msgs): idx
                    for idx, msgs in enumerate(messages_batch)
                }
                for future in tqdm(as_completed(futures), total=len(messages_batch), desc="Processing concurrent calls"):
                    index, result = future.result()
                    responses[index] = result

        return [r for r in responses]

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

    with open(data_path, "r", encoding='utf-8') as f:
        eval_data = json.load(f)

    labels = []
    for d in eval_data:
        assistant_responses = []
        for c in d["conversations"]:
            if c["from"] == "assistant":
                if isinstance(c["value"], list):
                    assistant_responses.extend(c["value"])
                else:
                    assistant_responses.append(c["value"])
        label_str = "\n".join(assistant_responses)
        labels.append(label_str)

    print(labels[0] if labels else "No labels found")

    print("[DEBUG] - main checkpoint 2...")
    if not is_api:
        llm = RoTBench(
            model=model, 
            tensor_parallel_size=tensor_parallel_size, 
            use_sharegpt_format=True,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
        )
    else:
        llm = RoTBench(model=model)

    print("[DEBUG] - main checkpoint 3...")
    # Run inference
    data_filename = os.path.splitext(os.path.basename(data_path))[0]
    output_path = f"benchmark_results/{model.split('/')[-1]}_rotbench_{data_filename}_results.json"

    def run_inference():
        print("[DEBUG] - run_inference checkpoint 1...")
        if os.path.exists(output_path):
            results = json.load(open(output_path, "r"))
        else:
            results = llm.batch_generate(eval_data, max_concurrent_calls=batch_size)
            json.dump(results, open(output_path, "w"), indent=4)
        print("[DEBUG] - run_inference checkpoint 2...")
        return results
        
    print("[DEBUG] - main checkpoint 4...")
    if not os.path.exists(output_path):
        print("[DEBUG] - main branch checkpoint 1...")
        if not is_api:
            print("[DEBUG] - main branch checkpoint 2...")
            with llm.start_server():
                results = run_inference()
        else:
            print("[DEBUG] - main branch checkpoint 3...")
            results = run_inference()
    else:
        print("[DEBUG] - main branch checkpoint 4...")
        results = json.load(open(output_path, "r"))

    print("[DEBUG] - main checkpoint 5...")
    # 这里可以进行后续的评估逻辑
    print("[DEBUG] - main checkpoint 6...")

if __name__ == "__main__":
    main()
