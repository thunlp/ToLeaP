import argparse
import os
import random
import mmengine
from tqdm import tqdm
import click
import sys
import time
from contextlib import nullcontext
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import copy

# 在文件开头设置GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"  # 使用0-7号卡

current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(current_dir, '..')
sys.path.append(utils_dir)
from utils.llm import LLM


thought_prompt = """You are an expert AI assistant that explains your reasoning step by step. For each step, provide a title that describes what you're doing in that step, along with the content. Decide if you need another step or if you're ready to give the final answer.concise final answer based on the comprehensive analysis USE AS MANY REASONING STEPS AS POSSIBLE. AT LEAST 3. BE AWARE OF YOUR LIMITATIONS AS AN LLM AND WHAT YOU CAN AND CANNOT DO. IN YOUR REASONING, INCLUDE EXPLORATION OF ALTERNATIVE ANSWERS. CONSIDER YOU MAY BE WRONG, AND IF YOU ARE WRONG IN YOUR REASONING, WHERE IT WOULD BE. FULLY TEST ALL OTHER POSSIBILITIES. YOU CAN BE WRONG. WHEN YOU SAY YOU ARE RE-EXAMINING, ACTUALLY RE-EXAMINE, AND USE ANOTHER APPROACH TO DO SO. DO NOT JUST SAY YOU ARE RE-EXAMINING. USE AT LEAST 3 METHODS TO DERIVE THE ANSWER. USE BEST PRACTICES."""


def is_valid_conversation(conversation: Dict) -> bool:
    """检查对话是否符合 human-gpt 交替格式"""
    turns = conversation.get("conversations", [])
    if not turns:
        return False
        
    if turns[0]["from"] != "human":
        return False
        
    for i in range(len(turns)):
        expected_role = "human" if i % 2 == 0 else "gpt"
        if turns[i]["from"] != expected_role:
            return False
            
    return True


class DatasetGeneratorLLM(LLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate(self, test_case, temperature=0.7):
        try:
            messages = self._create_messages_from_sharegpt(test_case)
            if not messages:
                return ""
            
            current_messages = messages[-1]
            
            # 生成回复
            response = self._single_inference(current_messages, temperature)
            return response
            
        except Exception as e:
            print(f"Error in generate: {e}")
            return ""

    def _create_messages_from_sharegpt(self, conversation_data: Dict) -> List[List[Dict]]:
        system_prompt = conversation_data.get("system", "")
        conversations = conversation_data["conversations"]
        
        base_messages = []
        if system_prompt:
            combined_system = f"{thought_prompt}\n\n{system_prompt}"
            base_messages.append({"role": "system", "content": combined_system})
        
        # 只创建一组消息，包含当前context的所有历史
        current_messages = base_messages.copy() 
        
        # 添加所有历史消息
        for turn in conversations:
            if turn["from"] == "human":
                current_messages.append({"role": "user", "content": turn["value"]})
            elif turn["from"] == "gpt":
                current_messages.append({"role": "assistant", "content": turn["value"]})
        
        return [current_messages]  # 返回单个消息列表

    def _single_inference(self, messages, temperature=0.7):
        """运行单次推理"""
        try:
            chat_output = self.client.chat.completions.create(
                    model=self.model_path_or_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=1024
                )
            return chat_output.choices[0].message.content
        except Exception as e:
            print(f"Inference error: {e}")
            return ""

    def _batch_inference(self, conversations_batch: List[Dict], max_concurrent_calls=4, temperature=0.7):
        """并行处理多个对话"""
        try:
            with ThreadPoolExecutor(max_workers=max_concurrent_calls) as executor:
                futures = []
                for conv in conversations_batch:
                    future = executor.submit(self._process_single_conversation, conv, temperature)
                    futures.append(future)
                
                all_responses = []
                for future in futures:
                    responses = future.result()
                    all_responses.append(responses)
                    
                return all_responses
            
        except Exception as e:
            print(f"Error in batch inference: {e}")
            return [[] for _ in range(len(conversations_batch))]

    def _process_single_conversation(self, conversation: Dict, temperature=0.7):
        responses = []
        messages = []
        
        if conversation.get("system"):
            combined_system = f"{thought_prompt}\n\n{conversation['system']}"
            messages.append({"role": "system", "content": combined_system})
        
        for turn in conversation["conversations"]:
            if turn["from"] == "human":
                messages.append({"role": "user", "content": turn["value"]})
                response = self._single_inference(messages.copy(), temperature)
                responses.append(response)
            elif turn["from"] == "gpt":
                messages.append({"role": "assistant", "content": turn["value"]})
                
        return responses


def load_dataset(file_path: str) -> List[Dict]:
    """Load dataset from file and filter invalid conversations"""
    try:
        data = mmengine.load(file_path)
        if not isinstance(data, list):
            data = [data]
            
        # 直接过滤掉无效对话
        valid_data = []
        invalid_count = 0
        empty_count = 0
        
        for conv in data:
            # 检查是否为空对话
            if not conv.get("conversations"):
                empty_count += 1
                continue
                
            # 检查是否符合格式要求
            if not is_valid_conversation(conv):
                invalid_count += 1
                continue
                
            valid_data.append(conv)
        
        print(f"Filtered stats for {file_path}:")
        print(f"- Total conversations: {len(data)}")
        print(f"- Empty conversations: {empty_count}")
        print(f"- Invalid format: {invalid_count}")
        print(f"- Valid conversations: {len(valid_data)}")
        
        return valid_data
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []


@click.command()
@click.option("--model", type=str, default="/hy-tmp/model-dir")
@click.option("--input_dir", type=str, help="Path to input directory")
@click.option("--is_api", type=bool, default=False)
@click.option("--out_dir", type=str, default="generated_datasets/")
@click.option("--tensor_parallel_size", type=int, default=8)
@click.option("--batch_size", type=int, default=2)
@click.option("--gpu_memory_utilization", type=float, default=0.9)
@click.option("--temperature", type=float, default=0.7)
@click.option("--resume", is_flag=True)
def main(
    model: str,
    input_dir: str,
    is_api: bool,
    out_dir: str,
    tensor_parallel_size: int,
    batch_size: int,
    gpu_memory_utilization: float,
    temperature: float,
    resume: bool
):
    os.makedirs(out_dir, exist_ok=True)
    
    # 获取输入目录中的所有json文件
    input_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    input_files.sort()
    print(f"Found {len(input_files)} json files in {input_dir}")
    
    # Initialize model
    generator = DatasetGeneratorLLM(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        use_api_model=is_api
    )
    
    context = generator.start_server() if not is_api else nullcontext()
    
    with context:
        for input_file in input_files:
            input_path = os.path.join(input_dir, input_file)
            output_path = os.path.join(out_dir, input_file)
            base_name = os.path.splitext(input_file)[0]
            
            print(f"\nProcessing file: {input_file}")
            
            dataset = mmengine.load(input_path)
            if not isinstance(dataset, list):
                dataset = [dataset]
            
            results = []
            pbar = tqdm(total=len(dataset), desc=f"Processing {input_file}")
            
            save_batch_interval = 20
            save_interval = batch_size * save_batch_interval
            
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i + batch_size]
                batch_results = []
                
                for conversation in batch:
                    new_conversation = {
                        "conversations": [],
                        "system": conversation.get("system", "")
                    }
                    
                    # 首先收集所有原始的gpt回复
                    original_gpt_responses = {}
                    for idx, turn in enumerate(conversation["conversations"]):
                        if turn["from"] == "human":
                            # 找到这个human turn后面的第一个gpt回复
                            for next_turn in conversation["conversations"][idx+1:]:
                                if next_turn["from"] == "gpt":
                                    original_gpt_responses[idx] = next_turn["value"]
                                    break
                    
                    # 然后处理对话
                    conversation_history = []
                    for idx, turn in enumerate(conversation["conversations"]):
                        if turn["from"] == "human":
                            # 添加human turn
                            new_conversation["conversations"].append(turn.copy())
                            conversation_history.append(turn.copy())
                            
                            # 准备生成context
                            current_context = {
                                "conversations": conversation_history.copy(),
                                "system": conversation.get("system", "")
                            }
                            
                            # 生成新的回复
                            response = generator.generate(
                                test_case=current_context,
                                temperature=temperature
                            )
                            
                            # 获取对应的原始回复
                            original_gpt = original_gpt_responses.get(idx, "")
                            
                            # 清理和组合回复
                            cleaned_response = response.replace("<thought>", "").replace("</thought>", "").replace("<output>", "").replace("</output>", "")
                            cleaned_original = original_gpt.replace("<thought>", "").replace("</thought>", "").replace("<output>", "").replace("</output>", "")
                            
                            combined_response = f"<thought> {cleaned_response} </thought> \n <output> {cleaned_original} </output>"
                            
                            # 添加到对话中
                            gpt_turn = {
                                "from": "gpt",
                                "value": combined_response
                            }
                            new_conversation["conversations"].append(gpt_turn)
                            conversation_history.append(gpt_turn)
                    
                    batch_results.append(new_conversation)
                
                results.extend(batch_results)
                pbar.update(len(batch))
                
                # 修改临时文件命名逻辑
                current_count = len(results)
                if current_count > 0 and current_count % save_interval == 0:
                    temp_output_path = os.path.join(out_dir, f"{base_name}_temp_{current_count}.json")
                    print(f'Saving at conversation {current_count}')
                    mmengine.dump(results, temp_output_path, indent=4)
                    print(f"\nSaved intermediate results ({current_count} conversations) to {temp_output_path}")
            
            # 保存最终结果
            mmengine.dump(results, output_path, indent=4)
            print(f"\nSaved final results to {output_path}")
        


if __name__ == "__main__":
    main( )
