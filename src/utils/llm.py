import time
import json
from typing import List, Dict
from tqdm import tqdm
from openai import OpenAI
import subprocess
import requests
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import os
import torch
from cfg.config import Config
from vllm import LLM as VLLM_LLM, SamplingParams

conf = Config()
STRICT_JSON_FORMAT_SYSTEM_PROMPT = "You are a helpful assistant that responds all questions in strict JSON format. Your response must be able to be directly parsed into a JSON object (i.e. start with '{' or '[' and end with '}' or ']')."

def extract_first_json(text):
    """
    Extracts the first JSON object or array from a string.
    
    Args:
        text (str): Input string containing JSON object(s)
        
    Returns:
        str: First complete JSON object/array found, or None if no valid JSON object is found
    """
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

# TODO: load config, use .env file
class LLM:
    def __init__(
        self,
        model: str = "/hy-tmp/3.1-8B", # 这个是vllm的model path，和huggingface保持统一
        gpu_memory_utilization: float = 0.9, # 0-1，代表用多少gpu。越高占显存越多但batch处理会加速
        is_api: bool = False,
        dtype: str = None, # 建议不设，用模型config自己的
        tensor_parallel_size: int = 1, # 用多少张gpu
        use_sharegpt_format: bool = False, # 是否使用sharegpt格式
        max_past_message_include: int = -1, # 历史记录看多少，-1代表全部
        max_input_tokens: int = None, # 输入最大token数，-1代表使用默认
        max_output_tokens: int = 4096, # 输出最大token数，-1代表使用默认
        batch_size: int = 1, # 批处理大小
    ):
        # env initialization
        self.port = conf.port
        self.host = conf.host
        self.api_key = conf.api_key
        self.use_chat = conf.use_chat
        # model initialization
        self.model_path_or_name = model
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.server_process = None
        self.tensor_parallel_size = tensor_parallel_size
        self.max_past_message_include = max_past_message_include
        self.use_sharegpt_format = use_sharegpt_format
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.batch_size = batch_size
        self.is_api = is_api
        
        if self.use_chat:
            if self.is_api:
                self.client = OpenAI(api_key=self.api_key, base_url=conf.api_base)
            else:
                self.client = OpenAI(api_key=self.api_key, base_url=f"http://{self.host}:{self.port}/v1")
        else:
            self.model = VLLM_LLM(
                model=self.model_path_or_name, 
                tensor_parallel_size=self.tensor_parallel_size,
                max_model_len=self.max_input_tokens
            )

    def estimate_max_input_tokens(self, all_messages: List[Dict]):
        if self.use_sharegpt_format:
            all_messages = [self._create_messages_from_sharegpt(case) for case in all_messages]
        else:
            all_messages = all_messages

        max_number_tokens = -1
        for message in all_messages:
            tokens = self.tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True)
            max_number_tokens = max(max_number_tokens, len(tokens))

        return max_number_tokens

    @contextmanager
    def start_server(self):
        """Context manager for starting and stopping the server"""
        cmd = ["vllm", "serve", self.model_path_or_name, "--gpu-memory-utilization", str(self.gpu_memory_utilization)]
        
        if self.dtype is not None:
            cmd.extend(["--dtype", self.dtype])
            
        if self.api_key is not None and self.api_key != "":
            cmd.extend(["--api-key", self.api_key])
            
        if self.port is not None:
            cmd.extend(["--port", str(self.port)])

        if self.host is not None:
            cmd.extend(["--host", self.host])
            
        if self.tensor_parallel_size is not None:
            cmd.extend(["--tensor-parallel-size", str(self.tensor_parallel_size)])
        
        # Start the server as a subprocess
        with open("serve.log", "w") as log_file:
            server_process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=log_file,
                text=True
            )
        
            # Wait for server to start up
            print("Starting server, waiting until ready...")
            while True:
                try:
                    # Try to connect to check if server is up
                    test_question = "True or False: 1+1=2?"
                    response = self.client.chat.completions.create(
                        model=self.model_path_or_name,
                        messages=[{"role": "user", "content": test_question}],
                        max_tokens=1,
                        timeout=10,
                    )
                    print("Test question: ", test_question, "Response: ", response.choices[0].message.content)
                    print("Successfully got response from server, evaluation will start soon...")
                    break
                except Exception as e:
                    print(f"Server not ready ({str(e)}), waiting...")
                    time.sleep(10)

            try:
                yield server_process
            finally:
                server_process.terminate()
                server_process.wait()

    # sharegpt format
    def _create_messages_from_sharegpt(self, conversation_data: Dict) -> List[Dict]:
        """Create messages list from conversation data"""
        messages = []
        
        # system prompt
        if "system" in conversation_data:
            messages.append({
                "role": "system",
                "content": conversation_data["system"]
            })

        if "tools" in conversation_data:
            messages[0]["content"] += f"\nAvailable tools: {conversation_data['tools']}"
        
        # getting the last as label
        conversations = conversation_data["conversations"][:-1]  
        
        for conv in conversations:
            if conv["from"] == "human":
                messages.append({
                    "role": "user",
                    "content": "USER: " + conv["value"]
                })
            elif conv["from"] == "gpt":
                messages.append({
                    "role": "assistant",
                    "content": "ASSISTANT: " + conv["value"]
                })
        
        return messages
        
    def batch_generate_chat(self, test_cases: List[Dict], temperature: float = 0) -> List[Dict]:
        """Run inference for a batch of test cases with concurrent processing, preserving order."""
        # Convert test cases to messages format if needed
        if self.use_sharegpt_format:
            messages_batch = [self._create_messages_from_sharegpt(case) for case in test_cases]
        else:
            messages_batch = test_cases

        responses = [None] * len(messages_batch)  # Pre-allocate a list to preserve order

        def process_single_message(index, messages):
            chat_output = self.client.chat.completions.create(
                model=self.model_path_or_name,
                messages=messages,
                temperature=temperature,
            )
            return index, chat_output.choices[0].message.content
        
        def process_single_message_hf(index, messages):
            outputs = self.pipeline(messages, max_length=self.max_output_tokens)
            return index, outputs[0]["generated_text"][-1]

        with ThreadPoolExecutor(max_workers=self.batch_size) as executor:
            futures = {executor.submit(process_single_message, idx, messages): idx 
                        for idx, messages in enumerate(messages_batch)}

            for future in tqdm(as_completed(futures), total=len(messages_batch), desc="Processing concurrent calls"):
                try:
                    index, result = future.result()
                    responses[index] = result
                except Exception as e:
                    print(f"An error occurred for batch {futures[future]}: {e}")
                    responses[futures[future]] = None
        
        return responses
    
    def batch_generate_complete(self, test_cases: List[str], temperature: float = 0) -> List[Dict]:
        gen_params = SamplingParams(temperature=temperature, max_tokens=self.max_output_tokens)
        all_outputs = []
        for i in tqdm(range(0, len(test_cases), self.batch_size), desc="Processing batch"):
            batch_messages = test_cases[i:i+self.batch_size]
            outputs = self.model.generate(batch_messages, gen_params, use_tqdm=False)
            for output in outputs:
                all_outputs.append(output.outputs[0].text)
        return all_outputs
