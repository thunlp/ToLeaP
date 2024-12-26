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
from cfg.config import Config


conf = Config()


# TODO: load config, use .env file
class LLM:
    def __init__(
        self,
        model: str = "/hy-tmp/3.1-8B", # 这个是vllm的model path，和huggingface保持统一
        gpu_memory_utilization: float = 0.9, # 0-1，代表用多少gpu。越高占显存越多但batch处理会加速
        dtype: str = None, # 建议不设，用模型config自己的
        tensor_parallel_size: int = 1, # 用多少张gpu
        use_api_model: bool = False, # 是否调用本地api model
        use_sharegpt_format: bool = False, # 是否使用sharegpt格式
        max_past_message_include: int = -1, # 历史记录看多少，-1代表全部
        max_output_tokens: int = 1024, # 输出最大token数，-1代表不限制
    ):
        # env initialization
        self.port = conf.port
        self.host = conf.host
        self.api_key = conf.api_key
        self.hf_raw = conf.hf_raw
        self.hf_pipeline = conf.hf_pipeline

        # model initialization
        self.model_path_or_name = model
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.server_process = None
        self.tensor_parallel_size = tensor_parallel_size
        self.max_past_message_include = max_past_message_include
        self.use_api_model = use_api_model
        self.use_sharegpt_format = use_sharegpt_format
        self.max_output_tokens = max_output_tokens

        # load model
        if self.hf_raw:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path_or_name,
                device_map="auto",
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path_or_name,
                trust_remote_code=True
            )
        elif self.hf_pipeline:
            self.pipeline = pipeline(
                "text-generation",
                model=self.model_path_or_name,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            if self.use_api_model:
                self.client = OpenAI(api_key=self.api_key, base_url=conf.api_base)
            else:
                self.client = OpenAI(api_key=self.api_key, base_url=f"http://{self.host}:{self.port}/v1")

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
    
    # openai format
    def _create_messages_from_user(self, user_prompt, system_prompt=None, former_messages=[], shrink_multiple_break=False):
        if shrink_multiple_break:
            while "\n\n\n" in user_prompt:
                user_prompt = user_prompt.replace("\n\n\n", "\n\n")
            while "\n\n\n" in system_prompt:
                system_prompt = system_prompt.replace("\n\n\n", "\n\n")

        messages = []

        if system_prompt is not None:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })

        if self.max_past_message_include > 0:
            messages.extend(former_messages[-1 * self.max_past_message_include :])
        else:
            messages.extend(former_messages)
        messages.append(
            {
                "role": "user",
                "content": user_prompt,
            }
        )
        return messages

    # single inference
    def _single_inference(self, messages: List[Dict], temperature: float = 0):
        if not self.use_hf:
            chat_output = self.client.chat.completions.create(
                model=self.model_path_or_name,
                messages=messages,
                temperature=temperature,
            )
            return chat_output.choices[0].message.content
        elif self.hf_pipeline:
            outputs = self.pipeline(messages)
            return outputs[0]["generated_text"][-1]
        else:
            inputs = self.tokenizer(messages, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(**inputs, max_length=self.max_output_tokens)
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    # batch inference
    def _batch_inference(self, messages_batch: List[List[Dict]], max_concurrent_calls: int = 2, temperature: float = 0) -> List[Dict]:
        """Run inference for a batch of messages with concurrent processing, preserving order."""
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

        with ThreadPoolExecutor(max_workers=max_concurrent_calls) as executor:
            if not self.use_hf:
                futures = {executor.submit(process_single_message, idx, messages): idx 
                        for idx, messages in enumerate(messages_batch)}
            elif self.hf_pipeline:
                futures = {executor.submit(process_single_message_hf, idx, messages): idx 
                        for idx, messages in enumerate(messages_batch)}
            else:
                raise ValueError("batch_generate only support openai format messages, please use single_generate for hf_raw inference.")

            for future in tqdm(as_completed(futures), total=len(messages_batch), desc="Processing concurrent calls"):
                try:
                    index, result = future.result()
                    responses[index] = result
                except Exception as e:
                    print(f"An error occurred for batch {futures[future]}: {e}")
                    responses[futures[future]] = None
        
        return responses

    def batch_generate(self, test_cases: List[Dict], max_concurrent_calls: int = 2, temperature: float = 0) -> List[Dict]:
        """Process test cases"""
        if self.use_sharegpt_format:
            all_messages = [self._create_messages_from_sharegpt(case) for case in test_cases]
        else:
            all_messages = test_cases

        return self._batch_inference(all_messages, max_concurrent_calls, temperature)
    
    def single_generate(self, user_prompt, system_prompt=None, former_messages=[], shrink_multiple_break=False, temperature: float = 0):
        if not self.hf_raw:
            messages = self._create_messages_from_user(user_prompt, system_prompt, former_messages, shrink_multiple_break)
        else:
            messages = user_prompt
        return self._single_inference(messages, temperature)
