import time
import json
from typing import List, Dict
from tqdm import tqdm
from openai import OpenAI
import subprocess
import requests
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed

#todo adding python -m vllm.entrypoints.openai.api_server --model /hy-tmp/3.1-8B --dtype bfloat16   --gpu-memory-utilization 0.9 --host 0.0.0.0 --port 8000
#inside the class

# model handler map
# offline infernce 

#when using, you need manually open the serve like this 
#vllm serve /home/test/test03/models/Qwen2.5-7B-Instruct \
    # --port 8000 \
    # --dtype bfloat16 \
    # --gpu-memory-utilization 0.8 \
    # --tensor-parallel-size 2 \

class LLM:
    def __init__(
        self,
        api_key: str = "",
        api_base: str = None,
        model: str = "/hy-tmp/3.1-8B",
        gpu_memory_utilization: float = 0.9,
        host: str = "0.0.0.0",
        dtype: str = None,
        port: int = 8000,
        tensor_parallel_size: int = 1,
    ):
        """Initialize LLM with API configurations"""
        self.api_key = api_key
        self.model_path_or_name = model
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.server_process = None
        self.port = port
        self.host = host
        self.tensor_parallel_size = tensor_parallel_size

        if api_base is not None:
            self.api_base = api_base
        else:   
            self.api_base = f"http://{host}:{port}/v1"

        self.client = OpenAI(api_key=api_key, base_url=self.api_base)

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

    def _create_messages(self, conversation_data: Dict) -> List[Dict]:
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

    def _batch_inference(self, messages_batch: List[List[Dict]], max_concurrent_calls: int = 2, temperature: float = 0) -> List[Dict]:
        """Run inference for a batch of messages with concurrent processing, preserving order."""
        time_start = time.time()
        
        responses = [None] * len(messages_batch)  # Pre-allocate a list to preserve order

        def process_single_message(index, messages):
            chat_output = self.client.chat.completions.create(
                model=self.model_path_or_name,
                messages=messages,
                temperature=temperature,
            )
            return index, chat_output.choices[0].message.content

        with ThreadPoolExecutor(max_workers=max_concurrent_calls) as executor:
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

    def __call__(self, test_cases: List[Dict], max_concurrent_calls: int = 2, temperature: float = 0) -> List[Dict]:
        """Process test cases"""
        all_messages = [self._create_messages(case) for case in test_cases]
        return self._batch_inference(all_messages, max_concurrent_calls, temperature)