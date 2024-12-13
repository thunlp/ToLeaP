import time
import json
from typing import List, Dict
from tqdm import tqdm
from openai import OpenAI
import subprocess
import requests
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed


# TODO: load config, use .env file
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
    
    def _create_messages_from_user(self, user_prompt, system_prompt=None, former_messages=[], shrink_multiple_break=False):
        if shrink_multiple_break:
            while "\n\n\n" in user_prompt:
                user_prompt = user_prompt.replace("\n\n\n", "\n\n")
            while "\n\n\n" in system_prompt:
                system_prompt = system_prompt.replace("\n\n\n", "\n\n")
        system_prompt = self.cfg.default_system_prompt if system_prompt is None else system_prompt
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]
        messages.extend(former_messages[-1 * self.cfg.max_past_message_include :])
        messages.append(
            {
                "role": "user",
                "content": user_prompt,
            }
        )
        return messages

    def _single_inference(self, user_prompt, system_prompt=None, former_messages=[], shrink_multiple_break=False, functions=None):
        # TODO: vllm single inference
        if self.cfg.use_llama:
            inputs = self.tokenizer(user_prompt, return_tensors="pt")
            outputs = self.model.generate(**inputs, max_length=1024)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            messages = self._create_messages_from_user(user_prompt, system_prompt, former_messages, shrink_multiple_break)
            if functions:
                completion = self.client.chat.completions.create(model=self.api_model,
                                                                messages=messages,
                                                                functions=functions,
                                                                function_call="auto")
                response = completion.choices[0].message.content
                if response is None:
                    response = [completion.choices[0].message.function_call.name,
                                completion.choices[0].message.function_call.arguments]
                return response
            else:
                completion = self.client.chat.completions.create(model=self.api_model,
                                                                messages=messages)
                response = completion.choices[0].message.content
        return response

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

        # TODO: OpenAI multiple workers
        # TODO: huggingface batch inference
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
        all_messages = [self._create_messages_from_sharegpt(case) for case in test_cases]
        return self._batch_inference(all_messages, max_concurrent_calls, temperature)