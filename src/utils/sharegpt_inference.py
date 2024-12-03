import time
import json
from typing import List, Dict
from tqdm import tqdm
from openai import OpenAI


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
        api_key: str = "Nah",
        api_base: str = "http://localhost:8000/v1",
        model_path: str = "/hy-tmp/3.1-8B",
        model_name: str = "llama-3-8b-instruct"
        stop_token: str = "<|eot_id|>"
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.9,
        host: str = "0.0.0.0",
        port: int = 8000
    ):
        """Initialize LLM with API configurations"""
        self.api_key = api_key
        self.api_base = api_base
        self.model_path = model_path
        self.model_name = model_name
        self.stop_token = stop_token
        self.dtype = dtype
        self.gpu_memory_utilization = gpu_memory_utilization
        self.host = host
        self.port = port
        self.server_process = None

        self.start_server()

        self.client = OpenAI(api_key=api_key, base_url=api_base)

    def create_messages(self, conversation_data: Dict) -> List[Dict]:
        """Create messages list from conversation data"""
        messages = []
        
        # system prompt
        messages.append({
            "role": "system",
            "content": conversation_data["system"]
        })
        messages[0]["content"] += f"\nAvailable tools: {conversation_data['tools']}"
        
        # getting the last as label
        conversations = conversation_data["conversations"][:-1]  
        
        for conv in conversations:
            if conv["from"] == "human":
                messages.append({
                    "role": "user",
                    "content": conv["value"]
                })
            elif conv["from"] == "gpt":
                messages.append({
                    "role": "assistant",
                    "content": conv["value"]
                })
        
        return messages

    def _batch_inference(self, messages_batch: List[List[Dict]], temperature: float = 0) -> List[Dict]:
        """Run inference for a batch of messages"""
        time_start = time.time()
        
        responses = []
        for messages in messages_batch:
            chat_output = self.client.chat.completions.create(
                model=self.model_path,
                messages=messages,
                temperature=temperature,
                stop=self.stop_token  #need to modify as differetn model i think
            )
            responses.append(
                chat_output.choices[0].message.content)
            
        time_end = time.time()
        print(f"Batch processing time: {time_end - time_start:.2f}s")
        return responses

    def __call__(self, test_cases: List[Dict], batch_size: int = 2, temperature: float = 0) -> List[Dict]:
        """Process test cases in batches"""
        all_messages = [self._create_messages(case) for case in test_cases]
        
        results = []
        for i in tqdm(range(0, len(all_messages), batch_size), desc="Processing batches"):
            batch = all_messages[i:i + batch_size]
            batch_results = self._batch_inference(batch, temperature=temperature)
            results.extend(batch_results)

            time.sleep(1)
            
        return results
    

