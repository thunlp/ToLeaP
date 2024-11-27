import time
import json
from typing import List, Dict
from tqdm import tqdm
from openai import OpenAI


#todo adding python -m vllm.entrypoints.openai.api_server --model /hy-tmp/3.1-8B --dtype bfloat16   --gpu-memory-utilization 0.9 --host 0.0.0.0 --port 8000
#inside the class

# model handler map
# offline infernce 


class LLM:
    def __init__(
        self,
        api_key: str = "Nah",
        api_base: str = "http://localhost:8000/v1",
        model_path: str = "/hy-tmp/3.1-8B",
        model_name: str = "llama-3-8b-instruct"
    ):
        """Initialize LLM with API configurations"""
        self.api_key = api_key
        self.api_base = api_base
        self.model_path = model_path
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key, base_url=api_base)

    def _create_messages(self, conversation_data: Dict) -> List[Dict]:
        """Create messages list from conversation data"""
        messages = []
        messages.append({
            "role": "system",
            "content": conversation_data["system"]
        })
        messages[0]["content"] += f"\nAvailable tools: {conversation_data['tools']}"
        
        human_message = next(
            conv["value"] for conv in conversation_data["conversations"]
            if conv["from"] == "human"
        )
        messages.append({
            "role": "user",
            "content": human_message
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
                stop="<|eot_id|>"
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
    

