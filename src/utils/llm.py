from typing import List, Dict, Tuple
from tqdm import tqdm
from openai import OpenAI
from cfg.config import Config
from vllm import LLM as VLLM_LLM, SamplingParams

conf = Config()

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

class LLM:
    def __init__(
        self,
        model: str = "/hy-tmp/3.1-8B", 
        gpu_memory_utilization: float = 0.9, 
        is_api: bool = False,
        tensor_parallel_size: int = 1, 
        use_sharegpt_format: bool = False,
        max_input_tokens: int = None,
        max_output_tokens: int = 512,
        batch_size: int = 16,
        special_tokens: List[Tuple[str, str]] = None,
        selected_special_tokens: List[str] = None,
        temperature: float = 0,
    ):
        # env initialization
        self.port = conf.port
        self.host = conf.host
        # model initialization
        self.model_path_or_name = model
        self.gpu_memory_utilization = gpu_memory_utilization
        self.server_process = None
        self.tensor_parallel_size = tensor_parallel_size
        self.use_sharegpt_format = use_sharegpt_format
        self.max_input_tokens = max_input_tokens
        self.max_output_tokens = max_output_tokens
        self.batch_size = batch_size
        self.is_api = is_api
        self.special_tokens = special_tokens
        self.temperature = temperature
        self.should_skip_special_tokens = special_tokens is None
        self.selected_special_tokens = selected_special_tokens

        # sampling params
        self.gen_params = SamplingParams(
            temperature=temperature,
            max_tokens=self.max_output_tokens,
            skip_special_tokens=self.should_skip_special_tokens,
        )

        if self.is_api:
            self.client = OpenAI(api_key=conf.api_key, base_url=conf.api_base)
            self.model = self.model_path_or_name
        else:
            self.model = VLLM_LLM(
                model=self.model_path_or_name,
                tensor_parallel_size=self.tensor_parallel_size,
                max_model_len=self.max_input_tokens,
                trust_remote_code=True
            )
            if self.special_tokens:
                # flatten the list of tuples
                self.model.llm_engine.tokenizer.tokenizer.add_special_tokens({"additional_special_tokens": [token for pair in self.special_tokens for token in pair]})
                self.special_token_id_map = self.get_special_token_ids()

    def get_special_token_ids(self):
        flatten_special_tokens = [token for pair in self.special_tokens for token in pair]
        special_token_ids = [self.model.llm_engine.tokenizer.tokenizer.encode(token, add_special_tokens=False)[0] for token in flatten_special_tokens]
        special_token_id_map = {token: id for token, id in zip(flatten_special_tokens, special_token_ids)}
        return special_token_id_map

    def parse_output_for_special_tokens(self, outputs):
        raw_texts = [output.outputs[0].text for output in outputs]
        if not self.special_tokens:
            return raw_texts
    
        # Only parse if there are special tokens and selected tokens
        all_token_ids = [output.outputs[0].token_ids for output in outputs]
        filtered_tokens = [token_tuple for token_tuple in self.special_tokens 
                         if token_tuple[0] in self.selected_special_tokens]
        all_responses = []
        for token_ids in all_token_ids:
            special_token_responses = {token_tuple[0]: "no text extracted" for token_tuple in filtered_tokens}
            for token_tuple in filtered_tokens:
                start_token, end_token = token_tuple
                start_token_id = self.special_token_id_map[start_token]
                end_token_id = self.special_token_id_map[end_token]
                token_start = 0
                for i, id in enumerate(token_ids):
                    if id == start_token_id:
                        token_start = i
                    if id == end_token_id:
                        token_end = i
                        special_token_responses[start_token] = self.model.llm_engine.tokenizer.tokenizer.decode(token_ids[token_start+1:token_end])
                        break
            all_responses.append(special_token_responses)
        return all_responses, raw_texts

    def set_temperature(self, temperature: float):
        self.gen_params = SamplingParams(
            temperature=temperature,
            max_tokens=self.max_output_tokens,
            skip_special_tokens=self.should_skip_special_tokens,
        )

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
        
    def batch_generate_chat(self, test_cases: List[List[Dict]]) -> List[Dict]:
        """Run inference for a batch of test cases using batched processing.
        
        Args:
            test_cases (List[List[Dict]]): List of test cases to process
            temperature (float): Sampling temperature for generation
            
        Returns:
            List[Dict]: List of generated responses
        """
        # Convert test cases to messages format if needed
        if self.use_sharegpt_format:
            messages_batch = [self._create_messages_from_sharegpt(case) for case in test_cases]
        else:
            messages_batch = test_cases

        all_outputs = []

        for i in tqdm(range(0, len(messages_batch), self.batch_size), desc="Processing batch"):
            batch_messages = messages_batch[i:i+self.batch_size]
            outputs = self.model.chat(batch_messages, self.gen_params, use_tqdm=False)
            all_outputs.extend(outputs)
                    
        return self.parse_output_for_special_tokens(all_outputs)
    
    def batch_generate_complete(self, messages_batch: List[str]) -> List[Dict]:
        gen_params = SamplingParams(temperature=self.temperature, max_tokens=self.max_output_tokens)
        all_outputs = []
        if self.is_api:
            for batch_msgs in tqdm(range(0, len(messages_batch), self.batch_size), desc="Processing API batch"):
                batch = messages_batch[batch_msgs:batch_msgs+self.batch_size]
                batch_responses = []
                for msgs in batch:
                    response = self.client.chat.completions.create(
                        model=self.model_path_or_name,
                        messages=msgs,
                        temperature=self.temperature,
                        max_tokens=self.max_output_tokens,
                    )
                    batch_responses.append(response.choices[0].message.content)
                all_outputs.extend(batch_responses)
            return all_outputs
        else:
            for i in tqdm(range(0, len(messages_batch), self.batch_size), desc="Processing batch"):
                batch_messages = messages_batch[i:i+self.batch_size]
                outputs = self.model.chat(batch_messages, self.gen_params, use_tqdm=False)
                all_outputs.extend(outputs)

            return self.parse_output_for_special_tokens(all_outputs)

    def single_generate_complete(
        self, 
        test_case: str, 
    ) -> str:
        gen_params = SamplingParams(
            temperature=self.temperature, 
            max_tokens=self.max_output_tokens,
        )
        output = self.model.generate([test_case], gen_params, use_tqdm=False)
        return output[0].outputs[0].text

    def single_generate_chat(
        self,
        messages: Dict,
    ) -> str:
        """Generate a single chat response.
        
        Args:
            messages (List[Dict]): List of message dictionaries with 'role' and 'content' keys
            temperature (float): Sampling temperature for generation
            
        Returns:
            str: Generated response text
        """
        if self.use_sharegpt_format:
            messages = self._create_messages_from_sharegpt(messages)
        # print(messages)
        if self.is_api:
            output = self.client.chat.completions.create(
                model=self.model_path_or_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_output_tokens,
            )
            output = output.choices[0].message.content
            return output
        else:   
            resp = self.model.chat(messages, self.gen_params, use_tqdm=False)
            return resp[0].outputs[0].text
        
