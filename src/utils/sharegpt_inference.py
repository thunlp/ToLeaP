from typing import Union, List, Dict, Optional
from dataclasses import dataclass
import json
from vllm import LLM as VLLM
from vllm import SamplingParams
from transformers import AutoTokenizer, AutoConfig
import torch

@dataclass
class Message:
    role: str
    content: str
    
    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content
        }

class LLM:
    def __init__(
        self,
        model_name: str,
        num_gpus: int = 1,
        gpu_memory_utilization: float = 0.9,
        temperature: float = 0,
        top_p: float = 0.95,
        model_len: int = 39440,
        tensor_parallel_size: Optional[int] = None,
        **kwargs
    ):
        """初始化LLM"""
        self.model_name = model_name
        self.model_len = model_len
        
        available_gpus = torch.cuda.device_count()
        if num_gpus > available_gpus:
            print(f"Warning: Requested {num_gpus} GPUs but only {available_gpus} available.")
            num_gpus = available_gpus
            
        if tensor_parallel_size is None:
            tensor_parallel_size = num_gpus
            
        self._init_model(
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            **kwargs
        )
        
        self.default_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=1024    #如果长就改这个
        )

    def _init_model(self, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        try:
            config = AutoConfig.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            if hasattr(self, "model_len"):
                self.max_length = self.model_len
            elif hasattr(config, "max_position_embeddings"):
                self.max_length = config.max_position_embeddings
            elif self.tokenizer.model_max_length is not None:
                self.max_length = self.tokenizer.model_max_length
            else:
                self.max_length = 2048
        except Exception as e:
            print(f"Warning: Failed to get model config: {e}")
            self.max_length = getattr(self, "model_len", 2048)
        
        self.engine = VLLM(
            model=self.model_name,
            trust_remote_code=True,
            max_model_len=self.max_length,
            dtype="bfloat16",
            **kwargs
        )

    def apply_chat_template(self, messages: List[Dict]) -> str:
        if hasattr(self.tokenizer, 'apply_chat_template'):
            formatted_messages = []
            for msg in messages:
                role = msg.get('role', '')
                content = msg.get('content', '')
                if role == 'user':
                    role = 'user'
                elif role == 'assistant':
                    role = 'assistant'
                elif role == 'system':
                    role = 'system'
                formatted_messages.append({
                    'role': role,
                    'content': content
                })
            
            try:
                result = self.tokenizer.apply_chat_template(
                    formatted_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return result
            except Exception as e:
                print(f"Warning: Error applying chat template: {e}")
                return self.format_chat_prompt(messages)
        else:
            return self.format_chat_prompt(messages)

    def format_chat_prompt(self, messages: List[Dict]) -> str:
        formatted_prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                formatted_prompt += f"System: {content}\n"
            elif role == "user":
                formatted_prompt += f"Human: {content}\n"
            elif role == "assistant":
                formatted_prompt += f"Assistant: {content}\n"
            
        return formatted_prompt.strip()

    def tokenize_prompt(self, prompt: str) -> List[int]:
        tokens = self.tokenizer.encode(prompt)
        if len(tokens) > self.max_length:
            print(f"Warning: Truncating prompt from {len(tokens)} to {self.max_length} tokens")
            tokens = tokens[:self.max_length]
        return tokens

    def prepare_prompts(
        self,
        test_cases: List[Dict],
        system_prompt: Optional[str] = None,
        tools_info: Optional[str] = None
    ) -> List[List[int]]:
        prompt_tokens = []
        for case in test_cases:
            messages = []
            
            # 添加系统提示词
            case_system = case.get('system', system_prompt)
            if case_system:
                messages.append({
                    "role": "system",
                    "content": case_system
                })
            
            # 添加工具信息
            case_tools = case.get('tools', tools_info)
            if case_tools:
                if isinstance(case_tools, list):
                    tool_content = json.dumps(case_tools, ensure_ascii=False)
                else:
                    tool_content = str(case_tools)
                messages.append({
                    "role": "system",
                    "content": tool_content
                })
            
            # 添加对话内容
            if 'conversations' in case and case['conversations']:
                conversation = case['conversations'][0]
                if isinstance(conversation, dict):
                    content = conversation.get('value', '')
                    # 假设对话内容是用户输入
                    messages.append({
                        "role": "user",
                        "content": content
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": str(conversation)
                    })
            
            # 应用chat template并tokenize
            formatted_prompt = self.apply_chat_template(messages)
            tokens = self.tokenize_prompt(formatted_prompt)
            prompt_tokens.append(tokens)
            
        return prompt_tokens

    def batch_generate(
        self,
        prompt_tokens: List[List[int]],
        batch_size: int,
        sampling_params: Optional[SamplingParams] = None
    ) -> List[str]:
        """批量生成回复"""
        if sampling_params is None:
            sampling_params = self.default_params
            
        all_outputs = []
        for i in range(0, len(prompt_tokens), batch_size):
            batch = prompt_tokens[i:i + batch_size]
            
            # for idx, tokens in enumerate(batch):
            #     # print(f"Batch {i} - Prompt {idx} tokens: {len(tokens)}")
            
            try:
                outputs = self.engine.generate(
                    prompt_token_ids=batch,
                    sampling_params=sampling_params
                )
                all_outputs.extend(outputs)
            except Exception as e:
                print(f"Generation error in batch {i}: {e}")
                continue
        
        results = []
        for output in all_outputs:
            text = output.outputs[0].text
            if len(text) < 10:
                print(f"Warning: Suspiciously short output: {text}")
            results.append(text)
            
        return results

    def __call__(
        self,
        input_data: Union[str, List[Dict], List[str]],
        system_prompt: Optional[str] = None,
        tools_info: Optional[str] = None,
        batch_size: Optional[int] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Union[str, List[str]]:
        if isinstance(input_data, str):
            messages = [{
                "role": "user",
                "content": input_data
            }]
            formatted_prompt = self.apply_chat_template(messages)
            prompt_tokens = [self.tokenize_prompt(formatted_prompt)]
        elif isinstance(input_data, list):
            if isinstance(input_data[0], dict):
                prompt_tokens = self.prepare_prompts(input_data, system_prompt, tools_info)
            else:
                messages_list = [[{"role": "user", "content": p}] for p in input_data]
                prompt_tokens = [
                    self.tokenize_prompt(self.apply_chat_template(msgs))
                    for msgs in messages_list
                ]
        else:
            raise ValueError("Unsupported input format")
            
        sampling_params_dict = {
            **self.default_params.__dict__,
            **kwargs
        }
        if max_tokens is not None:
            sampling_params_dict['max_tokens'] = max_tokens
            
        sampling_params = SamplingParams(**sampling_params_dict)
        
        if batch_size and len(prompt_tokens) > 1:
            results = self.batch_generate(prompt_tokens, batch_size, sampling_params)
        else:
            outputs = self.engine.generate(
                prompt_token_ids=prompt_tokens,
                sampling_params=sampling_params
            )
            results = [output.outputs[0].text for output in outputs]
            
        return results[0] if len(results) == 1 else results



import json

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

available_gpus = torch.cuda.device_count()
print(f"Available GPUs: {available_gpus}")


#初始化你的calss
llm = LLM(
    model_name='/hy-tmp/3.1-8B',  #change to your model path
    num_gpus=available_gpus,
    gpu_memory_utilization=0.9,
    tensor_parallel_size=available_gpus
)

test_cases = load_json('test.json')

responses = llm(
    test_cases,
    batch_size=2, 
    temperature=0 
)

for case, response in zip(test_cases, responses):
    print(f"Response: {response}")
    print("-" * 50)