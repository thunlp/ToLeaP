#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 THUNLP-MT/StableToolBench
# Modifications Copyright 2024 BodhiAgent
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import string, random
from termcolor import colored
from benchmark.stabletoolbench.toolbench.toolbench_utils import process_system_message
from benchmark.stabletoolbench.toolbench.model.model_adapter import get_conversation_template
from benchmark.stabletoolbench.toolbench.inference.inference_utils import react_parser
from utils.llm import LLM

class ToolLLaMA_vllm:
    def __init__(
            self, 
            model_name_or_path: str, 
            template: str="tool-llama-single-round",  
            max_sequence_length: int=4096,
            is_api: bool=False,
            tensor_parallel_size: int=1,
        ) -> None:
        self.template = template
        self.max_sequence_length = max_sequence_length
        self.is_api = is_api
        self.tensor_parallel_size = tensor_parallel_size
        self.model = LLM(
            model=model_name_or_path,
            tensor_parallel_size=tensor_parallel_size,
            use_sharegpt_format=False,
            max_input_tokens=self.max_sequence_length,
            is_api=is_api,
            gpu_memory_utilization=0.9,
            batch_size=32,
            max_output_tokens=512,
        )
    
    def prediction(self, prompt: str) -> str:
        if not self.is_api:
            result = self.model.single_generate_complete(prompt)
        else:
            result = self.model.single_generate_chat([{"role": "user", "content": prompt}])
        return result
        
    def add_message(self, message):
        self.conversation_history.append(message)

    def change_messages(self,messages):
        self.conversation_history = messages

    def display_conversation(self, detailed=False):
        role_to_color = {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "function": "magenta",
        }
        for message in self.conversation_history:
            print_obj = f"{message['role']}: {message['content']} "
            # if "function_call" in message.keys():
            #     print_obj = print_obj + f"function_call: {message['function_call']}"
            if "function_call" in message.keys():
                print_obj = print_obj + f"function_call: {message['function_call']}"
            if 'tool_calls' in message.keys():
                print_obj = print_obj + f"tool_calls: {message['tool_calls']}"
                print_obj = print_obj + f"number of tool calls: {len(message['tool_calls'])}"
            if detailed:
                print_obj = print_obj + f"function_call: {message['function_call']}"
                print_obj = print_obj + f"tool_calls: {message['tool_calls']}"
                print_obj = print_obj + f"function_call_id: {message['function_call_id']}"
            print_obj += ""
            print(
                colored(
                    print_obj,
                    role_to_color[message["role"]],
                )
            )

    def parse(self, tools, process_id, **args):
        conv = get_conversation_template(self.template)
        if self.template == "tool-llama":
            roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        elif self.template == "tool-llama-single-round" or self.template == "tool-llama-multi-rounds":
            roles = {"system": conv.roles[0], "user": conv.roles[1], "function": conv.roles[2], "tool": conv.roles[2], "assistant": conv.roles[3]}

        self.time = time.time()
        conversation_history = self.conversation_history

        if tools != []:
            functions = [tool['function'] for tool in tools]

        prompt = ''
        for message in conversation_history:
            role = roles[message['role']]
            content = message['content']
            if role == "System" and tools != []:
                content = process_system_message(content, functions=functions)
            prompt += f"{role}: {content}\n"
        prompt += "Assistant:\n"
        if tools != []:
            predictions = self.prediction(prompt)
        else:
            predictions = self.prediction(prompt)
        
        decoded_token_len = len(predictions)
        if process_id == 0:
            print(f"[process({process_id})]total tokens: {decoded_token_len}")

        # react format prediction
        thought, action, action_input = react_parser(predictions)
        random_id = ''.join([random.choice(string.ascii_letters + string.digits) for _ in range(8)])
        message = {
            "role": "assistant",
            "content": predictions,
            "tool_calls": [{
                'id': f"call_{random_id}",
                'type': "function",
                'function': {
                    'name': action,
                    'arguments': action_input
                }
            }]
        }
        return message, 0, decoded_token_len