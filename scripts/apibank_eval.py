# Copyright 2023 AlibabaResearch/DAMO-ConvAI
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

import json
import logging
import os
import re
import click
import numpy as np
from rouge import Rouge
from tqdm import tqdm
from benchmark.apibank.tool_manager import ToolManager
from benchmark.apibank.api_call_extraction import parse_api_call
from utils.llm import LLM

def calculate_rouge_l_score(reference, hypothesis):
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)
    rouge_l_score = scores[0]['rouge-l']['f']
    return rouge_l_score

class Sample:
    def __init__(self, chat_history, apis, ground_truth):
        self.chat_history = chat_history
        self.apis = apis
        self.ground_truth = ground_truth

    def __repr__(self):
        return 'Sample(chat_history={}, apis={}, ground_truth={})'.format(self.chat_history, self.apis, self.ground_truth)

    @classmethod
    def from_chat_history(cls, chat_history):
        apis = set()
        api_positions = []
        for i, item in enumerate(chat_history):
            if item['role'] == 'API':
                apis.add(item['api_name']) 
                api_positions.append(i)

        samples = []
        for i in api_positions:
            sample = cls(chat_history[:i], apis, chat_history[i])
            samples.append(sample)
            sample = cls(chat_history[:i + 1], apis, chat_history[i + 1])
            samples.append(sample)

        return samples

class Evaluator:
    def __init__(self, samples):
        self.dataset = samples
        self.sample_ids = list(range(len(self.dataset)))

    def get_all_sample_ids(self):
        return self.sample_ids
    
    def get_api_description(self, api_name):
        tool_manager = ToolManager()
        return tool_manager.get_api_description(api_name)

    
    def get_model_input(self, sample_id):
        sample = self.dataset[sample_id]
        apis = sample.apis
        chat_history = sample.chat_history
        tool_manager = ToolManager()
        api_descriptions = []
        for api_name in apis:
            try:
                desc = tool_manager.get_api_description(api_name)
            except Exception as e:
                print(f"Warning: skip invalid tool {api_name}: {e}")
                continue
            if desc:
                api_descriptions.append(desc)
        api_descriptions = [d for d in api_descriptions if isinstance(d, str)]
        merged = '\n'.join(api_descriptions)
        return merged, chat_history

    def evaluate(self, sample_id, model_output):
        tool_manager = ToolManager()

        sample = self.dataset[sample_id]
        ground_truth = sample.ground_truth
        if ground_truth['role'] == 'API':
            api_name, param_dict = parse_api_call(model_output)
            if api_name != ground_truth['api_name']:
                return False, 'API Name Mismatch: {} vs {}'.format(api_name, ground_truth['api_name'])
            try:
                result = tool_manager.api_call(api_name, **param_dict)
            except Exception as e:
                return False, str(e)
            api = tool_manager.init_tool(api_name)
            try:
                correct = api.check_api_call_correctness(result, ground_truth['result'])
            except KeyError:
                correct = False
                result = 'KeyError' + str(result)
            return correct, result
        elif ground_truth['role'] == 'AI':
            score = calculate_rouge_l_score(ground_truth['text'], model_output)
            return round(score, 4)
        
def get_api_call(model_output):
    api_call_pattern = r"\[(\w+)\((.*)\)\]"
    api_call_pattern = re.compile(api_call_pattern)
    match = api_call_pattern.search(model_output)
    if match:
        return match.group(0)
    else:
        return None

@click.command()
@click.option("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
@click.option("--is_api", type=bool, default=False)
@click.option("--tensor_parallel_size", type=int, default=1)
@click.option("--batch_size", type=int, default=128)
@click.option("--max_model_len", type=int, default=4096)
@click.option("--max_output_tokens", type=int, default=512)
def main(model: str, is_api: bool, tensor_parallel_size: int, batch_size: int, max_model_len: int, max_output_tokens: int):
    data_dirs = [
        '../data/apibank/lv1-lv2-samples/level-1-given-desc',
        '../data/apibank/lv1-lv2-samples/level-2-toolsearcher'
    ]

    llm = LLM(
        model=model, 
        tensor_parallel_size=tensor_parallel_size,
        is_api=is_api,
        use_sharegpt_format=False,
        max_input_tokens=max_model_len,
        batch_size=batch_size, 
        max_output_tokens=max_output_tokens
    )
    
    final_results = {}
    for data_dir in data_dirs:
        api_test_enabled = False
        dialog_test_enabled = not api_test_enabled

        if os.path.basename(data_dir).endswith('given-desc'):
            tool_search_enabled = False
        else:
            tool_search_enabled = True

        api_call_prompt = '''
    Based on the given API description and the existing conversation history 1..t, please generate the API request that the AI should call in step t+1 and output it in the format of [ApiName(key1='value1', key2='value2', ...)], replace the ApiName with the actual API name, and replace the key and value with the actual parameters. 
    Your output should start with a square bracket "[" and end with a square bracket "]". Do not output any other explanation or prompt or the result of the API call in your output. 
    This year is 2023.
    Input: 
    User: [User's utterence]
    AI: [AI's utterence]

    Expected output:
    [ApiName(key1='value1', key2='value2', ...)]

    API descriptions:
    '''

        response_prompt = '''
    Based on the given API description and the existing conversation history 1..t, please generate the next dialog that the AI should response after the API call t.
    This year is 2023.
    Input: 
    User: [User's utterence]
    AI: [AI's utterence]
    [ApiName(key1='value1', key2='value2', …)]

    Expected output:
    AI: [AI's utterence]

    API descriptions:
    '''

        total_api_calls = 0
        correct_api_calls = 0

        rougel_scores = []

        jsonl_files = [f for f in os.listdir(data_dir) if f.endswith('.jsonl')]

        for file in tqdm(jsonl_files, desc='Processing files', ncols=100):
            history = []
            with open(os.path.join(data_dir, file), 'r') as f:
                for line in f:
                    history.append(json.loads(line))
            samples = Sample.from_chat_history(history)
            evaluator = Evaluator(samples)

            for sample_id in evaluator.get_all_sample_ids():
                sample = evaluator.dataset[sample_id]
                if sample.ground_truth['role'] == 'API' and api_test_enabled:
                    if tool_search_enabled:
                        _, chat_history = evaluator.get_model_input(sample_id)
                        api_descriptions = evaluator.get_api_description('ToolSearcher')
                    else:
                        api_descriptions, chat_history = evaluator.get_model_input(sample_id)
                    prompt = api_call_prompt + api_descriptions
                    messages = [
                        {'role': 'system', 'content': prompt},
                    ]
                    for item in chat_history:
                        if item['role'] == 'User':
                            chat_role = 'user'
                            chat_content = item['text']
                        elif item['role'] == 'AI':
                            chat_role = 'assistant'
                            chat_content = item['text']
                        elif item['role'] == 'API':
                            chat_role = 'system'
                            chat_content = '[{}({})] Response: {}'.format(item['api_name'], ', '.join(['{}=\'{}\''.format(k, v) for k, v in item['param_dict'].items()]), str(item['result']['output']))
                        else:
                            raise ValueError('Invalid chat role: {}'.format(item['role']))
                        messages.append({'role': chat_role, 'content': chat_content})
                    
                    model_output = llm.single_generate_chat(messages)
                    print(model_output)

                    api_call = get_api_call(model_output)
                    if api_call:
                        try:
                            correct, model_output_result = evaluator.evaluate(sample_id, api_call)
                        except AssertionError as e:
                            if not 'The API name is not correct.' in str(e):
                                raise e
                            logging.info('AssertionError: {}'.format(e))
                            correct = False
                    else:
                        model_output_result = 'No API call found'
                        correct = False
                    if correct:
                        correct_api_calls += 1
                        logging.info('Correct API call: {} Ground truth: {}'.format(api_call, sample.ground_truth))
                    else:                    
                        logging.info('Incorrect model output: {} Result: {} Ground truth: {} File: {} Sample ID: {} Messages: {}'.format(model_output.replace('\n', ' '), model_output_result, sample.ground_truth, file, sample_id, messages[1:]))
                    total_api_calls += 1
                elif sample.ground_truth['role'] == 'AI' and dialog_test_enabled:
                    api_descriptions, chat_history = evaluator.get_model_input(sample_id)
                    prompt = response_prompt + api_descriptions
                    messages = [
                        {'role': 'system', 'content': prompt},
                    ]
                    for item in chat_history:
                        if item['role'] == 'User':
                            chat_role = 'user'
                            chat_content = item['text']
                        elif item['role'] == 'AI':
                            chat_role = 'assistant'
                            chat_content = item['text']
                        elif item['role'] == 'API':
                            chat_role = 'system'
                            chat_content = '[{}({})] Response: {}'.format(item['api_name'], ', '.join(['{}=\'{}\''.format(k, v) for k, v in item['param_dict'].items()]), str(item['result']['output']))
                        else:
                            raise ValueError('Invalid chat role: {}'.format(item['role']))
                        messages.append({'role': chat_role, 'content': chat_content})

                    model_output = llm.single_generate_chat(messages)
                    print(model_output)

                    if model_output:
                        score = evaluator.evaluate(sample_id, model_output)
                    else:
                        score = 0    
                    rougel_scores.append(score)
                    if score < 0.2:
                        logging.info('Low score: {} Score: {} Ground truth: {} File: {} Sample ID: {} Messages: {}'.format(model_output.replace('\n', ' '), score, sample.ground_truth, file, sample_id, messages[1:]))

        name = os.path.basename(data_dir)
        if dialog_test_enabled:
            final_results[name] = {
                'dialog_score': float(np.mean(rougel_scores)),
            }

        if api_test_enabled:
            final_results[name] = {
                'api_total_calls': total_api_calls,
                'api_correct_calls': correct_api_calls,
                'api_accuracy': correct_api_calls / total_api_calls,
            }

    output_dict = {}
    for name, res in final_results.items():
        score_percent = res.get("dialog_score", 0) * 100
        output_dict[name] = round(score_percent, 2)
    print(json.dumps(output_dict))

if __name__ == "__main__":
    main()
