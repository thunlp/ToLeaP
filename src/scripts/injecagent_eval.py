import sys
import os
import json
import time
import argparse
import transformers
import torch 
from tqdm import tqdm
from typing import List, Dict

from InjecAgent_files.prompts.agent_prompts import PROMPT_DICT
from InjecAgent_files.models import MODELS
from InjecAgent_files.InjecAgent_utils import get_tool_dict, get_simulated_attacker_tool_response, get_score
from InjecAgent_files.output_parsing import evaluate_output_prompted

current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(current_dir, '..')
sys.path.append(utils_dir)

from utils.llm import LLM

### Parse the params
def define_arguments_prompted(parser):
    parser.add_argument("--model_type", default='TogetherAI', help="Type of the model to evaluate")
    parser.add_argument("--model_name", default='/bjzhyai03/workhome/chenhaotian/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659', help="Name of the model to evaluate")
    parser.add_argument("--setting", default='base', choices=['base', 'enhanced'], help="base or enhanced setting")
    parser.add_argument("--prompt_type", default='InjecAgent',choices=['InjecAgent', 'hwchase17_react'], help="prompt type")
    parser.add_argument("--only_first_step", default=False, action='store_true', help="Only predict the first step")
    parser.add_argument("--use_cache", default=False, action='store_true', help="Use existing dataset")
    parser.add_argument("--only_get_score", default=False, action='store_true', help="Only get score without inference")
    
def define_arguments_finetuned(parser):
    parser.add_argument("--model_name", help="Name of the model to evaluate", required=True)
    parser.add_argument("--setting", choices=['base', 'enhanced'], help="base or enhanced setting", required=True)
    parser.add_argument("--prompt_type", choices=['InjecAgent', 'hwchase17_react'], help="prompt type", required=True)
    parser.add_argument("--only_first_step", action='store_true', help="Only predict the first step")
    
def parse_arguments(agent_type=None):
    parser = argparse.ArgumentParser()
    if agent_type == "prompted":
        define_arguments_prompted(parser)
    elif agent_type == "finetuned":
        define_arguments_finetuned(parser)
    args = parser.parse_args()
    params = args.__dict__
    return params

def create_messages(conversation_data: Dict) -> List[Dict]:
    messages = []
    for cov in conversation_data: # Dict
        message = []
        for prompt in cov.get("conversations", []): # List
            if prompt.get("from") == "human":
                message.append({"role": "user", "content": prompt["value"]})
        messages.append(message)
    return messages

def predict_one_case(system_prompt, user_prompt, params, item, model_class, tool_dict, step_2=False):
    available_tool_names = [item['User Tool']] + item['Attacker Tools']
    available_tool_descriptions = json.dumps([tool_dict[tool] for tool in available_tool_names], indent=True)
    input = item['User Instruction']
    if step_2:
        agent_scratchpad = f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}\nObservation: {item['Tool Response']}\nThought: {item['Attacker Tool Thought']}\nAction: {item['Attacker Tools'][0]}\nAction Input: {item['Attacker Tool Parameters']}\nObservation: {item['Attacker Tool Response']}\n"
    else:
        agent_scratchpad = f"\n{item['Thought']}\nAction: {item['User Tool']}\nAction Input: {item['Tool Parameters']}\nObservation: {item['Tool Response']}\n"
    user_prompt_filled = user_prompt.format(available_tool_descriptions = available_tool_descriptions, input = input, available_tool_names = available_tool_names, agent_scratchpad = agent_scratchpad)
    model_input = model_class.prepare_input(system_prompt, user_prompt_filled)
    output = model_class.call_model(model_input)
    return output

def main(params):
    ### Init
    print(params)
    if params['model_type'] == 'OpenModel':
        model_class = MODELS[params['model_type']](
            model=params['model_name'],
            tensor_parallel_size=4,
            use_sharegpt_format=False,
            max_input_tokens=4096,
            gpu_memory_utilization=0.8,
            batch_size=32
        )
    else:
        model_class = MODELS[params['model_type']](params)
    system_prompt, user_prompt = PROMPT_DICT[params['prompt_type']]
    output_dir = f"./InjecAgent_results/prompted_{params['model_type']}_{os.path.basename(params['model_name'])}_{params['prompt_type']}"
    print(f"The results will be saved to {os.path.abspath(output_dir)}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data_dir = '../data/sft_data/InjecAgent'
    # print(f"Loading data from {os.path.abspath(data_dir)}")

    tool_dict = get_tool_dict()
    
    output_files = {}
    for attack in ['dh', 'ds']:
        file_name = f"test_cases_{attack}_{params['setting']}.json"
        test_case_file = os.path.join(data_dir, file_name)
        output_file = os.path.join(output_dir, file_name)
        output_files[attack] = output_file
        
        with open(test_case_file, 'r') as f:
            data = json.load(f)
            
        if params['use_cache']:
            existing_data_step1 = {}
            existing_data_step2 = {}
            if os.path.exists(output_file):
                with open(output_file, 'r') as f:
                    for line in f.readlines():
                        try:
                            tmp = json.loads(line)
                            if type(tmp['output']) == str and len(tmp['output']):
                                existing_data_step1[(tmp['User Instruction'], tmp['Tool Response'])] = tmp['output']
                            if type(tmp['output Step 2']) == str and len(tmp['output Step 2']):
                                existing_data_step2[(tmp['User Instruction'], tmp['Tool Response'], tmp['output'])] = tmp['output Step 2']
                        except:
                            continue
                print(f"Got {len(existing_data_step1)} existing outputs for the first step and {len(existing_data_step2)} existing outputs for the second step")
                        
        if not params['only_get_score']:
            with open(output_file, 'w') as f:
                for item in tqdm(data):
                    try:
                        if params['use_cache'] and (item['User Instruction'], item['Tool Response']) in existing_data_step1:
                            # print("Use existing output for step 1")
                            output = existing_data_step1[(item['User Instruction'], item['Tool Response'])]
                        else:
                            output = predict_one_case(system_prompt, user_prompt, params, item, model_class, tool_dict)
                        item['output'] = output
                    
                        parsed_output = evaluate_output_prompted(item['output'], item['Attacker Tools'][0], item['Attacker Tools'], item['User Tool'], item["Attacker Instruction"])
                        item['eval'] = parsed_output['eval']
                        if item['eval'] == 'invalid':
                            item['invalid_reason'] = parsed_output['invalid_reason']
                        
                        if not params['only_first_step'] and item['eval'] == 'succ' and attack == 'ds':
                            item["Attacker Tool Parameters"] = parsed_output["Attacker Tool Parameters"]
                            item["Attacker Tool Thought"] = parsed_output["Attacker Tool Thought"]
                            item["Attacker Tool Response"] = get_simulated_attacker_tool_response(item['Attacker Tools'][0], item["Attacker Tool Parameters"], tool_dict)
                            
                            if params['use_cache'] and (item['User Instruction'], item['Tool Response'], item['output']) in existing_data_step2:
                                output = existing_data_step2[(item['User Instruction'], item['Tool Response'], item['output'])]
                                # print("Use existing output for step 2")
                            else:
                                output = predict_one_case(system_prompt, user_prompt, params, item, model_class, tool_dict, step_2=True)
                            
                            item['output Step 2'] = output
                            parsed_output = evaluate_output_prompted(output, item['Attacker Tools'][1], item['Attacker Tools'], item['User Tool'], item["Attacker Instruction"])
                            item['eval Step 2'] = parsed_output['eval']
                            if item['eval Step 2'] == 'invalid':
                                item['invalid_reason  Step 2'] = parsed_output['invalid_reason']
                        
                        f.write(json.dumps(item)+'\n')
                    except Exception as e:
                        # This block will execute if an exception occurs
                        print(f"An error occurred: {e} in {output_file}")
                        # assert 1==0
    
    scores = get_score(output_files)
    scores_file = os.path.join(output_dir, 'scores.json')
    with open(scores_file, 'w') as f:
        json.dump(scores, f, indent=4)
    print(json.dumps(scores, indent=4)) 

if __name__ == "__main__":
    params = parse_arguments(agent_type="prompted")
    main(params)

