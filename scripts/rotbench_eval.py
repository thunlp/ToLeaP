# This file comes from Junjie-Ye/RoTBench/Code/Evaluation/evaluate.py
# Copyright 2024 Junjie-Ye/RoTBench
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

import sys
import os
import click
import json
from typing import List, Dict
from ast import literal_eval
import re 

from cfg.config import Config
from utils.llm import LLM

def create_messages(conversation_data: List[Dict]) -> List[List[Dict]]:
    messages = []
    for conv in conversation_data:
        message = []
        for prompt in conv["conversations"]:
            if prompt["from"] == "system":
                message.append({
                    "role": "system",
                    "content": prompt["value"]
                })
            elif prompt["from"] == "user":
                message.append({
                    "role": "user",
                    "content": prompt["value"]
                })
        messages.append(message)
    return messages

conf = Config()

def match_square_bracket(text, pos_s):
    counter = -1
    for i in range(pos_s+1,len(text)):
        if text[i] == '{':
            counter -= 1
        elif text[i] == '}':
            counter += 1
        if counter == 0:
            return text[pos_s: i+1]
    return ""

def get_cata_list(answer_file):
    Text_Generation = []
    Real_Time_Search = []
    Data_Understanding = []
    Personal_Life = []
    Application_Manipulation = []
    Information_Retrieval = []
    Financial_Transactions = []
    # Record different scenarios
    with open(answer_file, encoding="utf-8") as f:
        data = json.load(f)
    for index, d in enumerate(data):
        sce = d["scenario"]
        if sce == "TG":
            Text_Generation.append(index)
            continue
        if sce == "RS":
            Real_Time_Search.append(index)
            continue
        if sce == "DU":
            Data_Understanding.append(index)
            continue
        if sce == "PL":
            Personal_Life.append(index)
            continue
        if sce == "AM":
            Application_Manipulation.append(index)
            continue
        if sce == "IR":
            Information_Retrieval.append(index)
            continue
        if sce == "FT":
            Financial_Transactions.append(index)
            continue
    cata_list = [Text_Generation, Real_Time_Search, Data_Understanding, Personal_Life, Application_Manipulation, Information_Retrieval, Financial_Transactions]
    return cata_list

def get_config(data):
    p_len = len(data["conversations"][0]["value"][:data["conversations"][0]["value"].find("[")])
    config = json.loads(data["conversations"][0]["value"][p_len:-13])
    return config

def get_answer_list(data):
    return data["conversations"][-1]["value"]

def get_raven_resultcall(data, version):
    if version == 1:
        result_call = data["result"]
        start_str = "Initial Answer: "
        end_str = "\nReflection: "
        start_idx = result_call.find(start_str) + len(start_str)
        end_idx = result_call.find(end_str)
        result_call = result_call[start_idx: end_idx]
    if version == 2:
        result_call = data["result"][6:data["result"].find("\nThought:") - 1]
        if result_call.find(";") != -1:
            result_call = result_call[:result_call.find(";")]
        if result_call.count("(") == 1:
            pass
        else:
            end_idx = result_call.find(")")
            start_idx = end_idx
            func = 0
            for char in result_call[:end_idx][::-1]:
                start_idx -= 1
                if char == "(":
                    func = 1
                if char == "=" and func:
                    break
            result_call = result_call[start_idx + 1: end_idx + 1]
    return result_call

def get_raven_action_input(action_input, test_action, config, version):
    if version == 1:
        if action_input.find("=") != -1:
            action_input = action_input.replace("(", "{").replace(")", "}").replace("=", "':")
            for idx, char in enumerate(action_input):
                if action_input[idx] == "{" and action_input[idx + 1] != "}":
                    action_input = action_input[:idx + 1] + "'" + action_input[idx + 1:]
                if idx > 0 and action_input[:idx + 1].count("'") % 2 == 0:
                    if (action_input[idx - 1] + action_input[idx] == ", ") and (action_input[idx - 1] + action_input[idx] + action_input[idx + 1] != ", '"):
                        action_input = action_input[:idx + 1] + "'" + action_input[idx + 1:]
            try:
                action_input = literal_eval(action_input)
            except SyntaxError:
                # print("SyntaxError")
                return 0
        else:
            match = re.search(r'\((.*)\)', action_input)
            if match:
                input_list = [item.strip() for item in match.group(1).split(',')] 
                input_list = [item for item in input_list if item]
            else:
                # print("MatchError")
                return 0
            
            paramlist = [] 
            for tools in config:
                if (tools["name"]) == test_action:
                    param_config = tools["parameters"]["properties"]
                    paramlist = list(param_config)
                    break
            action_input_dict = {} 
            try:
                if len(input_list) > len(paramlist) and paramlist: 
                     # print("IndexError: Too many arguments provided by model")
                     return 0

                for idx, input_val in enumerate(input_list):
                    if idx < len(paramlist): 
                        action_input_dict[paramlist[idx]] = input_val
                action_input = action_input_dict 
            except IndexError: 
                # print("UnboundLocalError/IndexError")
                return 0
    elif version == 2:
        action_input = action_input.replace("(", "{").replace(")", "}").replace("=", "':")
        for idx, char in enumerate(action_input):
            if action_input[idx] == "{" and action_input[idx + 1] != "}":
                action_input = action_input[:idx + 1] + "'" + action_input[idx + 1:]
            if idx > 0 and action_input[:idx + 1].count("'") % 2 == 0:
                if (action_input[idx - 1] + action_input[idx] == ", ") and (action_input[idx - 1] + action_input[idx] + action_input[idx + 1] != ", '"):
                    action_input = action_input[:idx + 1] + "'" + action_input[idx + 1:]
        try:
            action_input = literal_eval(action_input)
        except SyntaxError:
            # print("SyntaxError")
            return 0
            
    if isinstance(action_input, dict): 
        keys_to_delete = [key for key, value in action_input.items() if value == '']
        for key in keys_to_delete:
            del action_input[key]
    elif action_input == 0 and version == 1 and not match: 
        return 0
        
    return action_input


def get_test_value(data, config, version):
    if not version: # Assuming data is the raw model output string
        test_value_str = data # Rename to avoid confusion if data is dict later
        test_action = ""
        action_input_json_str = ""
        
        action_str_marker = "Action:"
        action_input_str_marker = "Action Input:"
        
        action_pos = test_value_str.find(action_str_marker)
        action_input_pos = test_value_str.find(action_input_str_marker)

        if action_pos != -1 and action_input_pos != -1 and action_pos < action_input_pos:
            test_action = test_value_str[action_pos + len(action_str_marker):action_input_pos].strip()
            
            # Try to find the start of the JSON for Action Input
            json_start_pos = action_input_pos + len(action_input_str_marker)
            # Find the opening brace '{' that starts the JSON object
            actual_json_start = test_value_str.find('{', json_start_pos)
            
            if actual_json_start != -1:
                action_input_json_str = match_square_bracket(test_value_str, actual_json_start)
                if not action_input_json_str: # match_square_bracket failed
                    return test_action, 0 
            else: # No '{' found after "Action Input:"
                 return test_action, 0
        else: # Markers not found or in wrong order
            return "", 0 # Or some indicator of failure

        if not test_action: # If action is empty after stripping
             return "", 0

        try:
            # Ensure action_input_json_str is not empty before parsing
            if not action_input_json_str:
                return test_action, 0
            test_action_input = json.loads(action_input_json_str)
        except json.decoder.JSONDecodeError:
            return test_action, 0
        
        if isinstance(test_action_input, str): # Should be a dict
            return test_action, 0
            
    else: # Raven versions
        # Ensure data is a dict as expected by get_raven_resultcall for Raven
        if not isinstance(data, dict) or "result" not in data:
             # print("Error: Raven mode expects a dictionary with 'result' key.")
             return "", 0 # Cannot proceed
        test_value = get_raven_resultcall(data, version)
        if not test_value or not isinstance(test_value, str) or test_value.find("(") == -1: # Check if test_value is valid
            return "", 0
        test_action = test_value[:test_value.find("(")]
        raw_action_input_str = test_value[test_value.find("("):] # Raw string like " (param1='value1')"
        test_action_input = get_raven_action_input(raw_action_input_str, test_action, config, version)
        # get_raven_action_input already returns 0 on failure
    
    return test_action, test_action_input


def delete_input_text(rc_file_path, test):
    new_test = []
    with open(rc_file_path, encoding="utf-8") as f:
        input_test = json.load(f)
    for i in range(len(input_test)):
        input_len = len(input_test[i]["content"])
        new_test.append(test[i][input_len:])
    return new_test

parsed_or_missing_indices = set()

def ts_eval(test, answer, version=0):
    global check_list, cata_list, error_cases, error_type_counts, parsed_or_missing_indices
    tool_selection = []
    for i in range(len(answer)):
        config = get_config(answer[i])
        answers = get_answer_list(answer[i])

        if test[i] is None:
            if i not in error_cases:
                error_cases[i] = []
            if "No Output Error" not in error_cases[i]: # Avoid duplicate error type string
                error_cases[i].append("No Output Error")
            if i not in parsed_or_missing_indices:
                error_type_counts["No Output Error"] += 1
                parsed_or_missing_indices.add(i)
            continue
        test_action, test_action_input = get_test_value(test[i], config, version)

        if not test_action_input: # test_action_input is 0 or other falsy value from get_test_value
            if i not in error_cases:
                error_cases[i] = []
            if "Parsing Error" not in error_cases[i]: # Avoid duplicate error type string
                 error_cases[i].append("Parsing Error")
            if i not in parsed_or_missing_indices: # Count only once per item for this type of error
                error_type_counts["Parsing Error"] += 1
                parsed_or_missing_indices.add(i)
            continue # Must skip further checks for this item

        right_status = 0
        for ans in answers:
            answer_action = ans[ans.find("Action:") + 8: ans.find("Action Input:")]
            if answer_action and answer_action[-1] == "\n": # Check if answer_action is not empty
                answer_action = answer_action[:-1]
            
            # Handle "finish" case for test_action, config might not always have -1 index if empty
            # Check if config is not empty and has at least one tool before accessing config[-1]
            final_tool_name = ""
            if config and isinstance(config, list) and len(config) > 0 and "name" in config[-1]:
                 final_tool_name = config[-1]["name"]

            if answer_action == final_tool_name and test_action == "finish":
                test_action = answer_action
            
            if not answer_action == test_action:
                continue
            if right_status < 1:
                right_status = 1
                break
        
        if right_status >= 1:
            tool_selection.append(i)
        else:
            if i not in error_cases:
                error_cases[i] = []
            if "Tool Selection Error" not in error_cases[i]: 
                 error_cases[i].append("Tool Selection Error")
            error_type_counts["Tool Selection Error"] += 1 
            
    a_list = []
    a_list.append(len(tool_selection))
    for cata in cata_list:
        a_list.append(len(list(set(cata) & set(tool_selection))))
    check_list.append(a_list)

def pi_eval(test, answer, version=0):
    global check_list, cata_list, error_cases, error_type_counts, parsed_or_missing_indices
    parameter_identification = []
    for i in range(len(answer)):
        config = get_config(answer[i])
        answers = get_answer_list(answer[i])

        if test[i] is None:
            if i not in error_cases:
                error_cases[i] = []
            if "No Output Error" not in error_cases[i]:
                error_cases[i].append("No Output Error")
            if i not in parsed_or_missing_indices:
                error_type_counts["No Output Error"] += 1
                parsed_or_missing_indices.add(i)
            continue

        test_action, test_action_input = get_test_value(test[i], config, version)

        if not test_action_input:
            if i not in error_cases:
                error_cases[i] = []
            if "Parsing Error" not in error_cases[i]:
                 error_cases[i].append("Parsing Error")
            if i not in parsed_or_missing_indices:
                error_type_counts["Parsing Error"] += 1
                parsed_or_missing_indices.add(i)
            continue

        right_status = 0
        for ans in answers:
            answer_action = ans[ans.find("Action:") + 8: ans.find("Action Input:")]
            if answer_action and answer_action[-1] == "\n":
                answer_action = answer_action[:-1]
            
            try: # Add try-except for json.loads on answer_action_input
                answer_action_input_str = ans[ans.find("Action Input:") + 14:]
                if not answer_action_input_str: answer_action_input = {}
                else: answer_action_input = json.loads(answer_action_input_str)
            except json.JSONDecodeError:
                continue


            final_tool_name = ""
            if config and isinstance(config, list) and len(config) > 0 and "name" in config[-1]:
                 final_tool_name = config[-1]["name"]

            if answer_action == final_tool_name and test_action == "finish":
                test_action = answer_action
            
            if not answer_action == test_action:
                continue
            if right_status < 1:
                right_status = 1
            
            if not isinstance(test_action_input, dict) or not isinstance(answer_action_input, dict):
                continue

            if not answer_action_input.keys() == test_action_input.keys():
                continue
            if right_status < 2:
                right_status = 2
                break
                
        if right_status >= 2:
            parameter_identification.append(i)
        else: # This implies right_status is 0 or 1
            if i not in error_cases:
                error_cases[i] = []
        
            if "Parameter Identification Error" not in error_cases[i]:
                 error_cases[i].append("Parameter Identification Error")
            error_type_counts["Parameter Identification Error"] += 1
            
    a_list = []
    a_list.append(len(parameter_identification))
    for cata in cata_list:
        a_list.append(len(list(set(cata) & set(parameter_identification))))
    check_list.append(a_list)

def cf_eval(test, answer, version=0):
    global check_list, cata_list, error_cases, error_type_counts, parsed_or_missing_indices
    content_filling = []
    for i in range(len(answer)):
        config = get_config(answer[i])
        answers = get_answer_list(answer[i])

        if test[i] is None:
            if i not in error_cases:
                error_cases[i] = []
            if "No Output Error" not in error_cases[i]:
                error_cases[i].append("No Output Error")
            if i not in parsed_or_missing_indices:
                error_type_counts["No Output Error"] += 1
                parsed_or_missing_indices.add(i)
            continue

        test_action, test_action_input = get_test_value(test[i], config, version)

        if not test_action_input:
            if i not in error_cases:
                error_cases[i] = []
            if "Parsing Error" not in error_cases[i]:
                 error_cases[i].append("Parsing Error")
            if i not in parsed_or_missing_indices:
                error_type_counts["Parsing Error"] += 1
                parsed_or_missing_indices.add(i)
            continue

        right_status = 0
        for ans in answers:
            answer_action = ans[ans.find("Action:") + 8: ans.find("Action Input:")]
            if answer_action and answer_action[-1] == "\n":
                answer_action = answer_action[:-1]

            try: # Add try-except for json.loads on answer_action_input
                answer_action_input_str = ans[ans.find("Action Input:") + 14:]
                if not answer_action_input_str: answer_action_input = {}
                else: answer_action_input = json.loads(answer_action_input_str)
            except json.JSONDecodeError:
                continue 

            final_tool_name = ""
            ask_user_tool_name = ""
            if config and isinstance(config, list):
                if len(config) > 0 and "name" in config[-1]:
                    final_tool_name = config[-1]["name"]
                if len(config) > 1 and "name" in config[-2]: 
                    ask_user_tool_name = config[-2]["name"]


            if answer_action == final_tool_name and test_action == "finish":
                test_action = answer_action 
            
            if not answer_action == test_action:
                continue
            if right_status < 1: # Tool name matches
                right_status = 1
            
            if not isinstance(test_action_input, dict) or not isinstance(answer_action_input, dict):
                continue # Should not happen if parsing is handled

            if not answer_action_input.keys() == test_action_input.keys():
                continue
            if right_status < 2: # Parameter keys match
                right_status = 2

            current_answer_action_for_value_check = answer_action 
            if current_answer_action_for_value_check == final_tool_name:
                current_answer_action_for_value_check = "finish"
            if current_answer_action_for_value_check == ask_user_tool_name: 
                current_answer_action_for_value_check = "ask_to_user"

            temp_answer_action_input = answer_action_input.copy()
            temp_test_action_input = test_action_input.copy()

            del_key_answer = []
            for key, value in temp_answer_action_input.items():
                if str(value) == "None": 
                    del_key_answer.append(key)
            
            for key in del_key_answer:
                del temp_answer_action_input[key]
                if key in temp_test_action_input:
                     del temp_test_action_input[key]

            values_match = False
            if current_answer_action_for_value_check == "finish" or current_answer_action_for_value_check == "ask_to_user":
                values_match = True
            else:
                if temp_answer_action_input == temp_test_action_input:
                    values_match = True
            
            if not values_match:
                continue

            if right_status < 3: # Values match (or skipped for finish/ask_to_user)
                right_status = 3
                break
                
        if right_status >= 3:
            content_filling.append(i)
        else:
            if i not in error_cases:
                error_cases[i] = []
            if "Content Filling Error" not in error_cases[i]:
                 error_cases[i].append("Content Filling Error")
            error_type_counts["Content Filling Error"] += 1
            
    a_list = []
    a_list.append(len(content_filling))
    for cata in cata_list:
        a_list.append(len(list(set(cata) & set(content_filling))))
    check_list.append(a_list)


def general_eval(test_data, answer_data):
    ts_eval(test_data, answer_data)
    pi_eval(test_data, answer_data)
    cf_eval(test_data, answer_data)

def raven_eval(test_data, answer_data, version):
    ts_eval(test_data, answer_data, version)
    pi_eval(test_data, answer_data, version)
    cf_eval(test_data, answer_data, version)

cata_list = None
check_list = None
error_cases = {}
parsed_or_missing_indices = set()

error_type_counts = {
    "Tool Selection Error": 0,
    "Parameter Identification Error": 0,
    "Content Filling Error": 0,
    "Parsing Error": 0,
    "No Output Error": 0
}


@click.command()
@click.option("--model", type=str, default="/bjzhyai03/workhome/songzijun/huggingface/llama3.1_8b_instruct")
@click.option("--datasets", type=list, default=["clean", "heavy", "medium", "slight", "union"])
@click.option("--is_api", type=bool, default=False)
@click.option("--tensor_parallel_size", type=int, default=8)
@click.option("--batch_size", type=int, default=128)
@click.option("--gpu_memory_utilization", type=float, default=0.9)
@click.option("--max_model_len", type=int, default=4096)
@click.option("--max_output_tokens", type=int, default=2048)
@click.option("--model_name", type=str)
@click.option("--debug", "debug_mode", is_flag=True, default=False, help="Run in debug mode with only one data sample.")
@click.option("--think_mode", "think_mode", is_flag=True, default=False)
@click.option("--think_special_tokens", "think_special_tokens", type=str, default="think")
def main(
    model: str,
    datasets: list,
    is_api: bool,
    tensor_parallel_size: int,
    batch_size: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    max_output_tokens: int, 
    model_name: str,
    debug_mode: bool,
    think_mode: bool,
    think_special_tokens: str
    ):
    ### Setup
    model_name = os.path.basename(model)
    
    llm = None
    need_llm = False
    for dataset in datasets:
        output_path = f"../results/rotbench/{model_name}/{dataset}_results.json"
        if not os.path.exists(output_path):
            need_llm = True
            break
    
    if need_llm:
        llm = LLM(
            model=model,
            tensor_parallel_size=tensor_parallel_size,
            is_api=is_api,
            use_sharegpt_format=False,
            max_input_tokens=max_model_len,
            batch_size=batch_size,
            max_output_tokens=max_output_tokens,
            think_mode=think_mode,
            think_special_tokens=think_special_tokens,
        )

    data_results = {}
    for dataset in datasets:
        raw_data_path = f"../data/rotbench/First_Turn/{dataset}.json"
        print(f"Loading data from {os.path.abspath(raw_data_path)}")
        with open(raw_data_path, "r", encoding='utf-8') as f:
            eval_data = json.load(f)

        if debug_mode:
            eval_data = eval_data[:1]
            print("[Debug] - in rotbench_eval.py - The first query sample is: ")
            print(eval_data[0]["conversations"][0]["value"])

        global cata_list, check_list, error_cases, error_type_counts, parsed_or_missing_indices 
        cata_list = get_cata_list(raw_data_path)
        check_list = []

        error_cases = {}
        error_type_counts = {
            "Tool Selection Error": 0,
            "Parameter Identification Error": 0,
            "Content Filling Error": 0,
            "Parsing Error": 0,
            "No Output Error": 0
        }
        parsed_or_missing_indices = set()

        output_path = f"../results/rotbench/{model_name}/{dataset}_results.json"
        if not os.path.exists(f"../results/rotbench/{model_name}"):
            os.makedirs(f"../results/rotbench/{model_name}")
        print(f"The raw result will be saved to {os.path.abspath(output_path)}...")
    
        def run_inference(current_eval_data) -> List:
            if os.path.exists(output_path):
                with open(output_path, "r") as f:
                    results = json.load(f)
            else:
                if not llm: 
                    raise ValueError("LLM not initialized but inference results are missing.")
                if not is_api:
                    prompts_for_llm = []
                    if current_eval_data and isinstance(current_eval_data, list) and \
                       all(isinstance(ed, dict) and "conversations" in ed for ed in current_eval_data):
                        prompts_for_llm = [({"role": "user", "content": ed["conversations"][0]["value"] + ed["conversations"][1]["value"]}) for ed in current_eval_data]
                    elif current_eval_data and isinstance(current_eval_data, list) and \
                         len(current_eval_data) > 0 and isinstance(current_eval_data[0], list) and \
                         all(isinstance(ed, dict) and "conversations" in ed for ed in current_eval_data[0]):
                         print("Warning: Using eval_data[0] structure for prompts. Verify data format.")
                         prompts_for_llm = [({"role": "user", "content": ed["conversations"][0]["value"] + ed["conversations"][1]["value"]}) for ed in current_eval_data[0]]
                    else:
                        raise ValueError("eval_data format not suitable for prompt extraction.")
                    
                    results = llm.batch_generate_complete(prompts_for_llm)
                else:
                    messages = create_messages(current_eval_data) 
                    results = llm.batch_generate_chat(messages)
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=4)
            return results

        test_data = run_inference(eval_data) 

        if debug_mode:
            print("[Debug] - in rotbench_eval.py - The answer is: ")
            print(test_data[0]) 
            assert False

        max_len = len(eval_data) # Use len(eval_data) as it's the number of items
        if isinstance(test_data, list) and len(test_data) != max_len:
            print(f"Warning: Length of test_data ({len(test_data)}) does not match eval_data ({max_len}). This might cause index errors.")
            if len(test_data) < max_len:
                print(f"Padding test_data with {max_len - len(test_data)} None values.")
                test_data.extend([None] * (max_len - len(test_data)))

        is_raven_model = "raven" in model_name.lower() 
        raven_version = 0
        if is_raven_model:
            if "v1" in model_name.lower(): raven_version = 1 
            elif "v2" in model_name.lower(): raven_version = 2 

        if raven_version > 0:
            raven_eval(test_data, eval_data, raven_version)
        else:
            general_eval(test_data, eval_data)


        error_type_count_path = f"../results/rotbench/{model_name}/error_type_counts_{dataset}.json"
        with open(error_type_count_path, "w", encoding="utf-8") as f:
            json.dump(error_type_counts, f, ensure_ascii=False, indent=4)
        print(f"Error type statistics have been saved to {os.path.abspath(error_type_count_path)}.")

        bad_cases = []
        for idx, errors in error_cases.items():
            scenario_val = eval_data[idx]["scenario"] if idx < len(eval_data) else "N/A (index out of bounds for eval_data)"
            test_data_val = test_data[idx] if idx < len(test_data) else "N/A (index out of bounds for test_data)"
            answer_data_val = eval_data[idx] if idx < len(eval_data) else "N/A (index out of bounds for eval_data)"

            bad_case = {
                "index": idx,
                "scenario": scenario_val,
                "test_data": test_data_val,
                "answer_data": answer_data_val,
                "errors": errors
            }
            bad_cases.append(bad_case)
        
        bad_cases_path = f"../results/rotbench/{model_name}/bad_cases_{dataset}.jsonl"
        with open(bad_cases_path, "w", encoding="utf-8") as f:
            for case in bad_cases:
                f.write(json.dumps(case, ensure_ascii=False) + "\n")
        print(f"The error cases have been saved to {os.path.abspath(bad_cases_path)}.")

        if not check_list or len(check_list) < 3:
             print(f"Warning: check_list is not fully populated for dataset {dataset}. Results might be inaccurate.")
             while len(check_list) < 3:
                 check_list.append([0] * (len(cata_list) + 1 if cata_list else 1))

        ts_accuracy = (
            "{:.2f}".format(check_list[0][0] / max_len * 100)
            if max_len > 0 and len(check_list) > 0 and len(check_list[0]) > 0
            else "0.00"
        )
        pi_accuracy = (
            "{:.2f}".format(check_list[1][0] / max_len * 100)
            if max_len > 0 and len(check_list) > 1 and len(check_list[1]) > 0
            else "0.00"
        )
        cf_accuracy = (
            "{:.2f}".format(check_list[2][0] / max_len * 100)
            if max_len > 0 and len(check_list) > 2 and len(check_list[2]) > 0
            else "0.00"
        )
        data_results[f"{dataset}"] = {
            "Tool Selection": ts_accuracy,
            "Parameter Identification": pi_accuracy,
            "Content Filling": cf_accuracy
        }
    print(json.dumps(data_results, indent=4))


if __name__ == "__main__":
    main()
