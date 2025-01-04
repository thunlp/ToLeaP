import multiprocessing
multiprocessing.set_start_method('spawn')

import sys
import os
import click
import json
import requests
from typing import List, Dict
from sklearn.metrics import f1_score
from ast import literal_eval
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__)) 
utils_dir = os.path.join(current_dir, '..')
sys.path.append(utils_dir)

from cfg.config import Config
from utils.llm import LLM

class RoTBench(LLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create_messages(self, conversation_data: List[Dict]) -> List[List[Dict]]:
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

def parse_model_name(model_path: str) -> str:
    model_mapping = {
        "bb46c15ee4bb56c5b63245ef50fd7637234d6f75": "Qwen2.5-7B-Instruct",
        "a2cb7a712bb6e5e736ca7f8cd98167f81a0b5bd8": "Llama-2-13b-chat-hf",
        "f5db02db724555f92da89c216ac04704f23d4590": "Llama-2-7b-chat-hf",
        "f66993d6c40a644a7d7885d4c029943861e06113": "ToolLLaMA-2-7b-v2",
        "checkpoint-10000": "afm10000",
        "checkpoint-20000": "afm20000",
        "checkpoint-30000": "afm30000",
        "checkpoint-40000": "afm40000",
        "checkpoint-50000": "afm50000",
        "01c7f73d771dfac7d292323805ebc428287df4f9": "Llama2-7b-hf",
    }
    model_split = os.path.basename(model_path)
    return model_mapping.get(model_split, model_split)

def initialize_llm(model: str, is_api: bool, conf: Config, tensor_parallel_size: int) -> LLM:
    if not is_api:
        if not conf.use_chat:
            print("Initializing LLM (hf batch path)...")
            llm = LLM(
                model=model,
                tensor_parallel_size=tensor_parallel_size,
                use_sharegpt_format=False,
            )
        else:
            print("Initializing RoTBenchLLM (vllm batch path)...")
            llm = RoTBench(
                model=model,
                tensor_parallel_size=tensor_parallel_size,
                use_sharegpt_format=False,
            )
    else:
        print("Initializing API model...")
        llm = LLM(model=model, is_api=is_api)
    return llm

def get_cata_list(answer_file: str) -> List[List[int]]:
    scenario_mapping = {
        "TG": "Text_Generation",
        "RS": "Real_Time_Search",
        "DU": "Data_Understanding",
        "PL": "Personal_Life",
        "AM": "Application_Manipulation",
        "IR": "Information_Retrieval",
        "FT": "Financial_Transactions",
    }
    categories = {key: [] for key in scenario_mapping.values()}
    with open(answer_file, encoding="utf-8") as f:
        data = json.load(f)
    for index, d in enumerate(data):
        sce = d.get("scenario")
        category = scenario_mapping.get(sce)
        if category:
            categories[category].append(index)
    return list(categories.values())

def get_config(data: Dict) -> Dict:
    value = data["conversations"][0].get("value", "")
    p_len = value.find("[")
    config_str = value[p_len:-13] if p_len != -1 else "{}"
    return json.loads(config_str)

def get_answer_list(data: Dict) -> List[Dict]:
    return data.get("conversations", [])[-1].get("value", "")

def match_square_bracket(text: str, pos_s: int) -> str:
    counter = -1
    for i in range(pos_s + 1, len(text)):
        if text[i] == '{':
            counter -= 1
        elif text[i] == '}':
            counter += 1
        if counter == 0:
            return text[pos_s: i + 1]
    return ""

def get_raven_resultcall(data, version):
    result_call = data.get("result", "")
    if version == 1:
        start_str = "Initial Answer: "
        end_str = "\nReflection: "
        start_idx = result_call.find(start_str) + len(start_str)
        end_idx = result_call.find(end_str)
        result_call = result_call[start_idx: end_idx]
    elif version == 2:
        result_call = data["result"][6:data["result"].find("\nThought:") - 1]
        if ";" in result_call:
            result_call = result_call.split(";")[0]
        if result_call.count("(") != 1:
            end_idx = result_call.find(")")
            start_idx = end_idx
            for char in reversed(result_call[:end_idx]):
                start_idx -= 1
                if char == "(":
                    break
            result_call = result_call[start_idx + 1: end_idx + 1]
    return result_call

def get_raven_action_input(action_input, test_action, config, version):
    try:
        if version == 1:
            if "=" in action_input:
                action_input = action_input.replace("(", "{").replace(")", "}").replace("=", "':")
                for idx, char in enumerate(action_input):
                    if action_input[idx] == "{" and action_input[idx + 1] != "}":
                        action_input = action_input[:idx + 1] + "'" + action_input[idx + 1:]
                    if idx > 0 and action_input[:idx + 1].count("'") % 2 == 0:
                        if (action_input[idx - 1] + action_input[idx] == ", ") and (action_input[idx - 1] + action_input[idx] + action_input[idx + 1] != ", '"):
                            action_input = action_input[:idx + 1] + "'" + action_input[idx + 1:]
                    action_input = literal_eval(action_input)
            else:
                match = re.search(r'\((.*)\)', action_input)
                input_list = match.group(1).split(', ') if match else []
                tools = next((tool for tool in config if tool["name"] == test_action), None)
                if tools:
                    paramlist = list(tools["parameters"]["properties"])
                    action_input = {paramlist[idx]: input for idx, input in enumerate(input_list)}
                else:
                    return 0
        elif version == 2:
            action_input = action_input.replace("(", "{").replace(")", "}").replace("=", "':")
            for idx, char in enumerate(action_input):
                if action_input[idx] == "{" and action_input[idx + 1] != "}":
                    action_input = action_input[:idx + 1] + "'" + action_input[idx + 1:]
                if idx > 0 and action_input[:idx + 1].count("'") % 2 == 0:
                    if (action_input[idx - 1] + action_input[idx] == ", ") and (action_input[idx - 1] + action_input[idx] + action_input[idx + 1] != ", '"):
                        action_input = action_input[:idx + 1] + "'" + action_input[idx + 1:]
            action_input = literal_eval(action_input)
        action_input = {k: v for k, v in action_input.items() if v != ''}
        return action_input
    except (SyntaxError, AttributeError, IndexError):
        print("Error processing action input.")
        return 0

def get_test_value(data, config, version):
    if version == 0:
        test_value = data.get("conversations", [])[-1].get("value", "") if isinstance(data, dict) else ""
        test_action = test_value.split("Action:")[1].split("Action Input:")[0].strip() if "Action:" in test_value else ""
        if not test_action:
            return "", 0
        if test_action.endswith("\n"):
            test_action = test_action[:-1]
        try:
            pos = test_value.find("Action Input:") + len("Action Input:")
            test_action_input_str = match_square_bracket(test_value, pos)
            test_action_input = json.loads(test_action_input_str)
        except json.decoder.JSONDecodeError:
            return test_action, 0
        return test_action, test_action_input if isinstance(test_action_input, dict) else 0
    else:
        test_value = get_raven_resultcall(data, version)
        test_action = test_value[:test_value.find("(")]
        test_action_input = test_value[test_value.find("("):]
        test_action_input = get_raven_action_input(test_action_input, test_action, config, version)
    return test_action, test_action_input

def get_evaluation_indices(test: List[Dict], answer: List[Dict], eval_type: str, version: int = 0) -> List[int]:
    evaluation_indices = []
    for i, ans in enumerate(answer):
        config = get_config(ans)
        answers = get_answer_list(ans)
        if not test[i]:
            continue
        test_action, test_action_input = get_test_value(test[i], config, version)
        if not test_action_input:
            continue
        for answer_entry in answers:
            answer_action = answer_entry.get("Action:", "").strip()
            if answer_action.endswith("\n"):
                answer_action = answer_action[:-1]
            if answer_action == config[-1]["name"] and test_action == "finish":
                test_action = answer_action
            if answer_action != test_action:
                continue
            if eval_type == "ts":
                evaluation_indices.append(i)
                break
            elif eval_type == "pi":
                answer_action_input = json.loads(answer_entry.get("Action Input:", "{}"))
                if set(answer_action_input.keys()) == set(test_action_input.keys()):
                    evaluation_indices.append(i)
                    break
            elif eval_type == "cf":
                answer_action_input = json.loads(answer_entry.get("Action Input:", "{}"))
                answer_action = answer_entry.get("Action:", "").strip()
                if answer_action == config[-1]["name"]:
                    answer_action = "finish"
                if answer_action == config[-2]["name"]:
                    answer_action = "ask_to_user"
                answer_action_input = {k: v for k, v in answer_action_input.items() if v != "None"}
                test_action_input = {k: v for k, v in test_action_input.items() if v != ""}
                if answer_action_input == test_action_input and answer_action not in ["finish", "ask_to_user"]:
                    evaluation_indices.append(i)
                    break
    return evaluation_indices

def general_eval(test_data: List[Dict], answer_data: List[Dict], check_list: List[List[int]], cata_list, version: int = 0):
    eval_types = ["ts", "pi", "cf"]
    for idx, eval_type in enumerate(eval_types):
        indices = get_evaluation_indices(test_data, answer_data, eval_type, version)
        a_list = [len(indices)]
        for cata in cata_list:
            a_list.append(len(set(cata) & set(indices)))
        check_list[idx].extend(a_list)

def show_stats(check_list: List[List[int]], max_len: int):
    print("Overall:")
    metrics = ["Tool Selection", "Parameter Identification", "Content Filling"]
    for idx, metric in enumerate(metrics):
        percentage = (check_list[idx][0] / max_len) * 100
        print(f"{metric}: {percentage:.2f}%")
    print(check_list)
    # All Scenarios
    scenarios = ["Text Generation", "Real-Time Search", "Data Understanding", 
                 "Personal Life", "Application Manipulation", "Information Retrieval", "Financial Transactions"]
    for id, sce in enumerate(scenarios):
        print(f"-----Acc_{sce}-----")
        div = max_len / 7
        for idx, metric in enumerate(metrics):
            percentage = (check_list[idx][id + 1] / div) * 100
            print(f"{metric}: {percentage:.2f}%")

@click.command()
@click.option("--model", type=str, default="/bjzhyai03/workhome/chenhaotian/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590")
@click.option("--datasets", type=list, default=["clean", "slight", "medium", "heavy", "union"])
@click.option("--is_api", type=bool, default=False)
@click.option("--tensor_parallel_size", type=int, default=2)
def main(
    model: str, 
    is_api: bool, 
    tensor_parallel_size: int, 
    datasets: list,
    ):
    ### Setup
    for dataset in datasets:
        if not conf.use_chat: # hf batch generate
            raw_data_path = f"../../src/data/input_data/RoTBench/First_turn_RC/{dataset}.json"
        else:
            raw_data_path = f"../../src/data/input_data/RoTBench/First_turn/{dataset}.json"
        print(f"Loading data from {raw_data_path}")
        with open(raw_data_path, "r", encoding='utf-8') as f:
            eval_data = json.load(f)
        model_name = parse_model_name(model)
        llm = initialize_llm(model, is_api, conf, tensor_parallel_size)
        cata_list = get_cata_list(raw_data_path)
        check_list = [[] for _ in range(3)]  # [ts, pi, cf]

        ### Run inference
        data_filename = os.path.splitext(os.path.basename(raw_data_path))[0]
        if not conf.use_chat:
            output_path = f"benchmark_results/hf_{model_name}_rotbench_{data_filename}_results.json"
        else:
            if is_api:
                output_path = f"benchmark_results/api_{model_name}_rotbench_{data_filename}_results.json"
            else: 
                output_path = f"benchmark_results/vllm_{model_name}_rotbench_{data_filename}_results.json"
        print(f"The raw result will be saved to {os.path.abspath(output_path)}...")

        def run_inference() -> List:
            if os.path.exists(output_path): # if exists
                with open(output_path, "r") as f:
                    results = json.load(f)
            else: # if not 
                if not conf.use_chat: # hf batch generate
                    # for ed in eval_data:
                    #     print(ed["content"])
                    #     print(type(ed["content"]))
                    #     assert False
                    results = llm.batch_generate_complete(
                        [ed["content"] for ed in eval_data],
                        temperature=0
                    )
                else:  # vllm batch generate
                    messages = llm._create_messages(eval_data)
                    if not is_api:
                        with llm.start_server():
                            results = llm.batch_generate_chat(messages)
                    else:
                        results = llm.batch_generate_chat(messages)
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=4)
            return results
        
        results = run_inference()

        ### Evaluation
        # with open(output_path, encoding="utf-8") as f:
        #     test_data = json.load(f)
        # max_len = len(test_data)
        # general_eval(test_data, eval_data, check_list, cata_list)
        # show_stats(check_list, max_len)

if __name__ == "__main__":
    main()
