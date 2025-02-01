import sys
import os
import click
import json
from typing import List, Dict
from ast import literal_eval

current_dir = os.path.dirname(os.path.abspath(__file__)) 
utils_dir = os.path.join(current_dir, '..')
sys.path.append(utils_dir)

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

def initialize_llm(model: str, is_api: bool, conf: Config, tensor_parallel_size: int,
                   max_model_len: int, gpu_memory_utilization: float, batch_size: int) -> LLM:
    if not is_api:
        llm = LLM(
                model=model,
                tensor_parallel_size=tensor_parallel_size,
                use_sharegpt_format=False,
                max_input_tokens=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                batch_size=batch_size
        )
    else:
        llm = LLM(model=model, is_api=is_api)
    return llm

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
                print("SyntaxError")
                return 0
        else:
            match = re.search(r'\((.*)\)', action_input)
            if match:
                input_list = [item for item in match.group(1).split(', ')]
            else:
                print("MatchError")
                return 0
            for tools in config:
                if (tools["name"]) == test_action:
                    param_config = tools["parameters"]["properties"]
                    paramlist = list(param_config)
                    break
            action_input = {}
            try:
                for idx, input in enumerate(input_list):
                    action_input[paramlist[idx]] = input
            except (UnboundLocalError, IndexError):
                print("UnboundLocalError/IndexError")
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
            print("SyntaxError")
            return 0
    for key in list(action_input.keys()):
        if action_input[key] == '':
            del action_input[key]
    return action_input

def get_test_value(data, config, version):
    if not version:
        test_value = data
        test_action = test_value[test_value.find("Action:") + 8: test_value.find("Action Input:")]
        if test_action == "":
            return "", 0
        if test_action[-1] == "\n":
            test_action = test_action[:-1]
        try:
            test_action_end_index = match_square_bracket(test_value, test_value.find("Action Input:") + 14)
            test_action_input = json.loads(test_action_end_index)
        except json.decoder.JSONDecodeError:
            return test_action, 0
        if isinstance(test_action_input, str):
            return test_action, 0
    else:
        test_value = get_raven_resultcall(data, version)
        test_action = test_value[:test_value.find("(")]
        test_action_input = test_value[test_value.find("("):]
        test_action_input = get_raven_action_input(test_action_input, test_action, config, version)
    return test_action, test_action_input

def delete_input_text(rc_file_path, test):
    new_test = []
    with open(rc_file_path, encoding="utf-8") as f:
        input_test = json.load(f)
    for i in range(len(input_test)):
        input_len = len(input_test[i]["content"])
        new_test.append(test[i][input_len:])
    return new_test

def ts_eval(test, answer, version=0):
    global check_list
    global cata_list
    global error_cases
    global error_type_counts
    tool_selection = []
    for i in range(len(answer)):
        config = get_config(answer[i])
        answers = get_answer_list(answer[i])
        # delete input text
        if test[i] == None:
            continue
        test_action, test_action_input = get_test_value(test[i], config, version)     
        if not test_action_input:
            continue
        # Check all possible answers
        right_status = 0
        for ans in answers:
            answer_action = ans[ans.find("Action:") + 8: ans.find("Action Input:")]
            if answer_action[-1] == "\n":
                answer_action = answer_action[:-1]
            if answer_action == config[-1]["name"] and test_action == "finish":
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
            error_cases[i].append("Tool Selection Error")
            error_type_counts["Tool Selection Error"] += 1
    a_list = []
    a_list.append(len(tool_selection))
    for cata in cata_list:
        a_list.append(len(list(set(cata) & set(tool_selection))))
    check_list.append(a_list)

def pi_eval(test, answer, version=0):
    global check_list
    global cata_list
    parameter_identification = []
    for i in range(len(answer)):
        config = get_config(answer[i])
        answers = get_answer_list(answer[i])
        if test[i] == None:
            continue
        test_action, test_action_input = get_test_value(test[i], config, version)
        if not test_action_input:
            continue
        # Check all possible answers
        right_status = 0
        for ans in answers:
            answer_action = ans[ans.find("Action:") + 8: ans.find("Action Input:")]
            if answer_action[-1] == "\n":
                answer_action = answer_action[:-1]
            answer_action_input = json.loads(ans[ans.find("Action Input:") + 14:])
            if answer_action == config[-1]["name"] and test_action == "finish":
                test_action = answer_action
            if not answer_action == test_action:
                continue
            if right_status < 1:
                right_status = 1
            if not answer_action_input.keys() == test_action_input.keys():
                continue
            if right_status < 2:
                right_status = 2
                # print("<Parameter Identification : Right>")
                break
        if right_status >= 2:
            parameter_identification.append(i)
        else:
            if i not in error_cases:
                error_cases[i] = []
            error_cases[i].append("Parameter Identification Error")
            error_type_counts["Parameter Identification Error"] += 1 
    a_list = []
    a_list.append(len(parameter_identification))
    for cata in cata_list:
        a_list.append(len(list(set(cata) & set(parameter_identification))))
    check_list.append(a_list)

def cf_eval(test, answer, version=0):
    global check_list
    
    content_filling = []
    for i in range(len(answer)):
        config = get_config(answer[i])
        answers = get_answer_list(answer[i])
        if test[i] == None:
            continue
        test_action, test_action_input = get_test_value(test[i], config, version)
        if not test_action_input:
            continue
        # Check all possible answers
        right_status = 0
        for ans in answers:
            answer_action = ans[ans.find("Action:") + 8: ans.find("Action Input:")]
            if answer_action[-1] == "\n":
                answer_action = answer_action[:-1]
            answer_action_input = json.loads(ans[ans.find("Action Input:") + 14:])
            if answer_action == config[-1]["name"] and test_action == "finish":
                test_action = answer_action
            if not answer_action == test_action:
                continue
            if right_status < 1:
                right_status = 1
            if not answer_action_input.keys() == test_action_input.keys():
                continue
            if right_status < 2:
                right_status = 2
            if answer_action == config[-1]["name"]:
                answer_action = "finish"
            if answer_action == config[-2]["name"]:
                answer_action = "ask_to_user"
            del_key = []
            for key, value in answer_action_input.items():
                if value == "None":
                    del_key.append(key)
            for key in del_key:
                del answer_action_input[key]
                del test_action_input[key]
            if not answer_action_input == test_action_input and answer_action != "finish" and answer_action != "ask_to_user":
                continue
            if right_status < 3:
                right_status = 3
                # print("<Content Filling : Right>")
                break
        if right_status >= 3:
            content_filling.append(i)
        else:
            if i not in error_cases:
                error_cases[i] = []
            error_cases[i].append("Content Filling Error")
            error_type_counts["Content Filling Error"] += 1  # 修改
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

def show_stats(check_list, max_len):
    print("*"*60)
    print("Overall:")
    print("Tool Selection: " + "{:.2f}".format(check_list[0][0] / max_len * 100))
    print("Parameter Identification: " + "{:.2f}".format(check_list[1][0] / max_len * 100))
    print("Content Filling: " + "{:.2f}".format(check_list[2][0] / max_len * 100))
    print(check_list)

cata_list = None
check_list = None
error_cases = {}
error_type_counts = {
    "Tool Selection Error": 0,
    "Parameter Identification Error": 0,
    "Content Filling Error": 0
}

@click.command()
@click.option("--model", type=str, default="/bjzhyai03/workhome/songzijun/huggingface/llama3.1_8b_instruct")
@click.option("--datasets", type=list, default=["clean", "heavy", "medium", "slight", "union"])
@click.option("--is_api", type=bool, default=False)
@click.option("--tensor_parallel_size", type=int, default=1)
@click.option("--batch_size", type=int, default=16)
@click.option("--gpu_memory_utilization", type=float, default=0.9)
@click.option("--max_model_len", type=int, default=8192)
def main(
    model: str, 
    datasets: list,
    is_api: bool, 
    tensor_parallel_size: int, 
    batch_size: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    ):
    ### Setup
    print("Begin to run RotBench")
    model_name = os.path.basename(model)
    llm = initialize_llm(model, is_api, conf, tensor_parallel_size, max_model_len, gpu_memory_utilization, batch_size)

    for dataset in datasets:
        raw_data_path = f"../../src/data/input_data/RoTBench/First_turn/{dataset}.json"
        print(f"Loading data from {raw_data_path}")
        with open(raw_data_path, "r", encoding='utf-8') as f:
            eval_data = json.load(f)
        print(len(eval_data))
        global cata_list
        global check_list 
        global error_cases
        global error_type_counts
        cata_list = get_cata_list(raw_data_path)
        check_list = [] 

        error_cases = {}
        error_type_counts = {
            "Tool Selection Error": 0,
            "Parameter Identification Error": 0,
            "Content Filling Error": 0
        }

        ### Run inference
        if not conf.use_chat:
            output_path = f"benchmark_results/rotbench/{model_name}/hf_{model_name}_rotbench_{dataset}_results.json"
        else:
            if is_api:
                output_path = f"benchmark_results/rotbench/{model_name}/api_{model_name}_rotbench_{dataset}_results.json"
            else: 
                output_path = f"benchmark_results/rotbench/{model_name}/vllm_{model_name}_rotbench_{dataset}_results.json"
        if not os.path.exists(f"benchmark_results/rotbench/{model_name}"):
            os.makedirs(f"benchmark_results/rotbench/{model_name}")
        print(f"The raw result will be saved to {os.path.abspath(output_path)}...")

        def run_inference() -> List:
            if os.path.exists(output_path): # if exists
                with open(output_path, "r") as f:
                    results = json.load(f)
            else: # if not 
                if not conf.use_chat: # hf batch generate
                    results = llm.batch_generate_complete(
                        [(ed["conversations"][0]["value"] + ed["conversations"][1]["value"]) for ed in eval_data],
                        temperature=0
                    )
                else:  # vllm batch generate
                    messages = create_messages(eval_data)
                    if not is_api:
                        with llm.start_server():
                            results = llm.batch_generate_chat(messages)
                    else:
                        print("You are using batch_generate_chat to execute inference")
                        results = llm.batch_generate_chat(messages)
                with open(output_path, "w") as f:
                    json.dump(results, f, indent=4)
            return results
        
        results = run_inference()

        ### Evaluation
        with open(output_path, encoding="utf-8") as f:
            test_data = json.load(f)
        max_len = len(test_data)
        general_eval(test_data, eval_data)
        show_stats(check_list, max_len)

        print("Error Type Statistics: ")
        for error_type, count in error_type_counts.items():
            print(f"{error_type}: {count}")

        error_type_count_path = f"benchmark_results/rotbench/{model_name}/error_type_counts_{dataset}.json"
        with open(error_type_count_path, "w", encoding="utf-8") as f:
            json.dump(error_type_counts, f, ensure_ascii=False, indent=4)
        print(f"Error type statistics have been saved to {error_type_count_path}.")

        bad_cases = []
        for idx, errors in error_cases.items():
            bad_case = {
                "index": idx,
                "scenario": eval_data[idx]["scenario"],
                "test_data": test_data[idx],
                "answer_data": eval_data[idx],
                "errors": errors
            }
            bad_cases.append(bad_case)
        
        bad_cases_path = f"benchmark_results/rotbench/{model_name}/bad_cases_{dataset}.jsonl"
        with open(bad_cases_path, "w", encoding="utf-8") as f:
            for case in bad_cases:
                f.write(json.dumps(case, ensure_ascii=False) + "\n")
        print(f"The error cases have been saved to {bad_cases_path}.")

if __name__ == "__main__":
    main()
