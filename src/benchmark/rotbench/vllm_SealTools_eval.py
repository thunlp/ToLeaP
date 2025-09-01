# This file comes from fairyshine/Seal-Tools/LLM_Evaluation/src/calculate_pluginscore_detailed.py
# Copyright 2024 fairyshine/Seal-Tools
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

import json, os, re

def transform_output_format(dataset_name, output_text):
    def match_square_bracket(text, pos_s):
        counter = -1
        for i in range(pos_s+1,len(text)):
            if text[i] == '[':
                counter -= 1
            elif text[i] == ']':
                counter += 1
            if counter == 0:
                return i
        return -1
        
    text = re.sub("'", '"', output_text)
    text = re.sub("\n", "", text)
    pattern = re.compile("\[\s*\{\s*\"api\"", re.DOTALL)

    search_result = re.search(pattern, text)

    if search_result != None:
        pos_s = search_result.span()[0]
        pos_e = match_square_bracket(text, pos_s)

        text = text[pos_s:pos_e+1]
        # if "api" in text and "parameters" in text and "responses" in text:
        if "api" in text and "response" in text:
            if "parameters" in text or "arguments" in text:
                try:
                    output = json.loads(text)
                    return output
                except:
                    return -1
            else:
                return -1
        else:
            return -1
    else:
        return -1  

def transform_thought_output_format(dataset_name, output_text):
    match dataset_name:
        case 'ToolLearning':
            def match_square_bracket(text, pos_s):
                counter = -1
                for i in range(pos_s+1,len(text)):
                    if text[i] == '[':
                        counter -= 1
                    elif text[i] == ']':
                        counter += 1
                    if counter == 0:
                        return i
                return -1
            
            if text.find("Output:"):
                text = text[text.find("Output:") + 8:]
            else:
                return -1
            
            text = re.sub("'", '"', output_text)
            text = re.sub("\n", "", text)
            pattern = re.compile("\[\s*\{\s*\"api\"", re.DOTALL)

            search_result = re.search(pattern, text)

            if search_result != None:
                pos_s = search_result.span()[0]
                pos_e = match_square_bracket(text, pos_s)

                text = text[pos_s:pos_e+1]
                # if "api" in text and "parameters" in text and "responses" in text:
                if "api" in text and "response" in text:
                    if "parameters" in text or "arguments" in text:
                        try:
                            output = json.loads(text)
                            return output
                        except:
                            return -1
                    else:
                        return -1
                else:
                    return -1
            else:
                return -1  
        case _:
            print("ERROR!")

def write_jsonl(data_path, dataset):
    with open(data_path,'w', encoding='UTF-8') as f:
        for data in dataset:
            f.writelines(json.dumps(data, ensure_ascii=False))
            f.write('\n')

def write_json(data_path, dataset,indent=0):
    with open(data_path,'w', encoding='UTF-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=indent)

def read_json(data_path):
    dataset=[]
    with open(data_path,'r', encoding='UTF-8') as f:
        dataset = json.load(f)
    return dataset

def get_all_json_file_names(directory_path):
    json_files = [file for file in os.listdir(directory_path) if file.endswith('.json')]
    return json_files

def read_jsonl(data_path):
    dataset=[]
    with open(data_path,'r', encoding='UTF-8') as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset

def calculate_score_ToolLearning(data_path):
    raw_dataset = read_jsonl(data_path)
    result_dict = {}

    correct_format_num = 0

    correct_api_num = 0
    predict_api_num = 0
    gold_api_num = 0

    correct_param_num = 0
    predict_param_num = 0
    gold_param_num = 0

    error_cases = []
    error_type_counts = {
        "Invalid format": 0,
        "Missing API": 0,
        "Incorrect API": 0,
        "Missing parameter": 0,
        "Invalid parameter value": 0
    }

    for data in raw_dataset:
        gold_answer = json.loads(json.dumps(eval(data['gold_data']["conversations"][1]["value"])))
        # e.g. [{'api': 'getFilmInfo', 'parameters': {'film_name': 'Inception'}, 'responses': ['API_call_0', 'API_call_1', 'API_call_2', 'API_call_3', 'API_call_4']}]"
        gold_api_num += len(gold_answer)
        for gold_api in gold_answer:
            gold_param_num += len(gold_api['parameters'])

        if data['predict'][0] != -1:
            predict_answer = data['predict'][0]
            # e.g. "[{"api": "getFilmDetails","parameters": {"title": "Pulp Fiction"},"responses": ["title","genre","director","release_date","rating"]},
            #       {"api": "getFilmInfo","parameters": {"film_name": "Pulp Fiction"},"responses": ["title","release_year","director","actors","plot"]}]"
            data_correct = True
            correct_format_num += 1
            for predict_api in predict_answer:
                if "api" in predict_api:
                    predict_api_num += 1
                    if "parameters" in predict_api and type(predict_api["parameters"])==dict:
                        predict_param_num += len(predict_api["parameters"])
                    if "arguments" in predict_api and type(predict_api["arguments"])==dict:
                        predict_param_num += len(predict_api["arguments"])
                    gold_idx = -1
                    for idx in range(len(gold_answer)):
                        if gold_answer[idx]["api"] == predict_api["api"]:
                            gold_idx = idx
                            break
                    if gold_idx != -1:
                        correct_api_num += 1
                        params_correct = True
                        if "arguments" in predict_api and type(predict_api["arguments"])==dict:
                            for parameter_name in predict_api["arguments"]:
                                if parameter_name in gold_answer[gold_idx]["parameters"] and str(predict_api["arguments"][parameter_name]) == str(gold_answer[gold_idx]["parameters"][parameter_name]):
                                    correct_param_num += 1
                                else:
                                    params_correct = False
                                    if parameter_name not in gold_answer[gold_idx]["parameters"]:
                                        error_type_counts["Missing parameter"] += 1
                                    else:
                                        error_type_counts["Invalid parameter value"] += 1
                        if "parameters" in predict_api and type(predict_api["parameters"])==dict:
                            for parameter_name in predict_api["parameters"]:
                                if parameter_name in gold_answer[gold_idx]["parameters"] and str(predict_api["parameters"][parameter_name]) == str(gold_answer[gold_idx]["parameters"][parameter_name]):
                                    correct_param_num += 1
                                else:
                                    params_correct = False
                                    if parameter_name not in gold_answer[gold_idx]["parameters"]:
                                        error_type_counts["Missing parameter"] += 1
                                    else:
                                        error_type_counts["Invalid parameter value"] += 1
                        if not params_correct:
                            data_correct = False
                    else:
                        data_correct = False
                        error_type_counts["Incorrect API"] += 1
                else:
                    data_correct = False
                    error_type_counts["Missing API"] += 1
            if not data_correct:
                error_cases.append(data)
        else:
            error_cases.append(data)
            error_type_counts["Invalid format"] += 1

    # 将 AMOUNT 转换为百分比并保留两位小数
    if correct_format_num > 0:
        amount_value = 1.0 * correct_format_num / len(raw_dataset)
        result_dict["AMOUNT"] = round(amount_value * 100, 2)

    # 计算 API 相关的 P, R, F1 指标
    if correct_api_num * predict_api_num * gold_api_num > 0:
        # 先计算出原始的 P (Precision) 和 R (Recall)
        p_api = 1.0 * correct_api_num / predict_api_num
        r_api = 1.0 * correct_api_num / gold_api_num
        
        # 基于原始的 P 和 R 计算 F1 score，防止分母为0
        if p_api + r_api > 0:
            f1_api = 2 * p_api * r_api / (p_api + r_api)
        else:
            f1_api = 0
        
        # 将所有结果乘以100并保留两位小数后存入字典
        result_dict["P_api"] = round(p_api * 100, 2)
        result_dict["R_api"] = round(r_api * 100, 2)
        result_dict["F1_api"] = round(f1_api * 100, 2)

    # 计算参数相关的 P, R, F1 指标 (逻辑同上)
    if correct_param_num * predict_param_num * gold_param_num > 0:
        # 先计算出原始的 P 和 R
        p_param = 1.0 * correct_param_num / predict_param_num
        r_param = 1.0 * correct_param_num / gold_param_num

        # 基于原始的 P 和 R 计算 F1 score
        if p_param + r_param > 0:
            f1_param = 2 * p_param * r_param / (p_param + r_param)
        else:
            f1_param = 0

        # 将所有结果乘以100并保留两位小数后存入字典
        result_dict["P_param"] = round(p_param * 100, 2)
        result_dict["R_param"] = round(r_param * 100, 2)
        result_dict["F1_param"] = round(f1_param * 100, 2)

        result_dict["Error Type Statistics"] = error_type_counts

    return result_dict, error_cases

def raw_to_pred(raw_data_path, label_data_path):
    raw_dataset = read_json(raw_data_path)
    label_dataset = read_json(label_data_path)
    pred_list = []
    for raw_data,label_data in zip(raw_dataset,label_dataset):
        pred_output = {
                        'id':label_data["id"],
                        'predict':[],
                        'gold_data':label_data,
                    }
        output_text = raw_data[:]
        pred_text = transform_output_format("ToolLearning", output_text)
        pred_output['predict'].append(pred_text)
        pred_list.append(pred_output)
    return pred_list

def raw_cot_to_pred(raw_data_path, label_data_path):
    raw_dataset = read_json(raw_data_path)
    label_dataset = read_json(label_data_path)
    pred_list = []
    for raw_data,label_data in zip(raw_dataset,label_dataset):
        pred_output = {
                        'id':label_data["id"],
                        'predict':[],
                        'gold_data':label_data,
                    }
        output_text = raw_data[:]
        pred_text = transform_thought_output_format("ToolLearning", output_text)
        pred_output['predict'].append(pred_text)
        pred_list.append(pred_output)
    return pred_list

if __name__ == "__main__":
    pred_folder_path = "src/data/pred_data/Seal-Tools"
    model_name = "20250108afm20000"
    os.makedirs('src/data/eval_result/Seal-Tools/' + model_name + '/', exist_ok=True)
    os.makedirs('src/data/pred_data/Seal-Tools/' + model_name + '/', exist_ok=True)
    output_dir = "src/data/eval_result/Seal-Tools/" + model_name
    # output_dir = "src/data/eval_result/Seal-Tools/" + model_name
    dataset_name_list = [
                         "dev", 
                        #  "test_in_domain", 
                        #  "test_out_domain",
                         ]

    for dataset_name in dataset_name_list:
        
        # raw file to pred file
        # raw_data_path = "src/data/pred_data/Seal-Tools" + "/" + model_name + '/raw_' + dataset_name +'.jsonl'
        raw_data_path = "src/data/vllm_pred_data/Seal-Tools" + "/" + model_name + '/' + dataset_name +'.json'
        label_data_path = "src/data/eval_data/Seal-Tools" +  "/" + dataset_name +'.json'
        pred_data = raw_to_pred(raw_data_path, label_data_path)
        pred_data_path = "src/data/pred_data/Seal-Tools" + "/" + model_name + '/pred_' + dataset_name +'.jsonl'
        write_jsonl(pred_data_path, pred_data)
        
        # evaluate pred file 
        result_path =  output_dir + '/result_' + dataset_name +'.json'
        pred_datapath = pred_folder_path + "/" + model_name + '/pred_' + dataset_name +'.jsonl'
        result = calculate_score_ToolLearning(pred_datapath)
        write_json(result_path, result, indent=4)

    
