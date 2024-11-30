import argparse
import json

def extract_nested_content(s):
    stack = []  # 用来存储左括号的索引
    result = None  # 存储匹配的内容
    start_index = -1  # 初始起点

    for i, char in enumerate(s):
        if char == '{':
            if not stack:  # 如果栈为空，记录最外层括号的起点
                start_index = i
            stack.append(i)  # 将当前括号的索引入栈
        elif char == '}':
            if stack:  # 栈不为空时匹配
                stack.pop()  # 出栈一个左括号
                if not stack:  # 如果栈为空，表示匹配完成
                    result = s[start_index + 1:i]  # 提取匹配的内容
                    break

    return result

def jsonlizer(predict):
    pretool = {}
    if "\"api\":" in predict:
        try:
            s = predict[predict.find("\"api\":")-20 :]
            result = "{" + extract_nested_content(s) + "}"
            pretool = json.loads(json.dumps(eval(result)))
        except: return ""
    return pretool

def calculate_score_ToolLearning(raw_dataset):
    result_dict = {}

    correct_format_num = 0

    correct_api_num = 0
    predict_api_num = 0
    gold_api_num = 0

    correct_param_num = 0
    predict_param_num = 0
    gold_param_num = 0

    for data in raw_dataset:
        gold_answer = json.loads(json.dumps(eval(data["label"])))

        gold_api_num += len(gold_answer)
        for gold_api in gold_answer:
            gold_param_num += len(gold_api['parameters'])

        if data['predict'][0] != -1:
            predict_answer = data['predict']
            predict_api =  jsonlizer(predict_answer)
            if predict_api != "":
                correct_format_num += 1
            
            if "api" in predict_api:
                predict_api_num += 1
                if "parameters" in predict_api and type(predict_api["parameters"])==dict:
                    predict_param_num += len(predict_api["parameters"])
                gold_idx = -1
                for idx in range(len(gold_answer)):
                    if gold_answer[idx]["api"] == predict_api["api"]:
                        gold_idx = idx
                        break
                if gold_idx != -1:
                    correct_api_num += 1
                    if "parameters" in predict_api and type(predict_api["parameters"])==dict:
                        for parameter_name in predict_api["parameters"]:
                            if parameter_name in gold_answer[gold_idx]["parameters"] and str(predict_api["parameters"][parameter_name])==str(gold_answer[gold_idx]["parameters"][parameter_name]):
                                correct_param_num += 1


    if correct_format_num > 0:
        result_dict["AMOUNT"] = 100.0*correct_format_num/len(raw_dataset)

    if correct_api_num * predict_api_num * gold_api_num > 0:
        result_dict["P_api"] = 100.0*correct_api_num/predict_api_num
        result_dict["R_api"] = 100.0*correct_api_num/gold_api_num
        result_dict["F1_api"] = 2*result_dict["P_api"]*result_dict["R_api"]/(result_dict["P_api"]+result_dict["R_api"])
    
    if correct_param_num * predict_param_num * gold_param_num > 0:
        result_dict["P_param"] = 100.0*correct_param_num/predict_param_num
        result_dict["R_param"] = 100.0*correct_param_num/gold_param_num
        result_dict["F1_param"] = 2*result_dict["P_param"]*result_dict["R_param"]/(result_dict["P_param"]+result_dict["R_param"])

    return result_dict


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, default="src/scripts/test_out_domain.jsonl")
    args = parser.parse_args()

    #Test_file follows generated_prediction format
    accuarcy = []
    f = open(args.test_file, encoding="utf-8")
    evalData = [json.loads(line) for line in f]
    result_dict = calculate_score_ToolLearning(evalData)
    print(result_dict)
    f.close