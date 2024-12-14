import sys
import json
import argparse
import re
sys.stdout.reconfigure(encoding='utf-8')


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


def get_answer_list(answer):
    answerList = {}
    answerMatch = re.split(r"(Thought:)", answer["label"][2:])
    for i in answerMatch:
        if i.find("Action:") >= 0:
            answerAction = i[i.find("Action:") + 8 : i.find("Action Input:")-2]
            answerParam = "{" + extract_nested_content(i[i.find("Action Input:"):]) + "}"
            answerList[answerAction] = json.loads(answerParam)
    return answerList




def get_Predict_value (preAnswer):
    preValue = preAnswer["predict"]
    if preValue.find("Action:") >= 0 and preValue.find("Action Input:")>=0:
        preAction = preValue[preValue.find("Action:") + 8 : preValue.find("Action Input:")-1]
        preParam = "{" + extract_nested_content(preValue[preValue.find("Action Input:") : ]) + "}"
        return preAction, preParam
    else:
        return "", ""


# Tool Selection (eval) aims to evaluate whether model call right function.
def ts_eval(evalData):
    global checkList
    tool_selection = []
    for i in range(len(evalData)):
        try:
            answers = get_answer_list(evalData[i])
        except: print("No answers")
        try:
            preAction, preParam = get_Predict_value(evalData[i])
        except: continue
        right_status = 0
        for ansAction in answers:
            if ansAction[-1] == "\n":
                ansAction = ansAction[:-1]
            if preAction == "finish":
                preAction = ansAction
            if not preAction == ansAction:
                continue
            if right_status < 1:
                right_status = 1
                break
        if right_status >= 1:
            tool_selection.append(i)
    evalList = []
    evalList.append(len(tool_selection))
    print(tool_selection)
    print("sucess ts num:",len(tool_selection)," tool num:", len(evalData), "Success Rate:", len(tool_selection)/len(evalData))
    evalList.append(tool_selection)
    checkList.append(evalList)


def pi_eval(evalData):
    global checkList
    parameter_identification = []
    for i in range(len(evalData)):
        try:
            answers = get_answer_list(evalData[i])
            preAction, preParam = get_Predict_value(evalData[i])
            preParam =  json.loads(preParam)
        except: 
            continue    
        right_status = 0
        for ansAction in answers:
            if ansAction[-1] == "\n":
                ansAction = ansAction[:-1]
            if preAction == "finish":
                preAction = ansAction
           
            if right_status < 1:
                right_status = 1
            if not preParam.keys() == answers[ansAction].keys():
                continue
            if right_status < 2:
                right_status = 2
                break
        if right_status >= 2:
            parameter_identification.append(i)
    evalList = []
    evalList.append(len(parameter_identification))
    print("sucess pi num:",len(parameter_identification)," tool num:", len(evalData), "Success Rate:", len(parameter_identification)/len(evalData))
    evalList.append(parameter_identification)
    checkList.append(evalList)


def cf_eval(evalData):
    global checkList
    content_filling = []
    for i in range(len(evalData)):
        try:
            answers = get_answer_list(evalData[i])
            preAction, preParam = get_Predict_value(evalData[i])
            preParam =  json.loads(preParam)
        except: 
            continue    
        right_status = 0
        for ansAction in answers:
            if ansAction[-1] == "\n":
                ansAction = ansAction[:-1]
            if preAction == "finish":
                preAction = ansAction
            if not preAction == ansAction:
                continue
            if not preParam.keys() == answers[ansAction].keys():
                continue
            for key, value in preParam.items():
                if value != answers[ansAction][key]:
                    continue
            if right_status < 3:
                right_status = 3
            if right_status >= 3:
                content_filling.append(i)
               
    evalList = []
    evalList.append(len(content_filling))
    evalList.append(content_filling)
    checkList.append(evalList)
    print("sucess cf num:",len(content_filling)," tool num:", len(evalData), "Success Rate:", len(content_filling)/len(evalData))


def general_eval(evalData,):
    ts_eval(evalData)
    pi_eval(evalData)
    cf_eval(evalData)



if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("--predict_file", type=str, default="src/scripts/new_Third_Turnmedium.jsonl")
    # parser.add_argument("--origin_file", type=str, default="src/scripts/new_Third_Turnclean.jsonl")
    args = parser.parse_args()

    #Test_file follows generated_prediction format
    checkList = []
    f = open(args.predict_file, encoding="utf-8")
    evalData = [json.loads(line) for line in f]
    max_len = len(evalData)
    general_eval(evalData)
    f.close
