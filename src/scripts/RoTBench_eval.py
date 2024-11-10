import sys
import json
import argparse
import re
from ast import literal_eval


sys.stdout.reconfigure(encoding='utf-8')




def get_config(data):
    config=[]
    for i in range(len(data["conversations"])):
        if i % 2 ==1:
            single_config_list=[]
            start =  data["conversations"][i]["value"].find("Action Input: ")
            single_config = json.loads(data["conversations"][i]["value"][start+14:])
            for j in single_config:
                single_config_list.append(single_config[j])
            config.append(single_config_list)
    return config


def get_answer_list(data):
    answer_list=[]
    for line in data["conversations"]:
        if line["from"]=="gpt":
            answer_list.append(line["value"])
    return answer_list


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


def get_test_value(data):
  
    test_value = data["label"]
    test_action = test_value[test_value.find("\"name\":") + 9: test_value.find("\"arguments\"")-3]
    if test_action[-1] == "\n":
        test_action = test_action[:-1]
    try:
        test_action_input=test_value[test_value.find("\"arguments\":") + 13:-1]
        test_action_input = json.loads(test_action_input)
    except json.decoder.JSONDecodeError:
        return test_action, 0
    # if isinstance(test_action_input, str):
    return test_action, test_action_input

# Tool Selection (eval) aims to evaluate whether model call right function.
def ts_eval(test, answer):
    global check_list
    tool_selection = []
    for i in range(len(answer)):
        answers = get_answer_list(answer[i])
        test_action, test_action_input = get_test_value(test[i])
        if not test_action_input:
            continue
        # Check all possible answers
        right_status = 0
        for ans in answers:
            answer_action = ans[ans.find("Action:") + 8: ans.find("Action Input:")]
            if answer_action[-1] == "\n":
                answer_action = answer_action[:-1]
            if test_action == "finish":
                test_action = answer_action
            if not answer_action == test_action:
                continue
            if right_status < 1:
                right_status = 1
                # print("<Tool Selection : Right>")
                break
        if right_status >= 1:
            tool_selection.append(i)
    a_list = []
    a_list.append(len(tool_selection))
    a_list.append(tool_selection)
    check_list.append(a_list)


def pi_eval(test, answer, version=0):
    global check_list
    parameter_identification = []
    for i in range(len(answer)):

        answers = get_answer_list(answer[i])
        test_action, test_action_input = get_test_value(test[i])
        if not test_action_input:
            continue
        # Check all possible answers
        right_status = 0
        for ans in answers:
            answer_action = ans[ans.find("Action:") + 8: ans.find("Action Input:")]
            if answer_action[-1] == "\n":
                answer_action = answer_action[:-1]
              
            answer_action_input = json.loads(ans[ans.find("Action Input:") + 14:])
            if test_action == "finish":
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
    a_list = []
    a_list.append(len(parameter_identification))
    a_list.append(parameter_identification)
    check_list.append(a_list)


def cf_eval(test, answer, version=0):
    global check_list
    content_filling = []
    for i in range(len(answer)):
        config = get_config(answer[i])
        answers = get_answer_list(answer[i])
        test_action, test_action_input = get_test_value(test[i])
        if not test_action_input:
            continue
        # Check all possible answers
        right_status = 0
        for ans in answers:
            answer_action = ans[ans.find("Action:") + 8: ans.find("Action Input:")]
            if answer_action[-1] == "\n":
                answer_action = answer_action[:-1]
            answer_action_input = json.loads(ans[ans.find("Action Input:") + 14:])
            
            
            if test_action == "finish":
                test_action = answer_action
            if not answer_action == test_action:
                continue
            if right_status < 1:
                right_status = 1
            if not answer_action_input.keys() == test_action_input.keys():
                continue
            if right_status < 2:
                right_status = 2

            # if answer_action == config[-1]["name"]:
            #     answer_action = "finish"
            # if answer_action == config[-2]["name"]:
            #     answer_action = "ask_to_user"
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
    a_list = []
    a_list.append(len(content_filling))
    a_list.append(content_filling)
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
    print("Overall:")
    print("Tool Selection: " + "{:.2f}".format(check_list[0][0] / max_len * 100))
    print("Parameter Identification: " + "{:.2f}".format(check_list[1][0] / max_len * 100))
    print("Content Filling: " + "{:.2f}".format(check_list[2][0] / max_len * 100))

    # # All Scenarios
    # scenarios = ["Text Generation", "Real-Time Search", "Data Understanding", "Personal Life", "Application Manipulation", "Information Retrieval", "Financial Transactions"]
    # for id, sce in enumerate(scenarios):
    #     print(f"-----Acc_{sce}-----")
    #     div = max_len / 7
    #     print("Tool Selection: " + "{:.2f}".format(check_list[0][id + 1] / div * 100))
    #     print("Parameter Identification: " + "{:.2f}".format(check_list[1][id + 1] / div * 100))
    #     print("Content Filling: " + "{:.2f}".format(check_list[2][id + 1] / div * 100))


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, default="src/scripts/generated_predictions.jsonl")
    parser.add_argument("--answer_file", type=str, default="src/scripts/first_turn_new_clean.json")
    args = parser.parse_args()

    #Test_file follows generated_prediction format
    test_file = args.test_file

    #Answer_file represents label and thus follows  share_gpt format
    answer_file = args.answer_file

    check_list = []
    #This evaluation method does not consider various distraction
    with open(test_file, encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f]
    with open(answer_file, encoding="utf-8") as f:
        answer_data = json.load(f)
    f.close

    max_len = len(answer_data)
    general_eval(test_data, answer_data)
    show_stats(check_list, max_len)
