import os
import json
import glob

def transferSharegpt(file_path):
    f = open(file_path, 'r', encoding='utf-8')
    data = json.load(f)
    for conversationIndex in data:
        del conversationIndex["id"]
        del conversationIndex["scenario"]
        rawData=conversationIndex["conversations"]
        for i in rawData:   
            if i["from"] == "assistant":
                i["from"] = "gpt"
                i["value"] = str(i["value"])
            elif i["from"] == "user":
                i["from"] = "human"
            elif i["from"] == "system":
                conversationIndex["system"] = str(i["value"])
                del conversationIndex["conversations"][0]
    return data

def transferSharegptthrid(file_path):
    f = open(file_path, 'r', encoding='utf-8')
    data = json.load(f)
    for conversationIndex in data:
        del conversationIndex["id"]
        del conversationIndex["scenario"]
        rawData=conversationIndex["conversations"]
        for i in rawData:   
            if i["from"] == "assistant":
                i["from"] = "gpt"
                i["value"] = str(i["value"])
            elif i["from"] == "user":
                i["from"] = "human"
            elif i["from"] == "system":
                conversationIndex["system"] = str(i["value"])
                del conversationIndex["conversations"][0]
            elif i["from"] == "function":
                i["from"] = "observation"
    return data

if __name__ == "__main__":
    # Check datasets exist
    if os.path.exists("First_Turn") and os.path.exists("Third_Turn"):
        print("Data file directory exist and start to process:")
    else:
        print("Data file directory do not exist, please check and retry")
        exit
    current_dir = os.getcwd()
    print("current_dir = os.getcwd(): ", current_dir)
    save_dir = "../../sft_data/RoTBench"
    print("save_dir:", save_dir)
    folders = [f for f in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, f))]
    for current_dir in folders:
        json_files =  glob.glob(os.path.join(current_dir, "*.json"))
        for file_path in json_files:
            if current_dir == "First_Turn":
                file_name_N = save_dir+ "/new_first_" + os.path.basename(file_path)
                dataNew = transferSharegpt(file_path)
                f = open(file_name_N, 'w', encoding='utf-8')
                json.dump(dataNew, f, ensure_ascii=False, indent=4)
                print(file_path, ":Finished")
            # 处理 thrid turn 文件
            if current_dir == "Third_Turn":
                file_name_N = save_dir+ "/new_third_" + os.path.basename(file_path)
                dataNew = transferSharegptthrid(file_path)
                with open(file_name_N, 'w', encoding='utf-8') as f:
                    json.dump(dataNew, f, ensure_ascii=False, indent=4)
                    print(file_path, ":Finished")
    print("**************!RoTBench Data Format Transtion Complete!**************")
