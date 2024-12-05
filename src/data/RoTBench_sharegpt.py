import os
import json
import glob
import copy
def transferSharegpt(file_path):
    f = open(file_path, 'r', encoding='utf-8')
    data = json.load(f)
    for conversationIndex in data:
        del conversationIndex["id"]
        del conversationIndex["scenario"]
        rawData= copy.deepcopy(conversationIndex["conversations"])
        for i in range(len(rawData)):   
            if rawData[i]["from"] == "assistant":
                conversationIndex["conversations"][i]["from"] = "gpt"
                conversationIndex["conversations"][i]["value"] = str(rawData[i]["value"])
            elif rawData[i]["from"]== "user":
                conversationIndex["conversations"][i]["from"] = "human"
            elif rawData[i]["from"] == "system":
                conversationIndex["system"] = str(conversationIndex["conversations"][i]["value"])
        del conversationIndex["conversations"][0]        
    return data

def transferSharegptthrid(file_path):
    f = open(file_path, 'r', encoding='utf-8')
    data = json.load(f)
    for conversationIndex in data:
        del conversationIndex["id"]
        del conversationIndex["scenario"]
        rawData=copy.deepcopy(conversationIndex["conversations"])
        for i in range(len(rawData)):   
            if rawData[i]["from"] == "assistant":
                conversationIndex["conversations"][i]["from"] = "gpt"
                conversationIndex["conversations"][i]["value"] = str(rawData[i]["value"])
            elif rawData[i]["from"]== "user":
                conversationIndex["conversations"][i]["from"] = "human"
            elif rawData[i]["from"] == "system":
                conversationIndex["system"] = str(conversationIndex["conversations"][i]["value"])
            elif rawData[i]["from"] == "function":
                conversationIndex["conversations"][i]["from"] = "observation"
        del conversationIndex["conversations"][0]
    return data

if __name__ == "__main__":
    # Check datasets exist
    if os.path.exists("First_Turn") and os.path.exists("Third_Turn"):
        print("Data file directory exist and start to process:")
    else:
        print("Data file directory do not exist, please check and retry")
        exit
    current_dir = os.getcwd()
    save_dir = "../../sft_data/RoTBench" #Get Save Path

    folders = [f for f in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, f))]
    for current_dir in folders:
        json_files =  glob.glob(os.path.join(current_dir, "*.json"))
        for file_path in json_files:
                file_name_N = save_dir+ "/new_" + current_dir + os.path.basename(file_path)

                try:
                    if current_dir == "First_Turn":
                        dataNew = transferSharegpt(file_path)
                    if current_dir == "Third_Turn":
                        dataNew = transferSharegptthrid(file_path)
                except: continue   
                f = open(file_name_N, 'w', encoding='utf-8')
                json.dump(dataNew, f, ensure_ascii=False, indent=4)
                print(file_path, ":Finished")
    print("**************!RoTBench Data Format Transtion Complete!**************")

