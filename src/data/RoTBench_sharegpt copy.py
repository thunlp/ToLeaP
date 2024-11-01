import os
import json
import glob
import copy
def insert(data, thought, index):
    human = {"from":"human","value":""}
    thought = {"from":"gpt","value":thought}
    
    data["conversations"].insert(index, thought)
    index +=1
    data["conversations"].insert(index, human)
  

def transferSharegpt(data):
    # dataNew={}
    for conversationIndex in data:
        del conversationIndex["id"]
        del conversationIndex["scenario"]
        rawData=copy.deepcopy(conversationIndex["conversations"])
        # print(type(rawData))
        #start to modify
        for i in range(len(rawData)):   
            if rawData[i]["from"] == "assistant":
                index = 2
                #  try to deal value
                for thought in rawData[i]["value"]:
                    insert(conversationIndex, thought, index)
                    index +=2
                conversationIndex["conversations"][i]["from"] = "gpt"
            elif rawData[i]["from"] == "user":
                conversationIndex["conversations"][i]["from"] = "human"
            elif rawData[i]["from"] == "system":
                conversationIndex["system"] = rawData[i]["value"]
        del conversationIndex["conversations"][0]
        del conversationIndex["conversations"][-1]
    return data

def transferSharegptthrid(data):

    for conversationIndex in data:
        del conversationIndex["id"]
        del conversationIndex["scenario"]
        rawData=copy.deepcopy(conversationIndex["conversations"])
        # print(type(rawData))
        #start to modify
        index = 2
        for i in range(len(rawData)):   
            if rawData[i]["from"] == "assistant":
                #  try to deal value
                if type(rawData[i]["value"]) == list:
                    for thought in rawData[i]["value"]:
                        insert(conversationIndex, thought, index)
                    next(i)
                elif type(rawData[i]["value"]) == str:
                    thought = rawData[i]["value"]
                    insert(conversationIndex, thought, index)
                    next(i)
                del conversationIndex["conversations"][index+2]
                index +=2
            elif rawData[i]["from"] == "user":
                conversationIndex["conversations"][i]["from"] = "human"
            elif rawData[i]["from"] == "system":
                conversationIndex["system"] = rawData[i]["value"]
            elif rawData[i]["from"] == "function":
                conversationIndex["conversations"][i]["from"] = "function_call"
        del conversationIndex["conversations"][0]
   
    return data

if __name__ == "__main__":

    
    # Check datasets exist
    if os.path.exists("First_Turn") and os.path.exists("Third_Turn"):
        print("Data file directory exist and start to process:")
    else:
        print("Data file directory do not exist, please check and retry")
        exit

    # 获取当前文件夹和当前文件夹下所有文件夹
    current_dir = os.getcwd()
    save_dir = "../../RoTBenchData"
    folders = [f for f in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, f))]
    # 获取所有的 .json 文件
    for current_dir in folders:
        json_files =  glob.glob(os.path.join(current_dir, "*.json"))
        # 读取每个 JSON 文件的内容
        for file_path in json_files:
              
                with open(file_path, 'r', encoding='utf-8') as f: 
                    data = json.load(f)
                    # 处理 first turn 文件
                    if current_dir == "First_Turn":
                        file_name_N = save_dir+ "//first_turn_new_" + os.path.basename(file_path)
                        try:
                            dataNew = transferSharegpt(data)
                            for wrong in dataNew:
                                del wrong["conversations"][-1]
                            with open(file_name_N, 'w', encoding='utf-8') as f:
                                json.dump(dataNew, f, ensure_ascii=False, indent=4)
                                print(file_path, ":Finished")
                        except:
                            print(file_path, ":Failed")

                    # 处理 thrid turn 文件
                    # if current_dir == "Third_Turn":
                    #     file_name_N = save_dir+ "//third_turn_new_" + os.path.basename(file_path)
                    #     try:
                    #         dataNew = transferSharegptthrid(data)
                    #         with open(file_name_N, 'w', encoding='utf-8') as f:
                    #             json.dump(dataNew, f, ensure_ascii=False, indent=4)
                    #             print(file_path, ":Finished")
                    #     except:
                    #         print(file_path, ":Failed")
    print("Data Complete")
