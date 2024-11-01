import os
import json
import glob


def transferSharegpt(data):
    dataNew=[]
    for conversationIndex in data:
        #start to modify
        for i in conversationIndex:   
            if i == "output":
                gpt = {"from":"gpt","value":conversationIndex[i]}
            elif i == "input":
                human = {"from":"human","value":conversationIndex[i]}
            elif i == "instruction":
                system =  conversationIndex[i]
        conversation={"conversations":[human, gpt], "system": system}
        dataNew.append(conversation)
    return dataNew


if __name__ == "__main__":

    # Check datasets exist
    if os.path.exists("test-data") and os.path.exists("training-data"):
        print("Data file directory exist and start to process:")
    else:
        print("Data file directory do not exist, please check and retry")
        exit

    # 获取当前文件夹和当前文件夹下所有文件夹
    current_dir = os.getcwd()
    
    save_dir = "../sft_data"

    folders = [f for f in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, f))]
    # 获取所有的 .json 文件
    for current_dir in folders:
        json_files =  glob.glob(os.path.join(current_dir, "*.json"))
        # 读取每个 JSON 文件的内容
        for file_path in json_files:
              
                with open(file_path, 'r', encoding='utf-8') as f: 
                    data = json.load(f)
                    # 处理 first turn 文件
                    if current_dir == "training-data":
                        file_name_N = save_dir+ "//sharegpt_format_" + os.path.basename(file_path)
                        try:
                            dataNew = transferSharegpt(data)
                            with open(file_name_N, 'w', encoding='utf-8') as f:
                                json.dump(dataNew, f, ensure_ascii=False, indent=4)
                                print(file_path, ":Finished")
                        except:
                            print(file_path, ":Failed")

    print("Data Complete")
