import os
import json
import glob

def transferSharegpt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f: 
        data = json.load(f)
        dataNew = []
        for conversationIndex in data:   #start to modify
            for i in conversationIndex:   
                if i == "expected_output" or i == "output":
                    gpt = {"from": "gpt", "value": conversationIndex[i]}
                elif i == "input":
                    human = {"from": "human", "value": conversationIndex[i]}
                elif i == "instruction":
                    system = conversationIndex[i]
            conversation={"conversations": [human, gpt], "system": system}
            dataNew.append(conversation)
        return dataNew

if __name__ == "__main__":
    # Check datasets exist
    if os.path.exists("test-data") and os.path.exists("training-data"):
        print("Data file directory exist and start to process:")
    else:
        print("Data file directory do not exist, please check and retry")
        exit
    current_dir = os.getcwd() # Get Current Path
    save_dir = "..\\sft_data\\APIBANK" #Get Save Path
    folders = [f for f in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, f))] # Get All Folders
    for current_dir in folders:
        json_files =  glob.glob(os.path.join(current_dir, "*.json")) # Load json file
        for file_path in json_files:
            file_name_N = save_dir+ "\\sharegpt_format_" + os.path.basename(file_path)
            if current_dir == "training-data" or "test-data":
                try:    
                    dataNew = transferSharegpt(file_path)
                    f = open(file_name_N, 'w', encoding='utf-8')
                    json.dump(dataNew, f, ensure_ascii=False, indent=4)
                    print(file_path, ":Finished")
                except: continue
    print("**************!API-BANK Data Format Transtion Complete!**************")
