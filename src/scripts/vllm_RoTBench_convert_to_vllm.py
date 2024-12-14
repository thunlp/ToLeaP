import json, os

def read_json(data_path):
    dataset=[]
    with open(data_path,'r', encoding='UTF-8') as f:
        dataset = json.load(f)
    return dataset

def convert(data_path):
    dataset = read_json(data_path)
    convert_list = []
    for data in dataset:
        content = ""
        for index in data["conversations"]:
            if index["from"] == "system" or index["from"] == "user":
                content += index["from"] + ": " + index["value"]
        template = {"role":"user", "content":content}
        convert_list.append(template)
    return convert_list

def write_json(data_path, dataset,indent=0):
    with open(data_path,'w', encoding='UTF-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=indent)

if __name__ == "__main__":
    raw_folder_path = "src/data/eval_data/RoTBench/First_turn"
    # output_dir = "src/data/eval_result/Seal-Tools/" + model_name
    os.makedirs("src/data/eval_data/RoTBench/First_turn_RC", exist_ok=True)
    dataset_name_list = ["clean", "heavy", "medium", "slight", "union"]
    for dataset_name in dataset_name_list:
        raw_data_path = raw_folder_path + "/" +dataset_name + ".json"
        vllm_data = convert(raw_data_path)
        vllm_folder_path = "src/data/eval_data/RoTBench/First_turn_RC/" + dataset_name + ".json"
        write_json(vllm_folder_path, vllm_data, indent= 4)