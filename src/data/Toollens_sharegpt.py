import os
import json
import glob

def transferSharegpt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f: 
            data=[]
            next(f)
            for line in f:
                # print(line)
                parts = line.strip().split('\t')
                for queryIndex in queries_data:
                    if int(queryIndex["_id"]) == int(parts[0]):
                        query = {
                            "from":"human",
                            "value": queryIndex["text"]
                        }
                        break
                answer = {
                    "from":"gpt",
                    "value": corpus_data[int(parts[1])]["text"] 
                }
                oneData = {"conversations":[query, answer]}
                data.append(oneData)
                # print("Finished building:", parts[0])
            return data



# Check datasets exist
if os.path.exists("ToolLens") and os.path.exists("ToolBenchG2") and os.path.exists("ToolBenchG3") :
    print("Data file directory exist and start to process:")
else:
    print("Data file directory do not exist, please check and retry")
    exit

# 获取当前文件夹和当前文件夹下所有文件夹
current_dir = os.getcwd()
new_dir = "../../sft_data/ToolLensData"
folders = [f for f in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, f))]

#读取Tsv文件
for current_dir in folders:
    tsv_files =  glob.glob(os.path.join(current_dir +"//qrels//", "*.tsv"))
    corpus_path = current_dir + "//corpus.jsonl"

    # 读取 json 文件，中间包括API data and Corpus
    with open(corpus_path, 'r', encoding='utf-8') as f: 
        corpus_data = []
        for line in f:
            # 解析每一行的 JSON 数据
            json_object = json.loads(line.strip())
            corpus_data.append(json_object)
    queries_path = current_dir + "//queries.jsonl"
    with open(queries_path, 'r', encoding='utf-8') as f:
        queries_data = []
        for line in f:
            # 解析每一行的 JSON 数据
            json_object = json.loads(line.strip())
            queries_data.append(json_object) 
    
    #开始转换
    for file_path in tsv_files:
        try:
            file_name, file_extension = os.path.splitext(os.path.basename(file_path))
            file_name_N = new_dir+ "//new_" + str(current_dir) + file_name + ".json"
            print("Data will be saved as:",file_name_N)
            dataNew = transferSharegpt(file_path)
            with open(file_name_N, 'w', encoding='utf-8') as f:
                        json.dump(dataNew, f, ensure_ascii=False, indent=4)
                        print(file_name_N, ":Finished")
                   
        except:
                print(file_name_N, ":Failed")                
        


