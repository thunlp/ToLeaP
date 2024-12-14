import argparse, json
def converter (conversations, result, history_sample=0):
    messages = []
    history_sample = history_sample if history_sample <= len(
    conversations) else len(conversations)
    messages += conversations[:history_sample + 1]
    try:
        messages.append({"from": "assistant", "value": result})
    except Exception as e:
        messages.append(
            {"from": "assistant", "value": "Thought: I cannot solve this task due to trying too many times.\nAction: finish\nAction Input: {\"answer\": \"I cannot handle the task.\"}"})
    return messages.copy()

def write_in(origin_data, raw_pred_text, output_file_path):
    conversations = origin_data["conversations"]
    result = converter(conversations, raw_pred_text, history_sample=0)
    with open(output_file_path, "a", encoding='utf8') as writer:
        writer.write(json.dumps(
            {"id": id, "conversations": result, "path": None, "scenario": origin_data["scenario"]}, ensure_ascii=False) + '\n')

def read_json(data_path):
    dataset=[]
    with open(data_path,'r', encoding='UTF-8') as f:
        dataset = json.load(f)
    return dataset

def read_jsonl(data_path):
    dataset=[]
    with open(data_path,'r', encoding='UTF-8') as f:
        for line in f:
            dataset.append(json.loads(line))
    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_file", type=str, default="src/data/eval_data/RoTBench/First_turn/clean.json")
    parser.add_argument("--answer_file", type=str, default="", help= "输入raw prediction的文件")
    parser.add_argument("--output_file", type=str, default="", help= "输出的目标文件")
    parser.add_argument("--version",type=int, default = 0)
    args = parser.parse_args()
    origin_dataset = read_jsonl(args.test_file)
    raw_pred_dataset = read_json(args.answer_file)
    for origin_data, raw_pred_data in zip(origin_dataset, raw_pred_dataset):
        raw_pred_text = "" ###这里需要raw predict的text part
        write_in(origin_data, raw_pred_text, args.output_file)