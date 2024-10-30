import json
import os

# 指定输入和输出文件路径
input_file = 'MetaTool_data/multi_tool_query_golden.json'   # 替换为您的输入文件路径
output_file = 'MetaTool_data/MetaTool_processed.json' # 替换为您希望保存的输出文件路径

def load_json(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Can't find: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved to: {file_path}")

def transform_data(data):

    transformed = []
    for index, item in enumerate(data):
        query = item.get("query", "")
        tools = item.get("tool", [])
        
        conversation = [
            {
                "from": "human",
                "value": query
            },
            {
                "from": "gpt",
                "value": json.dumps(tools, ensure_ascii=False)  # 工具列表转换为字符串
            },
        ]
        
        transformed_item = {
            "conversations": conversation,
        }
        
        transformed.append(transformed_item)
        
    return transformed

def main():
    try:
        print(f"Loading: {input_file}")
        input_data = load_json(input_file)
        
        if not isinstance(input_data, list):
            raise ValueError("The top-level structure of the input JSON file must be a list.")
        
        print("Start to transform...")
        output_data = transform_data(input_data)
        
        save_json(output_data, output_file)
        
        print("Success!")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
