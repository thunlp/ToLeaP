import json

def load_json(input_file): # 从指定的JSON文件加载数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Succeed loading {input_file}")
    return data

def save_json(data, output_file): # 将处理后的数据保存到新的JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"The output is saved to {output_file}")

def remove_id_add_system(data): # 移除id字段并添加system字段
    for item in data:
        if 'id' in item:
            del item['id']
        item['system'] = ""
    return data

def extract_system_from_conversations(data): # 迁移system字段
    for item in data:
        conversations = item.get('conversations', [])
        if conversations and conversations[0].get('from') == 'system':
            system_value = conversations[0].get('value', '')
            item['system'] = system_value
            # 删除第一个对话项
            del conversations[0]
    return data

def rearrange_conversations(data): # 满足奇偶要求
    for item in data:
        conversations = item.get('conversations', [])
        
        odd_conversations = [conv for conv in conversations if conv.get('from') in ['user', 'function']]
        even_conversations = [conv for conv in conversations if conv.get('from') in ['assistant']]
        
        rearranged = []
        max_length = max(len(odd_conversations), len(even_conversations))
        
        for i in range(max_length):
            if i < len(odd_conversations): # 奇数位
                rearranged.append(odd_conversations[i])
            else:
                rearranged.append({
                    "from": "user",
                    "value": ""
                })
            if i < len(even_conversations): # 偶数位
                rearranged.append(even_conversations[i])
            else:
                rearranged.append({
                    "from": "assistant",
                    "value": ""
                })
        item['conversations'] = rearranged
    return data

def modify_from_fields(data): # 修改键
    mapping = {
        'assistant': 'gpt',
        'function': 'observation',
        'user': 'human'
    }
    for item in data:
        conversations = item.get('conversations', [])
        for conv in conversations:
            from_value = conv.get('from')
            if from_value in mapping:
                conv['from'] = mapping[from_value]
    return data

if __name__ == '__main__':
    # 输入和输出文件的路径
    input_file = 'toolllama_G123_dfs_train.json'        
    output_file = 'toolllm_processed.json'  

    data = load_json(input_file)

    # 步骤1：移除id字段并添加system字段
    data = remove_id_add_system(data)
    print("Step 1 is finished!")

    # 步骤2：迁移system字段
    data = extract_system_from_conversations(data)
    print("Step 2 is finished!")

    # 步骤3：满足奇偶要求
    data = rearrange_conversations(data)
    print("Step 3 is finished!")

    # 步骤4：修改键
    data = modify_from_fields(data)
    print("Step 4 is finished!")
    
    save_json(data, output_file)
    print(f"Processing finished!")
