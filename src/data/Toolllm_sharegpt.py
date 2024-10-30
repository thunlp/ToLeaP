import json

def load_json(input_file): # Load data from the specified JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Succeed loading {input_file}")
    return data

def save_json(data, output_file): # Save the processed data to a new JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"The output is saved to {output_file}")

def remove_id_add_system(data): # Remove the ID field and add the system field
    for item in data:
        if 'id' in item:
            del item['id']
        item['system'] = ""
    return data

def extract_system_from_conversations(data): # Migration system field
    for item in data:
        conversations = item.get('conversations', [])
        if conversations and conversations[0].get('from') == 'system':
            system_value = conversations[0].get('value', '')
            item['system'] = system_value
            # Delete the first dialogue item
            del conversations[0]
    return data

def rearrange_conversations(data): # Meet parity requirements
    for item in data:
        conversations = item.get('conversations', [])
        
        odd_conversations = [conv for conv in conversations if conv.get('from') in ['user', 'function']]
        even_conversations = [conv for conv in conversations if conv.get('from') in ['assistant']]
        
        rearranged = []
        max_length = max(len(odd_conversations), len(even_conversations))
        
        for i in range(max_length):
            if i < len(odd_conversations): # Odd digits
                rearranged.append(odd_conversations[i])
            else:
                rearranged.append({
                    "from": "user",
                    "value": ""
                })
            if i < len(even_conversations): # Even bits
                rearranged.append(even_conversations[i])
            else:
                rearranged.append({
                    "from": "assistant",
                    "value": ""
                })
        item['conversations'] = rearranged
    return data

def modify_from_fields(data): # Modify key
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
    # Path of input and output files
    input_file = 'toolllm_data/toolllama_G123_dfs_train.json'     
    output_dir = 'sft_data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, 'toolllm_processed.json')

    data = load_json(input_file)

    # Step 1: Remove the ID field and add the system field
    data = remove_id_add_system(data)
    print("Step 1 is finished!")

    # Step 2: Migration of System Fields
    data = extract_system_from_conversations(data)
    print("Step 2 is finished!")

    # Step 3: Meet the parity requirement
    data = rearrange_conversations(data)
    print("Step 3 is finished!")

    # Step 4: Modify key
    data = modify_from_fields(data)
    print("Step 4 is finished!")
    
    save_json(data, output_file)
    print(f"Processing finished!")
