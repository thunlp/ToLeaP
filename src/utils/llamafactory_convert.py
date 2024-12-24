import json

#this code is use for converting corase data into dataset the fit for llamafactory sft
#input corase data output llf/corase data

def transform_json(input_file, output_file):
    # Read the JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process each item in the list
    for item in data:
        conversations = item['conversations']
        
        # Transform conversations
        for conv in conversations:
            if conv['from'] == 'function_call':
                conv['from'] = 'gpt'
            elif conv['from'] == 'observation':
                conv['from'] = 'human'
        
        # Process tools if they exist, otherwise set empty system
        if 'tools' in item and item['tools'] != '':
            tools_message = "available tools:\n" + item['tools']
            item['system'] += tools_message
            del item['tools']
        elif 'tools' in item:
            del item['tools']
        elif 'system' not in item:
            item['system'] = ""
        
    
    # Write the transformed JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


transform_json('interstellarninja-tool-calls-multiturn.json', 'llf/interstellarninja-tool-calls-multiturn.json')
transform_json('pandora-tool-calling-test.json', 'llf/pandora-tool-calling-test.json')
transform_json('pandora-tool-calling-train.json', 'llf/pandora-tool-calling-train.json')
transform_json('pandora-tool-calling-val.json', 'llf/pandora-tool-calling-val.json')
transform_json('Salesforce-xlam-function-calling-60k.json','llf/Salesforce-xlam-function-calling-60k.json')
transform_json('sharegpt_function-calling-chatml.json','llf/sharegpt_function-calling-chatml.json')
transform_json('sharegpt_glaive_toolcall_en.json','llf/sharegpt_glaive_toolcall_en.json')
transform_json('sharegpt_ToolACE.json','llf/sharegpt_ToolACE.json')
transform_json('sharegpt_ToolBench_toolllama_G123_dfs.json', 'llf/sharegpt_ToolBench_toolllama_G123_dfs.json')
transform_json('tool-calls-sampled-prompts_train-00000-of-00001.json', 'llf/tool-calls-sampled-prompts_train-00000-of-00001.json')
transform_json('tool-calls-singleturn_train-00000-of-00001.json','llf/tool-calls-singleturn_train-00000-of-00001.json')
transform_json('sharegpt_glaive-code-assistant-v3.json','llf/sharegpt_glaive-code-assistant-v3.json')


def transform_json_robot(input_file, output_file):
    # Read the JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process each item in the list
    for item in data:
        conversations = item['conversations']
        
        # Only keep human and function_call (transformed to gpt) messages
        item['conversations'] = [
            {'from': 'gpt', 'value': conv['value']} if conv['from'] == 'function_call'
            else conv
            for conv in conversations
            if conv['from'] in ['human', 'function_call']
        ]
        
        # Only process tools if they exist
        if 'tools' in item:
            tools_message = "available tools:\n" + item['tools']
            
            # Create system field at the same level as tools and conversations
            item['system'] = tools_message
            
            # Remove the original tools field
            del item['tools']


    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

transform_json_robot('roborovski-synthetic-tool-calls-v2-dpo-pairs.json', 'llf/roborovski-synthetic-tool-calls-v2-dpo-pairs.json')



def transform_json_bit(input_file, output_file):
    # Read the JSON file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process each item in the list
    for item in data:
        conversations = item['conversations']
        
        # Only keep human and function_call (transformed to gpt) messages
        item['conversations'] = [
            {'from': 'gpt', 'value': conv['value']} if conv['from'] == 'function_call'
            else conv
            for conv in conversations
            if conv['from'] in ['human', 'function_call']
        ]
        
        # Only process tools if they exist
        if 'tools' in item:
            tools_message = "available tools:\n" + item['tools']
            
            # Create system field at the same level as tools and conversations
            item['system'] = tools_message
            
            # Remove the original tools field
            del item['tools']


    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

transform_json_robot('BitAgent-tool-calling.json', 'llf/BitAgent-tool-calling.json')