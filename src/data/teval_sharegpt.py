import json
import ast
import re

def clean_api_spec(api_spec_str):
    try:
        api_dict = ast.literal_eval(api_spec_str)
        return api_dict
    except:
        print("Error parsing API spec:", api_spec_str)
        raise

def format_api_spec(api_spec_str):
    api_spec = clean_api_spec(api_spec_str)
    properties = {}
    required = []
    if 'required_parameters' in api_spec:
        for param in api_spec['required_parameters']:
            properties[param['name']] = {
                'type': param['type'].lower(),
                'description': param['description']
            }
            required.append(param['name'])
            
    if 'optional_parameters' in api_spec:
        for param in api_spec['optional_parameters']:
            properties[param['name']] = {
                'type': param['type'].lower(),
                'description': param['description']
            }
    
    tool = {
        'name': api_spec['name'],
        'description': api_spec['description'],
        'parameters': {
            'type': 'object',
            'properties': properties,
            'required': required
        }
    }
    return tool

def convert_to_sharegpt(query_data):
    try:
        format_type = query_data['meta_data']['response_format']
        system_content = query_data['origin_prompt'][0]['content']
        api_spec = system_content.split("You have access to the following API:\n")[1].split("\nPlease")[0]
        
        tool = format_api_spec(api_spec)
        tools = [tool]
        
        conversations = [
            {
                "from": "human",
                "value": query_data['origin_prompt'][1]['content']
            }
        ]
        
        # Add function call
        ground_truth = query_data['ground_truth']
        function_call = {
            "name": ground_truth['action'],
            "arguments": ground_truth['args']
        }
        
        # Add GPT response (same as function call)
        conversations.append({
            "from": "gpt",
            "value": json.dumps(function_call)
        })
        
        return {
            "conversations": conversations,
            "system": system_content,
            "tools": json.dumps(tools)
        }
        
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        raise

def process_Ins_json_file(input_file):
    with open(input_file) as f:
        data = json.load(f)
    
    results = []
    for key, query_data in data.items():
        format_type = query_data['meta_data']['response_format']
        if format_type == 'json':
            try:
                result = convert_to_sharegpt(query_data)
                results.append(result)
            except Exception as e:
                print(f"Error processing key {key}: {str(e)}")
                continue
    
    with open('sft_data/teval_sharegpt_format.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    



if __name__ == "__main__":
    process_Ins_json_file('data_instruct_v2.json')
