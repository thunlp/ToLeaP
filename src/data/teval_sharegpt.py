import json
def convert_to_sft_format_instruct(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    result = []
    for key, entry in data.items():
        conversations = []
        system_prompt = ""
        tool_description = ""

        origin_prompt = entry.get("origin_prompt", [])
        for prompt in origin_prompt:
            if prompt["role"] == "system":
                system_prompt = ""
            elif prompt["role"] == "user":
                conversations.append({"from": "human", "value": prompt["content"]})
        
        ground_truth = entry.get("ground_truth", {})
        conversations.append({"from": "function_call", "value": json.dumps(ground_truth)})
        conversations.append({"from": "observation", "value": ""})
        conversations.append({"from": "gpt", "value": json.dumps(ground_truth)})
        
        tool_description = json.dumps(entry.get("origin_prompt", [])[0]["content"].split('API:')[1].split('Please directly generate')[0].strip())
        
        converted_entry = {
            "conversations": conversations,
            "system": system_prompt,
            "tools": tool_description
        }
        result.append(converted_entry)
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

def convert_to_sft_format_plan(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    result = []
    for key, entry in data.items():
        conversations = []
        system_prompt = ""
        tool_description = ""

        origin_prompt = entry.get("origin_prompt", [])
        for prompt in origin_prompt:
            if prompt["role"] == "system":
                system_prompt = prompt["content"]
            elif prompt["role"] == "user":
                conversations.append({"from": "human", "value": prompt["content"]})
        
        ground_truth = entry.get("ground_truth", [])
        for action in ground_truth:
            function_call = {
                "from": "function_call",
                "value": json.dumps({"name": action["name"], "args": action["args"]})
            }
            conversations.append(function_call)
            conversations.append({"from": "observation", "value": ""})
            gpt_response = {
                "from": "gpt",
                "value": json.dumps({"name": action["name"], "args": action["args"]})
            }
            conversations.append(gpt_response)
        
        tool_description = json.dumps(entry.get("meta", {}).get("API_list", []))
        
        converted_entry = {
            "conversations": conversations,
            "system": system_prompt,
            "tools": tool_description
        }
        result.append(converted_entry)
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

def convert_to_sft_format_retrieve(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    result = []
    for key, entry in data.items():
        conversations = []
        system_prompt = ""
        tool_description = ""

        # Extract system prompt and prompts from origin_prompt
        origin_prompt = entry.get("origin_prompt", [])
        for prompt in origin_prompt:
            if prompt["role"] == "system":
                system_prompt = prompt["content"]
            else:
                conversations.append({"from": prompt["role"], "value": prompt["content"]})
        
        # Extract ground truth for function call
        ground_truth = entry.get("ground_truth", {})
        function_call = {
            "from": "function_call",
            "value": json.dumps({"name": ground_truth["name"], "args": ground_truth["args"]})
        }
        conversations.append(function_call)
        conversations.append({"from": "observation", "value": ""})
        gpt_response = {
            "from": "gpt",
            "value": json.dumps({"name": ground_truth["name"], "args": ground_truth["args"]})
        }
        conversations.append(gpt_response)
        
        # Extract tool descriptions from the system prompt in origin_prompt
        tool_description = ""
        if origin_prompt and origin_prompt[0]["role"] == "system":
            tool_description = origin_prompt[0]["content"].split('tools:\n')[1].strip()
        
        # Prepare the final converted entry
        converted_entry = {
            "conversations": conversations,
            "system": system_prompt,
            "tools": tool_description
        }
        result.append(converted_entry)
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)


def convert_to_sft_format_reason(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    result = []
    for key, entry in data.items():
        conversations = []
        system_prompt = ""
        tool_description = ""

        # Extract system prompt and prompts from origin_prompt
        origin_prompt = entry.get("origin_prompt", [])
        for prompt in origin_prompt:
            if prompt["role"] == "system":
                system_prompt = prompt["content"]
            else:
                conversations.append({"from": prompt["role"], "value": prompt["content"]})
        
        # Extract ground truth for function call and generate conversation steps
        ground_truth = entry.get("ground_truth", {})
        thought = ground_truth.get("thought", "")
        if thought:
            conversations.append({"from": "assistant", "value": thought})
        
        function_call = {
            "from": "function_call",
            "value": json.dumps({"name": ground_truth["name"], "args": ground_truth["args"]})
        }
        conversations.append(function_call)
        conversations.append({"from": "observation", "value": ""})
        gpt_response = {
            "from": "gpt",
            "value": json.dumps({"name": ground_truth["name"], "args": ground_truth["args"]})
        }
        conversations.append(gpt_response)
        
        # Extract tool descriptions from the system prompt in origin_prompt
        tool_description = ""
        if origin_prompt and origin_prompt[0]["role"] == "system":
            tool_description = origin_prompt[0]["content"].split('tools:\n')[1].strip()
        
        # Prepare the final converted entry
        converted_entry = {
            "conversations": conversations,
            "system": system_prompt,
            "tools": tool_description
        }
        result.append(converted_entry)
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    convert_to_sft_format_reason('teval/reason_str_v2.json', 'sft_teval_reason.json')
    convert_to_sft_format_retrieve('teval/retrieve_str_v2.json', 'stf_teval_retrieve.json')
    convert_to_sft_format_plan('teval/plan_json_v2.json', 'stf_teval_plan.json')
    convert_to_sft_format_instruct('teval/instruct_v2.json', 'stf_teval_ins.json')

