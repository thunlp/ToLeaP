import json
import re
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
        
        # Extract the ground_truth in the desired format
        ground_truth = entry.get("ground_truth", {})
        function_call_value = {
            "name": ground_truth.get("action", ""),
            "arguments": ground_truth.get("args", {})
        }
        conversations.append({"from": "function_call", "value": json.dumps(function_call_value)})
        conversations.append({"from": "observation", "value": ""})
        conversations.append({"from": "gpt", "value": json.dumps(function_call_value)})
        
        # Extract and format tool description as a list in string format
        tool_description_content = entry.get("origin_prompt", [])[0]["content"].split('API:')[1].split('Please directly generate')[0].strip()
        tool_description = f"[{tool_description_content}]"
        
        converted_entry = {
            "conversations": conversations,
            "system": system_prompt,
            "tools": tool_description
        }
        result.append(converted_entry)
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def convert_to_sft_format_plan(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    result = []
    for key, entry in data.items():
        conversations = []
        
        # Process user prompts
        origin_prompt = entry.get("origin_prompt", [])
        system_prompt_content = ""
        for prompt in origin_prompt:
            if prompt["role"] == "system":
                system_prompt_content = prompt["content"]
            elif prompt["role"] == "user":
                conversations.append({"from": "human", "value": prompt["content"]})
        
        # Process ground truth actions
        ground_truth = entry.get("ground_truth", [])
        for action in ground_truth:
            function_call = {
                "from": "function_call",
                "value": json.dumps({"name": action["name"], "arguments": action["args"]})
            }
            conversations.append(function_call)
            conversations.append({"from": "observation", "value": ""})
            gpt_response = {
                "from": "gpt",
                "value": json.dumps({"name": action["name"], "arguments": action["args"]})
            }
            conversations.append(gpt_response)
        
        # Extract the JSON structure for API list from system content
        # Using regular expression to isolate the API list
        match = re.search(r"\[\[.*?\]\]", system_prompt_content, re.DOTALL)
        tool_description = match.group(0) if match else "[]"
        
        # Reformat tool description as a JSON array string with each item formatted as JSON
        tool_description = tool_description.replace("'", "\"")  # Ensures JSON compatibility

        converted_entry = {
            "conversations": conversations,
            "system": "",  # Leave system empty
            "tools": tool_description
        }
        result.append(converted_entry)
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def convert_to_sft_format_retrieve(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    result = []
    for key, entry in data.items():
        conversations = []
        
        # Extract prompts from origin_prompt
        origin_prompt = entry.get("origin_prompt", [])
        system_prompt_content = ""
        for prompt in origin_prompt:
            if prompt["role"] == "system":
                system_prompt_content = prompt["content"]
            else:
                conversations.append({"from": prompt["role"], "value": prompt["content"]})
        
        # Extract ground truth for function call
        ground_truth = entry.get("ground_truth", {})
        function_call = {
            "from": "function_call",
            "value": json.dumps({"name": ground_truth.get("name"), "arguments": ground_truth.get("args")})
        }
        conversations.append(function_call)
        conversations.append({"from": "observation", "value": ""})
        gpt_response = {
            "from": "gpt",
            "value": json.dumps({"name": ground_truth.get("name"), "arguments": ground_truth.get("args")})
        }
        conversations.append(gpt_response)
        
        # Extract the tool descriptions from system content
        # Using regular expression to isolate the tool descriptions as JSON-compatible string
        match = re.search(r"\[(\{.*?\})\]", system_prompt_content, re.DOTALL)
        tool_description = match.group(0) if match else "[]"
        
        # Reformat to ensure JSON compatibility
        tool_description = tool_description.replace("'", "\"")  # Converts single to double quotes for JSON compatibility

        # Prepare the final converted entry
        converted_entry = {
            "conversations": conversations,
            "system": "",  # Leave system empty
            "tools": tool_description
        }
        result.append(converted_entry)
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


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
    # convert_to_sft_format_reason('teval/reason_str_v2.json', 'sft_data/sft_teval_reason.json')
    # convert_to_sft_format_retrieve('teval/retrieve_str_v2.json', 'sft_data/stf_teval_retrieve.json')
    # convert_to_sft_format_plan('teval/plan_json_v2.json', 'sft_data/stf_teval_plan.json')
    convert_to_sft_format_instruct('teval/instruct_v2.json', 'sft_data/stf_teval_ins.json')

