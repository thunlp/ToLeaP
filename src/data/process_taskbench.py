from datasets import load_dataset
import json

splits = ['huggingface', 'dailylifeapis', 'multimedia']
dt = load_dataset("microsoft/Taskbench", "huggingface")['test']

def convert_data(data):
    # Convert both messages and tools format
    # Messages part
    human_message = {
        "from": "human", 
        "value": data["instruction"]
    }
    
    tool_nodes = json.loads(data['tool_nodes'])
    sampled_nodes = json.loads(data['sampled_nodes'])
    
    # Map each tool node to its input type from sampled nodes
    input_type_map = {node["task"]: node["input-type"][0] for node in sampled_nodes}
    
    function_call = {
        "from": "function_call",
        "value": json.dumps([{
            "name": node["task"],
            "arguments": {input_type_map[node["task"]]: node["arguments"][0]}
        } for node in tool_nodes])
    }
    
    messages = [human_message, function_call]
    
    # Tools part 
    tools = []
    for node in sampled_nodes:
        tool = {
            "name": node["task"],
            "description": f"Perform {node['task']} task based on inputs",
            "parameters": {
                "type": "object",
                "properties": {
                    node["input-type"][0]: {
                        "type": "string",
                        "description": f"The {node['input-type'][0]} to use in {node['task']}"
                    }
                },
                "required": [node["input-type"][0]]
            }
        }
        tools.append(tool)
        
    return messages, tools

def convert_dailylifeapis(data):
    # Convert both messages and tools format
    # Messages part
    human_message = {
        "from": "human", 
        "value": data["instruction"]
    }
    
    tool_nodes = json.loads(data['tool_nodes'])
    sampled_nodes = json.loads(data['sampled_nodes'])
    
    function_call = {
        "from": "function_call",
        "value": json.dumps([{
            "name": node["task"],
            "arguments": {
                arg["name"]: arg["value"] for arg in node["arguments"]
            }
        } for node in tool_nodes])
    }
    
    messages = [human_message, function_call]
    
    # Tools part 
    tools = []
    for node in sampled_nodes:
        tool = {
            "name": node["task"],
            "description": f"Perform {node['task']} task based on inputs",
            "parameters": {
                "type": "object",
                "properties": {
                    **{
                        arg["name"]: {
                            "type": arg["type"],
                            "description": arg["desc"]
                        } for arg in node["arguments"]
                    }
                },
                "required": [arg["name"] for arg in node["arguments"]]
            }
        }
        tools.append(tool)
        
    return messages, tools

if __name__ == "__main__":
    all_datapoints = []
    errs = 0
    for split in splits:
        dt = load_dataset("microsoft/Taskbench", split)['test']
        if split == 'dailylifeapis':
            convert_fn = convert_dailylifeapis
        else:
            convert_fn = convert_data

        for i in range(len(dt)):
            try:
                messages, tools = convert_fn(dt[i])
                all_datapoints.append({
                    "conversations": [
                        {
                            "from": "human",
                            "value": messages[0]["value"]
                        },
                        {
                            "from": "function_call",
                            "value": messages[1]["value"]
                        }
                    ],
                    "tools": json.dumps(tools)
                })
            except:
                errs += 1
        
    print(f"Errors: {errs}")
    print(f"Total: {len(all_datapoints)}")
    
    with open('taskbench_data.json', 'w') as f:
        json.dump(all_datapoints, f, indent=2)
