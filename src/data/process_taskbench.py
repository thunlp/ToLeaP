from datasets import load_dataset, Dataset
import json

splits = ['huggingface', 'dailylifeapis', 'multimedia']
dt = load_dataset("microsoft/Taskbench", "huggingface")['test']

SYSTEM_PROMPT = """I want you to generate a series of actions to solve the user request. Each action uses a tool with inputs of strict formats. An example format of using one tool is given below, but you can use multiple tools if needed and write them in order. When an input argument is the output of a previous task, you should use the tag '<output of $TOOL_NAME>' to refer to the output of a previous task.

"""

def filter_huggingface(dt):
    # Function to check if any argument value contains a tag
    def has_tag_in_args(tool_step):
        if "arguments" not in tool_step:
            return "err"
        for arg in tool_step["arguments"]:
            if isinstance(arg, str) and arg.startswith("<") and arg.endswith(">"):
                if not arg.startswith("<node"):
                    return "not node"
                else:
                    return "node"
        return False

    # Find examples where any tool step has tag in arguments
    tagged_examples = []
    for i, example in enumerate(dt["tool_nodes"]):
        example = json.loads(example)
        if not isinstance(example, list):
            continue
        all_res = []
        err = False
        for step in example:
            res = has_tag_in_args(step)
            if res == "err":
                err = True
                break
            elif res:
                all_res.append(res)
        if not err:
            for res in all_res:
                tagged_examples.append((i, res))

    # Process tagged examples
    node_err_indices = []
    non_node_err_indices = []
    updated_items = []
    for i, res in tagged_examples:
        tool_nodes = dt[i]["tool_nodes"]
        if res == "node":
            # Parse tool nodes
            tool_nodes = json.loads(tool_nodes)
            
            # Find and replace <node-n> tags with corresponding task outputs
            for step in tool_nodes:
                for j, arg in enumerate(step["arguments"]):
                    if isinstance(arg, str) and arg.startswith("<node-"):
                        # Extract node number from tag
                        try:
                            node_num = int(arg[6:-1])  # Remove "<node-" prefix and ">" suffix
                        except:
                            node_err_indices.append(i)
                            continue
                        # Get corresponding task
                        try:
                            source_task = tool_nodes[node_num]["task"]
                        except:
                            node_err_indices.append(i)
                            continue
                        # Replace tag with task output reference
                        step["arguments"][j] = f"<output of {source_task}>"
                        
            # Update tool_nodes in dataset
            updated_items.append((i, json.dumps(tool_nodes)))
        else:
            # Parse tool nodes
            tool_nodes = json.loads(tool_nodes)
            
            # Find and replace <node-n> tags with corresponding task outputs
            for step in tool_nodes:
                for j, arg in enumerate(step["arguments"]):
                    if isinstance(arg, str) and arg.startswith("<") and arg.endswith(">") and not arg.startswith("<node"):
                        # Replace underscores with spaces
                        modified_arg = arg.replace("_", " ")
                        
                        # Check if starts with 'output of' (case insensitive)
                        if not modified_arg[1:].lower().startswith('output of'):
                            non_node_err_indices.append(i)
                            continue
                            
                        # Extract task name after "output of" and before ">"
                        task_name = modified_arg[1:-1].replace('output of ', '').strip()
                        
                        # Check task exists in sampled nodes
                        sampled_nodes = json.loads(dt[i]['sampled_nodes'])
                        task_found = False
                        for node in sampled_nodes:
                            if node['task'] == task_name:
                                task_found = True
                                break
                                
                        if not task_found:
                            non_node_err_indices.append(i)
                        else:
                            # Update argument if no errors found
                            step["arguments"][j] = modified_arg
            
            # Update tool_nodes in dataset
            updated_items.append((i, json.dumps(tool_nodes)))

    # Create new dataset with updated tool_nodes
    new_data = dt.to_dict()
    
    # Update tool_nodes at specified indices
    for idx, new_tool_nodes in updated_items:
        new_data['tool_nodes'][idx] = new_tool_nodes

    # Discard error indices
    for key in new_data:
        combined_err_indices = set(node_err_indices + non_node_err_indices)
        new_data[key] = [item for i, item in enumerate(new_data[key]) if i not in combined_err_indices]

    # Create new Dataset with updated data
    dt_updated = Dataset.from_dict(new_data)
    
    return dt_updated

def filter_dailylifeapis(dt):
    def has_tag_in_args(tool_step):
        for arg in tool_step["arguments"]:
            if "from" in arg:
                return True
        return False
    
    tagged_examples = []
    for i, example in enumerate(dt["tool_nodes"]):
        example = json.loads(example)
        for step in example:
            if has_tag_in_args(step):
                tagged_examples.append(i)
                break

    # Get all indices and remove the tagged ones
    all_indices = list(range(len(dt)))
    filtered_indices = [i for i in all_indices if i not in tagged_examples]
    return dt.select(filtered_indices)

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
            dt = filter_dailylifeapis(dt)
            convert_fn = convert_dailylifeapis
        else:
            dt = filter_huggingface(dt) # filter non node errors
            dt = filter_huggingface(dt) # filter node errors
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
                    "system": SYSTEM_PROMPT,
                    "tools": json.dumps(tools)
                })
            except:
                errs += 1
        
    print(f"Errors: {errs}")
    print(f"Total: {len(all_datapoints)}")
    
    with open('sft_data/taskbench_data.json', 'w') as f:
        json.dump(all_datapoints, f, indent=2)
