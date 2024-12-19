from datasets import load_dataset, Dataset
import json
import requests

splits = ['huggingface', 'dailylifeapis', 'multimedia']
tool_desc_url = "https://raw.githubusercontent.com/microsoft/JARVIS/refs/heads/main/taskbench/data_{}/tool_desc.json"

tool_descs = {}
for split in splits:
    url = tool_desc_url.format(split)
    try:
        response = requests.get(url)
        response.raise_for_status()
        tool_descs[split] = response.json()
        json.dump(tool_descs[split], open(f'sft_data/taskbench_tool_desc_{split}.json', 'w'), indent=2)
    except Exception as e:
        print(f"Error fetching tool description for {split}: {e}")


if __name__ == "__main__":
    for split in splits:
        all_datapoints = []
        errs = 0
        dt = load_dataset("microsoft/Taskbench", split)['test']
        # Create tool string
        tool_str = "# TASK LIST #:\n"
        for tool in tool_descs[split]['nodes']:
            tool_str += json.dumps(tool) + "\n"
        
        for item in dt:
            # Create prompt
            if split == 'dailylifeapis':
                prompt = """\n# GOAL #:\nBased on the above tools, I want you generate task steps and task nodes to solve the # USER REQUEST #. The format must in a strict JSON format, like: {"task_steps": [ "concrete steps, format as Step x: Call xxx tool with xxx: 'xxx' and xxx: 'xxx'" ], "task_nodes": [{"task": "task name must be from # TASK LIST #", "arguments": [ {"name": "parameter name", "value": "parameter value, either user-specified text or the specific name of the tool whose result is required by this node"} ]}], "task_links": [{"source": "task name i", "target": "task name j"}]}"""
                prompt += """\n\n# REQUIREMENTS #: \n1. the generated task steps and task nodes can resolve the given user request # USER REQUEST # perfectly. Task name must be selected from # TASK LIST #; \n2. the task steps should strictly aligned with the task nodes, and the number of task steps should be same with the task nodes; \n3. The task links (task_links) should reflect the temporal dependencies among task nodes, i.e. the order in which the APIs are invoked;"""
            else:
                prompt = """\n# GOAL #: Based on the above tools, I want you generate task steps and task nodes to solve the # USER REQUEST #. The format must in a strict JSON format, like: {"task_steps": [ step description of one or more steps ], "task_nodes": [{"task": "tool name must be from # TOOL LIST #", "arguments": [ a concise list of arguments for the tool. Either original text, or user-mentioned filename, or tag '<node-j>' (start from 0) to refer to the output of the j-th node. ]}]} """
                prompt += """\n\n# REQUIREMENTS #: \n1. the generated task steps and task nodes can resolve the given user request # USER REQUEST # perfectly. Task name must be selected from # TASK LIST #; \n2. the task steps should strictly aligned with the task nodes, and the number of task steps should be same with the task nodes; \n3. the dependencies among task steps should align with the argument dependencies of the task nodes; \n4. the tool arguments should be align with the input-type field of # TASK LIST #;"""
            prompt += f"""\n\n# USER REQUEST #: {item["instruction"]}\nnow please generate your result in a strict JSON format:\n# RESULT #:"""
            prompt = tool_str + prompt
            label = {"task_steps": item["tool_steps"], "task_nodes": item["tool_nodes"], "task_links": item["tool_links"]}
            all_datapoints.append({
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "gpt", "value": json.dumps(label)}
                ]
            })
        
        print(f"Split {split} - Errors: {errs}")
        print(f"Split {split} - Total: {len(all_datapoints)}")
        
        with open(f'sft_data/taskbench_data_{split}.json', 'w') as f:
            json.dump(all_datapoints, f, indent=2)
