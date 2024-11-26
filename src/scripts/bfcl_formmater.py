import json
import re


def parse_action_input(text):
    actions = []
    pattern = r'Action: (.*?)\nAction Input: ({.*?})'
    matches = re.findall(pattern, text)
    
    for func_name, params in matches:
        try:
            params_dict = json.loads(params)
            arguments = {k: [v] if not isinstance(v, list) else v 
                        for k, v in params_dict.items()}
            actions.append({
                "name": func_name,
                "arguments": arguments
            })
        except json.JSONDecodeError:
            continue
    return actions

def parse_label(label_str):
    try:
        label_data = json.loads(label_str)
        print(label_data)
        if isinstance(label_data, list):
            return label_data
        return [label_data]
    except json.JSONDecodeError:
        return None

def transform_line(line, id_prefix="simple"):
    try:
        data = json.loads(line.strip())
        actions = []

        if "predict" in data:
            try:
                predict_data = json.loads(data["predict"])
                if isinstance(predict_data, list):
                    actions = predict_data
                else:
                    actions = [predict_data]
            except json.JSONDecodeError:
                actions = parse_action_input(data["predict"])
                
            if not actions and "label" in data:
                label_actions = parse_label(data["label"])
                if label_actions:
                    actions = label_actions

        if not actions:
            return None
            
        id_value = data.get("id", f"{id_prefix}_0")
        
        def format_args(action):
            args = []
            for key, values in action["arguments"].items():
                if isinstance(values, list) and values:
                    value = values[0]
                    if isinstance(value, str):
                        if not value:
                            continue
                        args.append(f"{key}='{value}'")
                    else:
                        args.append(f"{key}={value}")
            return f"{action['name']}({', '.join(args)})"
        
        result = "[" + ", ".join(format_args(action) for action in actions) + "]"
        if len(actions) == 1:
            result = result[1:-1]


        return {
            "id": id_value,
            "result": result
        }
    except Exception as e:
        return {
            'id': id_value,
            'result': data["predict"]
        }

def transform_jsonl(input_file, output_file, id_prefix="simple"):
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for i, line in enumerate(f_in):
            if not line.strip():
                continue
            transformed = transform_line(line.strip(), id_prefix)
            if transformed:
                if "id" not in json.loads(line):
                    transformed["id"] = f"{id_prefix}_{i}"
                f_out.write(json.dumps(transformed, ensure_ascii=False) + '\n')



def transform_line_irr(line, id_prefix="simple"):
    try:
        data = json.loads(line.strip())
        actions = []

        if "predict" in data:
            # Try to directly parse the predict field if it's already in the desired format
            try:
                predict_data = json.loads(data["predict"])
                if isinstance(predict_data, list):
                    actions = predict_data
                else:
                    actions = [predict_data]
            except json.JSONDecodeError:
                # If direct parsing fails, try to parse as action input format
                actions = parse_action_input(data["predict"])

            # If no actions found and label exists, use label as fallback
            if not actions and "label" in data:
                label_actions = parse_label(data["label"])
                if label_actions:
                    actions = label_actions

        id_value = data.get("id", f"{id_prefix}_0")

        def format_args(action):
            args = []
            for key, values in action["arguments"].items():
                if isinstance(values, list) and values:
                    value = values[0]
                    if isinstance(value, str):
                        if not value:
                            continue
                        args.append(f"{key}='{value}'")
                    else:
                        args.append(f"{key}={value}")
            return f"{action['name']}({', '.join(args)})"

        result = "[]"
        if actions:
            result = "[" + ", ".join(format_args(action) for action in actions) + "]"
            if len(actions) == 1:
                result = result[1:-1]

        return {
            "id": id_value,
            "result": result
        }
    except Exception as e:
        print(f"Error processing line: {e}")
        return None

    
    
def transform_jsonl_irr(input_file, output_file, id_prefix="simple"):
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for i, line in enumerate(f_in):
            if not line.strip():
                continue
            transformed = transform_line_irr(line.strip(), id_prefix)
            if transformed:
                if "id" not in json.loads(line):
                    transformed["id"] = f"{id_prefix}_{i}"
                f_out.write(json.dumps(transformed, ensure_ascii=False) + '\n')
                
                
                
                
# transform_jsonl('bfcl_simple.jsonl', 'BFCL_v3_simple.json', id_prefix='simple') 
# transform_jsonl('bfcl_parallel.jsonl', 'BFCL_v3_parallel.json', id_prefix='parallel') 
transform_jsonl('bfcl_multiple.jsonl', 'BFCL_v3_multiple.json', id_prefix='multiple') 
# transform_jsonl('bfcl_parallel_multiple.jsonl', 'BFCL_v3_parallel_multiple.json', id_prefix='parallel_multiple') 
# transform_jsonl_irr('bfcl_irrelevance.jsonl', 'BFCL_v3_irrelevance.json', id_prefix='irrelevance') 