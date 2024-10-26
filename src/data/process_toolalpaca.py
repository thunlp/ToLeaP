import json
import re

# Helper function to parse parameter string and convert to JSON schema format
def parse_parameters(parameters_str):
    if not parameters_str:
        return {"type": "object", "properties": {}, "required": []}

    properties = {}
    required = []
    param_dict = json.loads(parameters_str)

    for key, description in param_dict.items():
        desc_split = description.split(". ")
        required_field = desc_split[0].strip() == "Required"
        type_str = desc_split[1].split(" ")[0].lower().strip()

        properties[key] = {
            "type": type_str,
            "description": ". ".join(desc_split[1:])
        }
        
        if required_field:
            required.append(key)

    return {"type": "object", "properties": properties, "required": required}

# Main function to convert descriptions
def convert_descriptions(function_descriptions):
    result = []

    for func_name, description in function_descriptions.items():
        # Extract function name and parameters
        desc_lines = description.split("\n")
        function_desc = desc_lines[0].strip()
        parameters_line = re.search(r'Parameters: (.*)', description)
        parameters_str = parameters_line.group(1).strip() if parameters_line else "{}"

        # Append converted function data to result list
        result.append({
            "name": func_name,
            "description": function_desc,
            "parameters": parse_parameters(parameters_str)
        })

    return result

def process_train(filename, output_filename, is_eval):
    raw_train_path = filename
    train_json = json.load(open(raw_train_path, 'r'))
    train_factory = []
    discarded = 0
    # Convert to sharegpt
    for topic in train_json:
        questions = topic["Instructions"] # First question
        golden_answers = topic["Golden_Answers"] if is_eval else topic["Instances"] # Golden answers
        # Ignore topic if cannot be parsed
        try:
            converted_funcs = convert_descriptions(topic["Function_Description"]) # This will be "tools"
        except:
            discarded += len(questions)
            continue

        for question, golden_answer in zip(questions, golden_answers):
            if is_eval:
                if len(golden_answer) > 1: # Only one action permitted
                    discarded += 1
                    continue
                call_name = golden_answer[0]["Action"]
                call_input = golden_answer[0]["Action_Input"]
                try:
                    train_factory.append({
                        "conversation": [
                            {"from": "human", "value": question},
                            {"from": "function_call", "value": json.dumps({"name": call_name, "arguments": json.loads(call_input)})}
                        ]
                    }) # Eval data only have question and golden function call
                except:
                    discarded += 1
                    continue
            else:
                try:
                    steps = golden_answer["intermediate_steps"][0]
                except:
                    continue
                try:
                    observation = steps[1].split("Response: ")[1]
                    observation = json.dumps(json.loads(observation)['response'])
                except:
                    observation = "{}"
                call_name = steps[0][0]
                call_input = steps[0][1]
                final_output = golden_answer["output"]
                try:
                    train_factory.append({
                        "conversation": [
                            {"from": "human", "value": question},
                            {"from": "function_call", "value": json.dumps({"name": call_name, "arguments": json.loads(call_input)})},
                            {"from": "observation", "value": observation},
                            {"from": "gpt", "value": final_output}
                        ]
                    }) # Train data have question, function call, observation, and final output
                except:
                    discarded += 1
                    continue

    json.dump(train_factory, open(output_filename, 'w'), indent=4)
    print(f"Discarded {discarded} examples")
    print(f"Total {len(train_factory)} examples")
    return train_factory

if __name__ == "__main__":
    process_train('ToolAlpaca/data/eval_simulated.json', 'toolalpaca_eval_simulated.json', True)
    process_train('ToolAlpaca/data/eval_real.json', 'toolalpaca_eval_real.json', True)
    process_train('ToolAlpaca/data/train_data.json', 'toolalpaca_train.json', False)