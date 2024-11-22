import json
import sys
import os

def sft_simple(input_file, output_file, ground_truth_file):
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]
    
    with open(ground_truth_file, 'r') as f:
        ground_truth_lines = f.readlines()
        ground_truth_data = {json.loads(line)["id"]: json.loads(line) for line in ground_truth_lines}
    
    result = []
    for entry in data:
        conversations = []
        system_prompt = "You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose. \n If none of the function can be used, point it out. If the given question lacks the parameters required by the function, also point it out.\n You should only return the function calls in your response.\n If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)] \n You SHOULD NOT include any other text in the response. \n At each turn, your should try your best to complete the tasks requested by the user within the current turn. Continue to output functions to call until you have fulfilled the user's request to the best of your ability. Once you have no more functions to call, the system will consider the current turn complete and proceed to the next turn or task."
        tool_description = json.dumps(entry.get("function", []))
        
        # Get the original question
        original_question = ""
        for qa_pair in entry.get("question", []):
            for q in qa_pair:
                if q["role"] == "user":
                    original_question = q["content"]
                    break

        # Get ground truth for the current entry
        ground_truth_entry = ground_truth_data.get(entry["id"], {}).get("ground_truth", [])
        
        # For each function call, create a complete conversation round
        for i, gt in enumerate(ground_truth_entry):
            # Add human message for each round
            conversations.append({
                "from": "human",
                "value": original_question
            })
            
            # Create function call value
            function_name = list(gt.keys())[0]
            function_arguments = gt[function_name]
            function_call_value = {
                "name": function_name,
                "arguments": function_arguments
            }
            
            # Add function call sequence
            conversations.extend([
                {
                    "from": "function_call",
                    "value": json.dumps([function_call_value])
                },
                {
                    "from": "observation",
                    "value": ""
                },
                {
                    "from": "gpt",
                    "value": json.dumps([function_call_value])
                }
            ])
        
        converted_entry = {
            "conversations": conversations,
            "system": system_prompt,
            "tools": tool_description
        }
        result.append(converted_entry)
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

def sft_multi(input_file, output_file, ground_truth_file):
    class_to_file_mapping = {
        "TicketAPI": "ticket_api.json",
        "GorillaFileSystem": "gorilla_file_system.json",
        "VehicleControlAPI": "vehicle_control.json",
        "MATHAPI": "math_api.json",
        "MessageAPI": "message_api.json",
        "TradingBot": "trading_bot.json",
        "TwitterAPI": "posting_api.json",
        "TravelAPI": "travel_booking.json"
    }

    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]
    
    with open(ground_truth_file, 'r') as f:
        ground_truth_lines = f.readlines()
        ground_truth_data = {json.loads(line)["id"]: json.loads(line) for line in ground_truth_lines}
    
    result = []
    for entry in data:
        conversations = []
        system_prompt = ""
        tool_description = ""

        question = entry.get("question", [])
        ground_truth_list = ground_truth_data.get(entry["id"], {}).get("ground_truth", [])

        # Process each turn
        for i, qa_pair in enumerate(question):
            # Add human message
            for q in qa_pair:
                if q["role"] == "user":
                    conversations.append({"from": "human", "value": q["content"]})
            
            # Add corresponding ground truth if available
            if i < len(ground_truth_list):
                ground_truth = ground_truth_list[i]
                conversations.extend([
                    {"from": "function_call", "value": json.dumps(ground_truth)},
                    {"from": "observation", "value": ""},
                    {"from": "gpt", "value": json.dumps(ground_truth)}
                ])
        
        # Add tool description from involved_classes
        involved_classes = entry.get("involved_classes", [])
        tool_descriptions = []
        for involved_class in involved_classes:
            file_name = class_to_file_mapping.get(involved_class)
            if file_name:
                tool_file_path = os.path.join('bfcl', 'multi_turn_func_doc', file_name)
                if os.path.exists(tool_file_path):
                    with open(tool_file_path, 'r') as tool_file:
                        tool_descriptions.extend([json.loads(line) for line in tool_file])
        
        tool_description = json.dumps(tool_descriptions)
        
        converted_entry = {
            "conversations": conversations,
            "system": system_prompt,
            "tools": tool_description
        }
        result.append(converted_entry)
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    input_prefix = sys.argv[1]
    if input_prefix == 'all':
        sft_simple(f'bfcl/BFCL_v3_simple.json', f'sft_data/sft_bfclsimple.json', f'bfcl/possible_answer/BFCL_v3_simple.json')
        sft_simple(f'bfcl/BFCL_v3_parallel.json', f'sft_data/sft_bfclparallel.json', f'bfcl/possible_answer/BFCL_v3_parallel.json')
        sft_simple(f'bfcl/BFCL_v3_multiple.json', f'sft_data/sft_bfclmultiple.json', f'bfcl/possible_answer/BFCL_v3_multiple.json')
        sft_simple(f'bfcl/BFCL_v3_parallel_multiple.json', f'sft_data/stf_bfclparallel_multiple.json', f'bfcl/possible_answer/BFCL_v3_parallel_multiple.json')
        # sft_simple(f'bfcl/BFCL_v3_live_simple.json', f'sft_data/sft_bfcllive_simple.json', f'bfcl/possible_answer/BFCL_v3_live_simple.json')
        # sft_simple(f'bfcl/BFCL_v3_live_parallel.json', f'sft_data/sft_bfcllive_parallel.json', f'bfcl/possible_answer/BFCL_v3_live_parallel.json')
        # sft_simple(f'bfcl/BFCL_v3_live_multiple.json', f'sft_data/sft_bfcllive_multiple.json', f'bfcl/possible_answer/BFCL_v3_live_multiple.json')
        #sft_simple(f'bfcl/BFCL_v3_live_parallel_multiple.json', f'sft_data/stf_bfcllive_parallel_multiple.json', f'bfcl/possible_answer/BFCL_v3_live_parallel_multiple.json')
    else:
        sft_simple(f'bfcl/BFCL_v3_{input_prefix}.json', f'stf_{input_prefix}.json', f'bfcl/possible_answer/BFCL_v3_{input_prefix}.json')
