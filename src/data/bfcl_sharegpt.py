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
        system_prompt = ""
        tool_description = ""

        question = entry.get("question", [])
        for qa_pair in question:
            for q in qa_pair:
                if q["role"] == "user":
                    conversations.append({"from": "human", "value": q["content"]})
        
        # Add function_call with ground truth JSON only once
        ground_truth = ground_truth_data.get(entry["id"], {}).get("ground_truth", [{}])[0]
        conversations.append({"from": "function_call", "value": json.dumps(ground_truth)})
        # Add corresponding observation as empty
        conversations.append({"from": "observation", "value": ""})
        # Add GPT response with ground truth
        conversations.append({"from": "gpt", "value": json.dumps(ground_truth)})
        
        # Add tool description with entire function JSON
        tool_description = json.dumps(entry.get("function", []))
        
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

        # Iterate over multi-turn conversations
        for i, qa_pair in enumerate(question):
            for q in qa_pair:
                if q["role"] == "user":
                    conversations.append({"from": "human", "value": q["content"]})
            
            if i < len(ground_truth_list):
                # Add function_call with ground truth JSON
                ground_truth = ground_truth_list[i]
                conversations.append({"from": "function_call", "value": json.dumps(ground_truth)})
                # Add corresponding observation as empty
                conversations.append({"from": "observation", "value": ""})
                # Add GPT response with ground truth
                conversations.append({"from": "gpt", "value": json.dumps(ground_truth)})
        
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
        sft_simple(f'bfcl/BFCL_v3_simple.json', f'sft_data/stf_bfclsimple.json', f'bfcl/possible_answer/BFCL_v3_simple.json')
        sft_simple(f'bfcl/BFCL_v3_parallel.json', f'sft_data/stf_bfclparallel.json', f'bfcl/possible_answer/BFCL_v3_parallel.json')
        sft_simple(f'bfcl/BFCL_v3_multiple.json', f'sft_data/stf_bfclmultiple.json', f'bfcl/possible_answer/BFCL_v3_multiple.json')
        sft_simple(f'bfcl/BFCL_v3_parallel_multiple.json', f'sft_data/stf_bfclparallel_multiple.json', f'bfcl/possible_answer/BFCL_v3_parallel_multiple.json')
        sft_simple(f'bfcl/BFCL_v3_live_simple.json', f'sft_data/stf_bfcllive_simple.json', f'bfcl/possible_answer/BFCL_v3_live_simple.json')
        sft_simple(f'bfcl/BFCL_v3_live_parallel.json', f'sft_data/stf_bfcllive_parallel.json', f'bfcl/possible_answer/BFCL_v3_live_parallel.json')
        sft_simple(f'bfcl/BFCL_v3_live_multiple.json', f'sft_data/stf_bfcllive_multiple.json', f'bfcl/possible_answer/BFCL_v3_live_multiple.json')
        sft_simple(f'bfcl/BFCL_v3_live_parallel_multiple.json', f'sft_data/stf_bfcllive_parallel_multiple.json', f'bfcl/possible_answer/BFCL_v3_live_parallel_multiple.json')
        sft_multi('bfcl/BFCL_v3_multi_turn_base.json', 'sft_data/stf_bfclmulti_base.json', 'bfcl/possible_answer/BFCL_v3_multi_turn_base.json')
    else:
        sft_simple(f'bfcl/BFCL_v3_{input_prefix}.json', f'stf_{input_prefix}.json', f'bfcl/possible_answer/BFCL_v3_{input_prefix}.json')
