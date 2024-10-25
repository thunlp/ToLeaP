import json
import sys

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

if __name__ == "__main__":
    input_prefix = sys.argv[1]
    if input_prefix == 'all':
        sft_simple(f'bfcl/BFCL_v3_simple.json', f'stf_simple.json', f'bfcl/possible_answer/BFCL_v3_simple.json')
        sft_simple(f'bfcl/BFCL_v3_parallel.json', f'stf_parallel.json', f'bfcl/possible_answer/BFCL_v3_parallel.json')
        sft_simple(f'bfcl/BFCL_v3_multiple.json', f'stf_multiple.json', f'bfcl/possible_answer/BFCL_v3_multiple.json')
        sft_simple(f'bfcl/BFCL_v3_parallel_multiple.json', f'stf_parallel_multiple.json', f'bfcl/possible_answer/BFCL_v3_parallel_multiple.json')
        sft_simple(f'bfcl/BFCL_v3_live_simple.json', f'stf_live_simple.json', f'bfcl/possible_answer/BFCL_v3_live_simple.json')
        sft_simple(f'bfcl/BFCL_v3_live_parallel.json', f'stf_live_parallel.json', f'bfcl/possible_answer/BFCL_v3_live_parallel.json')
        sft_simple(f'bfcl/BFCL_v3_live_multiple.json', f'stf_live_multiple.json', f'bfcl/possible_answer/BFCL_v3_live_multiple.json')
        sft_simple(f'bfcl/BFCL_v3_live_parallel_multiple.json', f'stf_live_parallel_multiple.json', f'bfcl/possible_answer/BFCL_v3_live_parallel_multiple.json')
    else:
        sft_simple(f'bfcl/BFCL_v3_{input_prefix}.json', f'stf_{input_prefix}.json', f'bfcl/possible_answer/BFCL_v3_{input_prefix}.json')
