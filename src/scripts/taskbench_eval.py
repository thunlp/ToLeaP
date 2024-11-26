import json
import os
from argparse import ArgumentParser
from llamafactory.train.sft.workflow import run_sft
from llamafactory.hparams import get_train_args
from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from llamafactory.hparams.data_args import DataArguments
from llamafactory.hparams.evaluation_args import EvaluationArguments
from llamafactory.hparams.finetuning_args import FinetuningArguments
from llamafactory.hparams.generating_args import GeneratingArguments
from llamafactory.hparams.model_args import ModelArguments

ALL_ARGS = [ModelArguments, DataArguments, Seq2SeqTrainingArguments, FinetuningArguments, GeneratingArguments]

parser = ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
parser.add_argument('--input_file', type=str, required=True, help='Path to input file')
parser.add_argument('--model_name', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct')

# Rest of the original functions remain unchanged
def extract_tool_name(action, tool_str):
    tool = json.loads(tool_str)
    names = [t['name'] for t in tool]
    names_lower = [n.lower() for n in names]
    # First check for exact match
    if action.lower() in names_lower:
        return action
    # Then check for partial match
    for name in names:
        if name.lower() in action.lower():
            return name
    return action

def get_tool_calls(result, src_tool_data):
    lines = [l.strip() for l in result.split('\n') if l.strip()]
    actions = []
    current_action = None
    current_action_input = None
    input_error = False
    for line in lines:
        if line.startswith('Action:'):
            current_action = line.split('Action:')[1].strip()
        elif line.startswith('Action Input:'):
            current_action_input = line.split('Action Input:')[1].strip()
        else:
            continue
        if current_action and current_action_input:
            current_action = extract_tool_name(current_action, src_tool_data)
            # Find the JSON object boundaries
            start_idx = 0
            end_idx = len(current_action_input)
            found_start = False

            for i in range(len(current_action_input)):
                if current_action_input[i] == '{' and not found_start:
                    start_idx = i
                    found_start = True
                elif current_action_input[i] == '}' and found_start:
                    end_idx = i

            if start_idx != -1 and end_idx != -1:
                # Extract only the JSON part
                current_action_input = current_action_input[start_idx:end_idx + 1]
            
            actions.append((current_action, current_action_input))
            current_action = None
            current_action_input = None
    
    return actions

def f1(pred_names, label_names):
    true_positives = len(set(pred_names) & set(label_names))
    precision = true_positives / len(pred_names) if pred_names else 0
    recall = true_positives / len(label_names) if label_names else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

if __name__ == "__main__":
    args = parser.parse_args()
    
    # Initialize Llama Factory
    yml_file = args.config
    parser = HfArgumentParser(ALL_ARGS)
    model_args, data_args, training_args, finetuning_args, generating_args = parser.parse_yaml_file(yml_file)
    model_args.model_name_or_path = args.model_name
    data_args.dataset_dir = 'llamafactory_data'
    run_sft(model_args, data_args, training_args, finetuning_args, generating_args)

    # Load results and source data
    output_file_path = os.path.join(training_args.output_dir, "generated_predictions.jsonl")
    results = [json.loads(s) for s in open(output_file_path)]
    tool_data = json.load(open(args.input_file))

    # Rest of the evaluation logic remains the same
    all_parsed = []
    for i, result in enumerate(results):
        parsed = get_tool_calls(result['predict'], tool_data[i]['tools'])
        all_parsed.append(parsed)

    labels = [t['conversations'][1]['value'] for t in tool_data]
    avg_node_f1 = 0
    avg_edge_f1 = 0
    
    # Tool name metrics: Node, Edge
    for parsed, label in zip(all_parsed, labels):
        label = json.loads(label)
        # Node F1
        label_names = [t['name'] for t in label]
        parsed_names = [t[0] for t in parsed]
        f1_score = f1(parsed_names, label_names)
        avg_node_f1 += f1_score
        # Edge F1
        if len(label) > 1 and len(parsed) > 1:
            label_edges = [f"{label[i]['name']} - {label[i+1]['name']}" for i in range(len(label) - 1)]
            parsed_edges = [f"{parsed[i][0]} - {parsed[i+1][0]}" for i in range(len(parsed) - 1)]
        else:
            label_edges = label_names
            parsed_edges = parsed_names
        f1_score = f1(parsed_edges, label_edges)
        avg_edge_f1 += f1_score

    avg_node_f1 /= len(all_parsed)
    avg_edge_f1 /= len(all_parsed)
    print(f'Node F1: {avg_node_f1:.4f}, Edge F1: {avg_edge_f1:.4f}')

    # Parameter metrics
    avg_name_f1 = 0
    avg_value_f1 = 0
    for parsed, label in zip(all_parsed, labels):
        label = json.loads(label)
        label_args = [t['arguments'] for t in label]
        try:
            parsed_params = [json.loads(p[1]) for p in parsed]
        except:
            continue
        try:
            label_args_keys = [k for t in label_args for k in t.keys()]
            parsed_args_keys = [k for t in parsed_params for k in t.keys()]
            f1_score = f1(parsed_args_keys, label_args_keys)
            avg_name_f1 += f1_score
            
            label_args_values = [t[k] for t in label_args for k in t.keys()]
            parsed_args_values = [t[k] for t in parsed_params for k in t.keys()]
            f1_score = f1(parsed_args_values, label_args_values)
            avg_value_f1 += f1_score
        except:
            continue

    avg_name_f1 /= len(all_parsed)
    avg_value_f1 /= len(all_parsed)
    print(f'Name F1: {avg_name_f1:.4f}, Value F1: {avg_value_f1:.4f}')