from utils.sharegpt_inference import LLM
import click
import json
import os
import requests
from typing import List, Dict
from sklearn.metrics import f1_score
from rouge_score import rouge_scorer

class TaskbenchLLM(LLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create_messages(self, conversation_data: Dict) -> List[Dict]:
        """Create messages list from conversation data"""
        messages = []
        
        # getting the last as label
        conversations = conversation_data["conversations"][:-1]  
        
        for conv in conversations:
            if conv["from"] == "human":
                messages.append({
                    "role": "user",
                    "content": conv["value"]
                })
            elif conv["from"] == "gpt":
                messages.append({
                    "role": "assistant",
                    "content": conv["value"]
                })
        
        return messages

@click.command()
@click.option("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
@click.option("--data_path", type=str, default="../data/sft_data/taskbench_data_dailylifeapis.json")
@click.option("--is_api", type=bool, default=False)
@click.option("--host", type=str, default="localhost")
@click.option("--port", type=int, default=13427)
@click.option("--tensor_parallel_size", type=int, default=1)
@click.option("--batch_size", type=int, default=20)
def main(model: str, data_path: str, is_api: bool, host: str, port: int, tensor_parallel_size: int, batch_size: int):
    # Initialize
    data_split = data_path.replace(".json", "").split("/")[-1].split("_")[-1]
    tool_desc_path = f"https://raw.githubusercontent.com/microsoft/JARVIS/refs/heads/main/taskbench/data_{data_split}/tool_desc.json"
    tool_desc = requests.get(tool_desc_path).json()
    eval_data = json.load(open(data_path, "r"))
    labels = [json.loads(d["conversations"][-1]["value"]) for d in eval_data]

    if not is_api:
        llm = LLM(model=model, tensor_parallel_size=tensor_parallel_size, use_sharegpt_format=True)
    else:
        llm = LLM(model=model)

    # Run inference
    output_path = f"{model.split('/')[-1]}_{data_split}_results.json"
    parsed_output_path = f"{model.split('/')[-1]}_{data_split}_parsed_results.json"

    def run_inference():
        if os.path.exists(output_path):
            results = json.load(open(output_path, "r"))
        else:
            results = llm.batch_generate(eval_data, max_concurrent_calls=batch_size)
            results = []
            for ed in eval_data:
                results.append(llm.single_generate(
                    ed["conversations"][-1]["value"], 
                    system_prompt=ed["system"],
                    former_messages=ed["conversations"][:-1],
                    shrink_multiple_break=True
                ))
            json.dump(eval_data, open(output_path, "w"), indent=4)
            json.dump(results, open(output_path, "w"), indent=4)
        parsed_results = parse_json(llm, results, data_split)
        json.dump(parsed_results, open(parsed_output_path, "w"), indent=4)
        return parsed_results

    if not os.path.exists(parsed_output_path):
        if not is_api:
            with llm.start_server():
                parsed_results = run_inference()
        else:
            parsed_results = run_inference()
    else:
        parsed_results = json.load(open(parsed_output_path, "r"))
    evaluate(parsed_results, labels, data_split, tool_desc)

def parse_json(llm: LLM, responses: List[str], data_split: str) -> List[Dict]:
    parsed_results = {}
    reformat_batch = []
    reformat_indices = []
    
    # First pass - try to parse each response
    for i, orig_content in enumerate(responses):
        orig_content = orig_content.replace("\n", "").replace("\_", "_")
        content = orig_content.replace("\\", "")
        
        # Extract content between brackets
        start_pos = content.find("RESULT #:")
        if start_pos != -1:
            content = content[start_pos + len("RESULT #:"):]
        content = content[content.find("{"):content.rfind("}") + 1]
        
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list) and len(parsed):
                merge_content = {}
                for c in parsed:
                    for k, v in c.items():
                        merge_content[k].extend(v) if k in merge_content else merge_content.update({k: v})
                parsed = merge_content

            parsed_results[i] = parsed
        except json.JSONDecodeError:
            if data_split == "dailylifeapis":
                reformat_batch.append({
                    "role": "user",
                    "content": """Please format the result # RESULT # to a strict JSON format # STRICT JSON FORMAT #. \nRequirements:\n1. Do not change the meaning of task steps, task nodes and task links;\n2. Don't tolerate any possible irregular formatting to ensure that the generated content can be converted by json.loads();\n3. Pay attention to the matching of brackets. Write in a compact format and avoid using too many space formatting controls;\n4. You must output the result in this schema: {"task_steps": [ "concrete steps, format as Step x: Call xxx tool with xxx: 'xxx' and xxx: 'xxx'" ], "task_nodes": [{"task": "task name must be from # TASK LIST #", "arguments": [ {"name": "parameter name", "value": "parameter value, either user-specified text or the specific name of the tool whose result is required by this node"} ]}], "task_links": [{"source": "task name i", "target": "task name j"}]}\n# RESULT #:{{illegal_result}}\n# STRICT JSON FORMAT #:""".replace("{{illegal_result}}", orig_content)
                })
            else:
                reformat_batch.append({
                    "role": "user",
                    "content": """Please format the result # RESULT # to a strict JSON format # STRICT JSON FORMAT #. \nRequirements:\n1. Do not change the meaning of task steps and task nodes;\n2. Don't tolerate any possible irregular formatting to ensure that the generated content can be converted by json.loads();\n3. You must output the result in this schema: {"task_steps": [ step description of one or more steps ], "task_nodes": [{"task": "tool name must be from # TOOL LIST #", "arguments": [ a concise list of arguments for the tool. Either original text, or user-mentioned filename, or tag '<node-j>' (start from 0) to refer to the output of the j-th node. ]}]}\n# RESULT #:{{illegal_result}}\n# STRICT JSON FORMAT #:""".replace("{{illegal_result}}", orig_content)
                })
            reformat_indices.append(i)
    
    # Second pass - reformat invalid responses
    if reformat_batch:
        reformatted = llm._batch_inference([[user_prompt] for user_prompt in reformat_batch])
        
        for idx, new_content in zip(reformat_indices, reformatted):
            new_content = new_content.replace("\n", "").replace("\_", "_")
            start_pos = new_content.find("STRICT JSON FORMAT #:")
            if start_pos != -1:
                new_content = new_content[start_pos + len("STRICT JSON FORMAT #:"):]
            new_content = new_content[new_content.find("{"):new_content.rfind("}") + 1]
            
            try:
                parsed_results[idx] = json.loads(new_content)
            except json.JSONDecodeError:
                parsed_results[idx] = ""
    
    # Reformat parsed results to list
    return [parsed_results[i] for i in sorted(parsed_results.keys())]

def get_content_type(content):
    content = content.strip('\'')
    assert isinstance(content, str), content
    # image
    for ext in ["jpg", "png", "jpeg", "gif", "bmp", "tiff", "svg", "ico"]:
        if "."+ext in content:
            return "image"
    # audio
    for ext in ["mp3", "wav", "wma", "ogg", "aac", "flac", "aiff", "au"]:
        if "."+ext in content:
            return "audio"
    # video
    for ext in ["mp4", "avi", "mov", "flv", "wmv", "mkv", "webm", "m4v", "mpg", "mpeg"]:
        if "."+ext in content:
            return "video"
    return "text"

def evaluate(predictions, labels, data_split, tool_desc):
    for label in labels:
        for k in label:
            label[k] = json.loads(label[k])
    for i in range(len(predictions)):
        if predictions[i] == "":
            predictions[i] = {"task_steps": [], "task_nodes": [], "task_links": []}
    # calculate task steps
    pred_tasksteps = []
    label_tasksteps = []
    for pred, label in zip(predictions, labels):
        # Convert all steps to string
        current_pred_steps = pred.get('task_steps', [])
        for i in range(len(current_pred_steps)):
            if isinstance(current_pred_steps[i], list):
                if all(isinstance(step, str) for step in current_pred_steps[i]):
                    current_pred_steps[i] = "\n".join(current_pred_steps[i])
                else:
                    current_pred_steps[i] = str(current_pred_steps[i])
            elif isinstance(current_pred_steps[i], dict):
                keys = ['description', 'step_description', 'step_name', 'step']
                current_pred_steps[i] = next((current_pred_steps[i][k] for k in keys if k in current_pred_steps[i]), 
                                            str(current_pred_steps[i]))
            elif isinstance(current_pred_steps[i], int):
                current_pred_steps[i] = str(current_pred_steps[i])
        pred_tasksteps.append("\n".join(current_pred_steps))
        label_tasksteps.append("\n".join(label['task_steps']))

    # calculate task nodes
    pred_node_names = []
    label_node_names = []
    pred_tasklinks = []
    label_tasklinks = []
    for pred, label in zip(predictions, labels):
        # names
        pred_possible_keys = ['task', 'name', 'task_name', 'task_node_name', 'node_name', 'task_node', 'tool']
        current_pred_nodes = pred.get('task_nodes', [])
        all_pred_names = []
        for node in current_pred_nodes:
            if isinstance(node, dict):
                node_name = next((node[k] for k in pred_possible_keys if k in node), None)
                if node_name:
                    all_pred_names.append(node_name)
        pred_node_names.append(all_pred_names)
        # names
        all_label_names = []
        for node in label['task_nodes']:
            if 'task' in node and isinstance(node, dict):
                all_label_names.append(node['task'])
        label_node_names.append(all_label_names)
        # links
        if data_split != "dailylifeapis":
            pred_tasklinks.append([])
            label_tasklinks.append([])
            for i in range(len(all_pred_names) - 1):
                pred_tasklinks[-1].append(all_pred_names[i] + " - " + all_pred_names[i+1])
            for i in range(len(all_label_names) - 1):
                label_tasklinks[-1].append(all_label_names[i] + " - " + all_label_names[i+1])

    # calculate task args
    pred_taskargnames = []
    label_taskargnames = []
    pred_taskargvalues = []
    label_taskargvalues = []
    for pred, label in zip(predictions, labels):
        # Label
        label_argnames = []
        label_argvalues = []
        for node in label['task_nodes']:
            task_name = node.get('task', '') if isinstance(node, dict) else 'PARSE ERROR'
            try:
                arguments = node.get('arguments', [])
            except Exception as e:
                arguments = []
            for arg in arguments:
                name = ""
                value = ""
                if isinstance(arg, str):
                    name = f"{task_name} - {get_content_type(arg)}"
                    value = f"{task_name} - {name}: {arg}"
                elif isinstance(arg, int) or isinstance(arg, float):
                    name = f"{task_name} - number"
                    value = f"{task_name} - {name}: {arg}"
                elif isinstance(arg, list):
                    name = f"{task_name} - list"
                    value = f"{task_name} - {name}: {arg}"
                else:
                    name = f"{task_name} - {arg.get('name', 'LABEL ERROR')}"
                    value = f"{task_name} - {name}: {arg.get('value', 'LABEL ERROR')}"
                label_argnames.append(name)
                label_argvalues.append(value)
        label_taskargnames.append(label_argnames)
        label_taskargvalues.append(label_argvalues)

        # Pred
        pred_argnames = []
        pred_argvalues = []
        current_pred_nodes = pred.get('task_nodes', [])
        for node in current_pred_nodes:
            if not isinstance(node, dict):
                continue
            pred_possible_keys = ['task', 'name', 'task_name', 'task_node_name', 'node_name', 'task_node', 'tool']
            task_name = next((node[k] for k in pred_possible_keys if k in node), '')
            arguments = node.get('arguments', [])
            if arguments is None:
                arguments = []
            for arg in arguments:
                name = ""
                value = []
                if isinstance(arg, str):
                    name = f"{task_name} - {get_content_type(arg)}"
                    value = f"{task_name} - {name}: {arg}"
                elif isinstance(arg, dict):
                    name = f"{task_name} - {arg.get('name', 'PRED ERROR')}"
                    value = f"{task_name} - {name}: {arg.get('value', 'PRED ERROR')}"
                elif isinstance(arg, list):
                    for item in arg:
                        name = f"{task_name} - {get_content_type(str(item))}"
                        value.append(f"{task_name} - {name}: {str(item)}")
                    value = "\n".join(value)
                else:
                    name = f"{task_name} - PRED ERROR"
                    value = f"{task_name} - PRED ERROR"
                pred_argnames.append(name)
                pred_argvalues.append(value)
        pred_taskargnames.append(pred_argnames)
        pred_taskargvalues.append(pred_argvalues)

    # calculate task links
    for pred, label in zip(predictions, labels):
        if data_split == "dailylifeapis":
            obj_pred_tasklinks = pred.get('task_links', [])
            try:
                obj_pred_tasklinks = [obj_pred_tasklinks[i]['source'] + " - " + obj_pred_tasklinks[i+1]['target'] for i in range(len(obj_pred_tasklinks) - 1)]
            except Exception as e:
                obj_pred_tasklinks = []
            pred_tasklinks.append(obj_pred_tasklinks)
            obj_label_tasklinks = label.get('task_links', [])
            try:
                obj_label_tasklinks = [obj_label_tasklinks[i]['source'] + " - " + obj_label_tasklinks[i+1]['target'] for i in range(len(obj_label_tasklinks) - 1)]
            except Exception as e:
                obj_label_tasklinks = []
            label_tasklinks.append(obj_label_tasklinks)

    # Calculate metrics

    # Rouge for task steps
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2'], use_stemmer=True)
    rouge_scores = [0, 0]
    for pred, label in zip(pred_tasksteps, label_tasksteps):
        rouge_scores[0] += rouge.score(pred, label)['rouge1'].fmeasure
        rouge_scores[1] += rouge.score(pred, label)['rouge2'].fmeasure
    rouge_scores[0] /= len(pred_tasksteps)
    rouge_scores[1] /= len(pred_tasksteps)
    print(rouge_scores)

    # F1 for task nodes
    name_f1 = 0
    for pred_name, label_name in zip(pred_node_names, label_node_names):
        ground_truth = set(label_name)
        prediction = set(pred_name)
        true_positive = ground_truth & prediction
        precision = 0 if len(prediction) == 0 else len(true_positive) / len(prediction)
        recall = 0 if len(ground_truth) == 0 else len(true_positive) / len(ground_truth)
        if precision == 0 or recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        name_f1 += f1
    name_f1 /= len(pred_node_names)
    print(name_f1)

    # F1 for task args
    t_f1 = 0
    v_f1 = 0
    for pred_argname, label_argname in zip(pred_taskargnames, label_taskargnames):
        ground_truth = set(label_argname)
        prediction = set(pred_argname)
        true_positive = ground_truth & prediction
        precision = 0 if len(prediction) == 0 else len(true_positive) / len(prediction)
        recall = 0 if len(ground_truth) == 0 else len(true_positive) / len(ground_truth)
        if precision == 0 or recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        t_f1 += f1
    t_f1 /= len(pred_taskargnames)
    print(t_f1)

    # F1 for task args
    for pred_argvalue, label_argvalue in zip(pred_taskargvalues, label_taskargvalues):
        # import pdb; pdb.set_trace()
        ground_truth = set(label_argvalue)
        prediction = set(pred_argvalue)
        true_positive = ground_truth & prediction
        precision = 0 if len(prediction) == 0 else len(true_positive) / len(prediction)
        recall = 0 if len(ground_truth) == 0 else len(true_positive) / len(ground_truth)
        if precision == 0 or recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        v_f1 += f1
    v_f1 /= len(pred_taskargvalues)
    print(v_f1)

    # F1 for task links
    link_f1 = 0
    for pred_tasklink, label_tasklink in zip(pred_tasklinks, label_tasklinks):
        ground_truth = set(label_tasklink)
        prediction = set(pred_tasklink)
        true_positive = ground_truth & prediction
        precision = 0 if len(prediction) == 0 else len(true_positive) / len(prediction)
        recall = 0 if len(ground_truth) == 0 else len(true_positive) / len(ground_truth)
        if precision == 0 or recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        link_f1 += f1
    link_f1 /= len(pred_tasklinks)
    print(link_f1)


if __name__ == "__main__":
    main()

