# Copyright 2023 StanfordLegion/task-bench
# Modifications Copyright 2024 BodhiAgent
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import click
import json
from typing import List, Dict
from rouge_score import rouge_scorer

from utils.llm import LLM, extract_first_json

def create_messages(conversation_data: Dict) -> List[Dict]:
    messages = []
    for cov in conversation_data: # Dict
        message = []
        for prompt in cov.get("conversations", []): # List
            if prompt.get("from") == "human":
                message.append({"role": "user", "content": prompt["value"]})
        messages.append(message)
    return messages

@click.command()
@click.option("--model", type=str, default="gpt-3.5-turbo")
@click.option("--data_paths", type=list, default=[
    "../data/taskbench/taskbench_data_dailylifeapis.json",
    "../data/taskbench/taskbench_data_huggingface.json",
    "../data/taskbench/taskbench_data_multimedia.json",
])
@click.option("--is_api", type=bool, default=False)
@click.option("--tensor_parallel_size", type=int, default=4)
@click.option("--batch_size", type=int, default=128)
@click.option("--max_model_len", type=int, default=4096)
@click.option("--max_output_tokens", type=int, default=512)
@click.option("--model_name", type=str)
def main(model: str, data_paths: str, is_api: bool, tensor_parallel_size: int, batch_size: int, max_model_len: int, max_output_tokens: int, model_name:str):
    data_results = {}
    
    llm = LLM(
        model=model, 
        tensor_parallel_size=tensor_parallel_size,
        is_api=is_api,
        use_sharegpt_format=False,
        max_input_tokens=max_model_len,
        batch_size=batch_size, 
        max_output_tokens=max_output_tokens
     )

    for data_path in data_paths:
        # Initialize
        data_split = data_path.replace(".json", "").split("/")[-1].split("_")[-1]
        tool_path = "../data/taskbench/"
        tool_desc_file = os.path.join(os.path.dirname(tool_path), f"tool_desc_{data_split}.json")
        tool_desc = json.load(open(tool_desc_file, "r"))
        eval_data = json.load(open(data_path, "r"))
        labels = [json.loads(d["conversations"][-1]["value"]) for d in eval_data]

        os.makedirs("../results/taskbench", exist_ok=True)
        output_path = f"../results/taskbench/{model_name}_{data_split}_results.json"
        parsed_output_path = f"../results/taskbench/{model_name}_{data_split}_parsed_results.json"

        # Run inference
        def run_inference():
            if os.path.exists(output_path):
                results = json.load(open(output_path, "r"))
            else:
                if is_api:
                    messages = create_messages(eval_data)
                    results = llm.batch_generate_chat(messages)
                else:
                    results = llm.batch_generate_complete(
                        [d["conversations"][0]["value"] for d in eval_data]
                    )
                json.dump(results, open(output_path, "w"), indent=4)
            return results

        if not os.path.exists(parsed_output_path):
            results = run_inference()
        else:
            results = json.load(open(output_path, "r"))

        parsed_results = []
        bad_cases = []
        for result in results:
            try:
                parsed_json = json.loads(extract_first_json(result))
                # Initialize with empty lists
                normalized_json = {
                    "task_steps": [],
                    "task_nodes": [],
                    "task_links": []
                }
                
                # Try to get each field individually
                try:
                    normalized_json["task_steps"] = parsed_json.get("task_steps", [])
                except:
                    pass
                    
                try:
                    normalized_json["task_nodes"] = parsed_json.get("task_nodes", [])
                except:
                    pass
                    
                try:
                    normalized_json["task_links"] = parsed_json.get("task_links", [])
                except:
                    pass
                    
                parsed_results.append(normalized_json)
            except Exception as e:
                bad_cases.append(result)
                parsed_results.append({"task_steps": [], "task_nodes": [], "task_links": []})

        # print("Bad Cases are those that cannot be parsed as JSON or has none of the required fields.")
        # print(f"Total bad cases: {len(bad_cases)}/{len(results)}")
        json.dump(bad_cases, open(f"{model.split('/')[-1]}_{data_split}_bad_cases.json", "w"), indent=4)

        rouge_1, rouge_2, name_f1, t_f1, v_f1, link_f1 = evaluate(parsed_results, labels, data_split, tool_desc)
        data_results[f"{data_split}"] = {
            "rouge_1": round(rouge_1  * 100, 2),
            "rouge_2": round(rouge_2  * 100, 2),
            "name_f1": round(name_f1 * 100, 2),
            "t_f1":   round(t_f1   * 100, 2),
            "v_f1":   round(v_f1   * 100, 2),
            "link_f1":round(link_f1* 100, 2),
        }
    print(json.dumps(data_results))
    

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
        try:
            current_pred_steps = pred.get('task_steps', [])
        except Exception as e:
            current_pred_steps = []
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
        try:
            current_pred_nodes = pred.get('task_nodes', [])
        except Exception as e:
            current_pred_nodes = []
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
                pred_tasklinks[-1].append(str(all_pred_names[i]) + " - " + str(all_pred_names[i+1]))
            for i in range(len(all_label_names) - 1):
                label_tasklinks[-1].append(str(all_label_names[i]) + " - " + str(all_label_names[i+1]))

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
    # print("Rouge 1", rouge_scores[0])
    # print("Rouge 2", rouge_scores[1])

    # F1 for task nodes
    name_f1 = 0
    for pred_name, label_name in zip(pred_node_names, label_node_names):
        ground_truth = set([str(name) for name in label_name])
        prediction = set([str(name) for name in pred_name])
        true_positive = ground_truth & prediction
        precision = 0 if len(prediction) == 0 else len(true_positive) / len(prediction)
        recall = 0 if len(ground_truth) == 0 else len(true_positive) / len(ground_truth)
        if precision == 0 or recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        name_f1 += f1
    name_f1 /= len(pred_node_names)
    # print("Name_F1", name_f1)

    # F1 for task args
    t_f1 = 0
    v_f1 = 0
    for pred_argname, label_argname in zip(pred_taskargnames, label_taskargnames):
        ground_truth = set([str(name) for name in label_argname])
        prediction = set([str(name) for name in pred_argname])
        true_positive = ground_truth & prediction
        precision = 0 if len(prediction) == 0 else len(true_positive) / len(prediction)
        recall = 0 if len(ground_truth) == 0 else len(true_positive) / len(ground_truth)
        if precision == 0 or recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        t_f1 += f1
    t_f1 /= len(pred_taskargnames)
    # print("t_f1", t_f1)

    # F1 for task args
    for pred_argvalue, label_argvalue in zip(pred_taskargvalues, label_taskargvalues):
        # import pdb; pdb.set_trace()
        ground_truth = set([str(value) for value in label_argvalue])
        prediction = set([str(value) for value in pred_argvalue])
        true_positive = ground_truth & prediction
        precision = 0 if len(prediction) == 0 else len(true_positive) / len(prediction)
        recall = 0 if len(ground_truth) == 0 else len(true_positive) / len(ground_truth)
        if precision == 0 or recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        v_f1 += f1
    v_f1 /= len(pred_taskargvalues)
    # print("v_f1", v_f1)

    # F1 for task links
    link_f1 = 0
    for pred_tasklink, label_tasklink in zip(pred_tasklinks, label_tasklinks):
        ground_truth = set([str(link) for link in label_tasklink])
        prediction = set([str(link) for link in pred_tasklink])
        true_positive = ground_truth & prediction
        precision = 0 if len(prediction) == 0 else len(true_positive) / len(prediction)
        recall = 0 if len(ground_truth) == 0 else len(true_positive) / len(ground_truth)
        if precision == 0 or recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        link_f1 += f1
    link_f1 /= len(pred_tasklinks)
    # print("edge_f1", link_f1)
    return rouge_scores[0], rouge_scores[1], name_f1, t_f1, v_f1, link_f1


if __name__ == "__main__":
    main()
