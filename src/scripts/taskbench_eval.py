from utils.sharegpt_inference import LLM
import click
import json
import os
import requests
from typing import List, Dict
from evaluate import load
import numpy as np
from scipy.optimize import linear_sum_assignment
import Levenshtein
from sklearn.metrics import precision_recall_fscore_support as prfs

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
        llm = LLM(model=model, port=port, host=host, api_key="taskbench", tensor_parallel_size=tensor_parallel_size)
    else:
        llm = LLM(model=model, api_base=f"http://toollearning.cn/v1", api_key=os.environ["OPENAI_API_KEY"])

    # Run inference
    output_path = f"{model.split('/')[-1]}_{data_split}_results.json"
    parsed_output_path = f"{model.split('/')[-1]}_{data_split}_parsed_results.json"

    def run_inference():
        if os.path.exists(output_path):
            results = json.load(open(output_path, "r"))
        else:
            results = llm(eval_data, max_concurrent_calls=batch_size)
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

def evaluate(predictions, labels, data_split, tool_desc):
    # Helpers
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
    
    def flatten(gt, pred, types = None):
        assert len(gt) == len(pred)

        gt_flat = []
        pred_flat = []

        for (sample_gt, sample_pred) in zip(gt, pred):
            union = set()

            union.update(sample_gt)
            union.update(sample_pred)

            for s in union:
                if types: 
                    if s in types:
                        if s in sample_gt:
                            gt_flat.append(types.index(s)+1)
                        else:
                            gt_flat.append(0)

                        if s in sample_pred:
                            pred_flat.append(types.index(s)+1)
                        else:
                            pred_flat.append(0)
                    else:
                        gt_flat.append(0)
                        pred_flat.append(0)
                else:
                    if s in sample_gt:
                        gt_flat.append(1)
                    else:
                        gt_flat.append(0)

                    if s in sample_pred:
                        pred_flat.append(1)
                    else:
                        pred_flat.append(0)
        return gt_flat, pred_flat
    # Hardcode Initialize
    metric = ["f1", "ed", "link", "argument", "rouge"]
    metric_dict = {}
    tool_map = {tool["id"]: i+1 for i, tool in enumerate(tool_desc["nodes"])}
    tool_map_reverse = {i+1: tool["id"] for i, tool in enumerate(tool_desc["nodes"])}
    tool_map_reverse[0] = "NEGATIVE"
    tool_map["<PAD>"] = -1
    if data_split != "dailylifeapis":
        tool_output_type_map = {tool["id"]: tool["output-type"][0] if len(tool["output-type"]) else "none" for tool in tool_desc["nodes"]}

    labels = {i: labels[i] for i in range(len(labels))}
    predictions = {i: predictions[i] for i in range(len(predictions))}

    prediction_task_steps = []
    label_task_steps = []
    prediction_names = []
    label_names = []
    label_graphs = []
    prediction_graphs = []
    label_links = []
    prediction_links = []
    label_task_arg_names = []
    prediction_task_arg_names = []
    label_task_arg_name_values = []
    prediction_task_arg_name_values = []

    for id in range(len(labels)):
        label = labels[id]
        label["task_steps"] = json.loads(label["task_steps"])
        label["task_nodes"] = json.loads(label["task_nodes"])
        label["task_links"] = json.loads(label["task_links"])
        
        prediction = predictions[id]
        if prediction == "":
            prediction = {
                "task_steps": [],
                "task_nodes": [],
                "task_links": []
            }
        else:
            prediction["task_steps"] = prediction.get("task_steps", [])
            prediction["task_nodes"] = prediction.get("task_nodes", [])
            prediction["task_links"] = prediction.get("task_links", [])

        if "rouge" in metric or "bertscore" in metric:
            prediction_task_step = prediction["task_steps"]
            label_task_step = label["task_steps"]
            
            try:
                if isinstance(prediction_task_step[0], str):
                    prediction_task_steps.append("\n".join(prediction_task_step))
                else:
                    if "task" in prediction_task_step[0]:
                        prediction_task_steps.append("\n".join([step["task"] for step in prediction_task_step]))
                    elif "step" in prediction_task_step[0]:
                        prediction_task_steps.append("\n".join([step["step"] for step in prediction_task_step]))
                    elif "id" in prediction_task_step[0]:
                        prediction_task_steps.append("\n".join([step["id"] for step in prediction_task_step]))
                    elif "step_name" in prediction_task_step[0]:
                        prediction_task_steps.append("\n".join([step["step_name"] for step in prediction_task_step]))
                    else:
                        prediction_task_steps.append("\n".join([step["description"] for step in prediction_task_step]))
            except Exception as e:
                prediction_task_steps.append(str(prediction_task_step))

            label_task_steps.append("\n".join(label_task_step))

        label_nodes = label["task_nodes"]
        prediction_nodes = prediction["task_nodes"] 

        try:
            label_node_name = [node["task"] for node in label_nodes]
            prediction_node_name = [node["task"] for node in prediction_nodes]
        except Exception as e:
            continue

        label_task_arg_name = []
        prediction_task_arg_name = []

        label_task_arg_name_value = []
        prediction_task_arg_name_value = []
            
        try:
            if data_split != "dailylifeapis":
                prediction_node_name = [name.replace("_", " ") for name in prediction_node_name]
                label_node_name = [name.replace("_", " ") for name in label_node_name]
                label_link = []
                prediction_link = []
                for inx, node in enumerate(label_nodes):
                    new_arguments = []
                    for i, argument in enumerate(node["arguments"]):
                        try:
                            if isinstance(argument, dict):
                                argument = list(argument.values())[0]
                            if isinstance(argument, list):
                                argument = " ".join(argument)
                            if "<node-" in argument:
                                index_start = argument.index("<node-") + 6
                                index_end = argument.index(">")
                                if int(argument[index_start: index_end]) == inx:
                                    continue
                                argument_tool_name = label_node_name[int(argument[index_start: index_end])]
                                label_link.append({"source": argument_tool_name, "target": node["task"]})
                                new_argument = {"name": tool_output_type_map.get(argument_tool_name, "other"), "value": argument_tool_name}
                            else:
                                new_argument = {"name": get_content_type(argument), "value": argument}
                        except Exception as e:
                            pass
                        new_arguments.append(new_argument)
                    node["arguments"] = new_arguments
                    
                for inx, node in enumerate(prediction_nodes):
                    new_arguments = []
                    for i, argument in enumerate(node.get("arguments", [])):
                        try:
                            if isinstance(argument, dict):
                                argument = list(argument.values())[0]
                            if isinstance(argument, list):
                                argument = " ".join(argument)
                            if isinstance(argument, str) and "<node-" in argument:
                                index_start = argument.index("<node-") + 6
                                index_end = argument.index(">")
                            
                                if int(argument[index_start: index_end]) == inx:
                                    continue
                                prediction_tool_name = prediction_node_name[int(argument[index_start: index_end])]
                                prediction_link.append({"source": prediction_tool_name, "target": node["task"]})
                                new_argument = {"name": tool_output_type_map.get(prediction_tool_name, "other"), "value": prediction_tool_name}
                            else:
                                new_argument = {"name": get_content_type(argument), "value": argument}

                        except Exception as e:
                            pass
                        new_arguments.append(new_argument)
                    node["arguments"] = new_arguments
            else:
                try:
                    prediction_link = prediction["task_links"]
                    label_link = label["task_links"]
                except Exception as e:
                    prediction_link = []
                    label_link = label["task_links"]
        except Exception as e:
            continue

        prediction_node_argument = [node.get("arguments", []) for node in prediction_nodes]
        label_node_argument = [node["arguments"] for node in label_nodes]
        # import pdb; pdb.set_trace()
        for task, arguments in zip (prediction_node_name, prediction_node_argument):
            for argument in arguments:
                try:
                    label_task_arg_name.append(f"{task}-{argument['name']}")
                    label_task_arg_name_value.append(f"{task}-{argument['name']}-{argument['value']}")
                except Exception as e:
                    label_task_arg_name.append(f"{task}-")
                    label_task_arg_name_value.append(f"{task}--{argument}")
        
        for task, arguments in zip (label_node_name, label_node_argument):
            for argument in arguments:
                try:
                    prediction_task_arg_name.append(f"{task}-{argument['name']}")
                    prediction_task_arg_name_value.append(f"{task}-{argument['name']}-{argument['value']}")
                except Exception as e:
                    name = argument.get("name", "")
                    value = argument.get("value", "")
                    prediction_task_arg_name.append(f"{task}-")
                    prediction_task_arg_name_value.append(f"{task}--{name}-{value}")

        label_graph = {
            "nodes": label_node_name,
            "links": label_link,
            "arguments": label_node_argument
        }
        prediction_graph = {
            "nodes": prediction_node_name,
            "links": prediction_link,
            "arguments": prediction_node_argument
        }

        label_graphs.append(label_graph)
        prediction_graphs.append(prediction_graph)

        for node_name in prediction_node_name:
            assert isinstance(node_name, str), node_name

        prediction_names.append(prediction_node_name)
        label_names.append(label_node_name)

        prediction_task_arg_names.append(prediction_task_arg_name)
        label_task_arg_names.append(label_task_arg_name)
    
        prediction_task_arg_name_values.append(prediction_task_arg_name_value)
        label_task_arg_name_values.append(label_task_arg_name_value)

        label_links.append(label_link)
        prediction_links.append(prediction_link)


    rouge = load("rouge")
    rouge_scores = rouge.compute(predictions=prediction_task_steps, references=label_task_steps, use_aggregator=True, use_stemmer=True)
    metric_dict["ROUGE-1"] = rouge_scores["rouge1"]
    metric_dict["ROUGE-2"] = rouge_scores["rouge2"]

    bertscore = load("bertscore")
    bertscore_scores = bertscore.compute(predictions=prediction_task_steps, references=label_task_steps, lang="en", model_type="roberta-large")
    metric_dict["BERTScore"] = np.mean(bertscore_scores["f1"])
    
    types = list(range(1, len(tool_desc["nodes"])+1))
    types_name = [tool_map_reverse[i] for i in types]
    gt_flat, pred_flat = flatten(label_names, prediction_names, types = types_name)

    micro = prfs(gt_flat, pred_flat, labels=types, average='micro')[:-1]
    metric_dict["node_f1"] = micro[2]


    gt_flat, pred_flat = flatten(label_task_arg_names, prediction_task_arg_names)
    micro = prfs(gt_flat, pred_flat, average="binary")[:-1]
    print(f"Argument Task-ArgName Binary F1: [ No Matching ]: {micro[-1]}")
    metric_dict["name_f1"] = micro[-1]

    gt_flat, pred_flat = flatten(label_task_arg_name_values, prediction_task_arg_name_values)
    micro = prfs(gt_flat, pred_flat, average="binary")[:-1]
    print(f"Argument Task-ArgName-Value Binary F1 [ No Matching ]: {micro[-1]}")
    metric_dict["value_f1"] = micro[-1]
    
    tuple_label_links = []
    tuple_prediction_links = []
    for label_link, prediction_link in zip(label_links, prediction_links):
        try:
            pred_tuples = [(link["source"], link["target"]) for link in prediction_link]
            label_tuples = [(link["source"], link["target"]) for link in label_link]
            
            # Pad shorter list with empty tuples to match lengths
            max_len = max(len(pred_tuples), len(label_tuples))
            pred_tuples.extend([("", "")] * (max_len - len(pred_tuples)))
            label_tuples.extend([("", "")] * (max_len - len(label_tuples)))
            
            tuple_prediction_links.append(pred_tuples)
            tuple_label_links.append(label_tuples)
        except Exception as e:
            pass
    
    gt_flat, pred_flat = flatten(tuple_label_links, tuple_prediction_links)


    micro = prfs(gt_flat, pred_flat, average="binary")[:-1]
    metric_dict["edge_f1"] = micro[-1]

    # Round F1 metrics to 2 decimal places and multiply by 100
    for key in list(metric_dict.keys()):
        if "f1" in key.lower():
            metric_dict[key] = round(metric_dict[key] * 100, 2)

    print(metric_dict)

if __name__ == "__main__":
    main()

