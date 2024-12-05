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

from utils.template import TOOLALPACA_EVAL
from tqdm import tqdm
from openai import OpenAI
from string import Template
import json
import os

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url="https://toollearning.cn/v1")

parser = ArgumentParser()
# Llama Factory Sharegpt format, default is generated_predictions.jsonl
parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
parser.add_argument("--input_file", type=str, required=True, help="Path to input file")
parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")

def parse_assistant_response(text):
    # Initialize empty dictionary for results
    result = {}
    
    # Split the text by newlines and remove empty strings
    parts = [part for part in text.split('\n') if part.strip()]
    
    try:
        for part in parts:
            if part.startswith('Action:'):
                result['action'] = part.replace('Action:', '').strip()
            elif part.startswith('Action Input:'):
                result['action_input'] = part.replace('Action Input:', '').strip()
    except: # return empty
        pass
    
    return result

def get_tools_text(text):
    start_marker = "You have access to the following tools:"
    end_marker = "Use the following format if using a tool:"
    
    try:
        start_idx = text.index(start_marker) + len(start_marker)
        end_idx = text.index(end_marker)
        return text[start_idx:end_idx].strip()
    except ValueError as e:
        print(f"Error extracting tools text: {str(e)}")
        return ""


if __name__ == "__main__":
    sys_args = parser.parse_args()
    yml_file = sys_args.config
    parser = HfArgumentParser(ALL_ARGS)
    model_args, data_args, training_args, finetuning_args, generating_args = parser.parse_yaml_file(yml_file)
    model_args.model_name_or_path = sys_args.model_name
    data_args.dataset_dir = 'llamafactory_data'
    run_sft(model_args, data_args, training_args, finetuning_args, generating_args)

    eval_template = Template(TOOLALPACA_EVAL)
    output_file_path = os.path.join(training_args.output_dir, "generated_predictions.jsonl")
    instances = json.load(open(sys_args.input_file, "r"))
    full_results = [json.loads(line) for line in open(output_file_path, "r")]
    error_stats = {
        "process_correct": 0,
        "final_correct": 0,
        "incorrect": 0,
    }

    print(f"Running evaluation for file: {sys_args.input_file}")
    output_file = open("gpt4_responses.txt", "w")
    for result, instance in tqdm(zip(full_results, instances)):
        answers = parse_assistant_response(result["predict"])
        if len(answers) == 0:
            error_stats["incorrect"] += 1
            continue
        solution = "Function: {}\nAction Input: {}".format(answers["action"], answers["action_input"])
        gold_answer = json.loads(instance["conversations"][1]["value"])
        gold_answer = "Function: {}\nAction Input: {}".format(gold_answer["name"], gold_answer["arguments"])
        eval_prompt = eval_template.substitute(
            documentation=get_tools_text(result["prompt"]), 
            instruction=instance["conversations"][0]["value"], 
            standard=gold_answer,
            solution=solution, 
            analysis="Brief analysis of the solution against the gold answer."
        )
        msg = [{"role": "user", "content": eval_prompt}]
        # Use gpt-4o to evaluate
        response = None
        while response is None:
            try:
                response = client.chat.completions.create(model="gpt-4o", messages=msg, temperature=0, max_tokens=1024)
            except Exception as e:
                print(f"Error occurred during evaluation: {str(e)}. Retrying...")
        # Extract evaluation results from GPT-4's response
        eval_text = response.choices[0].message.content
        output_file.write(eval_text + "\n---\n")
        process_correct = "404"
        final_correct = "404"
        
        for line in eval_text.split('\n'):
            if "Process Correctness:" in line:
                process_correct = line.split(':')[1].strip()
            elif "Final Response Correctness:" in line:
                final_correct = line.split(':')[1].strip()
        
        if process_correct.lower() == "yes" and final_correct.lower() == "yes":
            error_stats["process_correct"] += 1
            error_stats["final_correct"] += 1
        elif process_correct.lower() == "yes":
            error_stats["process_correct"] += 1
        elif final_correct.lower() == "yes":
            error_stats["final_correct"] += 1
        else:
            error_stats["incorrect"] += 1
        
        # print(response.choices[0].message.content)

    # Divide stats by total number of instances
    for key in error_stats:
        error_stats[key] = f"{error_stats[key]:.1f}%"

    output_file.flush()
    output_file.close()
    print(error_stats)