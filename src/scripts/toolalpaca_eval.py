from argparse import ArgumentParser
from utils.template import TOOLALPACA_EVAL
from tqdm import tqdm
from openai import OpenAI
from string import Template
import json
import os

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url="https://toollearning.cn/v1")

parser = ArgumentParser()
# Llama Factory Sharegpt format, default is generated_predictions.jsonl
parser.add_argument("--eval_file", type=str, required=True)
parser.add_argument("--data_file", type=str, default="../data/sft_data/toolalpaca_eval_real_sharegpt.json")
parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")

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
    except ValueError:
        return ""


if __name__ == "__main__":
    args = parser.parse_args()
    eval_template = Template(TOOLALPACA_EVAL)
    instances = json.load(open(args.data_file, "r"))
    full_results = [json.loads(line) for line in open(args.eval_file, "r")]
    error_stats = {
        "parsing_error": 0,
        "correct": 0,
        "incorrect": 0,
        "uncertain": 0,
    }

    print("Running evaluation...")
    for result, instance in tqdm(zip(full_results, instances)):
        answers = parse_assistant_response(result["predict"])
        if len(answers) == 0:
            error_stats["parsing_error"] += 1
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
        process_correct = "No"
        final_correct = "No"
        
        for line in eval_text.split('\n'):
            if "Process Correctness:" in line:
                process_correct = line.split(':')[1].strip()
            elif "Final Response Correctness:" in line:
                final_correct = line.split(':')[1].strip()
        
        if process_correct.lower() == "yes" and final_correct.lower() == "yes":
            error_stats["correct"] += 1
        elif process_correct.lower() == "no" or final_correct.lower() == "no":
            error_stats["incorrect"] += 1
        else:
            error_stats["uncertain"] += 1
        print(response.choices[0].message.content)
    
    print(error_stats)