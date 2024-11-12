from utils.formatter import format_sharegpt_tools
from utils.template import TOOLALPACA_SYSTEM, TOOLALPACA_EVAL
from argparse import ArgumentParser
from vllm import LLM, SamplingParams
from tqdm import tqdm
from openai import OpenAI
from string import Template
import json

client = OpenAI(api_key="", base_url="https://toollearning.cn/v1")

parser = ArgumentParser()
parser.add_argument("--eval_file", type=str, default="../data/sft_data/toolalpaca_eval_real_sharegpt.json")
parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")

def build_prompt(instance):
    tool_text = format_sharegpt_tools(instance["tools"])
    instruction = instance["conversations"][0]["value"]
    return TOOLALPACA_SYSTEM.format(tools=tool_text, instruction=instruction)

def parse_assistant_response(text):
    # Initialize empty dictionary for results
    result = {}
    
    # Split the text by newlines and remove empty strings
    parts = [part for part in text.split('\n') if part.strip()]
    
    try:
        for part in parts:
            if part.startswith('ASSISTANT Thought:'):
                result['thought'] = part.replace('ASSISTANT Thought:', '').strip()
            elif part.startswith('ASSISTANT Action:'):
                result['action'] = part.replace('ASSISTANT Action:', '').strip()
            elif part.startswith('ASSISTANT Action Input:'):
                # Parse the JSON-like string into a string
                result['action_input'] = part.replace('ASSISTANT Action Input:', '').strip()
    except: # return empty
        pass
    
    return result

if __name__ == "__main__":
    args = parser.parse_args()
    sampling_params = SamplingParams(temperature=0, max_tokens=256)
    llm = LLM(model=args.model)

    eval_items = json.load(open(args.eval_file, "r"))
    eval_template = Template(TOOLALPACA_EVAL)
    full_results = []
    error_stats = {
        "answer_generation_error": 0,
        "evaluation_error": 0,
        "correct": 0,
        "incorrect": 0,
        "uncertain": 0,
    }

    # Generate responses
    print("Generating responses...")
    for instance in tqdm(eval_items):
        system_prompt = build_prompt(instance)
        msg = [{"role": "user", "content": system_prompt}]
        vllm_msg = llm.llm_engine.tokenizer.tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        response = llm.generate([vllm_msg], sampling_params=sampling_params)
        response_text = response[0].outputs[0].text
        full_results.append({"response": response_text, "question": instance["conversations"][0]["value"], "gold_answer": instance["conversations"][1]["value"]})

    # Run evaluation
    print("Running evaluation...")
    for result in tqdm(full_results):
        answers = parse_assistant_response(result["response"])
        if len(answers) == 0:
            error_stats["answer_generation_error"] += 1
            continue
        solution = "Function: {}\nAction Input: {}".format(answers["action"], answers["action_input"])
        eval_prompt = eval_template.substitute(documentation=format_sharegpt_tools(instance["tools"]), instruction=result["question"], standard=result["gold_answer"], solution=solution, analysis="Brief analysis of the solution against the gold answer.")
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