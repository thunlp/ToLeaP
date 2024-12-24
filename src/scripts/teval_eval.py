import argparse
import os
import random
import shutil
import mmengine
import teval.evaluators as evaluator_factory
from tqdm import tqdm
import click
import sys
import time
current_dir = os.path.dirname(os.path.abspath(__file__)) 
utils_dir = os.path.join(current_dir, '..')
sys.path.append(utils_dir)
from utils.llm import LLM
from concurrent.futures import ThreadPoolExecutor, as_completed


class TevalLLM(LLM):
    def __init__(self, max_tokens=1024, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        self.max_tokens = max_tokens

    def _single_inference(self, messages, temperature=0):
        if not self.use_hf:
            chat_output = self.client.chat.completions.create(
                model=self.model_path_or_name,
                messages=messages,
                temperature=temperature,
                max_tokens=self.max_tokens
            )
            return chat_output.choices[0].message.content
        else:
            outputs = self.pipeline(messages, max_length=self.max_tokens)
            return outputs[0]["generated_text"][-1]

    def _batch_inference(self, messages_batch, max_concurrent_calls=4, temperature=0):
        responses = [None] * len(messages_batch)

        def process_single_message(index, messages):
            chat_output = self.client.chat.completions.create(
                model=self.model_path_or_name,
                messages=messages,
                temperature=temperature,
                max_tokens=self.max_tokens
            )
            return index, chat_output.choices[0].message.content

        def process_single_message_hf(index, messages):
            outputs = self.pipeline(messages, max_length=self.max_tokens)
            return index, outputs[0]["generated_text"][-1]

        with ThreadPoolExecutor(max_workers=max_concurrent_calls) as executor:
            if not self.use_hf:
                futures = {executor.submit(process_single_message, idx, messages): idx
                           for idx, messages in enumerate(messages_batch)}
            else:
                futures = {executor.submit(process_single_message_hf, idx, messages): idx
                           for idx, messages in enumerate(messages_batch)}

            for future in tqdm(as_completed(futures), total=len(messages_batch), desc="Processing concurrent calls"):
                try:
                    index, result = future.result()
                    responses[index] = result
                except Exception as e:
                    print(f"An error occurred for batch {futures[future]}: {e}")
                    responses[futures[future]] = ""  
        return responses



@click.command()
@click.option("--model", type=str, default="/hy-tmp/model-dir")
@click.option("--dataset_path", type=str, default="data/instruct_v2.json")
@click.option("--is_api", type=bool, default=False)
@click.option("--out_dir", type=str, default="work_dirs/")
@click.option("--out_name", type=str, default="tmp.json")
@click.option("--tensor_parallel_size", type=int, default=1)
@click.option("--batch_size", type=int, default=12)
@click.option("--gpu_memory_utilization", type=float, default=0.9) 
@click.option("--test_num", type=int, default=-1)
@click.option("--resume", is_flag=True)
@click.option("--eval", type=click.Choice(['instruct', 'reason', 'plan', 'retrieve', 'review', 'understand', 'rru']))
@click.option("--prompt_type", type=click.Choice(['json', 'str']), default='json')
@click.option("--model_name", type=str, default="qwen2.5")
def main(
   model: str,
   dataset_path: str,
   is_api: bool,
   out_dir: str,
   out_name: str,
   tensor_parallel_size: int, 
   batch_size: int,
   gpu_memory_utilization: float,
   test_num: int,
   resume: bool,
   eval: str,
   prompt_type: str,
   model_name: str
):
   os.makedirs(out_dir, exist_ok=True)
   tmp_folder_name = os.path.splitext(out_name)[0]
   os.makedirs(os.path.join(out_dir, tmp_folder_name), exist_ok=True)

   dataset, tested_num, total_num = load_dataset(dataset_path, out_dir, resume, tmp_folder_name)
   test_num = max(total_num - tested_num, 0) if test_num == -1 else max(min(test_num - tested_num, total_num - tested_num), 0)
   
   output_file_path = os.path.join(out_dir, out_name)
   if test_num > 0:
       llm = TevalLLM(
           model=model,
           tensor_parallel_size=tensor_parallel_size,
           gpu_memory_utilization=gpu_memory_utilization,
           use_api_model=is_api
       )
       
       print(f"Tested {tested_num} samples, left {test_num} samples, total {total_num} samples")
       context = llm.start_server() if not is_api else nullcontext()
       with context:
           prediction = infer(dataset, llm, out_dir, tmp_folder_name, test_num, batch_size)
           mmengine.dump(prediction, output_file_path)

   if eval:
       eval_mapping = {
           'instruct': "InstructEvaluator",
           'plan': "PlanningEvaluator", 
           'review': "ReviewEvaluator",
           'reason': "ReasonRetrieveUnderstandEvaluator",
           'retrieve': "ReasonRetrieveUnderstandEvaluator",
           'understand': "ReasonRetrieveUnderstandEvaluator",
           'rru': "ReasonRetrieveUnderstandEvaluator"
       }
       
       bert_score_model =  "all-mpnet-base-v2"
       json_path = os.path.join(out_dir, f"{model.split('/')[-1]}_{-1}_{'zh' if '_zh' in dataset_path else ''}.json")

       evaluator = getattr(evaluator_factory, eval_mapping[eval])(output_file_path, default_prompt_type=prompt_type, eval_type=eval, bert_score_model=bert_score_model)
       
       results = mmengine.load(json_path) if os.path.exists(json_path) else {}
       eval_results = evaluator.evaluate()
       results[f"{eval}_{prompt_type}"] = eval_results
       
       print(f"Evaluation Results:\n{eval_results}")
       print(f"Writing Results to {json_path}")
       mmengine.dump(results, json_path)



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='data/instruct_v2.json')
    parser.add_argument('--is_api', type=str, choices=['api','vllm'], default='vllm')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--out_name', type=str, default='tmp.json')
    parser.add_argument('--out_dir', type=str, default="work_dirs/")
    parser.add_argument('--model', type=str, help="path to huggingface model / api model name")
    parser.add_argument('--eval', type=str, choices=['instruct', 'reason', 'plan', 'retrieve', 'review', 'understand', 'rru'])
    parser.add_argument('--test_num', type=int, default=-1, help='number of samples to test, -1 means all')
    parser.add_argument('--prompt_type', type=str, default='json', choices=['json', 'str'])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--model_name', type=str, help="display name")
    args = parser.parse_args()
    return args

def load_dataset(dataset_path, out_dir, is_resume=False, tmp_folder_name='tmp'):
    dataset = mmengine.load(dataset_path)
    total_num = len(dataset)
    # possible filter here
    tested_num = 0
    if is_resume:
        file_list = os.listdir(os.path.join(out_dir, tmp_folder_name))
        for filename in file_list:
            if filename.split('.')[0] in dataset:
                tested_num += 1
                file_id = filename.split('.')[0]
                dataset.pop(file_id)
            else:
                print(f"Warning: {filename} not in dataset, remove it from cache")
                os.remove(os.path.join(out_dir, tmp_folder_name, filename))

    return dataset, tested_num, total_num

def infer(dataset, llm, out_dir, tmp_folder_name='tmp', test_num = 1, batch_size=10):
    random_list = list(dataset.keys())[:test_num]
    batch_infer_list = []; batch_infer_ids = []
    start_time = time.time()
    for idx in tqdm(random_list):
        prompt = dataset[idx]['origin_prompt']
        batch_infer_list.append(prompt)
        batch_infer_ids.append(idx)
        if len(batch_infer_ids) == batch_size or idx == random_list[-1]:
            predictions = llm.batch_generate(
                test_cases=batch_infer_list,
                max_concurrent_calls=batch_size,
                temperature=0  
            )
            for ptr, prediction in enumerate(predictions):
                data_ptr = batch_infer_ids[ptr]
                dataset[data_ptr]['prediction'] = prediction
                mmengine.dump(dataset[data_ptr], os.path.join(out_dir, tmp_folder_name, f'{data_ptr}.json'))
            batch_infer_ids = []; batch_infer_list = []

    # load results from cache
    results = dict()
    file_list = os.listdir(os.path.join(out_dir, tmp_folder_name))
    for filename in file_list:
        file_id = filename.split('.')[0]
        results[file_id] = mmengine.load(os.path.join(out_dir, tmp_folder_name, filename))
    return results


if __name__ == '__main__':
   from contextlib import nullcontext
   main()