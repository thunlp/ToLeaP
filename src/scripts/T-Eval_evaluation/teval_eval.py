import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import argparse
import os
import mmengine
from tqdm import tqdm
import click
import sys
import time
current_dir = os.path.dirname(os.path.abspath(__file__)) 
utils_dir = os.path.join(current_dir, '../..')
sys.path.append(utils_dir)
from utils.llm import LLM
from concurrent.futures import ThreadPoolExecutor, as_completed


class TevalLLM(LLM):
    def __init__(self, max_tokens=512, *args, **kwargs): 
        super().__init__(max_output_tokens=max_tokens, *args, **kwargs)


@click.command()
@click.option("--model", type=str, default="/hy-tmp/model-dir")
@click.option("--dataset_path", type=str, help="Dataset path list, format: [path1, path2, ...]")
@click.option("--is_api", type=bool, default= False)
@click.option("--out_dir", type=str, default="work_dirs/")
@click.option("--out_name", type=str, help="Output filename list, format: [name1, name2, ...]")
@click.option("--eval", type=str, help="Evaluation type list, format: [type1, type2, ...]")
@click.option("--prompt_type", type=str, help="Prompt type list, format: [type1, type2, ...]")
@click.option("--tensor_parallel_size", type=int, default=8)
@click.option("--batch_size", type=int, default=200)
@click.option("--gpu_memory_utilization", type=float, default=0.9) 
@click.option("--test_num", type=int, default=-1)
@click.option("--resume", is_flag=True)
@click.option("--model_name", type=str, default="")
@click.option("--special_tokens", is_flag=True, help="Whether to use special tokens")
@click.option("--selected_special_tokens", type=str, default=None, help="Selected special tokens")
def main(
    model: str,
    dataset_path: str,
    is_api: bool,
    out_dir: str,
    out_name: str,
    eval: str,
    prompt_type: str,
    tensor_parallel_size: int,
    batch_size: int,
    gpu_memory_utilization: float,
    test_num: int,
    resume: bool,
    special_tokens: bool,
    selected_special_tokens: str,
    model_name: str
):
    def parse_list_arg(arg):
        if arg and arg.startswith('[') and arg.endswith(']'):
            return [item.strip() for item in arg[1:-1].split(',')]
        return [arg]

    dataset_paths = parse_list_arg(dataset_path)
    out_names = parse_list_arg(out_name)
    eval_types = parse_list_arg(eval)
    prompt_types = parse_list_arg(prompt_type)

    if len(set(map(len, [dataset_paths, out_names, eval_types, prompt_types]))) > 1:
        raise ValueError("The length of all list parameters must be the same.")

    os.makedirs(out_dir, exist_ok=True)

    print("\n=== Task Overview ===")
    print("Total tasks:", len(dataset_paths))
    for i, (d_path, o_name, e_type, p_type) in enumerate(
        zip(dataset_paths, out_names, eval_types, prompt_types)
    ):
        print(f"\nTask {i+1}:")
        print(f"  Dataset: {d_path}")
        print(f"  Output:  {o_name}")
        print(f"  Eval:    {e_type}")
        print(f"  Prompt:  {p_type}")
    print("\n=== Starting Tasks ===\n")

    llm = TevalLLM(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        is_api=is_api,
        batch_size= batch_size
    )

    for i, (curr_dataset, curr_out_name, curr_eval, curr_prompt) in enumerate(
        zip(dataset_paths, out_names, eval_types, prompt_types)
    ):
        print(f"\n=== Processing task {i+1}/{len(dataset_paths)} ===")
        print(f"Dataset: {curr_dataset}")
        print(f"Eval type: {curr_eval}")
        
        tmp_folder_name = os.path.splitext(curr_out_name)[0]
        os.makedirs(os.path.join(out_dir, tmp_folder_name), exist_ok=True)

        dataset, tested_num, total_num = load_dataset(curr_dataset, out_dir, resume, tmp_folder_name)
        test_num_curr = max(total_num - tested_num, 0) if test_num == -1 else max(min(test_num - tested_num, total_num - tested_num), 0)
        
        if test_num_curr > 0:
            print(f"Tested {tested_num} samples, left {test_num_curr} samples, total {total_num} samples")
            
            output_file_path = os.path.join(out_dir, curr_out_name)
            prediction = infer(dataset, llm, out_dir, tmp_folder_name, test_num_curr, batch_size)
            mmengine.dump(prediction, output_file_path)

            if curr_eval:
                eval_mapping = {
                    'instruct': "InstructEvaluator",
                    'plan': "PlanningEvaluator", 
                    'review': "ReviewEvaluator",
                    'reason': "ReasonRetrieveUnderstandEvaluator",
                    'retrieve': "ReasonRetrieveUnderstandEvaluator",
                    'understand': "ReasonRetrieveUnderstandEvaluator",
                    'rru': "ReasonRetrieveUnderstandEvaluator"
                }
                
                bert_score_model = "/data3/models/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/9a3225965996d404b775526de6dbfe85d3368642"
                json_path = os.path.join(out_dir, f"{model.split('/')[-1]}_{-1}_{'zh' if '_zh' in curr_dataset else ''}.json")

                
                results = mmengine.load(json_path) if os.path.exists(json_path) else {}
                mmengine.dump(results, json_path)


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

def infer(dataset, llm, out_dir, tmp_folder_name='tmp', test_num=1, batch_size=10):
    random_list = list(dataset.keys())[:test_num]
    batch_infer_list = []; batch_infer_ids = []
    start_time = time.time()
    
    for idx in tqdm(random_list):
        prompt = dataset[idx]['origin_prompt']
        batch_infer_list.append(prompt)
        batch_infer_ids.append(idx)
        
        if len(batch_infer_ids) == batch_size or idx == random_list[-1]:
            try:
                predictions = llm.batch_generate_chat(test_cases=batch_infer_list)
                
                for ptr, prediction in enumerate(predictions):
                    if ptr < len(batch_infer_ids):
                        data_ptr = batch_infer_ids[ptr]
                        dataset[data_ptr]['prediction'] = prediction
                        mmengine.dump(dataset[data_ptr], 
                            os.path.join(out_dir, tmp_folder_name, f'{data_ptr}.json'))
                
                batch_infer_ids = []; batch_infer_list = []
            except Exception as e:
                print(f"处理批次时出错: {e}")
                import traceback
                traceback.print_exc()
    
    # 加载结果
    results = {}
    file_list = os.listdir(os.path.join(out_dir, tmp_folder_name))
    for filename in file_list:
        file_id = filename.split('.')[0]
        if file_id in dataset:
            results[file_id] = mmengine.load(os.path.join(out_dir, tmp_folder_name, filename))
    
    return results


if __name__ == '__main__':
   main()