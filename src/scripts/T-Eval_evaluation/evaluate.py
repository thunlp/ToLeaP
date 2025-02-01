import os
import mmengine
import click
from tqdm import tqdm
from evaluator import (
    InstructEvaluator, 
    PlanningEvaluator,
    ReviewEvaluator, 
    ReasonRetrieveUnderstandEvaluator
)
import json

@click.command()
@click.option("--model", type=str, help="模型名称,用于结果文件命名")
@click.option("--dataset_path", type=str, help="数据集路径列表，格式: [path1,path2,...]")
@click.option("--out_dir", type=str, default="work_dirs/")
@click.option("--out_name", type=str, help="输出文件名列表，格式: [name1,name2,...]")
@click.option("--eval", type=str, help="评估类型列表，格式: [type1,type2,...]")
@click.option("--prompt_type", type=str, help="提示类型列表，格式: [type1,type2,...]")
def main(
    model: str,
    dataset_path: str,
    out_dir: str,
    out_name: str,
    eval: str,
    prompt_type: str,
):
    # 解析列表形式的参数
    def parse_list_arg(arg):
        if arg and arg.startswith('[') and arg.endswith(']'):
            return [item.strip() for item in arg[1:-1].split(',')]
        return [arg]

    dataset_paths = parse_list_arg(dataset_path)
    out_names = parse_list_arg(out_name)
    eval_types = parse_list_arg(eval)
    prompt_types = parse_list_arg(prompt_type)

    # 验证参数长度一致
    if len(set(map(len, [dataset_paths, out_names, eval_types, prompt_types]))) > 1:
        raise ValueError("所有列表参数的长度必须相同")

    # 评估器映射
    eval_mapping = {
        'instruct': InstructEvaluator,
        'plan': PlanningEvaluator,
        'review': ReviewEvaluator,
        'reason': ReasonRetrieveUnderstandEvaluator,
        'retrieve': ReasonRetrieveUnderstandEvaluator,
        'understand': ReasonRetrieveUnderstandEvaluator,
        'rru': ReasonRetrieveUnderstandEvaluator
    }

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
    print("\n=== Starting Evaluation ===\n")

    for i, (curr_dataset, curr_out_name, curr_eval, curr_prompt) in enumerate(
        zip(dataset_paths, out_names, eval_types, prompt_types)
    ):
        print(f"\n=== Processing task {i+1}/{len(dataset_paths)} ===")
        print(f"Dataset: {curr_dataset}")
        print(f"Eval type: {curr_eval}")
        
        output_file_path = os.path.join(out_dir, curr_out_name)
        
        # 检查inference结果是否存在
        if not os.path.exists(output_file_path):
            print(f"Error: Inference result not found at {output_file_path}")
            continue

        # 评估
        bert_score_model = "/bjzhyai03/workhome/chenhaotian/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/9a3225965996d404b775526de6dbfe85d3368642"
        json_path = os.path.join(out_dir, f"{model}_{-1}_{'zh' if '_zh' in curr_dataset else ''}.json")
        
        evaluator = eval_mapping[curr_eval](
            dataset_path=output_file_path,
            bert_score_model=bert_score_model,
            default_prompt_type=curr_prompt,
            eval_type=curr_eval
        )
        
        results = mmengine.load(json_path) if os.path.exists(json_path) else {}
        eval_results, failed_cases = evaluator.evaluate()
        results[f"{curr_eval}_{curr_prompt}"] = eval_results
        
        print(f"Evaluation Results:\n{eval_results}")
        print(f"Writing Results to {json_path}")
        mmengine.dump(results, json_path)
        
        badcase_path = os.path.join(out_dir, f"{model}-teval-{curr_eval}.json")
        with open(badcase_path, 'w') as f:
            json.dump(failed_cases, f, indent=2)
        
        print(f"\nFailed Cases Analysis for {curr_eval}_{curr_prompt}:")
        for fail_type, cases in failed_cases.items():
            print(f"{fail_type}: {len(cases)} cases")

if __name__ == '__main__':
    main() 