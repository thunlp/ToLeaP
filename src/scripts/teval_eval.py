from lagent.llms.huggingface import HFTransformerCasualLM, HFTransformerChat
from lagent.llms.openai import GPTAPI
from dataclasses import asdict, dataclass, field
from typing import Any, Dict
import argparse
import mmengine
import os
from tqdm import tqdm
import shutil
import random
from collections import defaultdict
import json
from mmengine import load
import ast
import numpy as np
import openai
import argparse

def parse_string(template: str, input_string: str, allow_newline: bool=False) -> dict:
    """Return a dictionary whose keys are from input template and value is
    responding content from input_string.

    Args:
        template (str): Format template with keyword-only argument. For
            example '{who} like {what}'
        input_string (str): Input string will be parsed.
        allow_newline (boolen): Whether allow '\n' in {} during RE match, default to False.

    Returns:
        dict: Parsed data from input string according to format string. If
            input string doesn't match template, It will return None.

    Examples:
        >>> template = '{who} like {what}'
        >>> input_string = 'monkey like banana'
        >>> data = parse_string(template, input_string)
        >>> data
        >>> {'who': 'monkey', 'what': 'banana'}
        >>> input_string = 'monkey likes banana'
        >>> data = parse_string(template, input_string)
        >>> data
        >>> None
        >>> template = '{what} like {what}'
        >>> input_string = 'monkey like banana'
        >>> data = parse_string(template, input_string)
        >>> data
        >>> {'what': ['monkey', 'banana']}
    """

    formatter = Formatter()
    context = []
    keys = []
    for v in formatter.parse(template):
        # v is (literal_text, field_name, format_spec, conversion)
        if v[1] is not None:
            keys.append(v[1])
        context.append(v[0])
    pattern = template
    for k in keys:
        pattern = pattern.replace('{' + f'{k}' + '}', '(.*)')
    # pattern = re.compile(rf'{pattern}')
    values = re.findall(pattern, input_string, re.S if allow_newline else 0)
    if len(values) < 1:
        return None
    data = dict()
    for k, v in zip(keys, values[0]):
        if k in data:
            tmp = data[k]
            if isinstance(tmp, list):
                data[k].append(v)
            else:
                data[k] = [tmp, v]
        else:
            data[k] = v
    return data

def format_load(raw_data: str, start_character: str = '', end_character: str = ''):
    """Format the raw data into the format that can be evaluated.

    Args:
        raw_data (str): The raw data.
        start_character (str, optional): The start character. Defaults to '', if using it, the string will be sliced from the first start_character.
        end_character (str, optional): The end character. Defaults to '', if using it, the string will be sliced to the last end_character.

    Returns:
        str: The formatted data.
    """
    if type(raw_data) != str:
        # the data has been evaluated
        return raw_data
    if "```json" in raw_data:
        raw_data = raw_data[raw_data.find("```json") + len("```json"):]
        raw_data = raw_data.strip("`")
    if start_character != '':
        raw_data = raw_data[raw_data.find(start_character):]
    if end_character != '':
        raw_data = raw_data[:raw_data.rfind(end_character) + len(end_character)]
    successful_parse = False
    try:
        data = ast.literal_eval(raw_data)
        successful_parse = True
    except Exception as e:
        pass
    try:
        if not successful_parse:
            data = json.loads(raw_data)
        successful_parse = True
    except Exception as e:
        pass
    try:
        if not successful_parse:
            data = json.loads(raw_data.replace("\'", "\""))
        successful_parse = True
    except Exception as e:
        pass
    if not successful_parse:
        raise Exception("Cannot parse raw data")
    return data

@dataclass
class ResponseDataSample:
    """
    Args:
        template(str): Format string with keyword-only arguments. For
            example '{who} like {what}'
        pred(Any): Parsed data from LLM generating response.
        gt(Any): Ground truth data
        meta_data(dict, optional): Meta information will be used to evaluate
             LLM's response
    """
    template: str
    pred: Any
    gt: Any
    meta_data: dict = None
    
class InstructEvaluator:
    """Instruct Following Evaluation

    Args:
        dataset_path(str): File path of evaluation dataset.

    """

    def __init__(
        self,
        dataset_path: str,
        **kwargs,
    ) -> None:
        self.dataset_path = dataset_path

    def _load_dataset(self):
        self.dataset = []
        dataset = load(self.dataset_path)

        for key in dataset.keys():
            datum = dataset[key]
            data_sample = self._process_response(datum)
            
            self.dataset.append(
                dict(
                    origin_prompt=datum["origin_prompt"],
                    response_data_sample=data_sample))
        self.num_samples = len(self.dataset)

    def _process_response(
        self,
        datum: dict,
    ) -> ResponseDataSample:
        """Process the response to needed format.

        Args:
            datum(dict): inputs.

        Returns:
            dict: Processed response data sample.
        """

        # Dict with keyword-only arguments.
        template = datum['template']
        # Generated response.
        pred_data = datum['prediction']
        # Response of ground truth.
        gt_data = datum['ground_truth']
        meta_data = datum['meta_data']

        return ResponseDataSample(
            template=template, pred=pred_data, gt=gt_data, meta_data=meta_data)

    def _evaluate(self, data_sample: dict) -> dict:
        metrics_result = dict()
        response_format = data_sample.meta_data['response_format']
        if response_format == 'json':
            pred_data = self.json_format_parse(data_sample)
        else:
            pred_data = self.string_format_parse(data_sample)
        
        if pred_data is None:
            # directly set to 0 for all metrics
            metrics_result[f'{response_format}_format_metric'] = 0
            metrics_result[f'{response_format}_args_em_metric'] = 0
            return metrics_result

        # Exact matching
        metrics_result[f'{response_format}_format_metric'] = 1
        metrics_result[f'{response_format}_args_em_metric'] = self.compute_args_em_metric(
            gt_action=data_sample.gt['action'], pred_action=pred_data['action'],
            gt_args=data_sample.gt['args'], pred_args=pred_data['args']
        )
        return metrics_result
    
    def compute_args_em_metric(self, gt_action, pred_action, gt_args, pred_args):
        cnt = 0.
        if gt_action == pred_action:
            cnt += 1.
        num_args = len(gt_args) + 1     # 1 means action name match
        for gt_key in gt_args:
            pred_val = pred_args.get(gt_key, "")
            if pred_val == gt_args[gt_key]:
                cnt += 1.
        return cnt / num_args

    def string_format_parse(self, data_sample):
        pred_data = data_sample.pred
        template = data_sample.template
        thought_start = template['thought_start']
        thought_end = template['thought_end']
        action_start = template['action_start']
        action_end = template['action_end']
        args_start = template['args_start']
        args_end = template['args_end']

        parse_template = thought_start + "{thought}" + thought_end \
            + action_start + "{action}" + action_end \
            + args_start + "{args}" + args_end
        res = parse_string(parse_template, pred_data, allow_newline=True)
        try:
            if res is not None:
                args = ast.literal_eval(res['args'].strip())
                res['args'] = args if isinstance(args, dict) else {}
                res['action'] = res['action'].strip()
            return res
        except:
            return dict(thought=res['thought'], action=res['action'].strip(), args=dict())

    def json_format_parse(self, data_sample):
        try:
            pred_data = format_load(data_sample.pred)
            template = data_sample.template
            new_data = dict()
            new_data['thought'] = pred_data[template['thought']]
            new_data['action'] = pred_data[template['action']]
            args = pred_data[template['args']]
            new_data['args'] = args if isinstance(args, dict) else {}
        except Exception as e:
            return None

        return new_data

    def evaluate(self):
        self._load_dataset()
        results_list = []
        for data_sample in self.dataset:
            metrics_result = self._evaluate(data_sample['response_data_sample'])
            results_list.append(metrics_result)
        return self._post_process(results_list)

    def _post_process(self, results_list):
        # list of dict to dict of list
        results_dict = defaultdict(list)
        for sub in results_list:
            for key in sub:
                results_dict[key].append(sub[key])
        metric_list = ['json_format_metric', 'json_args_em_metric',
                       'string_format_metric', 'string_args_em_metric']
        for metric in metric_list:
            results_dict[metric] = np.round(np.mean(results_dict[metric]), decimals=4)
        return results_dict
    

meta_template_dict = dict(
    internlm = [
        dict(role='system', begin='<|System|>:', end='\n'),
        dict(role='user', begin='<|User|>:', end='\n'),
        dict(role='function', begin='<|System|>:', end='\n'),
        dict(
            role='assistant',
            begin='<|Bot|>:',
            end='<eoa>\n',
            generate=True)
    ],
    llama2 = [
        dict(role='system', begin='[INST]', end='[\INST]'),
        dict(role='user', begin='[INST]', end='[\INST]'),
        dict(role='function', begin='[INST]', end='[\INST]'),
        dict(role='assistant',
                begin='',
                end='</s>',
                generate=True),
    ],
    qwen = [
        dict(role='user', api_role='user', begin='\n<|im_start|>user\n', end='<|im_end|>'),
        dict(role='system', api_role='system', begin='\n<|im_start|>user\n', end='<|im_end|>'),
        dict(role='function', api_role='user', begin='\n<|im_start|>user\n', end='<|im_end|>'),
        dict(role='assistant',
                api_role='assistant',
                begin='\n<|im_start|>assistant\n',
                end='<|im_end|>',
                generate=True),
    ],
    vicuna = [
        dict(role='user', begin='user: ', end='\n'),
        dict(role='system', begin='user: ', end='\n'),
        dict(role='function', begin='user: ', end='\n'),
        dict(role='assistant',
                begin='assistant: ',
                end='\n',
                generate=True),
    ],
    llama3= [
        dict(role='system', begin='<|start_header_id|>system<|end_header_id|>', end='<|eot_id|>'),
        dict(role='user', begin='<|start_header_id|>user<|end_header_id|>', end='<|eot_id|>'),
        dict(role='function', begin='<|start_header_id|>function<|end_header_id|>', end='<|eot_id|>'),
        dict(role='assistant', begin='<|start_header_id|>assistant<|end_header_id|>', end='<|eot_id|>', generate=True)
    ],
    chatglm = [
        dict(role='system', api_role='user'),
        dict(role='user', api_role='user'),
        dict(role='function', api_role='user'),
        dict(role='assistant',
            api_role='assistant',
            generate=True)
    ],
    baichuan = [
        dict(role='user', begin='<reserved_106>', end='\n'),
        dict(role='system', begin='<reserved_106>', end='\n'),
        dict(role='function', begin='<reserved_106>', end='\n'),
        dict(role='assistant', begin='<reserved_107>', end='\n',
                generate=True),
    ]
)

# client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url="https://toollearning.cn/v1")
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', type=str, choices=['instruct', 'reason', 'plan', 'retrieve', 'review', 'understand', 'rru'])
    parser.add_argument("--dataset_path", type=str, default="../data/sft_data/stf_teval_ins.json")
    parser.add_argument('--model_display_name', type=str, default="")
    parser.add_argument('--out_name', type=str, default='tmp.json')
    parser.add_argument('--out_dir', type=str, default="work_dirs/")
    parser.add_argument('--model_path', type=str, help="path to huggingface model / api model name", default="/bjzhyai03/workhome/niuboye/model/llama3_instruct")
    parser.add_argument('--meta_template', type=str, default='llama3')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--model_type', type=str, choices=['api', 'hf'], default='hf')
    parser.add_argument('--test_num', type=int, default=-1, help='number of samples to test, -1 means all')
    args = parser.parse_args()
    return args

    

def load_dataset(dataset_path, out_dir, is_resume=False, tmp_folder_name='tmp'):
    raw_data = mmengine.load(dataset_path) 
    dataset = {}
    for idx, sample in enumerate(raw_data):
        sample['sample_id'] = sample.get('sample_id', str(idx))
        dataset[str(idx)] = sample 

    total_num = len(dataset)
    tested_num = 0
    if is_resume:
        file_list = os.listdir(os.path.join(out_dir, tmp_folder_name))
        for filename in file_list:
            file_id = filename.split('.')[0]
            if file_id in dataset:
                tested_num += 1
                dataset.pop(file_id)
            else:
                print(f"Warning: {filename} not in dataset, remove it from cache")
                os.remove(os.path.join(out_dir, tmp_folder_name, filename))
    
    return dataset, tested_num, total_num

def split_special_tokens(text):
    text = text.split('<eoa>')[0]
    text = text.split('<TOKENS_UNUSED_1>')[0]
    text = text.split('<|im_end|>')[0]
    text = text.split('\nuser')[0]
    text = text.split('\nassistant')[0]
    text = text.split('\nUSER')[0]
    text = text.split('[INST]')[0]
    text = text.split('<|user|>')[0]
    text = text.strip()
    if text.startswith('```json'):
        text = text[len('```json'):]
    text = text.strip('`').strip()
    return text

def infer(dataset, llm, out_dir, tmp_folder_name='tmp', test_num=1, batch_size=1):
    random_list = list(dataset.keys())[:test_num]
    batch_infer_list = []
    batch_infer_ids = []
    
    # 用于存储所有结果
    results = {}
    
    for idx in tqdm(random_list):
        prompt = ""
        if dataset[idx]['system']:
            system_prompt = dataset[idx]['system']
            prompt = f"{system_prompt}\n{prompt}"
            
        human_messages = [conv for conv in dataset[idx]['conversations'] 
                         if conv['from'] == 'human']
        if human_messages:
            prompt = f"{prompt}\n{human_messages[0]['value']}"
        else:
            print(f"Warning: No human message found in conversation {idx}")
            continue
        
        print(prompt)
        batch_infer_list.append(prompt)
        batch_infer_ids.append(idx)
        
        # Batch inference
        if len(batch_infer_ids) == batch_size or idx == random_list[-1]:
            try:
                predictions = llm.chat(batch_infer_list, do_sample=False)
                if not isinstance(predictions, list):
                    predictions = [predictions]
                
                # 确保预测结果和batch_ids长度匹配
                assert len(predictions) == len(batch_infer_ids), \
                    f"Predictions length ({len(predictions)}) != batch_infer_ids length ({len(batch_infer_ids)})"
                
                for ptr, prediction in enumerate(predictions):
                    if not isinstance(prediction, str):
                        prediction = str(prediction)
                        
                    prediction = split_special_tokens(prediction)
                    data_ptr = batch_infer_ids[ptr]
                    
                    # Create new conversation entry for model's prediction
                    new_conversation = {
                        'conversations': dataset[data_ptr]['conversations'] + [{
                            'from': 'assistant',
                            'value': prediction
                        }],
                        'system': dataset[data_ptr]['system'],
                        'tools': dataset[data_ptr]['tools']
                    }
                    
                    # 保存到结果字典
                    results[data_ptr] = new_conversation
                    
                    # Save the updated conversation
                    mmengine.dump(new_conversation, 
                                os.path.join(out_dir, tmp_folder_name, f'{data_ptr}.json'))
                
                batch_infer_ids = []
                batch_infer_list = []
                
            except Exception as e:
                print(f"Error in batch processing: {e}")
                print(f"Current batch_infer_ids: {batch_infer_ids}")
                print(f"Current batch_infer_list: {batch_infer_list}")
                continue
    
    # 不需要再次读取文件，直接返回内存中的结果
    return results



    results = {}
    file_list = os.listdir(os.path.join(out_dir, tmp_folder_name))
    for filename in file_list:
        file_id = filename.split('.')[0]
        results[file_id] = mmengine.load(os.path.join(out_dir, tmp_folder_name, filename))
    return results

    
if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tmp_folder_name = os.path.splitext(args.out_name)[0]
    os.makedirs(os.path.join(args.out_dir, tmp_folder_name), exist_ok=True)

    dataset, tested_num, total_num = load_dataset(args.dataset_path, args.out_dir, args.resume, tmp_folder_name=tmp_folder_name)
    test_num = (
        total_num - tested_num if args.test_num == -1
        else max(min(args.test_num - tested_num, total_num - tested_num), 0)
    )
    output_file_path = os.path.join(args.out_dir, args.out_name)

    if test_num != 0:
        if args.model_type == 'api':
            llm = GPTAPI(args.model_path)
        elif args.model_type == 'hf':
            meta_template = meta_template_dict.get(args.meta_template)
            if "chatglm" in args.model_display_name:
                llm = HFTransformerChat(path=args.model_path, meta_template=meta_template)
            else:
                llm = HFTransformerCasualLM(path=args.model_path, meta_template=meta_template, max_new_tokens=512)

        print(f"Tested {tested_num} samples, left {test_num} samples, total {total_num} samples")
        prediction = infer(dataset, llm, args.out_dir, tmp_folder_name=tmp_folder_name, test_num=test_num, batch_size=args.batch_size)
        mmengine.dump(prediction, os.path.join(args.out_dir, args.out_name))

    if args.eval:
        model_display_name = args.model_type if args.model_display_name == "" else args.model_display_name
        eval_mapping = dict(
            instruct="InstructEvaluator"
            # plan="PlanningEvaluator",
            # review="ReviewEvaluator",
            # reason="ReasonRetrieveUnderstandEvaluator",
            # retrieve="ReasonRetrieveUnderstandEvaluator",
            # understand="ReasonRetrieveUnderstandEvaluator",
            # rru="ReasonRetrieveUnderstandEvaluator"
        )
        bert_score_model = "all-mpnet-base-v2"
        json_path = os.path.join(args.out_dir, model_display_name + '_' + str(args.test_num) + '.json')

        evaluator_class = getattr(evaluator_factory, eval_mapping[args.eval])
        evaluator = evaluator_class(output_file_path, default_prompt_type=args.prompt_type, eval_type=args.eval, bert_score_model=bert_score_model)

        if os.path.exists(json_path):
            results = mmengine.load(json_path)
        else:
            results = dict()

        eval_results = evaluator.evaluate()
        print(eval_results)
        results[args.eval + '_' + args.prompt_type] = eval_results

        print(f"Writing Evaluation Results to {json_path}")
        mmengine.dump(results, json_path)
