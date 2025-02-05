from dataclasses import asdict, dataclass, field
from typing import Any, Dict
import ast
import json
import re
from string import Formatter
from collections import defaultdict
from mmengine import load
import numpy as np
from numpy import mean
import itertools
import networkx as nx
import copy
from tqdm import tqdm

from collections import defaultdict
import numpy as np
import copy


from sentence_transformers import SentenceTransformer, util
from termcolor import colored



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




def format_string(template: str, input_data: dict) -> str:
    """Return string with input content according input format template.

    Args:
        template (str): Format string with keyword-only argument. For
            example '{who} like {what}'
        input_data (dict): Input data to fill in the input template.

    Returns:
        str: Return string.
    """

    return template.format(**input_data)


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
        failed_cases = {
            'format_parse_failed': [],
            'format_metric_zero': [],
            'args_not_match': []
        }
        
        for idx, data_sample in enumerate(self.dataset):
            metrics_result = self._evaluate(data_sample['response_data_sample'])
            results_list.append(metrics_result)
            
            case_info = {
                'case_id': idx,
                'origin_prompt': data_sample['origin_prompt'],
                'prediction': data_sample['response_data_sample'].pred,
                'ground_truth': data_sample['response_data_sample'].gt,
                'template': data_sample['response_data_sample'].template,
                'meta_data': data_sample['response_data_sample'].meta_data,
                'metrics': metrics_result
            }
            
            try:
                parsed_pred = self.json_format_parse(data_sample['response_data_sample']) or self.string_format_parse(data_sample['response_data_sample'])
                if parsed_pred is None:
                    failed_cases['format_parse_failed'].append(case_info)
                    continue
                    
                # 检查所有format_metric和args_em_metric
                format_metrics = [v for k, v in metrics_result.items() if k.endswith('_format_metric')]
                args_metrics = [v for k, v in metrics_result.items() if k.endswith('_args_em_metric')]
                
                if any(m == 0 for m in format_metrics):
                    case_info['format_metrics'] = format_metrics
                    failed_cases['format_metric_zero'].append(case_info)
                    continue
                    
                if any(m != 1 for m in args_metrics):
                    case_info['args_metrics'] = args_metrics
                    failed_cases['args_not_match'].append(case_info)
                    
            except Exception as e:
                failed_cases['format_parse_failed'].append(case_info)

        return self._post_process(results_list), failed_cases

    def _post_process(self, results_list):
        # list of dict to dict of list
        results_dict = defaultdict(list)
        {
            results_dict[key].append(sub[key])
            for sub in results_list for key in sub
        }
        metric_list = ['json_format_metric', 'json_args_em_metric',
                       'string_format_metric', 'string_args_em_metric']
        for metric in metric_list:
            results_dict[metric] = np.round(np.mean(results_dict[metric]), decimals=4)
        return results_dict
    


class PlanningEvaluator:
    """Planning Evaluation
    Args:
        dataset_path(str): File path of evaluation dataset
        name_weight(float): the weight of action_name in bert_score match, default = 0.9
        args_weight(float): the weight of action_args in bert_score match, default = 0.1
        match_threshold(float): the threshold of matching
        match_strategy(str): matching method, can choose 'bertscore' or 'permutation' 
        bert_score_model(str): the bert_score model for sentence similarity, default = "all-mpnet-base-v2". 
            Refer to https://www.sbert.net/docs/pretrained_models.html for more models.
    """
    def __init__(
        self,
        dataset_path: str,
        name_weight = 0.75,
        args_weight = 0.25,
        match_threshold = 0.8,
        match_strategy: str = 'bertscore', # ["bertscore", "permutation"]
        bert_score_model: str = "all-mpnet-base-v2", # ['thenlper/gte-large-zh', 'all-mpnet-base-v2']
        default_prompt_type: str = 'json', # ["json", "ReWOO"]
        **kwargs,
    ) -> None:
        self.bert_score_model = bert_score_model
        print(bert_score_model)
        self.dataset_path = dataset_path
        self.name_weight = name_weight
        self.args_weight = args_weight
        self.match_threshold = match_threshold
        self.default_prompt_type = default_prompt_type # ["json", "ReWOO"]
        assert match_strategy in ["bertscore", "permutation"], f"match strategy must in [\"bertscore\", \"permutation\"], but get {match_strategy}"
        self.match_strategy = match_strategy
        self.valid_data_count = None
        self.sentence_model = SentenceTransformer(self.bert_score_model)

    def _load_dataset(self):
        self.dataset = []
        dataset = load(self.dataset_path)
        total_error = 0
        total_count = 0
        for key in dataset.keys():
            datum = dataset[key]
            data_sample, error = self._process_response(datum)
            total_error += error
            total_count += 1
            self.dataset.append(
                dict(response_data_sample=data_sample))

        self.num_samples = len(self.dataset)
        print("total_data_count:", total_count, "valid_data_count:", total_count - total_error)
        self.valid_data_count = total_count - total_error

    def format_load(self, data):
        r'''
            ensure evaluator can work correctly under any data input
        '''
        try:
            json_format = format_load(data, start_character='[', end_character=']')
        except Exception as e:
            return []
        if type(json_format) != list:
            return []
        for i in range(len(json_format)):
            try:
                json_format[i] = {
                    'name': str(json_format[i]['name']),
                    'id': int(json_format[i]['id']),
                    'args': str(json_format[i]['args'])
                }
            except Exception as e:
                return []
        return json_format

    def _process_response(
        self,
        datum,
    ) -> ResponseDataSample:
        """Process the response to needed format.
        Args:
            datum(dict): inputs.
        Returns:
            dict: Processed response data sample.
        """

        # Generated response, which can be a string or list
        pred_data = datum['prediction']
        # Response of ground truth, which can be a string or list
        gt_data = datum['ground_truth']
        # prompt_type: The type of planning prompt, supporting "json" and "ReWOO"
        if "meta" in datum:
            prompt_type = datum["meta"].get("prompt_type", self.default_prompt_type)
        else:
            prompt_type = self.default_prompt_type

        error = 0
        pred = dict()
        gt = dict()
        gt['planning'] = self.format_load(gt_data)
        if prompt_type == 'json':
            pred['planning'] = self.format_load(pred_data)
            if pred['planning'] == [] or gt['planning'] == []:
                error = 1

        elif prompt_type == 'ReWOO':
            """
                This type is deprecated
                The planning prediction data should in this format:
                    Plan 1: <str> description about the first action
                    Dependency 1: <list[number]> the first action depends on which previous actions
                    Action 1: #E1 = api_name1(args1)
                    ...
                Which will be passed only if "number of plan lines == number of dependency lines == number of action lines"
                The passed data's format is:
                    [
                        dict(
                            id = i,
                            name = curr_name,
                            args = args_str
                        )
                        ...
                    ]

                The golden answer prediction is a json that is the same as the json format.
            """
            thoughts = re.findall(r'(Plan [0-9]+: .+)', pred_data)
            dependencies = re.findall(r'(Dependency [0-9]+: .+)', pred_data)
            action_units = re.findall(r'Action [0-9]+: (.+)', pred_data)
            
            if not (len(thoughts) == len(dependencies) and len(thoughts) == len(action_units)):
                pred['planning'] = []
                gt['planning'] = []
                return ResponseDataSample(template = '', pred=pred, gt=gt), 1

            plan_action = []
            for i in range(len(action_units)):
                dependency_list = re.findall(r'Dependency [0-9]+: (.+)', dependencies[i])
                if action_units[i][0] == '#': 
                    # The action has a return #E
                    args_str_list = re.findall(r'#E[0-9]+ = .+\((.+)\)', action_units[i])
                    name_list = re.findall(r'#E[0-9]+ = (.+)\(', action_units[i])
                else: 
                    # The action does not have a return
                    args_str_list = re.findall(r'.+\((.+)\)', action_units[i])
                    name_list = re.findall(r'(.+)\(', action_units[i])
                if (len(name_list) > 0): 
                    curr_name = name_list[0]
                else: 
                    curr_name = ""
                if (len(args_str_list) > 0): 
                    args_str = "{" + args_str_list[0] + "}"
                else: 
                    args_str = "{}"
                if (len(dependency_list) > 0): 
                    dependency_str = dependency_list[0]
                else: 
                    dependency_str = ""
                dependency = re.findall('([0-9]+)', dependency_str)
                dependency = list(set([int(x) - 1 for x in dependency]))
                plan_action.append(
                    dict(
                        id = i,
                        name = curr_name,
                        prev = dependency,
                        args = args_str
                    ))
            pred['planning'] = plan_action
            #Turn dict into args str
            for i in range(len(gt['planning'])):
                args_str = ""
                if type(gt['planning'][i]['args']) == str:
                    args_dict = eval(gt['planning'][i]['args'])
                else:
                    assert type(gt['planning'][i]['args']) == dict
                    args_dict = gt['planning'][i]['args']
                for it in args_dict:
                    if args_str == "": args_str += f"{it}=\"{args_dict[it]}\""
                    else: args_str += f", {it}=\"{args_dict[it]}\""
                gt['planning'][i]['args'] = '{' + args_str + '}'

        elif prompt_type == 'str':
            pred_data_format = pred_data.replace('. ', '\n').split('\n')
            pred_actions = []
            for pred_step in pred_data_format:
                first_occur_time = 1e9
                pred_action = ""
                for api_name in datum['meta']['API_list']:
                    occur_time = pred_step.find(api_name)
                    if occur_time != -1 and occur_time < first_occur_time:
                        first_occur_time = occur_time
                        pred_action = api_name
                if pred_action != "":
                    pred_actions.append({
                        'id': len(pred_actions),
                        'name': pred_action,
                        'args': pred_step
                    })
            pred['planning'] = pred_actions
            if len(pred['planning']) == 0:
                error = 1
        else:
            raise NotImplementedError(f"Currently, we only support json and ReWOO format, but get {prompt_type}")

        return ResponseDataSample(template = '', pred=pred, gt=gt), error

    def _evaluate(self, data_sample) -> dict:
        if self.match_strategy == 'bertscore':
            metrics_result = self.bertscore_match(
                data_sample.pred['planning'], data_sample.gt['planning'])
        elif self.match_strategy == 'permutation':
            metrics_result = self.permutation_match(
                data_sample.pred['planning'], data_sample.gt['planning'])
        else:
            raise NotImplementedError
        if len(data_sample.pred['planning']) == 0 or len(data_sample.gt['planning']) == 0:
            metrics_result['parse_rate'] = 0
        else:
            metrics_result['parse_rate'] = 1
        return metrics_result

    def evaluate(self):
        self._load_dataset()
        results_list = []
        failed_cases = {
            'parse_failed': [],
            'low_f1': [],
            'low_precision': [],
            'low_recall': []
        }
        
        for idx, data_sample in enumerate(self.dataset):
            metrics_result = self._evaluate(data_sample['response_data_sample'])
            results_list.append(metrics_result)
            
            case_info = {
                'case_id': idx,
                'prediction': data_sample['response_data_sample'].pred,
                'ground_truth': data_sample['response_data_sample'].gt,
                'metrics': metrics_result
            }
            
            if metrics_result['parse_rate'] == 0:
                failed_cases['parse_failed'].append(case_info)
            if metrics_result['f1_score'] < 0.5:
                failed_cases['low_f1'].append(case_info)
            if metrics_result['precision'] < 0.5:
                failed_cases['low_precision'].append(case_info)
            if metrics_result['recall'] < 0.5:
                failed_cases['low_recall'].append(case_info)
                
        return self._post_process(results_list), failed_cases

    def permutation_match(self, pred_plan, gt_plan) -> dict:
        '''
            The function calculates all the permutation matches' score and selects the max f1_score;
            Since permutation is time consuming, we truncate the length of plans to 9
        '''
        if pred_plan[-1]['name'] != 'FinishAction':
            pred_plan.append(
                {'id': len(pred_plan), 'prev': [], 'name': 'FinishAction', 'args': r'\{\}'}
            )
        
        if gt_plan[-1]['name'] != 'FinishAction':
            gt_plan.append(
                {'id': len(gt_plan), 'prev': [], 'name': 'FinishAction', 'args': r'\{\}'}
            )

        # truncate plans to 9 since it is too long for permutation.
        if len(pred_plan) > 9: pred_plan = pred_plan[:9]
        if len(gt_plan) > 9: gt_plan = pred_plan[:9]

        pred_plan = sorted(pred_plan, key=lambda x: x['id'])
        gt_plan = sorted(gt_plan, key=lambda x: x['id'])
        len_pred = len(pred_plan)
        len_gt = len(gt_plan)
        map_id_max = max(len_pred, len_gt)
        numbers = [i for i in range(map_id_max)]
        perms = itertools.permutations(numbers, len_pred)
        gt_prev_count, pred_prev_count = 0, 0
        for i in range(len_gt):
            gt_plan[i]['prev'].append(i)
            gt_prev_count += len(gt_plan[i]['prev'])
        for i in range(len_pred):
            pred_plan[i]['prev'].append(i)
            pred_prev_count += len(pred_plan[i]['prev'])
        if gt_prev_count == 0 or pred_prev_count == 0:
            return {
                'precision': 0,
                'recall': 0,
                'f1_score': 0
            }
        max_recall, max_precision, max_f1 = 0, 0, 0
        for perm in perms:
            correct_count = 0
            for i in range(len_pred):
                if perm[i] >= len_gt: 
                    continue
                for j in pred_plan[i]['prev']:
                    if perm[j] in gt_plan[perm[i]]['prev']:
                        correct_count += 1
            now_recall, now_precision = correct_count / gt_prev_count, correct_count / pred_prev_count
            if now_recall + now_precision == 0: 
                continue
            now_f1 = 2 * now_recall * now_precision / (now_recall + now_precision)
            if now_f1 > max_f1:
                max_f1, max_recall, max_precision = now_f1, now_recall, now_precision
        return {
            'precision': max_precision,
            'recall': max_recall,
            'f1_score': max_f1
        }

    def bertscore_match(self, pred_plan, gt_plan) -> dict:
        """
            Calculate the similarity between predicted plan and golden answer,
            A plan can be regarded a sequence of actions, and each action has a name and args.
            Firstly, use bertscore to calculate pointwise similarity by:
                similarity(u, v) = bertscore(u.name, v.name) * name_weight + bertscore(u.args, v.args) * args_weight;
            Secondly, use Hungarian matching to match the points;
            Finally, use LIS to calculate the number of matched nodes.
        """
        if len(pred_plan) == 0 or len(gt_plan) == 0:
            return {
                'precision': 0,
                'recall': 0,
                'f1_score': 0
            }

        pred_plan = copy.deepcopy(sorted(pred_plan, key=lambda x: x['id']))
        gt_plan = copy.deepcopy(sorted(gt_plan, key=lambda x: x['id']))

        #Add end action
        #Currently it is hard-code
        if pred_plan[-1]['name'] == 'FinishAction':
            pred_plan = pred_plan[:-1]
        if gt_plan[-1]['name'] == 'FinishAction':
            gt_plan = gt_plan[:-1]
        #The total counts of nodes and edges.
        len_pred = len(pred_plan)
        len_gt = len(gt_plan)

        bert_score_matrix = np.zeros((len_pred, len_gt))
        name_pred, args_pred = [], []
        name_gt, args_gt = [], []
        for i in range(len_pred):
            name_pred.append(pred_plan[i]['name'])
            args_pred.append(str(pred_plan[i]['args']))
        for i in range(len_gt):
            name_gt.append(gt_plan[i]['name'])
            args_gt.append(str(gt_plan[i]['args']))
        name_pred_emb = self.sentence_model.encode(name_pred, convert_to_tensor=True)
        name_gt_emb = self.sentence_model.encode(name_gt, convert_to_tensor=True)
        args_pred_emb = self.sentence_model.encode(args_pred, convert_to_tensor=True)
        args_gt_emb = self.sentence_model.encode(args_gt, convert_to_tensor=True)
        name_cosine_scores = np.maximum(util.cos_sim(name_pred_emb, name_gt_emb).cpu().numpy(), 0)
        args_cosine_scores = np.maximum(util.cos_sim(args_pred_emb, args_gt_emb).cpu().numpy(), 0)
        for i in range(len_pred):
            for j in range(len_gt):
                bert_score_matrix[i][j] = \
                    name_cosine_scores[i][j] * self.name_weight \
                    + args_cosine_scores[i][j] * self.args_weight
        G = nx.Graph()
        for i in range(len_pred):
            for j in range(len_gt):
                if bert_score_matrix[i][j] > self.match_threshold:
                    G.add_edge(i, str(j), weight=bert_score_matrix[i][j])
        max_weight_matching = nx.max_weight_matching(G)

        pred_to_gt_mapping = dict()
        for key in max_weight_matching:
            if type(key[0]) == int:
                pred_to_gt_mapping[int(key[0])] = int(key[1])
            else:
                pred_to_gt_mapping[int(key[1])] = int(key[0])

        #If a prediction node does not match any golden answer node, we mark the node as -1.
        for i in range(len_pred):
            if i not in pred_to_gt_mapping:
                pred_to_gt_mapping[i] = -1
        #Calculate how many nodes are matched by Longest Increasing Subsequence (LIS)
        dp = np.ones(len_pred)
        for i in range(len_pred):
            for j in range(i):
                if pred_to_gt_mapping[i] == -1 or pred_to_gt_mapping[j] == -1:
                    continue
                if pred_to_gt_mapping[i] > pred_to_gt_mapping[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
        correct_count = int(max(dp))

        recall, precision = correct_count / len(gt_plan), correct_count / len(pred_plan)
        f1_score = 2 * recall * precision / (recall + precision)
        result = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
        return result

    def _post_process(self, results_list):
        # list of dict to dict of list
        results = dict()
        planning_metric_keys = ["precision", "recall", "f1_score", 'parse_rate']
        for key in planning_metric_keys:
            results[key] = mean([result[key] for result in results_list])
        return results



class ReasonRetrieveUnderstandEvaluator:
    """Planning Evaluation
    Args:
        dataset_path(str): File path of evaluation dataset
        bert_score_model(str): the bert_score model for sentence similarity, default = "all-mpnet-base-v2". 
            Refer to https://www.sbert.net/docs/pretrained_models.html for more models.
    """
    def __init__(
        self,
        dataset_path: str,
        bert_score_model: str = "all-mpnet-base-v2", # ['thenlper/gte-large-zh', 'all-mpnet-base-v2']
        default_prompt_type: str = 'json',
        eval_type: str = 'reason',
        **kwargs,
    ) -> None:
        self.bert_score_model = bert_score_model
        print(bert_score_model)
        self.dataset_path = dataset_path
        # self.bertscore = evaluate.load('bertscore')
        self.default_prompt_type = default_prompt_type # ["json", "str"]
        self.eval_type = eval_type
        self.valid_data_count = None
        self.sentence_model = SentenceTransformer(self.bert_score_model)

    def _load_dataset(self):
        self.dataset = []
        dataset = load(self.dataset_path)
        total_error = 0
        total_count = 0
        for key in dataset.keys():
            datum = dataset[key]
            data_sample, error = self._process_response(datum)
            total_error += error
            total_count += 1
            self.dataset.append(
                dict(response_data_sample=data_sample))

        self.num_samples = len(self.dataset)
        # print("total_data_count:", total_count, "valid_data_count:", total_count - total_error)
        self.valid_data_count = total_count - total_error

    def format_load(self, data):
        r'''
            ensure evaluator can work correctly under any data input
        '''
        try:
            json_format = format_load(data, start_character='{', end_character='}')
        except Exception as e:
            return {}
        if type(json_format) != dict:
            return {}
        prepared_json_format = dict()
        try:
            prepared_json_format['thought'] = str(json_format['thought'])
        except Exception as e:
            prepared_json_format['thought'] = ''
        try:
            prepared_json_format['name'] = str(json_format['name'])
        except Exception as e:
            prepared_json_format['name'] = ''

        if self.default_prompt_type == 'json':
            try:
                if isinstance(json_format['args'], dict):
                    prepared_json_format['args'] = json_format['args']
                else:
                    prepared_json_format['args'] = dict()
            except:
                prepared_json_format['args'] = dict()
        else:
            try:
                prepared_json_format['args'] = str(json_format['args'])
            except Exception as e:
                prepared_json_format['args'] = ""
        
        return prepared_json_format

    def _process_response(
        self,
        datum,
    ) -> ResponseDataSample:
        """Process the response to needed format.
        Args:
            datum(dict): inputs.
        Returns:
            dict: Processed response data sample.
        """

        # Generated response, which can be a string or list
        pred_data = datum['prediction']
        # Response of ground truth, which can be a string or list
        gt_data = datum['ground_truth']
        # prompt_type: The type of planning prompt, supporting "json" and "ReWOO"
        if "meta_data" in datum:
            prompt_type = datum["meta_data"].get("response_format", self.default_prompt_type)
        else:
            prompt_type = self.default_prompt_type

        error = 0
        gt = self.format_load(gt_data)
        
        if prompt_type == 'json':
            pred = self.format_load(pred_data)
            if pred == {} or gt == {}:
                error = 1
        elif prompt_type == 'str':
            # choose the first line
            pred = dict()
            if self.eval_type == 'reason':
                pred['thought'] = pred_data
            if self.eval_type == 'retrieve':
                pred['name'] = pred_data
            if self.eval_type == 'understand':
                pred['args'] = pred_data
        else:
            raise NotImplementedError(f"Currently, we only support json and str format, but get {prompt_type}")

        if error == 1:
            pred = dict()
        return ResponseDataSample(template = '', pred=pred, gt=gt), error

    def _evaluate(self, data_sample):
        """Evaluate the response data sample."""
        # 直接返回data_sample，评估逻辑放在post_process中
        return data_sample
        
    def evaluate(self):
        self._load_dataset()
        results_list = []
        failed_cases = {
            'parse_failed': [],
            'low_thought': [],
            'wrong_name': [],
            'low_args': []
        }
        
        # 先收集所有response_data
        for idx, data_sample in enumerate(self.dataset):
            response_data = self._evaluate(data_sample['response_data_sample'])
            results_list.append(response_data)
            
        # 计算评估指标
        metrics_results = []
        batch_data = []
        batch_id = []
        
        # 计算每个case的指标
        for idx, data in enumerate(results_list):
            curr_metrics = {'parse_rate': 0}
            if len(data.pred.keys()) != 0:
                curr_metrics['parse_rate'] = 1
            
            # 收集thought数据用于batch计算
            if 'thought' in data.pred and 'thought' in data.gt:
                if data.pred['thought'] is None or data.gt['thought'] is None:
                    curr_metrics['thought'] = 0  # 如果是None就直接给0分
                else:
                    batch_data.extend([data.pred['thought'], data.gt['thought']])
                    batch_id.append(idx)
            
            # 计算name
            if 'name' in data.pred and 'name' in data.gt:
                if self.default_prompt_type == 'json':
                    if data.pred['name'] is None:
                        curr_metrics['name'] = 0
                    else:
                        curr_metrics['name'] = 1 if data.pred['name'] == data.gt['name'] else 0
                else:
                    if data.pred['name'] is None:
                        curr_metrics['name'] = 0
                    else:
                        if data.gt['name'] not in data.pred['name']:
                            curr_metrics['name'] = 0
                        else:
                            curr_metrics['name'] = 1
                            find_all_name = self.find_a_dot_b_structure(data.pred['name']) + self.find_FinishAction(data.pred['name'])
                            for name in find_all_name:
                                if name != data.gt['name']:
                                    curr_metrics['name'] = 0
            
            # 计算args
            if 'args' in data.pred and 'args' in data.gt:
                if isinstance(data.gt['args'], dict):
                    # 计算args的precision, recall, f1
                    gt_num_keys = len(data.gt['args'].keys())
                    pred_num_keys = len(data.pred['args'].keys())
                    
                    if pred_num_keys == 0 and gt_num_keys == 0:
                        curr_metrics['args_precision'] = 1
                        curr_metrics['args_recall'] = 1
                        curr_metrics['args_f1_score'] = 1
                    elif pred_num_keys == 0 or gt_num_keys == 0:
                        curr_metrics['args_precision'] = 0
                        curr_metrics['args_recall'] = 0
                        curr_metrics['args_f1_score'] = 0
                    else:
                        correct_count = 0
                        for gt_arg_name in data.gt['args']:
                            if gt_arg_name in data.pred['args'] and str(data.pred['args'][gt_arg_name]) == str(data.gt['args'][gt_arg_name]):
                                correct_count += 1
                        curr_metrics['args_precision'] = correct_count / pred_num_keys
                        curr_metrics['args_recall'] = correct_count / gt_num_keys
                        if curr_metrics['args_precision'] + curr_metrics['args_recall'] == 0:
                            curr_metrics['args_f1_score'] = 0
                        else:
                            curr_metrics['args_f1_score'] = 2 * curr_metrics['args_precision'] * curr_metrics['args_recall'] / \
                                (curr_metrics['args_precision'] + curr_metrics['args_recall'])
                    
                    # 如果是json格式，还要计算整体args分数
                    if self.default_prompt_type == 'json':
                        curr_metrics['args'] = curr_metrics['args_f1_score']
                else:
                    if data.pred['args'] is None:
                        curr_metrics['args'] = 0
                    else:
                        data.pred['args'] = data.pred['args'].strip("'").strip('"')
                        curr_metrics['args'] = float(data.gt['args'] == data.pred['args'])
                
            metrics_results.append(curr_metrics)
        
        # 批量计算thought相似度
        if len(batch_data) > 0:
            pred_emb = self.sentence_model.encode(batch_data, convert_to_tensor=True)
            for i in range(0, len(batch_data), 2):
                cosine_score = float(np.maximum(util.cos_sim(pred_emb[i], pred_emb[i+1]).cpu().numpy(), 0)[0, 0])
                metrics_results[batch_id[i // 2]]['thought'] = cosine_score
            
        # 收集badcases
        for idx, response_data in enumerate(results_list):
            case_info = {
                'case_id': idx,
                'prediction': response_data.pred,
                'ground_truth': response_data.gt,
                'metrics': metrics_results[idx]
            }
            
            if not response_data.pred:
                failed_cases['parse_failed'].append(case_info)
            elif 'thought' in metrics_results[idx] and metrics_results[idx]['thought'] < 0.5:
                failed_cases['low_thought'].append(case_info)
            elif 'name' in response_data.pred and response_data.pred['name'] != response_data.gt['name']:
                failed_cases['wrong_name'].append(case_info)
            elif 'args' in response_data.pred and response_data.pred['args'] != response_data.gt['args']:
                failed_cases['low_args'].append(case_info)
            
        # 计算整体指标
        overall_metrics = {}
        if self.default_prompt_type == 'json':
            metric_keys = ['thought', 'name', 'args', 'parse_rate']
        elif self.default_prompt_type == 'str':
            if self.eval_type == 'reason':
                metric_keys = ['thought', 'parse_rate']
            elif self.eval_type == 'retrieve':
                metric_keys = ['name', 'parse_rate']
            elif self.eval_type == 'understand':
                metric_keys = ['args', 'parse_rate']
            
        for key in metric_keys:
            if any(key in m for m in metrics_results):
                overall_metrics[key] = float(mean([m.get(key, 0) for m in metrics_results]))
            
        return overall_metrics, failed_cases

    def find_a_dot_b_structure(self, text):
        # find a.b structure
        pattern = r'\w+\.\w+'
        return re.findall(pattern, text)
    
    def find_FinishAction(self, text):
        # find FinishAction
        pattern = r'FinishAction'
        return re.findall(pattern, text)

    def _post_process(self, results_list):
        # list of dict to dict of list
        if self.default_prompt_type == 'json':
            metric_keys = ['thought', 'name', 'args', 'parse_rate']
        if self.default_prompt_type == 'str':
            if self.eval_type == 'reason':
                metric_keys = ['thought', 'parse_rate']
            if self.eval_type == 'retrieve':
                metric_keys = ['name', 'parse_rate']
            if self.eval_type == 'understand':
                metric_keys = ['args', 'parse_rate']
        metrics_results = []
        batch_data = []; batch_arg_data = []
        batch_id = []; batch_arg_id = []
        BATCH_LIMIT = 32
        for id, data in enumerate(results_list):
            metrics_results.append(
                {metric_keys[x]: 0 for x in range(len(metric_keys))}
            )
            if len(data.pred.keys()) != 0:
                metrics_results[id]['parse_rate'] = 1
            if 'thought' in data.pred and 'thought' in data.gt:
                if data.pred['thought'] is None or data.gt['thought'] is None:
                    metrics_results[id]['thought'] = 0  # 如果是None就直接给0分
                else:
                    batch_data.extend([data.pred['thought'], data.gt['thought']])
                    batch_id.extend([id])
                    if len(batch_data) >= BATCH_LIMIT:
                        pred_emb = self.sentence_model.encode(batch_data, convert_to_tensor=True)
                        for i in range(0, len(batch_data), 2):
                            cosine_score = np.maximum(util.cos_sim(pred_emb[i], pred_emb[i+1]).cpu().numpy(), 0)
                            metrics_results[batch_id[i // 2]]['thought'] = cosine_score[0, 0]
                        batch_data = []
                        batch_id = []
            if 'name' in data.pred and 'name' in data.gt:
                if self.default_prompt_type == 'json':
                    if data.pred['name'] is None:
                        metrics_results[id]['name'] = 0
                    else:
                        if data.pred['name'] == data.gt['name']:
                            metrics_results[id]['name'] = 1
                        else:
                            metrics_results[id]['name'] = 0
                else:
                    if data.pred['name'] is None:
                        metrics_results[id]['name'] = 0
                    else:
                        if data.gt['name'] not in data.pred['name']:
                            metrics_results[id]['name'] = 0
                        else:
                            metrics_results[id]['name'] = 1
                            find_all_name = self.find_a_dot_b_structure(data.pred['name']) + self.find_FinishAction(data.pred['name'])
                            for name in find_all_name:
                                if name != data.gt['name']:
                                    metrics_results[id]['name'] = 0

            if 'args' in data.pred and 'args' in data.gt:
                if isinstance(data.gt['args'], dict):
                    for gt_arg_name in data.gt['args']:
                        if gt_arg_name in data.pred['args'] and str(data.pred['args'][gt_arg_name]) == str(data.gt['args'][gt_arg_name]):
                            metrics_results[id]['args'] += 1
                    metrics_results[id]['args'] /= (len(data.gt['args']) + 1e-5)
                    if len(data.gt['args']) == 0 and len(data.pred['args']) == 0:
                        metrics_results[id]['args'] = 1
                    if len(data.gt['args']) == 0 and len(data.pred['args']) != 0:
                        metrics_results[id]['args'] = 0
                else:
                    if data.pred['args'] is None:
                        metrics_results[id]['args'] = 0
                    else:
                        data.pred['args'] = data.pred['args'].strip("'").strip('"')
                        metrics_results[id]['args'] = float(data.gt['args'] == data.pred['args'])
                
        if len(batch_data) > 0:
            pred_emb = self.sentence_model.encode(batch_data, convert_to_tensor=True)
            for i in range(0, len(batch_data), 2):
                cosine_score = np.maximum(util.cos_sim(pred_emb[i], pred_emb[i+1]).cpu().numpy(), 0)
                metrics_results[batch_id[i // 2]]['thought'] = cosine_score[0, 0]    
            batch_data = []
            batch_id = []

        results = dict()
        for key in metric_keys:
            results[key] = mean([metrics_results[key] for metrics_results in metrics_results])
        return results




class ReviewEvaluator:
    """Review Capability Evaluation

    Args:
        dataset_path(str): File path of evaluation dataset.

    """

    def __init__(
        self,
        dataset_path: str,
        # bert_score_model: str = "all-mpnet-base-v2",
        **kwargs,
    ) -> None:
        self.dataset_path = dataset_path
        # self.bert_score_model = bert_score_model
        # self.sentence_model = SentenceTransformer(self.bert_score_model)

    def _load_dataset(self):
        self.dataset = []
        dataset = load(self.dataset_path)

        for key in dataset.keys():
            datum = dataset[key]
            data_sample = self._process_response(datum)
            
            self.dataset.append(
                dict(
                    origin_prompt=datum['origin_prompt'],
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

        template = datum['template']
        pred_data = datum['prediction']
        gt_data = datum['ground_truth']['answer']
        meta_data = datum['meta_data']

        if meta_data['response_format'] == 'json':
            pred_data = self.json_format_parse(pred_data)
        else:
            pred_data = pred_data[pred_data.find(":") + 1:]
            pred_data = pred_data.strip()
            if len(pred_data) > 0 and pred_data[0] in ['A', 'B', 'C', 'D', 'E']:
                pred_data = pred_data[0]
            else:
                pred_data = None

        return ResponseDataSample(
            template=template, pred=pred_data, gt=gt_data, meta_data=meta_data)

    def _evaluate(self, data_sample) -> dict:
        metrics_result = dict(
            parse_rate=0,
            review_quality=0,
        )
        
        pred_data = data_sample.pred
        if pred_data is not None:
            # import pdb; pdb.set_trace()
            metrics_result['review_quality'] = 1.0 if pred_data == \
                data_sample.gt else 0.0
            metrics_result['parse_rate'] = 1.0
        return metrics_result
    
    # def compute_sen_similarity(self, gt, pred):
    #     gt_embed = self.sentence_model.encode(gt, convert_to_tensor=True)
    #     pred_embed = self.sentence_model.encode(pred, convert_to_tensor=True)
    #     sen_sim = max(0, util.cos_sim(gt_embed, pred_embed).item())
    #     return sen_sim
    
    def json_format_parse(self, pred_data):
        try:
            data = format_load(pred_data)
        except Exception as e:
            return None
        try:
            new_data = dict()
            new_data['review'] = data['is_finished']
            assert new_data['review'] in [True, False]
        except Exception as e:
            return None
        return new_data

    def evaluate(self):
        self._load_dataset()
        results_list = []
        failed_cases = {
            'parse_failed': [],          # parse_rate = 0
            'wrong_review': []           # review_quality = 0
        }
        
        for idx, data_sample in enumerate(self.dataset):
            metrics_result = self._evaluate(data_sample['response_data_sample'])
            results_list.append(metrics_result)
            
            case_info = {
                'case_id': idx,
                'origin_prompt': data_sample['origin_prompt'],
                'prediction': data_sample['response_data_sample'].pred,
                'ground_truth': data_sample['response_data_sample'].gt,
                'metrics': metrics_result
            }
            
            if metrics_result['parse_rate'] == 0:
                failed_cases['parse_failed'].append(case_info)
            if metrics_result['review_quality'] == 0:
                failed_cases['wrong_review'].append(case_info)
                
        return self._post_process(results_list), failed_cases

    def _post_process(self, results_list):
        # list of dict to dict of list
        results_dict = defaultdict(list)
        {
            results_dict[key].append(sub[key])
            for sub in results_list for key in sub
        }
        metric_list = ['parse_rate', 'review_quality']
        for metric in metric_list:
            results_dict[metric] = np.round(np.mean(results_dict[metric]), decimals=4)
        return results_dict