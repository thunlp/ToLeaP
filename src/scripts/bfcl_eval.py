from bfcl_utlis import ast_file_runner,relevance_file_runner
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from utils.sharegpt_inference import LLM

model_path='/hy-tmp/3.1-8B'

def load_json(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def call_llm(file_path,model_path):
    llm = LLM(
        api_key="YOUR_API_KEY",
        api_base="http://localhost:8000/v1",
        model_path="/hy-tmp/3.1-8B"
    )
    test_cases = load_json(file_path)
    responses = llm(test_cases,batch_size=5, temperature=0 )
    model_results = []
    for response in responses:
        model_result = {
            'result': response
            }
        model_results.append(model_result)
    
    return model_results


def input_to_target_bfcl(file_path=None):
    if file_path:
        with open(file_path, 'r', encoding='utf-8') as file:
            json_data_list = json.load(file)
    if not json_data_list:
        raise ValueError("Either file_path or json_data_list must be provided")

    prompts = []
    gt = []
    test_categories = []

    def transform_response(gpt_response):
        """Transform response format"""
        if not gpt_response:
            return None
        if isinstance(gpt_response, list):
            return [{item['name']: item['arguments']} for item in gpt_response]
        else:
            return {gpt_response['name']: gpt_response['arguments']}

    def determine_category(ground_truth, functions):
        """Determine the category of the response"""
        # Get ground truth content
        gt_content = ground_truth['ground_truth']
        # Check if ground truth is None/empty
        if gt_content is None:
            return 'irrelevance'
            
        # Check if ground truth contains multiple dictionaries
        if isinstance(gt_content, list) and len(gt_content) > 1:
            # Get function names from ground truth items
            function_names = [item.get('name', '') for item in gt_content]
            # Check if all function names are the same
            if len(set(function_names)) == 1:
                return 'parallel'
            else:
                return 'parallel_multiple'
        else:
            # Single ground truth case
            if isinstance(functions, list) and len(functions) > 1:
                return 'multiple'
            else:
                return 'simple'

    for item in json_data_list:
        conversations = item['conversations']
        tools = json.loads(item['tools'])
        question = [[{
            'role': 'user' if conv['from'] == 'human' else 'assistant',
            'content': conv['value']
        } for conv in conversations if conv['from'] == 'human']]

        # Get GPT response with proper error handling
        gpt_messages = [conv for conv in conversations if conv['from'] == 'gpt']
        if gpt_messages:
            try:
                gpt_response = json.loads(gpt_messages[0]['value'])
            except (json.JSONDecodeError, IndexError):
                gpt_response = None
        else:
            gpt_response = None

        transformed_response = transform_response(gpt_response)
        formatted_item = {
            'question': question,
            'function': tools,
        }
        ground_truth = {
            'ground_truth': transformed_response
        }

        # Determine category for each item
        category = determine_category(ground_truth, tools)
        prompts.append(formatted_item)
        gt.append(ground_truth)
        test_categories.append(category)

    return prompts, gt, test_categories

def metric_cal_ast_bfcl(model_output, prompt, possible_answer, language, test_categories, model_name):
    return ast_file_runner(model_output, prompt, possible_answer, language, test_categories, model_name)

def metric_cal_irr_bfcl(model_output, prompt, test_categories, model_name):
    return relevance_file_runner(model_output, prompt, model_name, test_categories)

def evaluation(file_path):
    model_path='/hy-tmp/3.1-8B'
    language = "Python"
    model_name = "Test_model"
    prompt, possible_answer ,test_categories = input_to_target_bfcl(file_path)
    model_output = call_llm(file_path,model_path) 
    metric, _ = metric_cal_ast_bfcl(model_output, prompt, possible_answer, language, test_categories, model_name)
    # metric, _ = metric_cal_irr_bfcl(model_output, prompt, test_categories, model_name)
    return metric 

print(evaluation('BodhiAgent/src/data/sft_data/sft_bfcl_parallel.json'))
