import json
import re
from typing import List, Dict, Union, Any, Optional
from collections import defaultdict

def ast_checker(data_entry: Dict, mode: str = 'single') -> Dict:
    """
    Enhanced AST checker with manual mode selection.
    
    Args:
        data_entry (Dict): The data entry containing label and prediction
        mode (str): Execution mode - 'single', 'parallel', or 'multiple'
        
    Returns:
        Dict: Validation results
    """
    if mode not in ['single', 'parallel', 'multiple']:
        return {
            "valid": False,
            "error": [f"Invalid mode: {mode}. Must be one of: single, parallel, multiple"],
            "error_type": "invalid_mode",
        }
        
    label = json.loads(data_entry['label'])
    predict = data_entry['predict']
    
    if mode == 'single':
        # For single mode, if label is a list with one item, extract it
        if isinstance(label, list) and len(label) == 1:
            label = label[0]
        return check_single_function(label, predict)
    elif mode == 'parallel':
        # Ensure label is a list for parallel mode
        if not isinstance(label, list):
            label = [label]
        return check_parallel_functions(label, predict)
    else:  # multiple mode
        # Ensure label is a list for multiple mode
        if not isinstance(label, list):
            label = [label]
        return check_multiple_functions(label, predict)

def check_parallel_functions(label: List[Dict], predict: str) -> Dict:
    """
    处理并行函数调用（顺序不重要）
    """
    try:
        predicted_functions = parse_multiple_predictions(predict)
        
        if len(predicted_functions) != len(label):
            return {
                "valid": False,
                "error": [f"Number of function calls mismatch. Expected {len(label)}, got {len(predicted_functions)}"],
                "error_type": "function_count_mismatch",
            }
        
        # 使用集合来追踪匹配状态
        unmatched_predictions = set(range(len(predicted_functions)))
        unmatched_labels = set(range(len(label)))
        
        # 遍历每个标签，寻找匹配的预测
        for i in list(unmatched_labels):
            expected_func = label[i]
            match_found = False
            
            for j in list(unmatched_predictions):
                pred_func = predicted_functions[j]
                
                # 检查函数名和参数是否匹配
                if expected_func['name'] == pred_func['name']:
                    arguments_check = check_arguments(expected_func['arguments'], pred_func['arguments'])
                    if arguments_check['valid']:
                        unmatched_predictions.remove(j)
                        unmatched_labels.remove(i)
                        match_found = True
                        break
            
            if not match_found:
                return {
                    "valid": False,
                    "error": [f"No matching function call found for expected function at position {i+1}"],
                    "error_type": "parallel_function_no_match",
                }
        
        # 所有函数都应该被匹配
        if not unmatched_predictions and not unmatched_labels:
            return {
                "valid": True,
                "error": [],
            }
        else:
            return {
                "valid": False,
                "error": ["Not all functions were properly matched"],
                "error_type": "parallel_incomplete_match",
            }
            
    except Exception as e:
        return {
            "valid": False,
            "error": [f"Failed to parse parallel function predictions: {str(e)}"],
            "error_type": "parser_error",
        }

def check_multiple_functions(label: List[Dict], predict: str) -> Dict:
    """
    处理多重函数调用（顺序重要）
    """
    try:
        predicted_functions = parse_multiple_predictions(predict)
        
        if len(predicted_functions) != len(label):
            return {
                "valid": False,
                "error": [f"Number of function calls mismatch. Expected {len(label)}, got {len(predicted_functions)}"],
                "error_type": "function_count_mismatch",
            }
        
        # 按顺序检查每个函数
        for i, (pred_func, expected_func) in enumerate(zip(predicted_functions, label)):
            if expected_func['name'] != pred_func['name']:
                return {
                    "valid": False,
                    "error": [f"Function {i+1}: name mismatch. Expected '{expected_func['name']}', got '{pred_func['name']}'"],
                    "error_type": "function_name_mismatch",
                }
            
            arguments_check = check_arguments(expected_func['arguments'], pred_func['arguments'])
            if not arguments_check['valid']:
                error_msg = f"Function {i+1}: {arguments_check['error'][0]}"
                return {
                    "valid": False,
                    "error": [error_msg],
                    "error_type": f"function_{i+1}_{arguments_check['error_type']}",
                }
                
        return {
            "valid": True,
            "error": [],
        }
        
    except Exception as e:
        return {
            "valid": False,
            "error": [f"Failed to parse multiple function predictions: {str(e)}"],
            "error_type": "parser_error",
        }

def check_single_function(label: Dict, predict: str) -> Dict:
    """
    处理单个函数调用
    """
    expected_param_names = list(label['arguments'].keys())
    
    try:
        model_output = parse_prediction(predict, expected_param_names)
    except Exception as e:
        return {
            "valid": False,
            "error": [f"Failed to parse model prediction: {str(e)}"],
            "error_type": "parser_error",
        }

    if label['name'] != model_output['name']:
        return {
            "valid": False,
            "error": [f"Function name mismatch. Expected '{label['name']}', got '{model_output['name']}'"],
            "error_type": "function_name_mismatch",
        }

    arguments_check = check_arguments(label['arguments'], model_output['arguments'])
    if not arguments_check['valid']:
        return arguments_check

    return {
        "valid": True,
        "error": [],
    }

def parse_prediction(predict: str, expected_param_names: List[str]) -> Dict:
    """
    解析单个函数预测
    """
    function_call_pattern = r'(Action: )?(?P<name>[\w\.]+)\s*(?:\(|\nAction Input: )(?P<args>.*)'
    match = re.search(function_call_pattern, predict.strip(), re.DOTALL)
    if not match:
        raise ValueError("Prediction output does not contain a valid function call.")

    name = match.group('name').strip()
    args_str = match.group('args').strip()

    # Remove any trailing text after the arguments
    args_str = re.split(r'\n', args_str)[0].strip()

    # Check if arguments are in JSON format
    if args_str.startswith('{'):
        brace_count = 0
        for i, char in enumerate(args_str):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    args_str = args_str[:i+1]
                    break
        arguments = json.loads(args_str)
    else:
        # Handle function call arguments
        if args_str.endswith(')'):
            args_str = args_str[:-1]
        arguments = parse_arguments(args_str, expected_param_names)

    return {
        'name': name,
        'arguments': arguments,
    }

def parse_multiple_predictions(predict: str) -> List[Dict]:
    """
    解析多个函数调用
    """
    function_calls = re.findall(r'Action: .*?\nAction Input: .*?(?=\n(?:Action:|$)|\Z)', predict, re.DOTALL)
    
    if not function_calls:
        raise ValueError("No valid function calls found in prediction")
    
    parsed_functions = []
    for call in function_calls:
        match = re.match(r'Action: (?P<name>[\w\.]+)\s*\nAction Input: (?P<args>.*)', call.strip(), re.DOTALL)
        if not match:
            raise ValueError(f"Invalid function call format: {call}")
            
        name = match.group('name').strip()
        args_str = match.group('args').strip()
        
        try:
            arguments = json.loads(args_str)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in arguments: {args_str}")
            
        parsed_functions.append({
            'name': name,
            'arguments': arguments
        })
    
    return parsed_functions

def parse_arguments(args_str: str, expected_param_names: List[str]) -> Dict:
    """
    解析函数参数
    """
    args_list = re.split(r',(?![^\[\]]*\])', args_str)
    arguments = {}
    
    for i, arg in enumerate(args_list):
        arg = arg.strip()
        if '=' in arg:
            # Named argument
            key, value = arg.split('=', 1)
            key = key.strip()
            value = value.strip()
        else:
            # Positional argument
            if i < len(expected_param_names):
                key = expected_param_names[i]
                value = arg.strip()
            else:
                raise ValueError(f"Too many positional arguments provided: {arg}")
        
        # Process the value
        value = process_value(value)
        arguments[key] = value
        
    return arguments

def process_value(value: str) -> Union[str, int, float, List]:
    """
    处理和转换值的类型
    """
    value = value.strip()
    
    # Check if value is a list
    if value.startswith('[') and value.endswith(']'):
        try:
            value = json.loads(value.replace("'", '"'))
        except json.JSONDecodeError:
            value = value[1:-1].split(',')
            value = [process_value(v) for v in value]
    elif (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        value = value[1:-1]
    else:
        # Attempt to convert to int or float
        try:
            if '.' in value:
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            pass
    
    return value

def normalize_value(value: Any) -> Union[str, int, float, List]:
    """
    标准化值以进行比较
    """
    if isinstance(value, str):
        return value.strip().lower()
    elif isinstance(value, (int, float)):
        return value
    elif isinstance(value, list):
        if len(value) == 1:
            return normalize_value(value[0])
        return [normalize_value(v) for v in value]
    return value

def check_arguments(expected_args: Dict, predicted_args: Dict) -> Dict:
    """
    检查预测的参数是否匹配预期
    """
    expected_keys = set(expected_args.keys())
    predicted_keys = set(predicted_args.keys())

    # Identify required and optional parameters
    required_keys = {k for k, v in expected_args.items() if "" not in v}
    optional_keys = expected_keys - required_keys

    missing_keys = required_keys - predicted_keys
    extra_keys = predicted_keys - expected_keys

    if missing_keys:
        return {
            "valid": False,
            "error": [f"Missing required arguments: {', '.join(missing_keys)}"],
            "error_type": "missing_arguments",
        }
    if extra_keys:
        return {
            "valid": False,
            "error": [f"Unexpected arguments: {', '.join(extra_keys)}"],
            "error_type": "unexpected_arguments",
        }

    # Check values for each argument
    for key in expected_args:
        expected_values = expected_args[key]
        predicted_value = predicted_args.get(key, None)

        # Handle optional parameters
        if predicted_value is None:
            if key in optional_keys:
                continue
            else:
                return {
                    "valid": False,
                    "error": [f"Missing required argument: {key}"],
                    "error_type": "missing_argument",
                }

        # Normalize values for comparison
        predicted_value_normalized = normalize_value(predicted_value)
        expected_values_normalized = [normalize_value(val) for val in expected_values]

        # Handle single list value
        if len(expected_values) == 1 and isinstance(expected_values[0], list):
            expected_single_value = normalize_value(expected_values[0])
            if predicted_value_normalized == expected_single_value:
                continue

        # Check for match
        if predicted_value_normalized not in expected_values_normalized:
            matched = False
            for expected_value in expected_values_normalized:
                if isinstance(expected_value, list):
                    if predicted_value_normalized == normalize_value(expected_value):
                        matched = True
                        break
                    if isinstance(predicted_value_normalized, str) and predicted_value_normalized == normalize_value(expected_value[0]):
                        matched = True
                        break
                elif predicted_value_normalized == expected_value:
                    matched = True
                    break

            if not matched:
                return {
                    "valid": False,
                    "error": [f"Incorrect value for argument '{key}'. Expected one of {expected_values}, got '{predicted_value}'."],
                    "error_type": "argument_value_mismatch",
                }

    return {
        "valid": True,
        "error": [],
    }

def process_jsonl_file(file_path: str, mode: str = 'single') -> List[Dict]:
    """
    处理JSONL文件
    
    Args:
        file_path (str): JSONL文件路径
        mode (str): 执行模式 - 'single', 'parallel', 或 'multiple'
    """
    results = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, start=1):
            try:
                data_entry = json.loads(line.strip())
                result = ast_checker(data_entry, mode)
                results.append({
                    "id": data_entry.get("id", line_number),
                    "valid": result["valid"],
                    "errors": result.get("error", []),
                    "error_type": result.get("error_type", "")
                })
            except Exception as e:
                results.append({
                    "id": line_number,
                    "valid": False,
                    "errors": [f"Processing error: {str(e)}"],
                    "error_type": "processing_error"
                })
    return results


def save_results_to_file(results: List[Dict], output_file: str) -> None:
    """
    Save results to a JSONL file.
    """
    with open(output_file, 'w', encoding='utf-8') as file:
        for result in results:
            file.write(json.dumps(result) + '\n')

if __name__ == "__main__":
    input_file = "simple.jsonl"  # Your input file
    output_file = "bfcl_parrallel.jsonl"  # Your output file

    results = process_jsonl_file(input_file,  mode='single')
    save_results_to_file(results, output_file)
    print(f"AST checking complete! Results saved to {output_file}")
