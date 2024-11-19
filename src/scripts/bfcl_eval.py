import json
import re

def ast_checker(data_entry):
    label = json.loads(data_entry['label'])
    predict = data_entry['predict']
    test_category = data_entry.get('test_category', 'simple').lower()

    # Determine expected parameter names
    if isinstance(label, dict):
        expected_param_names = list(label['arguments'].keys())
    elif isinstance(label, list):
        expected_param_names = [list(l['arguments'].keys()) for l in label]
    else:
        expected_param_names = []

    try:
        model_output = parse_prediction(predict, expected_param_names)
    except Exception as e:
        return {
            "valid": False,
            "error": [f"Failed to parse model prediction: {str(e)}"],
            "error_type": "parser_error",
        }

    # Based on the test category, choose the appropriate checker
    if 'parallel' in test_category:
        return parallel_function_checker_no_order(label, model_output)
    elif 'multiple' in test_category:
        return multiple_function_checker(label, model_output)
    else:
        if not isinstance(model_output, dict):
            return {
                "valid": False,
                "error": ["Expected a single function call."],
                "error_type": "simple_function_checker:wrong_count",
            }
        return simple_function_checker(label, model_output)

def parse_prediction(predict, expected_param_names):
    # Handle expected_param_names for multiple functions
    if isinstance(expected_param_names, list) and all(isinstance(item, list) for item in expected_param_names):
        expected_param_names_flat = [param for sublist in expected_param_names for param in sublist]
    else:
        expected_param_names_flat = expected_param_names

    function_calls = []
    # Regex to match function calls
    function_call_pattern = r'(Action: )?(?P<name>[\w\.]+)\s*(?:\(|\nAction Input: )(?P<args>[\s\S]*?)(?=\nAction:|$)'
    matches = re.finditer(function_call_pattern, predict.strip(), re.MULTILINE)

    for match in matches:
        name = match.group('name').strip()
        args_str = match.group('args').strip()
        # Remove any trailing text after the arguments
        args_str = re.split(r'\n', args_str)[0].strip()
        # Parse arguments
        if args_str.startswith('{'):
            # JSON formatted arguments
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
            # Function call arguments
            if args_str.endswith(')'):
                args_str = args_str[:-1]
            arguments = parse_arguments(args_str, expected_param_names_flat)
        function_calls.append({'name': name, 'arguments': arguments})

    if len(function_calls) == 1:
        return function_calls[0]
    else:
        return function_calls

def parse_arguments(args_str, expected_param_names):
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

def process_value(value):
    value = value.strip()
    # Handle string representation of lists
    if value.startswith('[') and value.endswith(']'):
        try:
            # Replace single quotes with double quotes and parse as JSON list
            value = json.loads(value.replace("'", '"'))
            # Recursively process each element in the list
            value = [process_value(v) for v in value]
        except json.JSONDecodeError:
            # If JSON parsing fails, manually split and process
            inner_values = re.split(r',(?![^\[\]]*\])', value[1:-1])
            value = [process_value(v.strip()) for v in inner_values]
    elif (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        # Remove quotes from strings
        value = value[1:-1]
    else:
        # Attempt to convert to int or float
        try:
            if '.' in value:
                value = float(value)
            else:
                value = int(value)
        except ValueError:
            pass  # Keep as string if conversion fails
    return value

def simple_function_checker(label, model_output):
    # Compare function names
    if label['name'] != model_output['name']:
        return {
            "valid": False,
            "error": [f"Function name mismatch. Expected '{label['name']}', got '{model_output['name']}'."],
            "error_type": "function_name_mismatch",
        }
    # Compare arguments
    arguments_check = check_arguments(label['arguments'], model_output['arguments'])
    if not arguments_check['valid']:
        return arguments_check
    return {
        "valid": True,
        "error": [],
    }

def parallel_function_checker_no_order(labels, model_outputs):
    if not isinstance(labels, list) or not isinstance(model_outputs, list):
        return {
            "valid": False,
            "error": ["Labels and model outputs should be lists for parallel functions."],
            "error_type": "parallel_function_checker_no_order:type_mismatch",
        }
    if len(labels) != len(model_outputs):
        return {
            "valid": False,
            "error": ["Number of functions does not match."],
            "error_type": "parallel_function_checker_no_order:wrong_count",
        }
    unmatched_model_outputs = model_outputs.copy()
    errors = []

    for label in labels:
        match_found = False
        for model_output in unmatched_model_outputs:
            if label['name'] == model_output['name']:
                arguments_check = check_arguments(label['arguments'], model_output['arguments'])
                if arguments_check['valid']:
                    unmatched_model_outputs.remove(model_output)
                    match_found = True
                    break
                else:
                    errors.extend(arguments_check['error'])
        if not match_found:
            return {
                "valid": False,
                "error": ["No matching function found for label."],
                "error_type": "parallel_function_checker_no_order:no_match",
            }
    if unmatched_model_outputs:
        return {
            "valid": False,
            "error": ["Extra functions in model output."],
            "error_type": "parallel_function_checker_no_order:extra_functions",
        }
    return {
        "valid": True,
        "error": [],
    }

def multiple_function_checker(labels, model_outputs):
    if not isinstance(labels, list) or not isinstance(model_outputs, list):
        return {
            "valid": False,
            "error": ["Labels and model outputs should be lists for multiple functions."],
            "error_type": "multiple_function_checker:type_mismatch",
        }
    if len(labels) != len(model_outputs):
        return {
            "valid": False,
            "error": ["Number of functions does not match."],
            "error_type": "multiple_function_checker:wrong_count",
        }
    for label, model_output in zip(labels, model_outputs):
        # Compare function names
        if label['name'] != model_output['name']:
            return {
                "valid": False,
                "error": [f"Function name mismatch. Expected '{label['name']}', got '{model_output['name']}'."],
                "error_type": "function_name_mismatch",
            }
        # Compare arguments
        arguments_check = check_arguments(label['arguments'], model_output['arguments'])
        if not arguments_check['valid']:
            return arguments_check
    return {
        "valid": True,
        "error": [],
    }

def check_arguments(expected_args, predicted_args):
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
                continue  # Accept missing optional parameter
            else:
                return {
                    "valid": False,
                    "error": [f"Missing required argument: {key}"],
                    "error_type": "missing_argument",
                }

        # If predicted value is a single-element list, unpack it
        if isinstance(predicted_value, list) and len(predicted_value) == 1:
            predicted_value = predicted_value[0]

        # Normalize values for comparison
        predicted_value_normalized = normalize_value(predicted_value)
        expected_values_normalized = [normalize_value(val) for val in expected_values]

        # If predicted value is a list, check if any element matches
        if isinstance(predicted_value_normalized, list):
            if not any(val in expected_values_normalized for val in predicted_value_normalized):
                return {
                    "valid": False,
                    "error": [
                        f"Incorrect value for argument '{key}'. Expected one of {expected_values}, got '{predicted_value}'."
                    ],
                    "error_type": "argument_value_mismatch",
                }
        else:
            if predicted_value_normalized not in expected_values_normalized:
                return {
                    "valid": False,
                    "error": [
                        f"Incorrect value for argument '{key}'. Expected one of {expected_values}, got '{predicted_value}'."
                    ],
                    "error_type": "argument_value_mismatch",
                }
    return {
        "valid": True,
        "error": [],
    }

def normalize_value(value):
    # Convert value to a standardized format for comparison
    if isinstance(value, str):
        return value.strip().lower()
    elif isinstance(value, list):
        return [normalize_value(v) for v in value]
    else:
        return value  # For numbers and other types

def process_jsonl_file(file_path):
    results = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, start=1):
            try:
                data_entry = json.loads(line.strip())
                result = ast_checker(data_entry)
                results.append({
                    "id": data_entry.get("id", line_number),
                    "valid": result["valid"],
                    "errors": result.get("error", []),
                })
            except Exception as e:
                results.append({
                    "id": line_number,
                    "valid": False,
                    "errors": [f"Processing error: {str(e)}"],
                })
    return results

def save_results_to_file(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for result in results:
            file.write(json.dumps(result) + '\n')

if __name__ == "__main__":
    input_file = "par.jsonl"
    output_file = "ast_results.jsonl"

    results = process_jsonl_file(input_file)
    save_results_to_file(results, output_file)

    print(f"AST checking complete! Results saved to {output_file}")
