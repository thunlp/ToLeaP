import json
import re

def ast_checker(data_entry):
    label = json.loads(data_entry['label'])
    predict = data_entry['predict']

    expected_param_names = list(label['arguments'].keys())

    try:
        model_output = parse_prediction(predict, expected_param_names)
    except Exception as e:
        return {
            "valid": False,
            "error": [f"Failed to parse model prediction: {str(e)}"],
            "error_type": "parser_error",
        }

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

def parse_prediction(predict, expected_param_names):
    function_call_pattern = r'(Action: )?(?P<name>[\w\.]+)\s*(?:\(|\nAction Input: )(?P<args>.*)'
    match = re.search(function_call_pattern, predict.strip(), re.DOTALL)
    if not match:
        raise ValueError("Prediction output does not contain a valid function call.")

    name = match.group('name').strip()
    args_str = match.group('args').strip()

    # Remove any trailing text after the arguments (e.g., 'Output:')
    args_str = re.split(r'\n', args_str)[0].strip()

    # Check if arguments are in JSON format
    if args_str.startswith('{'):
        # Handle JSON-formatted arguments
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
        # Handle function call arguments (e.g., func(a=1, b=2) or func(1, 2))
        if args_str.endswith(')'):
            args_str = args_str[:-1]
        arguments = parse_arguments(args_str, expected_param_names)

    return {
        'name': name,
        'arguments': arguments,
    }

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
    # Check if value is a list in string form
    if value.startswith('[') and value.endswith(']'):
        # Attempt to parse as JSON list
        try:
            value = json.loads(value.replace("'", '"'))
        except json.JSONDecodeError:
            # If JSON parsing fails, handle as a list of values
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
            pass  # Keep as string if conversion fails
    return value

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

        # Normalize values for comparison
        if isinstance(predicted_value, list) and len(predicted_value) == 1:
            predicted_value = predicted_value[0]
        predicted_value_normalized = normalize_value(predicted_value)
        expected_values_normalized = [normalize_value(val) for val in expected_values]

        if predicted_value_normalized not in expected_values_normalized:
            return {
                "valid": False,
                "error": [f"Incorrect value for argument '{key}'. Expected one of {expected_values}, got '{predicted_value}'."],
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
    input_file = "simple.jsonl"
    output_file = "bfcl_simple_result.jsonl"

    results = process_jsonl_file(input_file)
    save_results_to_file(results, output_file)

    print(f"AST checking complete! Results saved to {output_file}")
