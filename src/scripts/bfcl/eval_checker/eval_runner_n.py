import argparse

from bfcl.constant import (
    DOTENV_PATH,
    POSSIBLE_ANSWER_PATH,
    PROMPT_PATH,
    RESULT_PATH,
    SCORE_PATH,
    TEST_COLLECTION_MAPPING,
    TEST_FILE_MAPPING,
    VERSION_PREFIX,
)
from bfcl.eval_checker.ast_eval.ast_checker import ast_checker
from bfcl.eval_checker.eval_runner_helper import *
from bfcl.eval_checker.executable_eval.custom_exception import BadAPIStatusError
from bfcl.eval_checker.executable_eval.executable_checker import (
    executable_checker_non_rest,
    executable_checker_rest,
)
from bfcl.eval_checker.multi_turn_eval.multi_turn_checker import (
    multi_turn_checker,
    multi_turn_irrelevance_checker,
)
from bfcl.eval_checker.multi_turn_eval.multi_turn_utils import is_empty_execute_response
from bfcl.model_handler.handler_map import HANDLER_MAP
from bfcl.utils import *
from dotenv import load_dotenv
from tqdm import tqdm



def multi_turn_runner(
    handler, model_result, prompt, possible_answer, model_name, test_category
):
    assert (
        len(model_result) == len(prompt) == len(possible_answer)
    ), f"The length of the model result ({len(model_result)}) does not match the length of the prompt ({len(prompt)}) or possible answer ({len(possible_answer)}). Please check the input files for completeness."

    result = []
    correct_count = 0
    for i in range(len(model_result)):
        index: str = model_result[i]["id"]
        # Model result is stored as a list of list of model responses. Each inner list represents a turn.
        multi_turn_model_result_list: list[list] = model_result[i]["result"]
        multi_turn_ground_truth_list: list[list[str]] = possible_answer[i]["ground_truth"]
        test_entry: dict = prompt[i]

        # Remove the function doc from the score file for better readability; they are repeated and way too long
        if "function" in test_entry:
            del test_entry["function"]

        if type(multi_turn_model_result_list) != list:
            result.append(
                {
                    "id": index,
                    "model_name": model_name,
                    "test_category": test_category,
                    "valid": False,
                    "error": {
                        "error_message": [
                            "Error during inference phase. Model did not output a list of model responses."
                        ],
                        "error_type": "multi_turn:inference_error",
                    },
                    "prompt": test_entry,
                    "model_result": multi_turn_model_result_list,
                    "possible_answer": multi_turn_ground_truth_list,
                }
            )
        # Check if force-terminated during inference phase.
        # This happens when the model has retried too many times and still haven't figured out the answer.
        # When force-terminated, no further evaluation is needed. This whole entry will be failed.
        if len(multi_turn_model_result_list) != len(multi_turn_ground_truth_list):
            result.append(
                {
                    "id": index,
                    "model_name": model_name,
                    "test_category": test_category,
                    "valid": False,
                    "error": {
                        "error_message": [
                            f"Model was force-terminated during inference phase. The length of the model result turns ({len(multi_turn_model_result_list)}) does not match the length of the ground truth turns ({len(multi_turn_ground_truth_list)})."
                        ],
                        "error_type": "multi_turn:force_terminated",
                    },
                    "prompt": test_entry,
                    "model_result": multi_turn_model_result_list,
                    "possible_answer": multi_turn_ground_truth_list,
                }
            )
            continue

        multi_turn_model_result_list_decoded: list[list[list[str]]] = (
            []
        )  # decode_execute returns a list of strings
        # Try decoding the model results into executable function calls
        for single_turn_model_result_list in multi_turn_model_result_list:
            single_turn_model_result_list_decoded = []
            for model_result_item in single_turn_model_result_list:
                # model_result_item is per step
                try:
                    decoded_result: list[str] = handler.decode_execute(model_result_item)
                    if is_empty_execute_response(decoded_result):
                        # Empty output is not considered as a valid function call
                        continue

                    single_turn_model_result_list_decoded.append(decoded_result)

                except Exception as e:
                    # Ignore any failed decoding and continue to the next message
                    # We only care about the decoded function call, not the error message or if the model is chatting
                    continue
            multi_turn_model_result_list_decoded.append(
                single_turn_model_result_list_decoded
            )

        # Check if the model output the correct function calls
        accuracy_checker_result = multi_turn_checker(
            multi_turn_model_result_list_decoded,
            multi_turn_ground_truth_list,
            test_entry,
            test_category,
            model_name,
        )
        
        # Perform additional check for multi-turn irrelevance
        # This happens when the model is expected to not output any function calls in a certain turn due to miss parameters or miss functions
        # irrelevance_checker_result = multi_turn_irrelevance_checker(
        #     multi_turn_model_result_list_decoded,
        #     multi_turn_ground_truth_list,
        # )

        if not accuracy_checker_result["valid"]:
            temp = {}
            temp["id"] = index
            temp["model_name"] = model_name
            temp["test_category"] = test_category
            temp["valid"] = accuracy_checker_result.pop("valid")
            temp["error"] = accuracy_checker_result
            temp["prompt"] = test_entry
            temp["model_result_raw"] = multi_turn_model_result_list
            temp["model_result_decoded"] = multi_turn_model_result_list_decoded
            temp["possible_answer"] = multi_turn_ground_truth_list
            temp["inference_log"] = model_result[i].get("inference_log", "")
            result.append(temp)
        else:
            correct_count += 1

    accuracy = correct_count / len(model_result)
    result.insert(
        0,
        {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": len(model_result),
        },
    )
    output_file_name = f"{VERSION_PREFIX}_{test_category}_score.json"
    output_file_dir = SCORE_PATH / model_name
    write_list_of_dicts_to_file(output_file_name, result, output_file_dir)

    return accuracy, len(model_result)


def executable_file_runner(handler, model_result, prompt, model_name, test_category):
    assert len(model_result) == len(prompt)

    result = []
    correct_count = 0
    for i in tqdm(range(len(model_result)), desc="Running tests"):
        raw_result = model_result[i]["result"]
        try:
            decoded_result = handler.decode_execute(raw_result)
        except Exception as e:
            result.append(
                {
                    "id": i + 1,
                    "model_name": model_name,
                    "test_category": test_category,
                    "valid": False,
                    "error": [f"Failed to decode executable. {str(e)}"],
                    "error_type": "executable_decoder:decoder_failed",
                    "prompt": prompt[i],
                    "model_result_raw": raw_result,
                }
            )
            continue

        if "rest" in test_category:
            # REST is always single-functioned. Therefore we take the first one and pass it to the REST checker.
            if not is_rest_format_output(decoded_result):
                result.append(
                    {
                        "id": i + 1,
                        "model_name": model_name,
                        "test_category": test_category,
                        "valid": False,
                        "error": [
                            "Did not output in the specified format. Note: the model_result is wrapped in a string to ensure json serializability."
                        ],
                        "error_type": "executable_decoder:rest_wrong_output_format",
                        "prompt": prompt[i],
                        "model_result_raw": str(raw_result),
                        "model_result_decoded": str(decoded_result),
                    }
                )
                continue

            checker_result = executable_checker_rest(decoded_result[0], i)

        else:
            if not is_executable_format_output(decoded_result):
                result.append(
                    {
                        "id": i + 1,
                        "model_name": model_name,
                        "test_category": test_category,
                        "valid": False,
                        "error": [
                            "Did not output in the specified format. Note: the model_result is wrapped in a string to ensure json serializability."
                        ],
                        "error_type": "executable_decoder:wrong_output_format",
                        "prompt": prompt[i],
                        "model_result_raw": str(raw_result),
                        "model_result_decoded": str(decoded_result),
                    }
                )
                continue

            prompt_item = prompt[i]
            checker_result = executable_checker_non_rest(
                decoded_result, prompt_item, test_category
            )

        if checker_result["valid"]:
            correct_count += 1
        else:
            temp = {}
            temp["id"] = i + 1
            temp["model_name"] = model_name
            temp["test_category"] = test_category
            temp["valid"] = checker_result["valid"]
            temp["error"] = checker_result["error"]
            temp["error_type"] = checker_result["error_type"]
            temp["prompt"] = prompt[i]
            temp["model_result_raw"] = raw_result
            temp["model_result_decoded"] = decoded_result
            if "model_executed_output" in checker_result:
                temp["model_executed_output"] = checker_result["model_executed_output"]
            result.append(temp)

    accuracy = correct_count / len(model_result)
    result.insert(
        0,
        {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": len(model_result),
        },
    )
    output_file_name = f"{VERSION_PREFIX}_{test_category}_score.json"
    output_file_dir = SCORE_PATH / model_name
    write_list_of_dicts_to_file(output_file_name, result, output_file_dir)

    return accuracy, len(model_result)


def relevance_file_runner(handler, model_result, prompt, model_name, test_category):
    # This function serves for both relevance and irrelevance tests, which share the exact opposite logic.
    # If `test_category` is "irrelevance", the model is expected to output no function call.
    # No function call means either the AST decoding fails (a error message is generated) or the decoded AST does not contain any function call (such as a empty list, `[]`).
    # If `test_category` is "relevance", the model is expected to output to a function call, and empty list doesn't count as a function call.
    result = []
    correct_count = 0
    for i in range(len(model_result)):
        model_result_item = model_result[i]["result"]
        contain_func_call = False
        decoded_result = None
        decode_error = None

        try:
            decoded_result = handler.decode_ast(model_result_item, language="Python")
            # Decode successfully, which means the model output is in valid function call format
            contain_func_call = True
            if is_empty_output(decoded_result):
                # Empty output is not considered as a valid function call
                contain_func_call = False

        except Exception as e:
            # Decode failed, which means the model output is not in valid function call format
            contain_func_call = False
            decode_error = str(e)

        # irrelevance test means no function call outputted
        if "irrelevance" in test_category:
            success = not contain_func_call
        else:
            success = contain_func_call

        if success:
            correct_count += 1
        else:
            temp = {}
            temp["id"] = i + 1
            temp["model_name"] = model_name
            temp["test_category"] = test_category
            temp["valid"] = success
            if "irrelevance" in test_category:
                temp["error"] = [
                    f"Valid syntax. Successfully decode AST when it should not."
                ]
                temp["error_type"] = "irrelevance_error:decoder_success"
            else:
                temp["error"] = [
                    f"Invalid syntax. Failed to decode AST when it should have. {decode_error}"
                ]
                temp["error_type"] = "relevance_error:decoder_failed"
            temp["prompt"] = prompt[i]
            temp["model_result"] = model_result_item
            temp["decoded_result"] = decoded_result

            result.append(temp)

    accuracy = correct_count / len(model_result)
    result.insert(
        0,
        {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": len(model_result),
        },
    )
    output_file_name = f"{VERSION_PREFIX}_{test_category}_score.json"
    output_file_dir = SCORE_PATH / model_name
    write_list_of_dicts_to_file(output_file_name, result, output_file_dir)

    return accuracy, len(model_result)


# model_result, prompt, possible_answer 都是一个file 就行 user 后面的就是 prompt 然后 possible answer 是label， model asnwe是 peridct
#传入就是一整个大的json
def ast_file_runner(
    handler, model_result, prompt, possible_answer, language, test_category, model_name
):
    # print(model_result[-1])
    # print(possible_answer[-1])
    assert (
        len(model_result) == len(prompt) == len(possible_answer)
    ), f"The length of the model result ({len(model_result)}) does not match the length of the prompt ({len(prompt)}) or possible answer ({len(possible_answer)}). Please check the input files for completeness."

    result = []
    correct_count = 0
    for i in range(len(model_result)):
        model_result_item = model_result[i]["result"]
        prompt_item = prompt[i]["function"]
        possible_answer_item = possible_answer[i]["ground_truth"]

        try:
            model_result_item_raw = model_result_item
            model_result_item = handler.decode_ast(model_result_item, language)
        except Exception as e:
            result.append(
                {
                    "id": i + 1,
                    "model_name": model_name,
                    "test_category": test_category,
                    "valid": False,
                    "error": [f"Invalid syntax. Failed to decode AST. {str(e)}"],
                    "error_type": "ast_decoder:decoder_failed",
                    "prompt": prompt[i],
                    "model_result_raw": model_result_item_raw,
                    "possible_answer": possible_answer_item,
                }
            )
            continue

        decoder_output_valid = is_function_calling_format_output(model_result_item)
        if not decoder_output_valid:
            result.append(
                {
                    "id": i + 1,
                    "model_name": model_name,
                    "test_category": test_category,
                    "valid": False,
                    "error": [
                        "Did not output in the specified format. Note: the model_result is wrapped in a string to ensure json serializability."
                    ],
                    "error_type": "ast_decoder:decoder_wrong_output_format",
                    "prompt": prompt[i],
                    "model_result_raw": str(model_result_item_raw),
                    "model_result_decoded": str(model_result_item),
                    "possible_answer": possible_answer_item,
                }
            )
            continue

        checker_result = ast_checker(
            prompt_item,
            model_result_item,
            possible_answer_item,
            language,
            test_category,
            model_name,
        )

        if checker_result["valid"]:
            correct_count += 1
        else:
            temp = {}
            temp["id"] = i + 1
            temp["model_name"] = model_name
            temp["test_category"] = test_category
            temp["valid"] = checker_result["valid"]
            temp["error"] = checker_result["error"]
            temp["error_type"] = checker_result["error_type"]
            temp["prompt"] = prompt[i]
            temp["model_result_raw"] = model_result_item_raw
            temp["model_result_decoded"] = model_result_item
            temp["possible_answer"] = possible_answer_item
            result.append(temp)

    accuracy = correct_count / len(model_result)
    result.insert(
        0,
        {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": len(model_result),
        },
    )
    output_file_name = f"{VERSION_PREFIX}_{test_category}_score.json"
    output_file_dir = SCORE_PATH / model_name
    write_list_of_dicts_to_file(output_file_name, result, output_file_dir)

    return accuracy, len(model_result)


#### Main runner function ####
def runner(model_names, test_categories, api_sanity_check):
    # A flag to indicate if the API has been tested.
    # We should always test the API with ground truth first before running the executable tests.
    # Sometimes the API may not be working as expected and we want to catch that before running the evaluation to ensure the results are accurate.
    API_TESTED = False
    API_STATUS_ERROR_REST = None
    API_STATUS_ERROR_EXECUTABLE = None

    # Before running the executable evaluation, we need to get the expected output from the ground truth.
    # So we need a list of all the test categories that we have ran the ground truth evaluation on.
    # We only get the expected output once for each test category.
    EXECUTABLE_TEST_CATEGORIES_HAVE_RUN = []

    # Get a list of all entries in the folder
    entries = RESULT_PATH.iterdir()

    # Filter out the subdirectories
    subdirs = [entry for entry in entries if entry.is_dir()]

    # Traverse each subdirectory
    for subdir in tqdm(subdirs, desc="Number of models evaluated"):

        model_name = subdir.relative_to(RESULT_PATH).name
        if model_names is not None and model_name not in model_names:
            continue

        model_name_escaped = model_name.replace("_", "/")

        print(f"🦍 Model: {model_name}")

        # Find and process all JSON files in the subdirectory
        for model_result_json in subdir.glob("*.json"):
            test_category = extract_test_category(model_result_json)
            if test_categories is not None and test_category not in test_categories:
                continue

            handler = get_handler(model_name_escaped)

            # We don't evaluate chatable and SQL models in our current leaderboard
            if is_chatable(test_category) or is_sql(test_category):
                continue

            language = "Python"
            if is_java(test_category):
                language = "Java"
            if is_js(test_category):
                language = "JavaScript"

            print(f"🔍 Running test: {test_category}")

            model_result = load_file(model_result_json)
            # print(model_result)
            # print(type(model_result[0]))

            prompt_file = find_file_with_suffix(PROMPT_PATH, test_category)
            prompt = load_file(prompt_file)

            # prompt 就是 直接就是用 jsonl的就行

            if is_relevance_or_irrelevance(test_category):
                accuracy, total_count = relevance_file_runner(
                    handler, model_result, prompt, model_name, test_category
                )
                print(f"✅ Test completed: {test_category}. 🎯 Accuracy: {accuracy}")
                continue

            if is_executable(test_category):
                # We only test the API with ground truth once
                if not API_TESTED and api_sanity_check:
                    print("---- Sanity checking API status ----")
                    try:
                        api_status_sanity_check_rest()
                    except BadAPIStatusError as e:
                        API_STATUS_ERROR_REST = e

                    try:
                        api_status_sanity_check_executable()
                    except BadAPIStatusError as e:
                        API_STATUS_ERROR_EXECUTABLE = e

                    display_api_status_error(
                        API_STATUS_ERROR_REST,
                        API_STATUS_ERROR_EXECUTABLE,
                        display_success=True,
                    )
                    print("Continuing evaluation...")

                    API_TESTED = True

                if (
                    test_category not in EXECUTABLE_TEST_CATEGORIES_HAVE_RUN
                    and not is_rest(test_category)
                ):
                    print(
                        f"---- Getting real-time execution result from ground truth for {test_category} ----"
                    )
                    get_executable_expected_output(prompt_file)
                    print(
                        f"---- Ground truth real-time execution result obtained for {test_category} 🌟 ----"
                    )
                    EXECUTABLE_TEST_CATEGORIES_HAVE_RUN.append(test_category)
                    # Need to re-load the prompt file after getting the expected output, as the prompt file has been updated
                    prompt = load_file(prompt_file)

                accuracy, total_count = executable_file_runner(
                    handler, model_result, prompt, model_name, test_category
                )
                print(f"✅ Test completed: {test_category}. 🎯 Accuracy: {accuracy}")

                continue

            # Find the corresponding possible answer file
            possible_answer_file = find_file_with_suffix(
                POSSIBLE_ANSWER_PATH, test_category
            )
            possible_answer = load_file(possible_answer_file)


            if is_multi_turn(test_category):
                accuracy, total_count = multi_turn_runner(
                    handler,
                    model_result,
                    prompt,
                    possible_answer,
                    model_name,
                    test_category,
                )
                print(f"✅ Test completed: {test_category}. 🎯 Accuracy: {accuracy}")
            # Single turn test
            else:
                print(prompt[0])
                accuracy, total_count = ast_file_runner(
                    handler,
                    model_result,
                    prompt,
                    possible_answer,
                    language,
                    test_category,
                    model_name,
                )
                # record_result(
                #     LEADERBOARD_TABLE, model_name, test_category, accuracy, total_count
                # )
                print(f"✅ Test completed: {test_category}. 🎯 Accuracy: {accuracy}")

    # This function reads all the score files from local folder and updates the leaderboard table.
    # This is helpful when you only want to run the evaluation for a subset of models and test categories.
    # update_leaderboard_table_with_score_file(LEADERBOARD_TABLE, SCORE_PATH)
    # # Write the leaderboard table to a file
    # generate_leaderboard_csv(LEADERBOARD_TABLE, SCORE_PATH, model_names, test_categories)

    # Clean up the executable expected output files
    # They should be re-generated the next time the evaluation is run
    clean_up_executable_expected_output(PROMPT_PATH, EXECUTABLE_TEST_CATEGORIES_HAVE_RUN)

    display_api_status_error(
        API_STATUS_ERROR_REST, API_STATUS_ERROR_EXECUTABLE, display_success=False
    )

    print(
        f"🏁 Evaluation completed. See {SCORE_PATH / 'data_overall.csv'} for overall evaluation results on BFCL V3."
    )
    print(
        f"See {SCORE_PATH / 'data_live.csv'}, {SCORE_PATH / 'data_non_live.csv'} and {SCORE_PATH / 'data_multi_turn.csv'} for detailed evaluation results on each sub-section categories respectively."
    )


def main(model, test_category, api_sanity_check):
    test_categories = None
    if test_category is not None:
        test_categories = []
        for category in test_category:
            if category in TEST_COLLECTION_MAPPING:
                test_categories.extend(TEST_COLLECTION_MAPPING[category])
            else:
                test_categories.append(category)

    model_names = None
    if model is not None:
        model_names = []
        for model_name in model:
            # Runner takes in the model name that contains "_", instead of "/", for the sake of file path issues.
            # This is differnet than the model name format that the generation script "openfunctions_evaluation.py" takes in (where the name contains "/").
            # We patch it here to avoid confusing the user.
            model_names.append(model_name.replace("/", "_"))

    runner(model_names, test_categories, api_sanity_check)


def get_handler(model_name):
    return HANDLER_MAP[model_name](
        model_name, temperature=0
    )  # Temperature doesn't matter for evaluation



import json
import re

def process_jsonl(file_path):
    """Process a JSONL file and return three lists: prompts, results, and ground_truths"""
    prompts = []
    results = []
    ground_truths = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in enumerate(file):
            if not line[1].strip():
                continue
            
            data = json.loads(line[1].strip())
            
            # Transform original data to new format
            new_data = {
                'question': [[{'role': 'user', 'content': extract_user_prompt(data['prompt'])}]],
                'function': []
            }
            
            # Extract function information from the original prompt
            system_part = data['prompt'].split('user\n\n')[0]
            tool_matches = re.findall(r'Tool Name: (.*?)\nTool Description: (.*?)\nTool Args:(.*?)(?=\n\n|$)', 
                                    system_part, re.DOTALL)
            
            for tool_name, description, args in tool_matches:
                param_matches = re.findall(r'-\s+(\w+)\s+\(([\w,\s]+)(?:,\s*required)?\):\s*(.*?)(?=\n|$)', args, re.DOTALL)
                properties = {}
                required_params = []
                
                for param_name, param_type, param_desc in param_matches:
                    # Clean up param_type by removing any "required" text
                    clean_type = param_type.split(',')[0].strip()
                    
                    # Check if parameter is required
                    if 'required' in param_type:
                        required_params.append(param_name)
                    
                    # Check for enum values in description
                    enum_match = re.search(r"Allowed values: '(.*?)'", param_desc)
                    if enum_match:
                        enum_values = [val.strip("'") for val in enum_match.group(1).split("', '")]
                        properties[param_name] = {
                            'type': clean_type,
                            'description': param_desc.strip(),
                            'enum': enum_values
                        }
                    # Check if parameter description contains items information
                    elif clean_type == 'array' and 'where each item should be' in param_desc:
                        properties[param_name] = {
                            'type': 'array',
                            'items': {
                                'type': 'string'  # or extract from description if needed
                            },
                            'description': param_desc.strip()
                        }
                    else:
                        properties[param_name] = {
                            'type': clean_type,
                            'description': param_desc.strip()
                        }
                
                new_data['function'].append({
                    'name': tool_name.strip(),
                    'description': description.strip(),
                    'parameters': {
                        'type': 'dict',
                        'properties': properties,
                        'required': required_params
                    }
                })
            
            prompts.append(new_data)
            
            # Extract and process label (ground truth)
            if 'label' in data:
                label_actions = parse_label(data['label'])
                if label_actions:
                    ground_truth = []
                    for action in label_actions:
                        ground_truth.append({
                            action['name']: action['arguments']
                        })
                    ground_truths.append({
                        'ground_truth': ground_truth
                    })
                else:
                    ground_truths.append({
                        'ground_truth': []
                    })
            else:
                ground_truths.append({
                    'ground_truth': []
                })
            
            # Extract and format result
            if 'predict' in data:
                try:
                    predict_data = json.loads(data['predict'])
                    if isinstance(predict_data, list):
                        actions = predict_data
                    else:
                        actions = [predict_data]
                except json.JSONDecodeError:
                    actions = parse_action_input(data['predict'])
                
                action_strings = []
                if actions:
                    for action in actions:
                        args = []
                        for key, values in action['arguments'].items():
                            if isinstance(values, list) and values:
                                value = values[0]
                                if isinstance(value, str):
                                    if not value:
                                        continue
                                    args.append(f"{key}='{value}'")
                                else:
                                    args.append(f"{key}={value}")
                        action_strings.append(f"{action['name']}({', '.join(args)})")
                
                results.append({
                    'result': '[' + ', '.join(action_strings) + ']' if action_strings else '[]'
                })
            else:
                results.append({
                    'result': '[]'
                })
    
    return prompts, results, ground_truths

def extract_user_prompt(prompt_text):
    """Extract the text between 'user' and first 'assistant' markers"""
    parts = prompt_text.split('\nuser\n\n')
    if len(parts) >= 2:
        user_part = parts[1].split('\nAction')[0]
        user_part = user_part.split('assistant')[0]
        return user_part.strip()
    return prompt_text

def parse_label(label_str):
    """Parse label string into structured format"""
    try:
        label_data = json.loads(label_str)
        if isinstance(label_data, list):
            return label_data
        return [label_data]
    except json.JSONDecodeError:
        return None

def parse_action_input(text):
    """Parse action input text into structured format"""
    actions = []
    pattern = r'Action: (.*?)\nAction Input: ({.*?})'
    matches = re.findall(pattern, text)
    for func_name, params in matches:
        try:
            params_dict = json.loads(params)
            arguments = {k: [v] if not isinstance(v, list) else v
                        for k, v in params_dict.items()}
            actions.append({
                "name": func_name,
                "arguments": arguments
            })
        except json.JSONDecodeError:
            continue
    return actions






handler = get_handler('Llama-3.1-8B-Instruct')
# prompt, model_result, possible_answer = process_jsonl('gp_ToolACE_sharegpt.jsonl')
prompt, model_result, possible_answer = process_jsonl('bfcl_multiple.jsonl')
# prompt, model_result, possible_answer = process_jsonl('bfcl_parallel.jsonl')
language = 'Python'
test_category = 'parallel'
model_name = 'Llama-3.1-8B-Instruct'
accuracy, total_count = ast_file_runner(handler,model_result,prompt,possible_answer,language,test_category,model_name)
print(accuracy)