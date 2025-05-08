# Copyright 2023 open-compass/T-Eval
# Modifications Copyright 2024 BodhiAgent
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import mmengine

def format_percentage(value):
    return f"{value * 100:.2f}%"

def format_results(result_path):
    result = mmengine.load(result_path)
    
    instruct_json = (result['instruct_json']['json_format_metric'] + result['instruct_json']['json_args_em_metric']) / 2
    instruct_str = (result['instruct_json']['string_format_metric'] + result['instruct_json']['string_args_em_metric']) / 2
    
    formatted_results = {
        'Instruct': {
            'JSON': format_percentage(instruct_json),
            'String': format_percentage(instruct_str)
        },
        'Plan': {
            'String': format_percentage(result['plan_str']['f1_score']),
            'JSON': format_percentage(result['plan_json']['f1_score'])
        },
        'Reason': {
            'String': format_percentage(result['reason_str']['thought']),
            'JSON': format_percentage(result['rru_json']['thought'])
        },
        'Retrieve': {
            'String': format_percentage(result['retrieve_str']['name']),
            'JSON': format_percentage(result['rru_json']['name'])
        },
        'Understand': {
            'String': format_percentage(result['understand_str']['args']),
            'JSON': format_percentage(result['rru_json']['args'])
        },
        'Review': {
            'String': format_percentage(result['review_str']['review_quality']),
            'JSON': format_percentage(result['review_str']['review_quality'])
        }
    }
    
    return formatted_results

def print_results(formatted_results):
    """print the result"""
    for category, values in formatted_results.items():
        print(f"{category}:")
        print(f"  String: {values['String']}")
        print(f"  JSON: {values['JSON']}")
        print()

if __name__ == '__main__':
    import sys
    from mmengine import load
    
    if len(sys.argv) != 2:
        print("Usage: python format_results.py <result_path>")
        sys.exit(1)
        
    result_path = sys.argv[1]
    formatted = format_results(result_path)
    print_results(formatted) 