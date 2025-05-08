# Copyright 2023 AlibabaResearch/DAMO-ConvAI
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

import re

def fn(**kwargs):
    return kwargs

def get_api_call(model_output):
    api_call_pattern = r"\[(\w+)\((.*)\)\]"
    api_call_pattern = re.compile(api_call_pattern)
    match = api_call_pattern.search(model_output)
    if match:
        return match.group(0)
    else:
        return None

def parse_api_call(text):
    pattern = r"\[(\w+)\((.*)\)\]"
    match = re.search(pattern, text, re.MULTILINE)

    api_name = match.group(1)
    params = match.group(2)

    # params = params.replace('\'[', '[')
    # params = params.replace(']\'', ']')
    # params = params.replace('\'{', '{')
    # params = params.replace('}\'', '}')
    
    # param_dict = eval('fn(' + params + ')')

    param_pattern = r"(\w+)\s*=\s*['\"](.+?)['\"]|(\w+)\s*=\s*(\[.*\])|(\w+)\s*=\s*(\w+)"
    param_dict = {}
    for m in re.finditer(param_pattern, params):
        if m.group(1):
            param_dict[m.group(1)] = m.group(2)
        elif m.group(3):
            param_dict[m.group(3)] = m.group(4)
        elif m.group(5):
            param_dict[m.group(5)] = m.group(6)
    return api_name, param_dict

