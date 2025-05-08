# Copyright 2024 THUNLP-MT/StableToolBench
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

class base_env:

    def __init__(self):
        self.task_description = ""
        self.input_description = ""
        self.tool_names = []
        self.functions = []

    def restart(self):
        '''
        Restrat the environment
        '''
        raise NotImplementedError
    
    def get_score(self):
        '''
        Get the value of the current state
        A fake function, used to search in oracle mode, which is not actually used (and impossible to obtain)
        '''
        raise NotImplementedError

    def step(self, action, input_str):
        '''
        Perform an interaction in natural language mode
        return value (output str, status code)
        '''
        raise NotImplementedError
    
    def check_success(self):
        '''
        Returns 1 if successful, otherwise returns 0
        '''
        raise NotImplementedError
    
    def to_json(self):
        raise NotImplementedError