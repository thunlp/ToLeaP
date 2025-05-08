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

from .api import API
import datetime

class GetToday(API):
    description = 'This API gets the current date.'
    input_parameters = {}
    output_parameters = {
        'date': {'type': 'str', 'description': 'The current date. Format: %Y-%m-%d'},
    }
    def call(self, **kwargs) -> dict:
        # today is 2023-03-31
        return {'api_name': self.__class__.__name__, 'input': None, 'output': "2023-03-31", 'exception': None}
        # return {'api_name': self.__class__.__name__, 'input': None, 'output': datetime.datetime.now().strftime('%Y-%m-%d'), 'exception': None}
    def check_api_call_correctness(self, response, groundtruth) -> bool:
        return response['output'] != None and response['exception'] == None and response['input'] == None