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

from benchmark.apibank.apis.api import API
import json

class CheckToken(API):

    description = 'Check the user token.'
    input_parameters = {
        'token': {'type': 'str', 'description': 'The token of the user.'},
    }
    output_parameters = {
        'username': {'type': 'str', 'description': 'The username of the user.'},
    }
    database_name = 'Account'
    def __init__(self, init_database=None) -> None:
        if init_database != None:
            self.database = init_database
        else:
            self.database = {}

    def check_token(self, token: str) -> str:
        """
        Checks the user token.

        Parameters:
        - token (str): the token of the user.

        Returns:
        - username (str): the username of the user.
        """
        for username in self.database:
            if self.database[username]['token'] == token:
                return username
        raise Exception('The token is invalid.')
    
    def call(self, token: str) -> dict:
        """
        Calls the API with the given parameters.

        Parameters:
        - token (str): the token of the user.

        Returns:
        - response (dict): the response from the API call.
        """
        input_parameters = {
            'token': token,
        }
        try:
            username = self.check_token(token)
        except Exception as e:
            exception = str(e)
            return {
                'input': input_parameters,
                'output': None,
                'exception': exception,
            }
        return {
            'input': input_parameters,
            'output': {
                'username': username,
            },
            'exception': None,
        }
    
    def check_api_call_correctness(self, response, groundtruth) -> bool:
        """
        Checks the correctness of the API call.

        Parameters:
        - response (dict): the response from the API call.
        - groundtruth (dict): the groundtruth response from the API call.

        Returns:
        - correctness (bool): whether the API call is correct.
        """
        if response['input'] == groundtruth['input'] and response['output'] == groundtruth['output'] and response['exception'] == groundtruth['exception']:
            return True
        else:
            return False
        
        