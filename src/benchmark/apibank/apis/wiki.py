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

class Wiki(API):
    description = 'This API for searching a keyword in Wikipedia.'
    input_parameters = {
        "keyword": {'type': 'str', 'description': 'The keyword to search.'},
    }
    output_parameters = {
        "results": {'type': 'dict', 'description': 'The list of results. Format be like {"url": "xxx", "summary": "xxx", "content": "xxx"}'},
    }
    database_name = 'Wiki'


    def __init__(self, init_database=None) -> None:
        if init_database != None:
            self.database = init_database
        else:
            self.database = {}

    def call(self, keyword: str) -> dict:
        """
        Calls the API with the given parameters.

        Parameters:
        - keyword (str): the keyword to search.

        Returns:
        - response (dict): the response from the API call.
        """
        input_parameters = {
            'keyword': keyword,
        }
        try:
            results = self.search(keyword)
        except Exception as e:
            exception = str(e)
            return {
                'api_name': self.__class__.__name__,
                'input': input_parameters,
                'output': None,
                'exception': exception,
            }
        return {
            'api_name': self.__class__.__name__,
            'input': input_parameters,
            'output': results,
            'exception': None,
        }
    
    def search(self, keyword: str) -> dict:
        """
        Search for a given keyword.

        Parameters:
        - keyword (str): the keyword to search.

        Returns:
        - results (dict): the results from the search.
        """
        keyword = keyword.replace('_', ' ').strip().lower()
        if keyword in self.database:
            return {
                "url": self.database[keyword]["url"],
                "summary": self.database[keyword]["summary"],
                # "content": self.database[keyword]["content"],
            }
        else:
            raise Exception('Keyword not found.')
        
    def check_api_call_correctness(self, response, groundtruth):
        """
        Checks if the API call is correct.

        Parameters:
        - response (dict): the response from the API call.
        - groundtruth (dict): the groundtruth.

        Returns:
        - correctness (bool): whether the API call is correct.
        """
        if response['api_name'] != groundtruth['api_name']:
            return False
        if response['output'] != groundtruth['output']:
            return False
        if response['exception'] != groundtruth['exception']:
            return False
        return True
        