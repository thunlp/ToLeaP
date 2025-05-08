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

class DocumentQA(API):
    description = 'This API answers the question from a given document url.'
    input_parameters = {
        "url": {'type': 'str', 'description': 'The url to download the document. It should end with .txt.'},
        "question": {'type': 'str', 'description': 'The question to be answered.'},
    }
    output_parameters = {
        "answer": {'type': 'str', 'description': 'The answer to the question.'},
    }
    database_name = 'QuestionAnswering'
    """
    database = {
        'url': {
            'question': 'answer',
        },
    }
    """

    def __init__(self, init_database=None) -> None:
        if init_database != None:
            self.database = init_database
        else:
            self.database = {}

    def call(self, url: str, question: str) -> dict:
        """
        Calls the API with the given parameters.

        Parameters:
        - url (str): the url to download the document. It should end with .txt.
        - question (str): the question to be answered.

        Returns:
        - response (dict): the response from the API call.
        """
        input_parameters = {
            'url': url,
            'question': question,
        }
        try:
            answer = self.answer_question(url, question)
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
            'output': answer,
            'exception': None,
        }
    
    def answer_question(self, url: str, question: str) -> str:
        """
        Answer the question from a given document url.

        Parameters:
        - url (str): the url to download the document. It should end with .txt.
        - question (str): the question to be answered.

        Returns:
        - answer (str): the answer to the question.
        """
        url = url.strip()
        question = question.strip()
        if question == '':
            raise Exception('The question is empty.')
        if url not in self.database:
            raise Exception('The document of this url failed to be processed.')
        if question not in self.database[url]:
            return "The question is too difficult to answer."
        return self.database[url][question]
    
    def check_api_call_correctness(self, response, groundtruth):
        """
        Check the correctness of the API call.
        
        Parameters:
        - response (dict): the response from the API call.
        - groundtruth (dict): the groundtruth of the API call.

        Returns:
        - correctness (bool): whether the response is correct.
        """
        
        if response['input']['url'].strip() != groundtruth['input']['url'].strip():
            return False
        if response['input']['question'].strip() != groundtruth['input']['question'].strip():
            return False
        if response['output'] != groundtruth['output']:
            return False
        if response['exception'] != groundtruth['exception']:
            return False
        return True