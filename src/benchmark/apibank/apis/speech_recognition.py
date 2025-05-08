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

class SpeechRecognition(API):
    description = 'This API recognizes the speech from a given audio url.'
    input_parameters = {
        "url": {'type': 'str', 'description': 'The url to download the audio. It should end with .wav.'},
    }
    output_parameters = {
        "transcript": {'type': 'str', 'description': 'The transcript of the audio.'},
    }
    database_name = 'SpeechRecognition'

    def __init__(self, init_database=None) -> None:
        if init_database != None:
            self.database = init_database
        else:
            self.database = {}

    def call(self, url: str) -> dict:
        """
        Calls the API with the given parameters.

        Parameters:
        - url (str): the url to download the audio. It should end with .wav.

        Returns:
        - response (dict): the response from the API call.
        """
        input_parameters = {
            'url': url,
        }
        try:
            transcript = self.recognize_speech(url)
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
            'output': transcript,
            'exception': None,
        }
    
    def recognize_speech(self, url: str) -> str:
        """
        Recognize the speech from a given audio url.

        Parameters:
        - url (str): the url to download the audio. It should end with .wav.

        Returns:
        - transcript (str): the transcript of the audio.
        """

        url = url.strip()
        if not url.endswith('.wav'):
            raise Exception('The url should end with .wav.')
        if url in self.database:
            return self.database[url]
        else:
            raise Exception('The audio of this url failed to be processed.')
        

    def check_api_call_correctness(self, response, groundtruth):
        """
        Check if the response is correct.
        
        Parameters:
        - response (dict): the response from the API call.
        - groundtruth (dict): the groundtruth of the API call.

        Returns:
        - correctness (bool): True if the response is correct, False otherwise.
        """

        if response['input'] != groundtruth['input']:
            return False
        if response['output'] != groundtruth['output']:
            return False
        if response['exception'] != groundtruth['exception']:
            return False
        return True