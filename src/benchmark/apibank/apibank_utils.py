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

from tenacity import (
    retry,
    wait_random_exponential,
    retry_if_exception_type,
)

import requests
import json
from requests.exceptions import ConnectionError

class RateLimitReached(Exception):
    pass

class OfficialError(Exception):
    pass

class RecoverableError(Exception):
    pass

class KeysBusyError(Exception):
    pass

class ChatGPTWrapper:
    def __init__(self, api_key='', proxies=None) -> None:
        # Set the request parameters
        self.url = 'https://api.openai.com/v1/chat/completions'
        # Set the header
        self.header = {
            "Content-Type": "application/json",
            "Authorization": 'Bearer {}'.format(api_key)
        }
        self.proxies = proxies

    @retry(wait=wait_random_exponential(min=1, max=60), retry=retry_if_exception_type((RateLimitReached, RecoverableError, OfficialError, ConnectionError)))
    def call(self, messages, **kwargs):
        query = {
            "model": "gpt-3.5-turbo",
            "messages": messages
        }
        query.update(kwargs)

        # Make the request
        if self.proxies:
            response = requests.post(self.url, headers=self.header, data=json.dumps(query), proxies=self.proxies)
        else:
            response = requests.post(self.url, headers=self.header, data=json.dumps(query))
        response = response.json()
        if 'error' in response and 'Rate limit reached' in response['error']['message']:
            raise RateLimitReached()
        elif 'choices' in response:
            return response
        else:
            if 'error' in response:
                print(response['error']['message'])
                if response['error']['message'] == 'The server had an error while processing your request. Sorry about that!':
                    raise RecoverableError(response['error']['message'])
                else:
                    raise OfficialError(response['error']['message'])
            else:
                raise Exception('Unknown error occured. Json: {}'.format(response))
            
            
class DavinciWrapper:
    def __init__(self, api_key='') -> None:
        # Set the request parameters
        self.url = 'https://api.openai.com/v1/completions'
        # Set the header
        self.header = {
            "Content-Type": "application/json",
            "Authorization": 'Bearer {}'.format(api_key)
        }

    @retry(wait=wait_random_exponential(min=1, max=60), retry=retry_if_exception_type((RateLimitReached, RecoverableError, OfficialError, ConnectionError)))
    def call(self, messages, **kwargs):
        # messages to prompt
        prompt = ''
        for message in messages:
            prompt += message['role'] + ': ' + message['content'] + '\n'

        query = {
            "model": "davinci",
            "prompt": prompt
        }
        query.update(kwargs)

        # Make the request
        response = requests.post(self.url, headers=self.header, data=json.dumps(query))
        response = response.json()
        if 'error' in response and 'Rate limit reached' in response['error']['message']:
            raise RateLimitReached()
        elif 'choices' in response:
            return response
        else:
            if 'error' in response:
                print(response['error']['message'])
                if response['error']['message'] == 'The server had an error while processing your request. Sorry about that!':
                    raise RecoverableError(response['error']['message'])
                else:
                    raise OfficialError(response['error']['message'])
            else:
                raise Exception('Unknown error occured. Json: {}'.format(response))

class GPT4Wrapper(ChatGPTWrapper):
    @retry(wait=wait_random_exponential(min=1, max=60), retry=retry_if_exception_type((RateLimitReached, RecoverableError, OfficialError, ConnectionError)))
    def call(self, messages, **kwargs):
        query = {
            "model": "gpt-4-0314",
            "messages": messages
        }
        query.update(kwargs)

        # Make the request
        if self.proxies:
            response = requests.post(self.url, headers=self.header, data=json.dumps(query), proxies=self.proxies)
        else:
            response = requests.post(self.url, headers=self.header, data=json.dumps(query))        
        response = response.json()
        if 'error' in response and 'Rate limit reached' in response['error']['message']:
            raise RateLimitReached()
        elif 'choices' in response:
            return response
        else:
            if 'error' in response:
                print(response['error']['message'])
                if response['error']['message'] == 'The server had an error while processing your request. Sorry about that!':
                    raise RecoverableError(response['error']['message'])
                else:
                    raise OfficialError(response['error']['message'])
            else:
                raise Exception('Unknown error occured. Json: {}'.format(response))

