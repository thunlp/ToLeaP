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
import datetime

class TimedSwitch(API):
    
    description = 'This API for setting a timed switch for a smart device.'
    input_parameters = {
        "name": {'type': 'str', 'description': 'The name of the smart device.'},
        "time": {'type': 'str', 'description': 'The time to switch the device on or off. Format: %Y-%m-%d %H:%M:%S'},
        "on": {'type': 'bool', 'description': 'Whether to switch the device on or off.'},
    }

    output_parameters = {
        'status': {'type': 'str', 'description': 'Whether the time switch is successful.'},
    }

    database_name = 'TimeSwitch'

    def __init__(self, init_database=None) -> None:
        if init_database != None:
            self.database = init_database
        else:
            self.database = {}

    def call(self, name: str, time: str, on: bool) -> dict:
        """
        Calls the API with the given parameters.

        Parameters:
        - name (str): the name of the smart device.
        - time (str): the time to switch the device on or off.
        - on (bool): whether to switch the device on or off.

        Returns:
        - response (dict): the response from the API call.
        """
        input_parameters = {
            'name': name,
            'time': time,
            'on': on,
        }
        try:
            status = self.timed_switch(name, time, on)
        except Exception as e:
            exception = str(e)
            return {
                'api_name': self.__class__.__name__,
                'input': input_parameters,
                'output': None,
                'exception': exception,
            }
        else:
            return {
                'api_name': self.__class__.__name__,
                'input': input_parameters,
                'output': status,
                'exception': None,
            }
        
    def format_check(self, time):
        time = time.strip()
        split_time = time.split('-')
        if len(split_time) == 3:
            if len(split_time[0]) == 4:
                pass
            else:
                split_time[0] = split_time[0].zfill(4)
            time = '-'.join(split_time)
        try:
            time = datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')
        except Exception as e:
            time = e
        return time
    
    def timed_switch(self, name: str, time: str, on: bool) -> bool:
        """
        Switches a smart device on or off at a specified time.

        Parameters:
        - name (str): the name of the smart device.
        - time (str): the time to switch the device on or off.
        - on (bool): whether to switch the device on or off.

        Returns:
        - status (str): whether the time switch is successful.
        """
        
        name = name.strip().lower()
        if name == '':
            raise Exception('Name cannot be empty.')
        time = self.format_check(time)
        if isinstance(time, Exception):
            raise time
        if name not in self.database:
            self.database[name] = []
            self.database[name].append(
                {
                    'time': time,
                    'on': on,
                }
            )
            return "success"
        else:
            for i in range(len(self.database[name])):
                if self.database[name][i]['time'] == time:
                    self.database[name][i]['on'] = on
                    return "success"
            self.database[name].append(
                {
                    'time': time,
                    'on': on,
                }
            )
            return "success"
        
    
    def check_api_call_correctness(self, response, groundtruth) -> bool:
        response_name = response['input']['name']
        groundtruth_name = groundtruth['input']['name']
        response_time = response['input']['time']
        groundtruth_time = groundtruth['input']['time']
        response_on = response['input']['on']
        groundtruth_on = groundtruth['input']['on']

        response_name = response_name.strip().lower()
        groundtruth_name = groundtruth_name.strip().lower()
        response_time = self.format_check(response_time)
        groundtruth_time = self.format_check(groundtruth_time)

        if response_name != groundtruth_name:
            return False
        if response_time != groundtruth_time:
            return False
        if response_on != groundtruth_on:
            return False
        if response['output'] != groundtruth['output']:
            return False
        if response['exception'] != groundtruth['exception']:
            return False
        return True