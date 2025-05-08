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

import os
import importlib

# Get the directory path of the "apis" folder
apis_dir = os.path.dirname(os.path.abspath(__file__))

# Get a list of all the files in the "apis" folder
api_files = [f[:-3] for f in os.listdir(apis_dir) if f.endswith(".py") and f != "__init__.py"]

# Import all classes in the files in the "apis" folder
for api_file in api_files:
    try:
        module = importlib.import_module(f".{api_file}", package="apis")
        globals().update({k: v for k, v in module.__dict__.items() if not k.startswith("__")})
    except ImportError:
        # Handle ImportError, if necessary
        pass
