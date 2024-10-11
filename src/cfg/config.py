# This file is ...
# Author: Haotian Chen, ...
# Date: 2024-08
# Copyright (c) THUNLP, Tsinghua University. All rights reserved. 
# See LICENSE file in the project root for license information. 

import os
from dotenv import load_dotenv

# loading variables stored in .env
load_dotenv()

class Config:
    api_key = os.getenv("API_KEY")
    api_model = os.getenv("API_MODEL")
    database_url = os.getenv("DATABASE_URL")
    max_past_message_include = 128000
    default_system_prompt = ''



