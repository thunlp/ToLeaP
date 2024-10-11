

# 留下掌握的，去掉没掌握的，写出通过api取用信息的各种功能


import datetime
import hashlib
import importlib
import json
import multiprocessing
import os
import re
import sqlite3
import ssl
import time
import urllib.request
import uuid
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple, Union, Any, Literal
from datetime import datetime, timezone

import numpy as np
import tiktoken
from scipy.spatial.distance import cosine

from cfg.config import Config
from utils.log import Logger, LogColors

try:
    import openai
except ImportError:
    Logger().warning("openai is not installed.")


class Message:
    def __init__(
        self,
        api_key=None,
        api_model=None,
        database_url=None,
    ) -> None:
        self.cfg = Config()

        self.api_key = api_key if api_key is not None else self.cfg.api_key
        self.api_model = api_model if api_model is not None else self.cfg.api_model
        self.encoder = tiktoken.encoding_for_model(self.api_model)
        self.database_url = database_url if database_url is not None else self.cfg.database_url

        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.database_url)

    def build_messages(
        self,
        user_prompt,
        system_prompt=None,
        former_messages=[],
        shrink_multiple_break=False,
    ):
        """build the messages to avoid implementing several redundant lines of code"""
        # shrink multiple break will recursively remove multiple breaks(more than 2)
        if shrink_multiple_break:
            while "\n\n\n" in user_prompt:
                user_prompt = user_prompt.replace("\n\n\n", "\n\n")
            while "\n\n\n" in system_prompt:
                system_prompt = system_prompt.replace("\n\n\n", "\n\n")
        system_prompt = self.cfg.default_system_prompt if system_prompt is None else system_prompt
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]
        messages.extend(former_messages[-1 * self.cfg.max_past_message_include :])
        messages.append(
            {
                "role": "user",
                "content": user_prompt,
            }
        )
        return messages

    def get_response(
        self,
        user_prompt,
        system_prompt=None,
        former_messages=[],
        shrink_multiple_break=False,
        functions=None,
        **kwargs,
    ):
        messages = self.build_messages(user_prompt, system_prompt, former_messages, shrink_multiple_break)
        if functions:
            completion = self.client.chat.completions.create(model=self.api_model,
                                                            messages=messages,
                                                            functions=functions,
                                                            function_call="auto")
            response = completion.choices[0].message.content
            if response is None:
                response = [completion.choices[0].message.function_call.name,
                            completion.choices[0].message.function_call.arguments]
            return response
        else:
            completion = self.client.chat.completions.create(model=self.api_model,
                                                             messages=messages)
            response = completion.choices[0].message.content
            return response


