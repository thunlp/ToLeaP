# The function of this file is to download the glaiveai/glaive-function-calling data and convert it into a simplified version of the sharegpt format for the purpose of evaluation and supervised fine-tuning.
# Author: Zijun Song
# Date: 2025-04
# Copyright (c) THUNLP, Tsinghua University. All rights reserved.
# See LICENSE file in the project root for license information.

import os
import json
import re
from datasets import load_dataset

PROMPT = """You must answer in the format: 
{"name":"<specific function name>", "arguments": {"<specific parameter name>": "<specific parameter value>"}}"""

def convert_json_to_sharegpt(output_file):
    # Load dataset directly from Hugging Face
    dataset = load_dataset("glaiveai/glaive-function-calling")
    data = dataset['train']  # assuming you're working with the 'train' split

    # store all converted data
    all_conversations = []

    for item in data:
        # Regular expression to extract the complete SYSTEM content
        system_message_match = re.match(r"SYSTEM: (.*?)(?=\nUSER:|$)", item['sample'], re.DOTALL)
        if system_message_match:
            system_message = system_message_match.group(1).strip()

        # Use regular expressions to strictly extract the USER and ASSISTANT sections, and only extract the content within <functioncall>
        user_messages = []
        assistant_function_calls = []

        # Extract all <functioncall> sections from both USER and ASSISTANT
        user_messages += re.findall(r"USER: (.*?)\n", item['sample'], re.DOTALL)
        assistant_function_calls += re.findall(r"ASSISTANT: <functioncall> (.*?)\n", item['sample'], re.DOTALL)

        # If no <functioncall> in the ASSISTANT message is found, skip that data
        if not assistant_function_calls:
            continue

        # Keep only the first <functioncall> and extract the JSON content within it
        conversation_data = []
        # Keep only the first round of user messages and the first round of assistant function calls
        if user_messages:
            conversation_data.append({"from": "human", "value": user_messages[0] + '\n' + PROMPT})
        if assistant_function_calls:
            # Extract the JSON content of the first <functioncall>
            function_call_json = assistant_function_calls[0]

            # Check if the GPT content is valid JSON
            try:
                json.loads(function_call_json)
                conversation_data.append({"from": "gpt", "value": function_call_json})
            except json.JSONDecodeError:
                continue

        sharegpt_entry = {
            "conversations": conversation_data,
            "system": system_message
        }

        all_conversations.append(sharegpt_entry)

    # Save the processed data to output file
    print(f"The transformed data will be saved at {os.path.abspath(output_file)}")
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(all_conversations, outfile, ensure_ascii=False, indent=2)

# Call the function with the desired output file
convert_json_to_sharegpt("../../../data/glaive/glaive-function-calling-sharegpt.json")
