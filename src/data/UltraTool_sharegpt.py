import json

def convert_to_sharegpt_format(data_list):
    sharegpt_format = []

    for data in data_list:
        # Extracting the question as a conversation
        conversation = []
        conversation.append({
            "from": "human",
            "value": data["question"]
        })

        # Adding function calls and observations for each tool
        if "tools" in data and data["tools"]:
            for tool in data["tools"]:
                function_call_value = {
                    "name": tool["name"],
                    "arguments": {key: "" for key in tool["arguments"]["properties"].keys()}
                }
                conversation.append({
                    "from": "function_call",
                    "value": json.dumps(function_call_value)
                })
                conversation.append({
                    "from": "observation",
                    "value": ""
                })

        # Converting plan steps into alternating human and GPT responses
        for step in data["plan"]:
            conversation.append({
                "from": "gpt",
                "value": f"Step: {step['step']}, Tool: {step['tool']}"
            })
            conversation.append({
                "from": "human",
                "value": ""
            })

        # Removing the last empty human response if unnecessary
        if conversation[-1]["from"] == "human" and conversation[-1]["value"] == "":
            conversation.pop()

        # Ensuring the conversation ends with GPT
        if conversation[-1]["from"] == "human":
            conversation.pop()

        # Preparing the tools information
        tools_info = json.dumps([tool["description"] for tool in data.get("tools", [])])

        # Adding the conversation and tools to the final format
        sharegpt_format.append({
            "conversations": conversation,
            "tools": tools_info
        })

    return sharegpt_format

# Reading input from a JSON file
with open('dev.json', 'r') as infile:
    data_list = json.loads("[" + infile.read().replace("}\n{", "},{") + "]") # Handle multiple JSON objects and take the first two items for testing

# Converting the data to ShareGPT format
sharegpt_data = convert_to_sharegpt_format(data_list)

# Writing the result to an output JSON file
with open('output.json', 'w') as outfile:
    json.dump(sharegpt_data, outfile, indent=2)

