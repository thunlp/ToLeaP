from cfg.config import Config
from utils.llm import Message

cfg = Config()

user_prompt = "What's the weather like in Boston today?"

functions = [
  {
    "name": "get_current_weather",
    "description": "Get the current weather in a given location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["location"],
    },
  }
]

message = Message()

response = message.get_response(user_prompt=user_prompt,functions=functions)

print(response)