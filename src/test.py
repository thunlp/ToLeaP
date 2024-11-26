from utils.llm import Message

user_prompt = "What is the capital of France?"

message = Message()
response = message.get_response(user_prompt)

print("Response from Llama Model:", response)
