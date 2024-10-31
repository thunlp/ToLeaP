import openai
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import Config


def get_response_from_llama(query):
    print(f"Loading Llama model from: {Config.llama_model_path}")
    model = AutoModelForCausalLM.from_pretrained(Config.llama_model_path)
    tokenizer = AutoTokenizer.from_pretrained(Config.llama_model_path)

    inputs = tokenizer(query, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def get_model_response(query):
    if Config.use_llama:
        print("Using Llama model for inference")
        return get_response_from_llama(query)
    else:
        print("Using OpenAI API for inference")


if __name__ == "__main__":
    query = input("Please enter your query: ")
    response = get_model_response(query)
    print(f"Model Response: {response}")
