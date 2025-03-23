import subprocess
import time
import json
from io import BytesIO
from contextlib import contextmanager

import requests
from PIL import Image
from transformers import AutoProcessor

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(current_dir, '..')
sys.path.append(utils_dir)
from utils.llm import LLM
from vllm import SamplingParams

MODEL_PATH = "/share_data/data2/models/Qwen/Qwen2.5-VL-7B-Instruct"
HOST = "0.0.0.0"
PORT = 11111
SERVER_URL = f"http://{HOST}:{PORT}"

processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    min_pixels=256 * 28 * 28,
    max_pixels=1024 * 28 * 28,
)

llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=4,
    gpu_memory_utilization=0.75,       # 降低预分配比例
    max_input_tokens=4096,               # 缩短上下文长度，大幅降低 KV-cache 需求
)

sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.001,
    repetition_penalty=1.05,
    max_tokens=1024,
    stop_token_ids=[],
)

def multimodal_infer(image_path: str, text_prompt: str, max_tokens: int = 128) -> str:
    conv = [{
        "role": "user",
        "content": [
            {"type": "image", "path": image_path},
            {"type": "text", "text": text_prompt}
        ]
    }]
    prompt = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)

    buf = BytesIO()
    Image.open(image_path).convert("RGB").save(buf, format="PNG")
    buf.seek(0)

    payload = {
        "prompt": prompt,
        "sampling_params": {
            "max_tokens": max_tokens,
            "temperature": sampling_params.temperature,
            "top_p": sampling_params.top_p,
            "repetition_penalty": sampling_params.repetition_penalty
        }
    }
    files = {"image": ("image.png", buf, "image/png")}
    headers = {"Authorization": f"Bearer {llm.api_key}"}

    url = f"http://{HOST}:{PORT}/v1/chat/completions"
    resp = requests.post(url, json=payload, files=files, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

if __name__ == "__main__":
    examples = [
        ("d8152ad6-e4d5-4c12-8bb7-8d57dc10c6de.png", "I have the Standard plan in the image below, and I just uploaded 60 equally sized files and got a message that I'm 100GB over the limit. I have 980 more files of the same size to upload. What is the average additional cost per file in dollar that goes over my current plan limit rounded to the nearest cent if I have to upgrade to the minimum possible plan to store them all? Answer with the following format: x.xx"),
        ("df6561b2-7ee5-4540-baab-5095f742716a.png", "When you take the average of the standard population deviation of the red numbers and the standard sample deviation of the green numbers in this image using the statistics module in Python 3.11, what is the result rounded to the nearest three decimal points?"),
    ]

    with llm.start_server():
        for img, prompt in examples:
            print(f"\n== {img} | Prompt: {prompt} ==")
            print(multimodal_infer(img, prompt))
