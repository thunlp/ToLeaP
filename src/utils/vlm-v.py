from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

MODEL_PATH = "/share_data/data2/models/Qwen/Qwen2.5-VL-7B-Instruct"

# 1. AssertionError: assert "factor" in rope_scaling 
# 根源在于 vLLM 在读取模型的 config.json 时发现了一个不完整（或过时）的 RoPE‑scaling 配置
# 最简单的修复方法是手动补全 config.json 中的 rope_scaling 字段
import json, pathlib

config_path = pathlib.Path("/share_data/data2/models/Qwen/Qwen2.5-VL-7B-Instruct/config.json")
cfg = json.loads(config_path.read_text())
cfg["rope_scaling"] = {
    "type": "yarn",
    "factor": 4.0,
    "original_max_position_embeddings": cfg.get("max_position_embeddings", 32768)
}
config_path.write_text(json.dumps(cfg, indent=2))
print("✅ Updated rope_scaling in config.json")

# 1️⃣ Load processor + vLLM engine
processor = AutoProcessor.from_pretrained(MODEL_PATH, min_pixels=256*28*28, max_pixels=1024*28*28)
llm = LLM(
    model=MODEL_PATH,
    trust_remote_code=True,
    limit_mm_per_prompt={"image": 1}, # 每个 prompt 最多 1 张图
)
# 2. ValueError: Model architectures ['Qwen2_5_VLForConditionalGeneration'] are not supported 
# pip install --upgrade https://vllm-wheels.s3.us-west-2.amazonaws.com/nightly/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl

# 2️⃣ Build batch of prompts + images
conversations = [
    [{"role": "user", "content": [{"type": "image", "path": "towerchristchurchpower.jpg"}, {"type": "text", "text": "Describe this image."}]}],
    [{"role": "user", "content": [{"type": "image", "path": "australia.jpg"}, {"type": "text", "text": "Describe this image."}]}],
]

requests = []
for conv in conversations:
    prompt_str = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
    image_path = conv[0]["content"][0]["path"]
    image = Image.open(image_path).convert("RGB")
    requests.append({"prompt": prompt_str, "multi_modal_data": {"image": image}})

# 3️⃣ Generate
outputs = llm.generate(requests, sampling_params=SamplingParams(max_tokens=128))

# 4️⃣ Extract text
for out in outputs:
    print(out.outputs[0].text.strip())
