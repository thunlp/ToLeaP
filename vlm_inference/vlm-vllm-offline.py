import json
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams # vllm >= 0.7.3
from tqdm import tqdm

INPUT_JSONL = "gaia_valid_with_text_or_pic_tasks.jsonl"                   
OUTPUT_JSON = "Qwen2.5-VL-7B-Instruct_valid_text_or_pic_outputs.json"
MODEL_PATH = "/share_data/data2/models/Qwen/Qwen2.5-VL-7B-Instruct"

# 1️⃣ Load processor + vLLM engine
processor = AutoProcessor.from_pretrained(
    MODEL_PATH, 
    min_pixels=256*28*28, 
    max_pixels=1024*28*28, 
    trust_remote_code=True
)

llm = LLM(
    model=MODEL_PATH,
    trust_remote_code=True,
    tensor_parallel_size=4,
    device="cuda",
    gpu_memory_utilization=0.75,       # 降低预分配比例
    max_model_len=4096,               # 缩短上下文长度，大幅降低 KV-cache 需求
    max_num_seqs=1,                   # 限制同时处理的序列数
    max_num_batched_tokens=2048,      # chunked prefill 推荐值
    enable_chunked_prefill=True,       # 减少内存碎片
    # kv_cache_dtype="fp8",  # 将KV缓存精度降为FP8
    dtype="bfloat16",  # 模型计算使用BF16混合精度
    limit_mm_per_prompt={"image": 1, "video": 0}, # 每个 prompt 最多 1 张图
)

sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.001,
    repetition_penalty=1.05,
    max_tokens=1024,
    stop_token_ids=[],
)

def build_requests(path):
    reqs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            fname = obj.get("file_name","").strip()
            question = obj.get("Question","").strip()
            if not question:
                continue

            content = [{"type": "text", "text": question}]
            if fname:
                content.insert(0, {"type": "image", "path": fname})

            conv = [{"role": "user", "content": content}]
            prompt = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            entry = {"prompt": prompt}
            if fname:
                entry["multi_modal_data"] = {"image": Image.open(fname).convert("RGB")}
            reqs.append((obj, entry))
    return reqs

requests = build_requests(INPUT_JSONL)
print(f"🔍 Total prompts: {len(requests)}") # OK

results = []
skipped = 0
for original, req in tqdm(requests, desc="Generating", unit="item"):
    try:
        output = llm.generate(req, sampling_params=sampling_params, use_tqdm=False)[0]
        results.append({
            "Question": original["Question"],
            "file_name": original["file_name"],
            "response": output.outputs[0].text.strip()
        })
    except ValueError as e:
        if "too long to fit into the model" in str(e):
            skipped += 1
            print(f"⚠️ Prompt too long, skipped: {original['Question'][:50]}...")
            continue
        else:
            raise

with open(OUTPUT_JSON, "w", encoding="utf-8") as out:
    json.dump(results, out, ensure_ascii=False, indent=2)

print(f"✅ Done — saved {len(results)} entries to {OUTPUT_JSON}")
print(f"⚠️ Skipped {skipped} prompts due to length") 