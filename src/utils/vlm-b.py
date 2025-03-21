import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from PIL import Image

model_path = "/share_data/data2/workhome/chenhaotian/models/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/1b989f2c63999d7344135894d3cfa8f494116743"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, 
    device_map="auto", # TODO
    # device_map=None,
    torch_dtype="auto",
)

processor = AutoProcessor.from_pretrained(
    model_path,
    use_fast=True
)

conversations = [
    [{"role": "user", "content": [{"type": "image", "path": "towerchristchurchpower.jpg"}, {"type": "text", "text": "Describe this image."}]}],
    [{"role": "user", "content": [{"type": "image", "path": "australia.jpg"}, {"type": "text", "text": "Describe this image."}]}],
    [{"role": "user", "content": [{"type": "image", "path": "towerchristchurchpower.jpg"}, {"type": "text", "text": "Describe this image."}]}],
    [{"role": "user", "content": [{"type": "image", "path": "australia.jpg"}, {"type": "text", "text": "Describe this image."}]}]
]

# 1. Qwen2.5‑VL 视觉编码器只能接受 patch grid 整齐划一的图像，而不是任意大小
# 2. Qwen2.5‑VL 的视觉部分是基于 patch（14×14 pixels）和 temporal patch（2）的组合——合并尺寸因子就是 14×2=28
# 因此，无论宽度还是高度，都必须被 28 整除，否则模型无法把图像正确切成 patch grid
# 3. 保证图像至少有 16×16 个 patch（16×28=448），对应至少 256 个视觉 token，避免输入过小造成模型理解能力下降
def safe_image_processing(image_path):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    
    new_w = max(448, (w // 28) * 28)
    new_h = max(448, (h // 28) * 28)
    resized_img = img.resize((new_w, new_h))

    return resized_img

def batch_inference(conversations, batch_size=2):
    all_outputs = []
    device = next(model.parameters()).device
    for i in range(0, len(conversations), batch_size):
        print(i)
        batch = conversations[i:i+batch_size]
        
        images = [safe_image_processing(conv[0]["content"][0]["path"]) for conv in batch] # -> PIL.Image
        # Qwen2.5‑VL 的 processor 在把图像和文本拼在一起时，会自动在文本流中插入它专用的 image‑placeholder token（通常是 <ImageHere> 或 tokenizer.image_token_id 对应的 ID）
        prompts = [
            processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            for conv in batch
        ]

        inputs = processor(
            text=prompts,
            images=images, # PIL.Image -> Base64
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            top_p=0.9,
            temperature=0.6,
            use_cache=False,
        )

        decoded = processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True
        )
        
        all_outputs.extend(decoded)
    return all_outputs

results = batch_inference(conversations, batch_size=2)
print(results)
