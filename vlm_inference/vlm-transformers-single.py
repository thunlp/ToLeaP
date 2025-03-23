import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor

model_path = "/share_data/data2/workhome/chenhaotian/models/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/1b989f2c63999d7344135894d3cfa8f494116743"
# Load the model in half-precision on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, 
    device_map="auto",
)

min_pixels = 256*28*28
max_pixels = 1024*28*28 
processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)

conversations = [
    [{"role": "user", "content": [{"type": "image", "path": "towerchristchurchpower.jpg"}, {"type": "text", "text": "Describe this image."}]}],
    [{"role": "user", "content": [{"type": "image", "path": "australia.jpg"}, {"type": "text", "text": "Describe this image."}]}],
    [{"role": "user", "content": [{"type": "image", "path": "towerchristchurchpower.jpg"}, {"type": "text", "text": "Describe this image."}]}],
    [{"role": "user", "content": [{"type": "image", "path": "australia.jpg"}, {"type": "text", "text": "Describe this image."}]}]
]

# Preparation for batch inference
for i in range(0, len(conversations)):
    inputs = processor.apply_chat_template(
        conversations[i],
        video_fps=1,
        add_generation_prompt=True,
        tokenize=True,
        padding=True,
        truncation=True,
        return_dict=True,
        return_tensors="pt"
    )

    # Batch Inference
    output_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print(output_text)