import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

VLM_PATH = 'Qwen/Qwen2.5-VL-7B-Instruct'
IMAGE_SIZE = 448
IMAGE_TOKEN = '<|vision_start|><|image_pad|><|vision_end|>'


def load(device):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(VLM_PATH, torch_dtype=torch.bfloat16).to(device).eval()
    processor = AutoProcessor.from_pretrained(VLM_PATH)
    return model, processor


@torch.no_grad()
def generate(model, processor, prompt, images=None, max_new_tokens=1024):
    text = processor.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]


def describe(model, processor, vr, frame_indices, prompt):
    images = [Image.fromarray(vr[i].asnumpy()).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE)) for i in frame_indices]
    data = ''.join(f'Frame{i + 1}: {IMAGE_TOKEN}\n' for i in range(len(images)))
    return generate(model, processor, prompt.replace('$Data', data), images)
