import argparse
import functools
import json
import os
import logging as std_logging
import sys

import numpy as np
import torch

import common
import qwen_vlm

ANOMALY_TYPE = 'A worker crouches or reaches down to pick up something below'
BASE_DATASET = 'shtech'
MODEL_PATH = 'Qwen/Qwen3.5-27B'
QUANTIZATION = '8bit'
KEY_FRAME_ONLY = True
THRESHOLD = 0.5
PROMPT_PATH = None

REPO = common.ROOT / 'third_party' / 'anyanomaly'
WORK = common.WORK_DIR / 'anyanomaly'
USE_QWEN = MODEL_PATH.startswith('Qwen/')
USE_QWEN35 = MODEL_PATH.startswith('Qwen/Qwen3.5')
SCORE_DIR = WORK / ('scores' + ('_keyframe' if KEY_FRAME_ONLY else '') + ('_' + MODEL_PATH.split('/')[-1] if USE_QWEN else '')
                    + ('_' + PROMPT_PATH.stem if PROMPT_PATH else ''))
SCORE_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)
sys.path.insert(0, str(REPO))

import clip
from scipy.ndimage import gaussian_filter1d
from config import update_config
from data_loader import clip_path_loader
from utils import generate_output, load_keyword_list
from functions.text_func import make_text_embedding
from functions.MiniCPM_func import load_lvlm, lvlm_test, make_instruction
from functions.attn_func import winclip_attention
from functions.grid_func import grid_generation
from functions.key_func import KFS
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, logging
logging.set_verbosity_error()
std_logging.getLogger('bitsandbytes').setLevel(std_logging.ERROR)


def qwen_test(model, processor, instruction, image_path, image=None):
    if image is None:
        image = Image.open(image_path)
    image = image.convert('RGB')
    image.thumbnail((qwen_vlm.IMAGE_SIZE, qwen_vlm.IMAGE_SIZE))
    return qwen_vlm.generate(model, processor, qwen_vlm.IMAGE_TOKEN + '\n' + instruction, [image], max_new_tokens=64)


def load_qwen35():
    from transformers import AutoModelForMultimodalLM, BitsAndBytesConfig
    quant = {
        '8bit': BitsAndBytesConfig(load_in_8bit=True, llm_int8_skip_modules=['visual', 'lm_head']),
        '4bit': BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16,
                                   llm_int8_skip_modules=['visual', 'lm_head']),
    }.get(QUANTIZATION)
    model = AutoModelForMultimodalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, quantization_config=quant, device_map={'': 0}).eval()
    return model, AutoProcessor.from_pretrained(MODEL_PATH)


@torch.no_grad()
def qwen35_test(model, processor, instruction, image_path, image=None):
    if image is None:
        image = Image.open(image_path)
    image = image.convert('RGB')
    image.thumbnail((qwen_vlm.IMAGE_SIZE, qwen_vlm.IMAGE_SIZE))
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    return processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]


def main():
    assert ANOMALY_TYPE, "ANOMALY_TYPE に検出したい異常のテキストを設定してください"
    cfg = update_config(argparse.Namespace(
        dataset=BASE_DATASET, type=ANOMALY_TYPE, multiple='-' in ANOMALY_TYPE, prompt_type=3, anomaly_detect=True,
        calc_auc=False, calc_video_auc=False, clip_length=24, template_adaption=True, class_adaption=True, kfs_num=4,
        lge_scale=True, mid_scale=True, sml_scale=True, stride=False, model_path=MODEL_PATH, grid_search=False,
        sigma_range='1,30', weight_step=0.1, sigma=15, alpha=0.6, beta=0.3, gamma=0.1))
    keyword_list = load_keyword_list(cfg)

    todo = [n for n in common.video_names() if not (SCORE_DIR / f"{n}.npy").exists()]
    if todo:
        device = torch.device("cuda")
        if USE_QWEN35:
            model, processor = load_qwen35()
            lvlm = functools.partial(qwen35_test, model, processor)
        elif USE_QWEN:
            qwen_vlm.VLM_PATH = MODEL_PATH
            model, processor = qwen_vlm.load(device)
            lvlm = functools.partial(qwen_test, model, processor)
        else:
            tokenizer, model = load_lvlm(cfg.model_path, device)
            model.chat = functools.partial(model.chat, processor=AutoProcessor.from_pretrained(f"openbmb/{cfg.model_path}", trust_remote_code=True))
            lvlm = functools.partial(lvlm_test, tokenizer, model)
        clip_model, preprocess = clip.load('ViT-B/32', device=device)
        kfs = KFS(cfg.kfs_num, cfg.clip_length, clip_model, preprocess, device)

    for n in tqdm(todo, desc="videos", position=0):
        num_frames = common.extract_frames(n)
        cp_loader = clip_path_loader(str(common.FRAME_DIR / n), cfg.clip_length)
        predicted, predicted_wa, predicted_tc = [], [], []
        log = []

        for c in tqdm(range(len(cp_loader)), desc=n, position=1, leave=False):
            cp = cp_loader[c]
            max_score = 0
            max_score_wa = 0
            max_score_tc = 0

            for k_i, keyword in enumerate(keyword_list):
                instruction, instruction_tc = make_instruction(cfg, keyword, True)
                if PROMPT_PATH:
                    instruction = PROMPT_PATH.read_text(encoding='utf-8')

                indice = kfs.call_function(cp, keyword)
                key_image_path = cp[indice[0]]
                image_paths = [cp[idx] for idx in indice[1:]]

                response = lvlm(instruction, key_image_path, None)
                max_score = max(max_score, generate_output(response)['score'])
                log.append({'clip': c, 'keyword': keyword, 'key_frame': key_image_path, 'response': response})
                if KEY_FRAME_ONLY:
                    continue

                text_embedding = make_text_embedding(clip_model, device, text=keyword, type_list=cfg.type_list,
                                                      class_adaption=cfg.class_adaption, template_adaption=cfg.template_adaption)

                wa_image = winclip_attention(cfg, key_image_path, text_embedding, clip_model, device, cfg.class_adaption, cfg.type_ids[k_i])
                grid_image = grid_generation(cfg, image_paths, keyword, clip_model, device)

                response_wa = lvlm(instruction, None, wa_image)
                response_tc = lvlm(instruction_tc, None, grid_image)

                max_score_wa = max(max_score_wa, generate_output(response_wa)['score'])
                max_score_tc = max(max_score_tc, generate_output(response_tc)['score'])

            for _ in range(cfg.clip_length):
                predicted.append(max_score)
                predicted_wa.append(max_score_wa)
                predicted_tc.append(max_score_tc)

        if KEY_FRAME_ONLY:
            score = np.array(predicted, dtype=np.float64)
        else:
            score = cfg.alpha * np.array(predicted) + cfg.beta * np.array(predicted_wa) + cfg.gamma * np.array(predicted_tc)
        if len(score) == 0:
            score = np.zeros(num_frames)
        score = gaussian_filter1d(score, sigma=cfg.sigma)
        score = np.pad(score, (0, num_frames - len(score)), mode='edge')
        np.save(SCORE_DIR / f"{n}.npy", score)
        (SCORE_DIR / f"{n}.json").write_text(json.dumps(log, ensure_ascii=False, indent=1))

    common.cross_validate(SCORE_DIR, THRESHOLD)


if __name__ == "__main__":
    main()
