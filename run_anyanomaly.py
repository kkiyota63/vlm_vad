import argparse
import functools
import os
import sys

import numpy as np
import torch

import common

ANOMALY_TYPE = 'worker squatting down-worker bending over to pick up something from the floor'
BASE_DATASET = 'shtech'
MODEL_PATH = 'MiniCPM-Llama3-V-2_5'
KEY_FRAME_ONLY = True

REPO = common.ROOT / 'third_party' / 'anyanomaly'
WORK = common.WORK_DIR / 'anyanomaly'
SCORE_DIR = WORK / ('scores_keyframe' if KEY_FRAME_ONLY else 'scores')
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
from tqdm import tqdm
from transformers import AutoProcessor, logging
logging.set_verbosity_error()


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
        tokenizer, model = load_lvlm(cfg.model_path, device)
        model.chat = functools.partial(model.chat, processor=AutoProcessor.from_pretrained(f"openbmb/{cfg.model_path}", trust_remote_code=True))
        clip_model, preprocess = clip.load('ViT-B/32', device=device)
        kfs = KFS(cfg.kfs_num, cfg.clip_length, clip_model, preprocess, device)

    for n in tqdm(todo, desc="videos", position=0):
        num_frames = common.extract_frames(n)
        cp_loader = clip_path_loader(str(common.FRAME_DIR / n), cfg.clip_length)
        predicted, predicted_wa, predicted_tc = [], [], []

        for c in tqdm(range(len(cp_loader)), desc=n, position=1, leave=False):
            cp = cp_loader[c]
            max_score = 0
            max_score_wa = 0
            max_score_tc = 0

            for k_i, keyword in enumerate(keyword_list):
                instruction, instruction_tc = make_instruction(cfg, keyword, True)

                indice = kfs.call_function(cp, keyword)
                key_image_path = cp[indice[0]]
                image_paths = [cp[idx] for idx in indice[1:]]

                response = lvlm_test(tokenizer, model, instruction, key_image_path, None)
                max_score = max(max_score, generate_output(response)['score'])
                if KEY_FRAME_ONLY:
                    continue

                text_embedding = make_text_embedding(clip_model, device, text=keyword, type_list=cfg.type_list,
                                                      class_adaption=cfg.class_adaption, template_adaption=cfg.template_adaption)

                wa_image = winclip_attention(cfg, key_image_path, text_embedding, clip_model, device, cfg.class_adaption, cfg.type_ids[k_i])
                grid_image = grid_generation(cfg, image_paths, keyword, clip_model, device)

                response_wa = lvlm_test(tokenizer, model, instruction, None, wa_image)
                response_tc = lvlm_test(tokenizer, model, instruction_tc, None, grid_image)

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

    common.cross_validate(SCORE_DIR)


if __name__ == "__main__":
    main()
