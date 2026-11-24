import json
import os
import pathlib
import sys

import numpy as np
import torch

import common

CLIP_CHUNK = 512
PROMPT_PATH = common.ROOT / 'prompts' / 'corevad_squat.txt'

REPO = common.ROOT / 'third_party' / 'corevad'
WORK = common.WORK_DIR / 'corevad'
SCORE_DIR = WORK / 'scores'
SCORE_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(WORK)
sys.path.insert(0, str(REPO / 'src' / 'ucf'))

import clip
from decord import VideoReader, cpu
from tqdm import tqdm
import qwen_vlm
import extract_clip_features as ecf
import generate_VLM_response as gen
import LRC
import eval_ucf

FEAT_DIR = pathlib.Path('./CLIP_feats/ucf_test')
RESPONSE_PATH = pathlib.Path('./outputs/VLM_responses_ucf.json')
LRC_PATH = pathlib.Path('./outputs/VLM_responses_LRC_ucf.json')


def main():
    names = common.video_names()
    device = torch.device('cuda:0')
    FEAT_DIR.mkdir(parents=True, exist_ok=True)

    todo = [n for n in names if not (FEAT_DIR / f"{n}_CLIP_features.npy").exists()]
    if todo:
        ecf.device = device
        clip_model = ecf.load_clip(device="cuda", model_name="ViT-B/16")
        for n in tqdm(todo, desc="CLIP features", position=0):
            vr = VideoReader(str(common.video_path(n)), ctx=cpu(0), num_threads=1)
            feats = [ecf.get_clip_frame_features(vr, list(range(i, min(i + CLIP_CHUNK, len(vr)))), clip_model,
                                                 batch_size=8, input_size=224, max_num=1).cpu().numpy()
                     for i in tqdm(range(0, len(vr), CLIP_CHUNK), desc=n, position=1, leave=False)]
            np.save(FEAT_DIR / f"{n}_CLIP_features.npy", np.concatenate(feats))
        del clip_model

    done = json.loads(RESPONSE_PATH.read_text(encoding="utf-8")) if RESPONSE_PATH.exists() else {}
    todo = [n for n in names if n not in done]
    if todo:
        prompt = PROMPT_PATH.read_text(encoding='utf-8')
        model, processor = qwen_vlm.load(device)
        RESPONSE_PATH.parent.mkdir(parents=True, exist_ok=True)
        for n in tqdm(todo, desc="VLM responses", position=0):
            vr = VideoReader(str(common.video_path(n)), ctx=cpu(0), num_threads=1)
            try:
                texts = []
                for chunk in tqdm(gen.build_second_chunks(len(vr), 30, 8), desc=n, position=1, leave=False):
                    text = qwen_vlm.describe(model, processor, vr, chunk, prompt).strip().split("\n")
                    texts.append(text[1].strip() + " " + text[2].strip())
            except Exception as e:
                print(f"[WARN] Eval failed on {n}: {e}")
                continue
            done[n] = texts
            RESPONSE_PATH.write_text(json.dumps(done, ensure_ascii=False, indent=2), encoding="utf-8")
        del model
        torch.cuda.empty_cache()
        failed = [n for n in names if n not in done]
        if failed:
            raise RuntimeError(f"VLM 応答の生成に失敗した動画: {failed}")

    LRC.Local_Response_Cleaning(clip_feat_dir=str(FEAT_DIR), device="cuda")

    clip_model, _ = clip.load("ViT-B/16", device=device)
    clip_model.eval()
    cleaned = json.loads(LRC_PATH.read_text(encoding="utf-8"))
    for n in tqdm(names, desc="Scoring"):
        num_frames = len(np.load(FEAT_DIR / f"{n}_CLIP_features.npy", mmap_mode='r'))
        score = eval_ucf.process_video(video=n, clean_response=cleaned[n], clip_feature_root_test=str(FEAT_DIR),
                                       vr_len=num_frames, clip_model=clip_model, device=device,
                                       tau=0.05, size=9, sigma=5, frame_len=30)
        np.save(SCORE_DIR / f"{n}.npy", score)

    common.cross_validate(SCORE_DIR)


if __name__ == "__main__":
    main()
