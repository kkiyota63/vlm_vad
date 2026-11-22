import json
import subprocess
import sys
import time

import numpy as np
from tqdm import tqdm

import common

FPS = 30
BATCH_SIZE = 32
FRAME_INTERVAL = 16
CAPTION_FRAME_INTERVAL = 16
NUM_NEIGHBORS = 10
LLM_MODEL = "llama-2-13b-chat"
LLM_NPROC = 2
CAP_MODELS = ["blip2-opt-6.7b-coco", "blip2-opt-6.7b", "blip2-flan-t5-xxl", "blip2-flan-t5-xl", "blip2-flan-t5-xl-coco"]
CONTEXT_PROMPT = "If you were a law enforcement agency, how would you rate the scene described on a scale from 0 to 1, with 0 representing a standard scene and 1 denoting a scene with suspicious activities?"
FORMAT_PROMPT = "Please provide the response in the form of a Python list and respond with only one number in the provided list below [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] without any textual explanation. It should begin with '[' and end with  ']'."
SUMMARY_PROMPT = "Please summarize what happened in few sentences, based on the following temporal description of a scene. Do not include any unnecessary details or descriptions."

REPO = common.ROOT / 'third_party' / 'lavad'
WORK = common.WORK_DIR / 'lavad'
SCORE_DIR = WORK / 'scores_npy'
ANNOTATION = WORK / 'annotations' / 'test.txt'


def run(desc, todo, out, suffix, *args):
    log_path = WORK / 'logs' / f"{desc.replace(' ', '_').replace('/', '_')}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    count = lambda: sum((out / f"{n}{suffix}").exists() for n in todo)
    with open(log_path, 'w') as log, tqdm(total=len(todo), desc=desc) as bar:
        proc = subprocess.Popen([sys.executable, *map(str, args)], cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            bar.n = count()
            bar.refresh()
            time.sleep(5)
        bar.n = count()
        bar.refresh()
    if proc.returncode:
        print("".join(log_path.read_text(errors='replace').splitlines(keepends=True)[-30:]))
        raise RuntimeError(f"{desc} が失敗しました。ログ: {log_path}")


def main():
    SCORE_DIR.mkdir(parents=True, exist_ok=True)
    ANNOTATION.parent.mkdir(parents=True, exist_ok=True)
    todo = [n for n in common.video_names() if not (SCORE_DIR / f"{n}.npy").exists()]
    num_frames = {n: common.extract_frames(n) for n in tqdm(todo, desc="extract frames")}

    if todo:
        ANNOTATION.write_text("".join(f"{n} 0 {num_frames[n] - 1} 0\n" for n in todo))
        data = ["--root_path", common.FRAME_DIR, "--annotationfile_path", ANNOTATION, "--batch_size", BATCH_SIZE]
        base = [*data, "--frame_interval", FRAME_INTERVAL]
        raw_caps = WORK / 'captions' / 'raw' / 'Salesforce'
        llama = ["--ckpt_dir", f"libs/llama/{LLM_MODEL}/", "--tokenizer_path", "libs/llama/tokenizer.model"]
        torchrun = ["-m", "torch.distributed.run", "--nproc_per_node", LLM_NPROC, "--nnodes", 1, "-m", "src.models.llm_anomaly_scorer"]

        for i, m in enumerate(CAP_MODELS, 1):
            run(f"1/6 caption {i}/{len(CAP_MODELS)} {m}", todo, raw_caps / m, ".json",
                "-m", "src.models.image_captioner", *data, "--frame_interval", CAPTION_FRAME_INTERVAL,
                "--pretrained_model_name", f"Salesforce/{m}", "--output_dir", raw_caps / m)
        run("2/6 caption index", todo, WORK / 'index' / 'captions', ".bin",
            "-m", "src.models.create_index", *base, "--index_dim", 1024, "--output_dir", WORK / 'index' / 'captions',
            "--captions_dirs", *[raw_caps / m for m in CAP_MODELS])
        run("3/6 clean captions", todo, WORK / 'captions' / 'clean', ".json",
            "-m", "src.models.image_text_caption_cleaner", *base, "--output_dir", WORK / 'captions' / 'clean',
            "--captions_dir_template", f"{raw_caps}/{{}}/", "--index_dir", WORK / 'index' / 'captions',
            "--fps", FPS, "--clip_duration", 10, "--num_samples", 10, "--num_neighbors", 1)
        run("4/6 LLM summary", todo, WORK / 'captions' / 'summary', ".json",
            *torchrun, *base, *llama, "--summary_prompt", SUMMARY_PROMPT,
            "--output_summary_dir", WORK / 'captions' / 'summary', "--captions_dir", WORK / 'captions' / 'clean')
        run("4/6 LLM score", todo, WORK / 'scores' / 'raw', ".json",
            *torchrun, *base, *llama, "--output_summary_dir", WORK / 'captions' / 'summary',
            "--context_prompt", CONTEXT_PROMPT, "--format_prompt", FORMAT_PROMPT,
            "--output_scores_dir", WORK / 'scores' / 'raw', "--score_summary")
        run("5/6 summary index", todo, WORK / 'index' / 'summary', ".bin",
            "-m", "src.models.create_summary_index", *base, "--index_dim", 1024,
            "--captions_dir", WORK / 'captions' / 'summary', "--output_dir", WORK / 'index' / 'summary')
        run("6/6 refine scores", todo, WORK / 'scores' / 'refined', ".json",
            "-m", "src.models.video_text_score_refiner", *base,
            "--output_scores_dir", WORK / 'scores' / 'refined', "--output_summary_dir", WORK / 'captions' / 'clean_summary',
            "--output_similarity_dir", WORK / 'similarity', "--output_filenames_dir", WORK / 'filenames',
            "--captions_dir", WORK / 'captions' / 'summary', "--index_dir", WORK / 'index' / 'summary',
            "--scores_dir", WORK / 'scores' / 'raw', "--fps", FPS, "--clip_duration", 10, "--num_samples", 10,
            "--num_neighbors", NUM_NEIGHBORS)

        sys.path.insert(0, str(REPO))
        from src.eval import calculate_weighted_scores
        for n in todo:
            scores = json.loads((WORK / 'scores' / 'refined' / f"{n}.json").read_text())
            similarity = json.loads((WORK / 'similarity' / f"{n}.json").read_text())
            score = calculate_weighted_scores(scores, similarity, NUM_NEIGHBORS, FRAME_INTERVAL)[:num_frames[n]]
            np.save(SCORE_DIR / f"{n}.npy", score)

    common.cross_validate(SCORE_DIR)


if __name__ == "__main__":
    main()
