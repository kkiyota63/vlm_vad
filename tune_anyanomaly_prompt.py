import random
import shutil

import cv2
import gepa
import numpy as np
import torch
from gepa.core.adapter import EvaluationBatch

import common
import run_anyanomaly as ra

SEGMENT_LEN = 24
NUM_TRAIN = 100
NUM_VAL = 100
MAX_METRIC_CALLS = 1000
MINIBATCH_SIZE = 5
SEED = 0
KEYWORD = ra.ANOMALY_TYPE
OUT_PATH = common.ROOT / 'prompts' / 'anyanomaly_squat_gepa.txt'
WORK = common.WORK_DIR / 'anyanomaly_gepa'
KEY_DIR = WORK / 'key_frames'

OUTPUT_RULE = ("- **Response**: Provide the score as a float, rounded to one decimal place, "
               "including a brief reason for the score in **one short sentence**.")

SEED_INSTRUCTION = (
    f"- **Task**: Evaluate whether the given image includes **{KEYWORD}** on a scale from 0 to 1. "
    f"A score of 1 means **{KEYWORD}** is clearly present in the image, while a score of 0 means **{KEYWORD}** is not present at all. "
    f"For intermediate cases, assign a value between 0 and 1 based on the degree to which **{KEYWORD}** is visible.\n"
    f"- **Consideration**: The key is whether **{KEYWORD}** is present in the image, not its focus. "
    f"Thus, if **{KEYWORD}** is present, even if it is not the main focus, assign a higher score like 1.0."
)

REFLECTION_TEMPLATE = """I provided a vision-language model with the following task instruction. The model sees one key frame from a one-second segment of a factory video and must output an anomaly score from 0 to 1 followed by a one-sentence reason. A fixed output-format line is appended after the instruction, so the instruction itself must not describe the output format.
```
<curr_param>
```

The following are examples of segments, the model's response for each of them, and feedback with the ground truth:
```
<side_info>
```

Your task is to write a new task instruction that makes the model give high scores (close to 1) to segments that contain the target behavior and low scores (close to 0) to segments that do not.

Identify the systematic mistakes (missed target behaviors and false alarms) and write clear visual criteria for what counts as the target behavior and what does not. The target behavior must stay the same: """ + KEYWORD + """. Do not describe the output format, and do not use backticks inside the instruction.

Provide the new instruction within ``` blocks."""


def frame_labels(name, num_frames):
    path = common.LABEL_DIR / f"{name}.npy"
    if not path.exists():
        return np.zeros(num_frames, dtype=np.int64)
    lab = np.load(path).reshape(-1).astype(np.int64)
    return np.pad(lab[:num_frames], (0, max(0, num_frames - len(lab))))


def build_dataset():
    split_files = sorted(common.SPLIT_DIR.glob("splits_*.txt"), key=lambda p: common.natural_keys(p.name))
    all_names = sorted({n for p in split_files for n in common.read_txt(p)})
    used = set(common.video_names())
    pool = [n for n in all_names if n not in used and common.video_path(n).exists()]
    print(f"最適化に使う動画: {len(pool)} 本(交差検証で使う {len(used)} 本は除外)")

    rng = random.Random(SEED)
    rng.shuffle(pool)
    half = len(pool) // 2
    sets = []
    for names, size in ((pool[:half], NUM_TRAIN), (pool[half:], NUM_VAL)):
        pos, neg = [], []
        for n in names:
            cap = cv2.VideoCapture(str(common.video_path(n)))
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            lab = frame_labels(n, num_frames)
            for s, start in enumerate(range(0, num_frames - SEGMENT_LEN + 1, SEGMENT_LEN)):
                seg = {"video": n, "segment": s, "label": int(lab[start:start + SEGMENT_LEN].mean() >= 0.5)}
                (pos if seg["label"] else neg).append(seg)
        if not pos:
            raise RuntimeError("最適化用の動画に異常セグメントがありません")
        k = min(size // 2, len(pos))
        data = rng.sample(pos, k) + rng.sample(neg, min(size - k, len(neg)))
        rng.shuffle(data)
        print(f"  {len(data)} セグメント(異常 {k} / 正常 {len(data) - k})")
        sets.append(data)
    return sets


def attach_key_frames(segments, kfs):
    tmp = WORK / 'tmp'
    for seg in ra.tqdm(segments, desc="key frames"):
        path = KEY_DIR / seg["video"] / f"{seg['segment']:05d}.jpg"
        if not path.exists():
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
            cap = cv2.VideoCapture(str(common.video_path(seg["video"])))
            cap.set(cv2.CAP_PROP_POS_FRAMES, seg["segment"] * SEGMENT_LEN)
            paths = []
            for i in range(SEGMENT_LEN):
                ok, frame = cap.read()
                if not ok:
                    break
                paths.append(str(tmp / f"{i:02d}.jpg"))
                cv2.imwrite(paths[-1], frame)
            cap.release()
            idx = kfs.call_function(paths, KEYWORD)[0] if len(paths) == SEGMENT_LEN else len(paths) // 2
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(paths[idx], path)
        seg["key_frame"] = str(path)


@torch.no_grad()
def generate_text(model, processor, prompt):
    msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    inputs = processor(text=[text], return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    return processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]


class AnyAnomalyAdapter:
    propose_new_texts = None

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        self.cache = {}

    def predict(self, instruction, seg):
        key = (instruction, seg["video"], seg["segment"])
        if key not in self.cache:
            self.cache[key] = ra.qwen35_test(self.model, self.processor, instruction + "\n" + OUTPUT_RULE, seg["key_frame"])
        return self.cache[key]

    def evaluate(self, batch, candidate, capture_traces=False):
        outputs, scores, trajectories = [], [], []
        for seg in batch:
            response = self.predict(candidate["instruction"], seg)
            pred = min(max(ra.generate_output(response)["score"], 0.0), 1.0)
            score = 1.0 - abs(pred - seg["label"])
            truth = "the target behavior IS present (ideal score 1.0)" if seg["label"] else "the target behavior is NOT present (ideal score 0.0)"
            verdict = "Good" if score >= 0.5 else "Bad"
            outputs.append(response)
            scores.append(score)
            trajectories.append({"Generated Outputs": response.strip(),
                                 "Feedback": f"{verdict}: the model gave {pred:.1f}. Ground truth: {truth}."})
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories if capture_traces else None)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        return {c: eval_batch.trajectories for c in components_to_update}


def main():
    assert ra.USE_QWEN35, "run_anyanomaly.py の MODEL_PATH を Qwen3.5 にしてください"
    assert '-' not in KEYWORD, "キーワードは 1 つにしてください"

    trainset, valset = build_dataset()
    device = torch.device("cuda")
    clip_model, preprocess = ra.clip.load('ViT-B/32', device=device)
    attach_key_frames(trainset + valset, ra.KFS(4, SEGMENT_LEN, clip_model, preprocess, device))
    del clip_model
    torch.cuda.empty_cache()

    model, processor = ra.load_qwen35()
    result = gepa.optimize(
        seed_candidate={"instruction": SEED_INSTRUCTION}, trainset=trainset, valset=valset,
        adapter=AnyAnomalyAdapter(model, processor), reflection_lm=lambda prompt: generate_text(model, processor, prompt),
        reflection_prompt_template=REFLECTION_TEMPLATE, reflection_minibatch_size=MINIBATCH_SIZE,
        max_metric_calls=MAX_METRIC_CALLS, run_dir=str(WORK / 'run' / ra.MODEL_PATH.split('/')[-1]),
        display_progress_bar=True, seed=SEED)

    best = result.best_candidate["instruction"]
    OUT_PATH.write_text(best + "\n" + OUTPUT_RULE + "\n", encoding='utf-8')
    print(f"seed の val スコア: {result.val_aggregate_scores[0]:.3f}")
    print(f"best の val スコア: {result.val_aggregate_scores[result.best_idx]:.3f}")
    print(f"保存先: {OUT_PATH}")
    print("run_anyanomaly.py の PROMPT_PATH にこのパスを設定すると本番の評価に使えます")


if __name__ == "__main__":
    main()
