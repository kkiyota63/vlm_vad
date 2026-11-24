import random
import sys

import numpy as np
import torch

import common

MLLM_PATH = 'OpenGVLab/InternVL2-8B'
REFLECTION_PATH = 'Qwen/Qwen2.5-14B-Instruct'
SEGMENT_LEN = 30
NUM_PER_SEGMENT = 8
NUM_TRAIN = 100
NUM_VAL = 100
MAX_METRIC_CALLS = 1500
MINIBATCH_SIZE = 5
SEED = 0
SEED_PROMPT_PATH = common.ROOT / 'prompts' / 'corevad_squat.txt'
OUT_PATH = common.ROOT / 'prompts' / 'corevad_squat_gepa.txt'

TEMPLATE = """You are the model.

**Model Description:**
You are designed for video anomaly detection in a factory. The input is a sequence of video frames sampled from a short video segment.

**Task:**
<instruction>

**Input:**
[$Data]

**Output Rules (STRICT):**
- Based on the analysis above, reply with ONLY a fenced code block (no language tag) that has EXACTLY two lines.
  1) write **"Anomalous scenes."** if the frames show the target behavior defined in the Task; otherwise write **"Normal scenes."**
  2) write ONE short sentence describing what is happening.
- Do NOT add any other text before or after the code block."""

SEED_INSTRUCTION = """Answer the following questions based on what you see from the video frames:
- First, decide whether any worker is squatting down or bending over to reach for or pick up something from the floor.
- Then, write ONE short description of what the scene shows, including the posture of each worker (e.g., standing, walking, squatting, bending over).
The target behavior is: any worker squatting down or bending over to reach for or pick up something from the floor."""

REFLECTION_TEMPLATE = """I provided a vision-language model with the following task instruction. It is inserted into the **Task** section of a fixed prompt. The model sees 8 frames from a one-second segment of a factory video and must answer in a fixed two-line format: line 1 is "Anomalous scenes." if the target behavior is present, otherwise "Normal scenes."; line 2 is one sentence describing the scene.
```
<curr_param>
```

The following are examples of segments, the model's response for each of them, and feedback with the ground truth:
```
<side_info>
```

Your task is to write a new task instruction that makes the model's line-1 decision match the ground truth more often.

Identify the systematic mistakes (missed target behaviors and false alarms) and write clear criteria for what counts as the target behavior and what does not. The target behavior must stay the same: a worker squatting down or bending over to reach for or pick up something from the floor. Do not change or restate the output format, and do not use backticks inside the instruction.

Provide the new instruction within ``` blocks."""

REPO = common.ROOT / 'third_party' / 'corevad'
WORK = common.WORK_DIR / 'corevad_gepa'
WORK.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / 'src' / 'ucf'))

import gepa
from decord import VideoReader, cpu
from gepa.core.adapter import EvaluationBatch
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
import generate_VLM_response as gen


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
    pool = [n for n in all_names if n not in used]
    print(f"検証用の動画: {len(pool)} 本(交差検証で使う {len(used)} 本は除外)")

    rng = random.Random(SEED)
    rng.shuffle(pool)
    half = len(pool) // 2
    sets = []
    for names, size in ((pool[:half], NUM_TRAIN), (pool[half:], NUM_VAL)):
        pos, neg = [], []
        for n in tqdm(names, desc="segments"):
            num_frames = len(VideoReader(str(common.video_path(n)), ctx=cpu(0), num_threads=1))
            lab = frame_labels(n, num_frames)
            for s, start in enumerate(range(0, num_frames, SEGMENT_LEN)):
                seg = {"video": n, "segment": s, "label": int(lab[start:start + SEGMENT_LEN].mean() >= 0.5)}
                (pos if seg["label"] else neg).append(seg)
        if not pos:
            raise RuntimeError("検証用の動画に異常セグメントがありません")
        k = min(size // 2, len(pos))
        data = rng.sample(pos, k) + rng.sample(neg, min(size - k, len(neg)))
        rng.shuffle(data)
        print(f"  {len(data)} セグメント(異常 {k} / 正常 {len(data) - k})")
        sets.append(data)
    return sets


class CoReVADAdapter:
    propose_new_texts = None

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.cache = {}

    def predict(self, instruction, seg):
        key = (instruction, seg["video"], seg["segment"])
        if key not in self.cache:
            vr = VideoReader(str(common.video_path(seg["video"])), ctx=cpu(0), num_threads=1)
            chunk = gen.build_second_chunks(len(vr), SEGMENT_LEN, NUM_PER_SEGMENT)[seg["segment"]]
            pixel_values, num_patches = gen.get_pixel_values(vr, chunk)
            prompt = TEMPLATE.replace('<instruction>', instruction)
            self.cache[key] = gen.make_VLM_response(self.model, self.tokenizer, dict(max_new_tokens=1024, do_sample=False),
                                                    pixel_values, num_patches, self.device, prompt)
        return self.cache[key]

    def evaluate(self, batch, candidate, capture_traces=False):
        outputs, scores, trajectories = [], [], []
        for seg in batch:
            response = self.predict(candidate["instruction"], seg)
            lines = response.strip().split("\n")
            first = lines[1].strip().lower() if len(lines) > 2 else ""
            if first.startswith("anomalous scenes"):
                pred = 1
            elif first.startswith("normal scenes"):
                pred = 0
            else:
                pred = None
            score = float(pred == seg["label"])
            truth = "a worker IS squatting or bending over to the floor" if seg["label"] else "NO worker is squatting or bending over to the floor"
            if pred is None:
                feedback = f"Incorrect: the response did not follow the two-line format. Ground truth: {truth}."
            elif score:
                feedback = f"Correct. Ground truth: {truth}."
            else:
                feedback = f"Incorrect: the model answered {'Anomalous' if pred else 'Normal'}. Ground truth: {truth}."
            outputs.append(response)
            scores.append(score)
            text = "\n".join(l for l in response.strip().split("\n") if not l.strip().startswith("```"))
            trajectories.append({"Generated Outputs": text, "Feedback": feedback})
        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories if capture_traces else None)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        return {c: eval_batch.trajectories for c in components_to_update}


def main():
    trainset, valset = build_dataset()
    device = torch.device('cuda:0')
    model = AutoModel.from_pretrained(MLLM_PATH, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        use_flash_attn=True, trust_remote_code=True).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(MLLM_PATH, trust_remote_code=True, use_fast=False)

    r_model = AutoModelForCausalLM.from_pretrained(REFLECTION_PATH, torch_dtype=torch.bfloat16, device_map="auto").eval()
    r_tokenizer = AutoTokenizer.from_pretrained(REFLECTION_PATH)

    @torch.no_grad()
    def reflection_lm(prompt):
        ids = r_tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                              return_tensors="pt").to(r_model.device)
        out = r_model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=1024, do_sample=False)
        return r_tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    result = gepa.optimize(
        seed_candidate={"instruction": SEED_INSTRUCTION}, trainset=trainset, valset=valset,
        adapter=CoReVADAdapter(model, tokenizer, device), reflection_lm=reflection_lm,
        reflection_prompt_template=REFLECTION_TEMPLATE, reflection_minibatch_size=MINIBATCH_SIZE,
        max_metric_calls=MAX_METRIC_CALLS, run_dir=str(WORK / 'run' / REFLECTION_PATH.split('/')[-1]),
        display_progress_bar=True, seed=SEED)

    best = result.best_candidate["instruction"]
    OUT_PATH.write_text(TEMPLATE.replace('<instruction>', best) + "\n", encoding='utf-8')
    print(f"seed の val 正解率: {result.val_aggregate_scores[0]:.3f}")
    print(f"best の val 正解率: {result.val_aggregate_scores[result.best_idx]:.3f}")
    print(f"保存先: {OUT_PATH}")


if __name__ == "__main__":
    main()
