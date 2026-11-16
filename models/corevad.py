"""CoReVAD (ICPR 2026)  https://github.com/Muk-00/CoReVAD

学習なし（単一の凍結 VLM）。処理の流れは公式実装に従う:
  1. generate_VLM_response.py : 30 フレームごとのセグメントから 8 枚を等間隔に取り、
     InternVL2 に P_VLM_format.txt のプロンプトで「Anomalous/Normal scenes. + 1 文の説明」を生成させる
  2. extract_clip_features.py : 全フレームの CLIP ViT-B/16 画像特徴（InternVL の前処理で 224 にリサイズ、L2 正規化）
  3. LRC.py                   : Local Response Cleaning（各セグメントの視覚特徴と最も整合する近傍の応答に置換）
  4. eval_*.py                : 応答の 1 文目から初期スコア(0/1)、説明文との類似度による softmax 重み付け、
                                ガウス平滑化、フレームへの展開、位置重み付け（動画中央を重視する Gaussian）
1〜3 と 4 の類似度計算を extract（キャッシュ対象）で、4 のスコア計算を predict で行う。
"""
import re

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from common.data import iter_chunks
from models.base import VADModel

# 公式 P_VLM_format.txt
P_VLM_FORMAT = """You are the model.

**Model Description:**
You are designed for video anomaly detection. The input is a sequence of video frames for identifying whether there is an anomaly in the video.

**Prompt Questions:**
Answer the following questions based on what you see from the video frames:
- First, decide whether the scene shows abnormal/anomalous activity or just normal/no unusual activity.
- Then, write ONE short description of what the scene shows.
Based on the analysis above, please conclude your answer.

**Input:**
[$Data]

**Output Rules (STRICT):**
- Based on the analysis above, reply with ONLY a fenced code block (no language tag) that has EXACTLY two lines.
  1) write **"Anomalous scenes."** if you detected any abnormal or unusual human/vehicle/object activity in the frames; otherwise write **"Normal scenes."**
  2) write ONE short sentence describing what is happening.
- Do NOT add any other text before or after the code block.
"""

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    """公式 utils/internvl_utils.py。max_num=1 の dynamic_preprocess は単純な正方形リサイズと等価"""
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def parse_response(resp):
    """公式は text[1] + " " + text[2]（コードブロックの 2 行）。フェンスの有無に頑健にした版"""
    lines = [l.strip() for l in resp.strip().split("\n") if l.strip() and not l.strip().startswith("```")]
    if not lines:
        return ""
    return " ".join(lines[:2])


# ---- 公式 eval_ucf.py より ----
def split_first_sentence(text):
    t = re.sub(r'\s+', ' ', text).strip()
    m = re.search(r'\.(\s|$)', t)
    if m:
        first = t[:m.end()].strip()
        body = t[m.end():].strip()
    else:
        first, body = t, ""
    if not body:
        body = first
    return first, body


def first_sentence_to_label(first_sentence):
    s = first_sentence.lower().strip("*\"' ")
    score = 0.0
    if s.startswith("anomalous scenes"):
        score = 1.0
    if s.startswith("normal scenes"):
        score = 0.0
    return score


def gaussian_kernel_original(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def gaussian_smoothing(data, sigma):
    """位置重み付け（動画中央を中心とする Gaussian を掛ける）"""
    n = len(data)
    x = np.arange(n)
    centroid_index = int(n / 2)
    kernel_values = gaussian_kernel_original(x, centroid_index, max(sigma, 1))
    return kernel_values * data


def gaussian_kernel(size, sigma):
    kernel = np.exp(-np.linspace(-size // 2, size // 2, size) ** 2 / (2 * sigma ** 2))
    return kernel / kernel.sum()


def gaussian_smooth_1d(data, size=5, sigma=1.0):
    # np.convolve(mode='same') と同じ結果。ただしデータがカーネルより短い場合も長さを保つ
    full = np.convolve(data, gaussian_kernel(size, sigma), mode='full')
    off = (size - 1) // 2
    return full[off:off + len(data)]


class CoReVAD(VADModel):
    name = "corevad"

    @staticmethod
    def add_args(p):
        g = p.add_argument_group("CoReVAD")
        g.add_argument("--corevad-vlm", default="OpenGVLab/InternVL2-8B", help="公式: InternVL2-8B")
        g.add_argument("--corevad-frame-len", type=int, default=30, help="セグメント長（フレーム）。公式 30")
        g.add_argument("--corevad-num-frames", type=int, default=8, help="セグメントあたり VLM に入れる枚数。公式 8")
        g.add_argument("--corevad-input-size", type=int, default=448)
        g.add_argument("--corevad-clip-stride", type=int, default=1, help="CLIP 特徴を取るフレーム間隔。公式は全フレーム(1)")
        g.add_argument("--corevad-lrc-window", type=int, default=1)
        g.add_argument("--corevad-tau", type=float, default=0.05)
        g.add_argument("--corevad-size", type=int, default=9)
        g.add_argument("--corevad-sigma", type=float, default=5)
        g.add_argument("--corevad-no-pos-weight", action="store_true",
                       help="位置重み付け（動画中央を重視）を無効化。公式では有効")
        g.add_argument("--corevad-max-new-tokens", type=int, default=1024)
        g.add_argument("--use-flash-attn", action="store_true")

    def __init__(self, args, device):
        super().__init__(args, device)
        self._vlm = None
        self._clip = None

    def cache_config(self):
        a = self.args
        return {"vlm": a.corevad_vlm, "fl": a.corevad_frame_len, "nf": a.corevad_num_frames,
                "is": a.corevad_input_size, "cs": a.corevad_clip_stride, "w": a.corevad_lrc_window, "v": 1}

    def _load(self):
        if self._vlm is None:
            from transformers import AutoModel, AutoTokenizer
            dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
            model = AutoModel.from_pretrained(self.args.corevad_vlm, torch_dtype=dtype, low_cpu_mem_usage=True,
                                              use_flash_attn=self.args.use_flash_attn, trust_remote_code=True).to(self.device).eval()
            tok = AutoTokenizer.from_pretrained(self.args.corevad_vlm, trust_remote_code=True, use_fast=False)
            self._vlm = (model, tok, dtype)
            import clip
            cm, _ = clip.load("ViT-B/16", device=self.device, jit=False)
            for p in cm.parameters():
                p.requires_grad = False
            self._clip = cm.eval()
        return self._vlm, self._clip

    @torch.no_grad()
    def _vlm_response(self, frames):
        (model, tok, dtype), _ = self._load()
        tf = build_transform(self.args.corevad_input_size)
        pv = torch.stack([tf(Image.fromarray(f)) for f in frames]).to(dtype).to(self.device)
        num_patches_list = [1] * len(frames)
        prefix = ''.join([f'Frame{i + 1}: <image>\n' for i in range(len(frames))])
        q = P_VLM_FORMAT.replace('$Data', prefix)
        gen = dict(max_new_tokens=self.args.corevad_max_new_tokens, do_sample=False)
        return model.chat(tok, pv, num_patches_list=num_patches_list, question=q, generation_config=gen, return_history=False)

    @torch.no_grad()
    def _clip_feats(self, frames):
        _, cm = self._load()
        tf = build_transform(224)  # 公式 extract_clip_features.py と同じ前処理
        x = torch.stack([tf(Image.fromarray(f)) for f in frames]).to(self.device)
        x = x.to(cm.visual.conv1.weight.dtype)
        f = cm.encode_image(x)
        return torch.nn.functional.normalize(f, dim=-1).float().cpu().numpy()

    @torch.no_grad()
    def _text_encode(self, text):
        import clip
        _, cm = self._load()
        if not text:
            return np.zeros(512, np.float32)
        tokens = clip.tokenize(text, truncate=True).to(self.device)
        t = cm.encode_text(tokens).float()
        t = t / t.norm(dim=-1, keepdim=True)
        return t[0].cpu().numpy()

    def extract(self, video_path):
        a = self.args
        responses, vis, n = [], [], 0
        for start, frames in iter_chunks(video_path, a.corevad_frame_len):
            L = len(frames)
            if L <= a.corevad_num_frames:
                sampled = list(range(L))
            else:
                sampled = np.linspace(0, L - 1, num=a.corevad_num_frames, dtype=np.int32).tolist()
            raw = self._vlm_response([frames[i] for i in sampled])
            responses.append(parse_response(raw))
            vis.append(self._clip_feats(frames[::a.corevad_clip_stride]).mean(axis=0))
            n += L
        vis = np.stack(vis).astype(np.float32)                     # (M, 512) セグメントごとの視覚特徴

        # --- Local Response Cleaning（公式 LRC.py）---
        resp = [" ".join(r.splitlines()).strip() if r else "" for r in responses]
        tf = np.stack([self._text_encode(r) for r in resp])        # (M, 512)
        sims_TM = vis @ tf.T
        M = len(resp)
        w = a.corevad_lrc_window
        cleaned = []
        for i in range(len(vis)):
            cand = np.arange(max(0, i - w), min(M, i + w + 1))
            cleaned.append(resp[cand[np.argmax(sims_TM[i, cand])]])

        # --- eval_ucf.py の類似度部分 ---
        y_init, body_emb = [], []
        for text in cleaned:
            first, body = split_first_sentence(text)
            y_init.append(first_sentence_to_label(first))
            body_emb.append(self._text_encode(body))
        sims = vis @ np.stack(body_emb).T                          # (M, R)
        return {"n_frames": n, "responses": responses, "responses_lrc": cleaned,
                "y_init": np.asarray(y_init, np.float32), "sims": sims.astype(np.float32)}

    def predict(self, extracted):
        a = self.args
        sims, y_init, n = extracted["sims"].astype(np.float64), extracted["y_init"].astype(np.float64), extracted["n_frames"]
        sc = sims - sims.max(axis=1, keepdims=True)
        exps = np.exp(sc / float(a.corevad_tau))
        weights = exps / (exps.sum(axis=1, keepdims=True) + 1e-12)
        y_refined = (weights * y_init[None, :]).sum(axis=1)
        y_pred = gaussian_smooth_1d(y_refined, a.corevad_size, a.corevad_sigma)
        y_pred = np.repeat(y_pred, a.corevad_frame_len)[:n]
        if not a.corevad_no_pos_weight:
            y_pred = gaussian_smoothing(y_pred, int(len(y_pred) * 0.5))
        return y_pred.astype(np.float64)
