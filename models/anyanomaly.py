"""AnyAnomaly (WACV 2026)  https://github.com/SkiddieAhn/Paper-AnyAnomaly  (MIT License)

学習なし（LVLM はファインチューニングしない）の Customizable VAD。ユーザー定義の異常テキスト（キーワード）を
Context-aware VQA で 0〜1 に採点する。公式 vad_proposed_Qwen_vllm.py の処理を移植:
  - clip_length(24) フレームのセグメントごとに
    * KFS (functions/key_func.py, kfs_num=4: CLIP→grouping) でキーフレームを選択
    * 元画像 (key), WinCLIP 風の注意画像 (position context, functions/attn_func.py),
      4 枚の時系列グリッド画像 (temporal context, functions/grid_func.py) の 3 つで VQA
    * キーワードごとの最大値をセグメントのスコアとし、セグメント内の全フレームに割り当て
  - 評価時 (functions/eval_func.py): alpha*key + beta*wa + gamma*tc を Gaussian 平滑化
公式との違い:
  - 公式は評価時にテストデータ上で sigma と重みをグリッド探索するが、ここでは既定値（公式の非探索時の値）を使う。
    --aa-tune を付けると、学習 fold の動画で同じグリッド探索を行う（テストデータは使わない）。
  - 公式は平滑化後に全データ連結で min-max 正規化するが、単調変換で AUC に影響しないため省略し、平滑化は動画ごとに行う。
  - 公式は端数フレーム（clip_length 未満）を捨てるが、ここでは最後のフレームを複製して採点する。
"""
import re

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from sklearn import metrics

from common.data import iter_chunks, scores_to_labels
from models.base import VADModel

STYLE = {  # 公式 config.py
    "shtech": dict(lge=(120, 120), mid=(80, 80), sml=(48, 48), out_prompt="Response"),
    "avenue": dict(lge=(240, 240), mid=(80, 80), sml=(48, 48), out_prompt="Output"),
}
IMG_SIZE = (240, 240)

TEMPLATES = [  # 公式 functions/text_func.py
    "a cropped photo of the {}.", "a close-up photo of a {}.", "a close-up photo of the {}.",
    "a bright photo of a {}.", "a bright photo of the {}.", "a dark photo of the {}.", "a dark photo of a {}.",
    "a jpeg corrupted photo of the {}.", "a jpeg corrupted photo of the {}.", "a blurry photo of the {}.",
    "a blurry photo of a {}.", "a photo of a {}.", "a photo of the {}.", "a photo of a small {}.",
    "a photo of the small {}.", "a photo of a large {}.", "a photo of the large {}.",
    "a photo of the {} for visual inspection.", "a photo of a {} for visual inspection.",
    "a photo of the {} for anomaly detection.", "a photo of a {} for anomaly detection.",
]


def make_instruction(keyword, out_prompt):
    """公式 functions/qwen_vllm_func.py"""
    instruction = f"""\
- **Task**: Evaluate whether the given image includes **{keyword}** on a scale from 0 to 1.
A score of 1 means **{keyword}** is clearly present in the image, while a score of 0 means **{keyword}** is not present at all.
For intermediate cases, assign a value between 0 and 1 based on the degree to which **{keyword}** is visible.
- **Consideration**: The key is whether **{keyword}** is present in the image, not its focus. Thus, if **{keyword}** is present, even if it is not the main focus, assign a higher score like 1.0.
- **{out_prompt}**: Provide the score as a float, rounded to one decimal place, including a brief reason for the score in **one short sentence**.
"""
    tc_instruction = f"""\
- **Task**: Evaluate whether the given image includes **{keyword}** on a scale from 0 to 1.
A score of 1 means **{keyword}** is clearly present in the image, while a score of 0 means **{keyword}** is not present at all.
For intermediate cases, assign a value between 0 and 1 based on the degree to which **{keyword}** is visible.
- **Context**: The given image represents a sequence (row 1 column 1 → row 1 column 2 → row 2 column 1 -> row 2 column 2) illustrating temporal progression.
- **Consideration**: The key is whether **{keyword}** is present in the image, not its focus. Thus, if **{keyword}** is present, even if it is not the main focus, assign a higher score like 1.0.
- **{out_prompt}**: Provide the score as a float, rounded to one decimal place, including a brief reason for the score in **one short sentence**.
"""
    return instruction, tc_instruction


def extract_score(text):
    """公式 utils.py"""
    match = re.search(r'(\d+(\.\d+)?)', text)
    return float(match.group(0)) if match else 0.0


def calculate_patches(image_size, kernel_size, stride_size):
    return ((image_size[0] - kernel_size[0]) // stride_size[0] + 1,
            (image_size[1] - kernel_size[1]) // stride_size[1] + 1)


def _to_tensor240(img):
    """公式: cv2.imread → resize(240,240) → RGB → /255  （入力はすでに RGB）"""
    image = cv2.resize(img, IMG_SIZE).astype('float32') / 255
    return torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float()


def split_one_image_with_unfold(img, kernel_size, stride_size=None):
    image = _to_tensor240(img)
    stride_size = stride_size or kernel_size
    org = F.unfold(image, kernel_size=kernel_size, stride=stride_size)
    patches = org.permute(0, 2, 1).reshape(-1, 3, kernel_size[0], kernel_size[1])
    return F.interpolate(patches, size=(224, 224), mode='bilinear')


def split_images_with_unfold(imgs, kernel_size, stride_size=None):
    all_patches = [split_one_image_with_unfold(im, kernel_size, stride_size) for im in imgs]
    num = all_patches[0].shape[0]
    return [torch.stack([ps[i] for ps in all_patches]) for i in range(num)]


def transform2pil(image, transform=True):
    if transform:
        image = (np.transpose(image * 255, [1, 2, 0])).astype(np.uint8)
    return Image.fromarray(image)


class AnyAnomaly(VADModel):
    name = "anyanomaly"

    @staticmethod
    def add_args(p):
        g = p.add_argument_group("AnyAnomaly")
        g.add_argument("--aa-vlm", default="Qwen/Qwen2.5-VL-7B-Instruct", help="公式スクリプト既定は 3B。7B も公式に記載")
        g.add_argument("--aa-backend", default="auto", choices=["auto", "vllm", "hf"],
                       help="vllm: 公式実装 / hf: transformers で生成（auto は CUDA かつ vllm があれば vllm）")
        g.add_argument("--aa-keywords", default="accident,falling,fighting",
                       help="検出したい異常イベントのテキスト（カンマ区切り）。データに合わせて必ず設定すること")
        g.add_argument("--aa-extra-classes", default="walking,standing,car,bicycle,running",
                       help="class adaption の softmax に入れる比較用クラス（公式はデータセットのクラス一覧）")
        g.add_argument("--aa-style", default="shtech", choices=list(STYLE), help="パッチサイズ等の設定（公式 config.py）")
        g.add_argument("--aa-clip-length", type=int, default=24)
        g.add_argument("--aa-kfs", type=int, default=4, help="1: random, 2: clip, 3: grouping->clip, 4: clip->grouping")
        g.add_argument("--aa-alpha", type=float, default=0.6)
        g.add_argument("--aa-beta", type=float, default=0.3)
        g.add_argument("--aa-gamma", type=float, default=0.1)
        g.add_argument("--aa-sigma", type=float, default=15)
        g.add_argument("--aa-tune", action="store_true", help="学習 fold で sigma と重みをグリッド探索（公式の grid_search 相当）")
        g.add_argument("--aa-tune-stride", type=int, default=4, help="探索時の AUC 計算に使うフレーム間引き")
        g.add_argument("--aa-max-side", type=int, default=None, help="読み込み時の長辺リサイズ（メモリ節約用）")

    def __init__(self, args, device):
        super().__init__(args, device)
        self.keywords = [k.strip() for k in args.aa_keywords.split(",") if k.strip()]
        extra = [k.strip() for k in args.aa_extra_classes.split(",") if k.strip()]
        self.type_list = self.keywords + [e for e in extra if e not in self.keywords]
        self.type_ids = [self.type_list.index(k) for k in self.keywords]
        st = STYLE[args.aa_style]
        self.sizes = [st["sml"], st["mid"], st["lge"]]
        self.patch_nums = [calculate_patches(IMG_SIZE, s, s) for s in self.sizes]
        self.out_prompt = st["out_prompt"]
        self.weights = (args.aa_alpha, args.aa_beta, args.aa_gamma)
        self.sigma = args.aa_sigma
        self._lvlm = None
        self._clip = None
        self._text_emb = None

    @property
    def trainable(self):
        return self.args.aa_tune

    def cache_config(self):
        a = self.args
        return {"vlm": a.aa_vlm, "kw": self.keywords, "types": self.type_list, "style": a.aa_style,
                "cl": a.aa_clip_length, "kfs": a.aa_kfs, "side": a.aa_max_side, "v": 1}

    # ---------- モデル読み込み ----------
    def _clip_model(self):
        if self._clip is None:
            import clip
            model, preprocess = clip.load('ViT-B/32', device=self.device, jit=False)
            model.eval()
            self._clip = (model, preprocess)
            # 公式 make_text_embedding(class_adaption=True, template_adaption=True)
            with torch.no_grad():
                embs = []
                for t in self.type_list:
                    tok = clip.tokenize([tp.format(t) for tp in TEMPLATES]).to(self.device)
                    embs.append(model.encode_text(tok).float().mean(dim=0, keepdim=True))
                self._text_emb = torch.cat(embs)          # (C, D)
        return self._clip

    def _backend(self):
        b = self.args.aa_backend
        if b == "auto":
            b = "hf"
            if self.device.type == "cuda":
                try:
                    import vllm  # noqa: F401
                    b = "vllm"
                except ImportError:
                    pass
        return b

    def _load_lvlm(self):
        if self._lvlm is not None:
            return self._lvlm
        from transformers import AutoProcessor
        backend = self._backend()
        processor = AutoProcessor.from_pretrained(self.args.aa_vlm)
        if backend == "vllm":
            from vllm import LLM, SamplingParams
            model = LLM(model=self.args.aa_vlm, limit_mm_per_prompt={"image": 10, "video": 10})
            sp = SamplingParams(temperature=0.1, top_p=0.001, repetition_penalty=1.05, max_tokens=50, stop_token_ids=[])
            self._lvlm = ("vllm", model, processor, sp)
        else:
            from transformers import AutoModelForImageTextToText
            dtype = torch.bfloat16 if self.device.type in ("cuda", "mps") else torch.float32
            model = AutoModelForImageTextToText.from_pretrained(self.args.aa_vlm, dtype=dtype).to(self.device).eval()
            self._lvlm = ("hf", model, processor, None)
        print(f"    [anyanomaly] LVLM backend: {self._lvlm[0]}")
        return self._lvlm

    @staticmethod
    def _messages(image, instruction):
        return [
            {"role": "system", "content": "You are a vision anomaly detector."},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": instruction}]},
        ]

    @torch.no_grad()
    def _lvlm_test(self, message_list):
        kind, model, processor, sp = self._load_lvlm()
        if kind == "vllm":
            from qwen_vl_utils import process_vision_info
            llm_inputs = []
            for m in message_list:
                prompt = processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                image_inputs, _, _ = process_vision_info(m, return_video_kwargs=True)
                llm_inputs.append({"prompt": prompt, "multi_modal_data": {"image": image_inputs}})
            outputs = model.generate(llm_inputs, sampling_params=sp, use_tqdm=False)
            return [o.outputs[0].text for o in outputs]
        outs = []
        for m in message_list:
            text = processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            img = m[1]["content"][0]["image"]
            inputs = processor(text=[text], images=[img], return_tensors="pt").to(self.device)
            gen = model.generate(**inputs, max_new_tokens=50, do_sample=False, repetition_penalty=1.05)
            outs.append(processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0])
        return outs

    # ---------- Context 生成（公式 functions/*.py を配列入力に変更）----------
    @torch.no_grad()
    def _kfs(self, images, keyword):
        import clip
        model, preprocess = self._clip_model()
        L = self.args.aa_clip_length
        if self.args.aa_kfs == 1:
            idx = sorted(np.random.choice(L, 4, replace=False).tolist())
            return [idx[0]] + idx
        x = torch.cat([preprocess(Image.fromarray(im)).unsqueeze(0) for im in images]).to(self.device)
        imf = model.encode_image(x).float()
        tf = model.encode_text(clip.tokenize([keyword]).to(self.device)).float()
        imf /= imf.norm(dim=-1, keepdim=True)
        tf /= tf.norm(dim=-1, keepdim=True)
        sim = (tf @ imf.T).cpu().numpy()[0]
        max_idx = int(np.argmax(sim))
        g = L // 4
        if self.args.aa_kfs == 2:
            top = np.argsort(sim)[::-1][:4]
            return [int(top[0])] + sorted(int(t) for t in top)
        if self.args.aa_kfs == 3:
            return [max_idx] + [k * g + int(np.argmax(sim[k * g:(k + 1) * g])) for k in range(4)]
        d = max_idx % g
        return [max_idx] + [k * g + d for k in range(4)]

    @torch.no_grad()
    def _winclip_attention(self, img, type_id):
        model, _ = self._clip_model()
        te = self._text_emb / self._text_emb.norm(dim=-1, keepdim=True)
        maps = []
        for size, (r, c) in zip(self.sizes, self.patch_nums):
            patches = split_one_image_with_unfold(img, size, size).to(self.device)
            ie = model.encode_image(patches).float()
            ie /= ie.norm(dim=-1, keepdim=True)
            sim = (te @ ie.T).cpu().softmax(dim=0)[type_id].view(1, 1, r, c)
            maps.append(F.interpolate(sim, size=(224, 224), mode='bilinear').squeeze(0))
        total = (sum(maps) / len(maps)).squeeze(0).numpy()
        rng = total.max() - total.min()
        mask = (total - total.min()) / rng if rng > 0 else np.ones_like(total)
        base = cv2.resize(img, (224, 224))
        attn = (base * np.repeat(mask[:, :, None], 3, axis=2)).astype(np.uint8)
        return transform2pil(attn, False)

    @torch.no_grad()
    def _grid(self, key_images, keyword):
        import clip
        model, _ = self._clip_model()
        gpatches = []
        for size in self.sizes:
            gpatches += split_images_with_unfold(key_images, size, size)
        tf = model.encode_text(clip.tokenize([keyword]).to(self.device)).float()
        tf /= tf.norm(dim=-1, keepdim=True)
        best = []
        for gp in gpatches:
            ie = model.encode_image(gp.to(self.device)).float()
            ie /= ie.norm(dim=-1, keepdim=True)
            best.append(float((tf @ ie.T).max()))
        gp = gpatches[int(np.argmax(best))]
        grid = np.zeros((3, 224, 224), dtype=np.float32)
        for k, (y, x) in enumerate([(0, 0), (0, 112), (112, 0), (112, 112)]):
            grid[:, y:y + 112, x:x + 112] = F.interpolate(gp[k].unsqueeze(0), size=(112, 112), mode='bilinear').numpy()
        return transform2pil(grid)

    # ---------- 抽出 ----------
    def extract(self, video_path):
        L = self.args.aa_clip_length
        org, wa, tc, reasons, n = [], [], [], [], 0
        for _, frames in iter_chunks(video_path, L, self.args.aa_max_side):
            n += len(frames)
            frames = frames + [frames[-1]] * (L - len(frames))      # 端数セグメントは最終フレームで埋める
            s_max = [0.0, 0.0, 0.0]
            rs = []
            for k_i, kw in enumerate(self.keywords):
                ins, ins_tc = make_instruction(kw, self.out_prompt)
                key = self._kfs(frames, kw)
                main_key = Image.fromarray(frames[key[0]])
                wa_img = self._winclip_attention(frames[key[0]], self.type_ids[k_i])
                grid_img = self._grid([frames[i] for i in key[1:]], kw)
                res = self._lvlm_test([self._messages(main_key, ins), self._messages(wa_img, ins),
                                       self._messages(grid_img, ins_tc)])
                s_max = [max(m, extract_score(r)) for m, r in zip(s_max, res)]
                rs.append(res)
            org.append(s_max[0])
            wa.append(s_max[1])
            tc.append(s_max[2])
            reasons.append(rs)
        return {"n_frames": n, "org": np.array(org, np.float32), "wa": np.array(wa, np.float32),
                "tc": np.array(tc, np.float32), "responses": reasons}

    # ---------- スコア ----------
    def _components(self, ext):
        L, n = self.args.aa_clip_length, ext["n_frames"]
        return [np.repeat(ext[k].astype(np.float64), L)[:n] for k in ("org", "wa", "tc")]

    def predict(self, extracted):
        a, b, g = self.weights
        o, w, t = self._components(extracted)
        return gaussian_filter1d(a * o + b * w + g * t, sigma=self.sigma)

    def fit(self, items):
        """--aa-tune: 学習 fold で公式 evaluate_auc と同じ探索（sigma 1..29, 重み 0.1 刻み）"""
        st = self.args.aa_tune_stride
        comps, labels = [], []
        for ext, label, _ in items:
            c = self._components(ext)
            comps.append(c)
            labels.append(label)
        y = np.concatenate([l[::st] for l in labels])
        if len(np.unique(y)) < 2:
            print("    [anyanomaly] 学習 fold に両クラスが無いため既定値を使用")
            return
        best = (-1, None, None)
        for sigma in range(1, 30):
            sm = [[gaussian_filter1d(x, sigma=sigma) for x in c] for c in comps]
            mapped = [[scores_to_labels(np.arange(len(x)), x, len(l), len(x))[::st] for x in c] for c, l in zip(sm, labels)]
            O, W, T = (np.concatenate([m[k] for m in mapped]) for k in range(3))
            for a1 in np.arange(0.0, 1.1, 0.1):
                for a2 in np.arange(0.0, 1.1, 0.1):
                    a3 = 1 - a1 - a2
                    if 0 <= a3 + 1e-9 and a3 <= 1:
                        auc = metrics.roc_auc_score(y, a1 * O + a2 * W + a3 * T)
                        if auc > best[0]:
                            best = (auc, sigma, (float(a1), float(a2), float(max(a3, 0.0))))
        _, self.sigma, self.weights = best
        print(f"    [anyanomaly] tuned on train: AUC {best[0]:.4f}  sigma {self.sigma}  weights {tuple(round(x, 2) for x in self.weights)}")
