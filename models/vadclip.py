"""VadCLIP (AAAI 2024)  https://github.com/nwpu-zxr/VadCLIP

凍結 CLIP ViT-B/16 のフレーム特徴に、LGT-Adapter（局所 Transformer + GCN）と
テキストプロンプト学習・視覚プロンプトによる 2 ブランチを学習する（CLIP 本体は学習しない）。

公式実装との対応:
  - モデル本体: third_party/vadclip/model.py (CLIPVAD) をそのまま使用
  - 特徴抽出: 公式 crop.py と同様に 340x256 へリサイズ → 中央 224 クロップ → CLIP encode_image。
    16 フレームのスニペットごとに中央フレーム 1 枚の特徴を使う（公式は 10-crop 特徴。ここでは中央クロップのみ）
  - 学習: 公式 ucf_train.py の損失（CLAS2 + CLASM + テキスト直交損失）・AdamW・MultiStepLR
    クラスは 2 つ（normal / 異常テキスト）。動画単位ラベルの弱教師あり学習が既定。
    --vadclip-supervision frame でフレームラベルの BCE を追加できる。
    公式はテスト AUC で最良エポックを選ぶが、ここではリークを避けるため最終エポックを使う。
  - 推論: 公式 ucf_test.py と同様に 256 スニペットごとに分割して推論し、スコアを 16 フレームに展開
"""
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim.lr_scheduler import MultiStepLR

from common.data import iter_chunks
from models.base import VADModel


# ---- 公式 src/utils/tools.py より ----
def get_batch_mask(lengths, maxlen):
    batch_size = lengths.shape[0]
    mask = torch.empty(batch_size, maxlen)
    mask.fill_(0)
    for i in range(batch_size):
        if lengths[i] < maxlen:
            mask[i, lengths[i]:maxlen] = 1
    return mask.bool()


def uniform_extract(feat, t_max, avg=True):
    new_feat = np.zeros((t_max, feat.shape[1])).astype(np.float32)
    r = np.linspace(0, len(feat), t_max + 1, dtype=np.int32)
    if avg:
        for i in range(t_max):
            if r[i] != r[i + 1]:
                new_feat[i, :] = np.mean(feat[r[i]:r[i + 1], :], 0)
            else:
                new_feat[i, :] = feat[r[i], :]
    else:
        r = np.linspace(0, feat.shape[0] - 1, t_max, dtype=np.uint16)
        new_feat = feat[r, :]
    return new_feat


def pad(feat, min_len):
    clip_length = feat.shape[0]
    if clip_length <= min_len:
        return np.pad(feat, ((0, min_len - clip_length), (0, 0)), mode='constant', constant_values=0)
    return feat


def process_feat(feat, length):
    if feat.shape[0] > length:
        return uniform_extract(feat, length), length
    return pad(feat, length), feat.shape[0]


def process_split(feat, length):
    clip_length = feat.shape[0]
    if clip_length < length:
        return pad(feat, length), clip_length
    split_num = int(clip_length / length) + 1
    for i in range(split_num):
        if i == 0:
            split_feat = feat[i * length:i * length + length, :].reshape(1, length, feat.shape[1])
        elif i < split_num - 1:
            split_feat = np.concatenate([split_feat, feat[i * length:i * length + length, :].reshape(1, length, feat.shape[1])], axis=0)
        else:
            split_feat = np.concatenate([split_feat, pad(feat[i * length:i * length + length, :], length).reshape(1, length, feat.shape[1])], axis=0)
    return split_feat, clip_length


# ---- 公式 src/ucf_train.py より ----
def CLASM(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = labels / torch.sum(labels, dim=1, keepdim=True)
    labels = labels.to(device)
    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True, dim=0)
        instance_logits = torch.cat([instance_logits, torch.mean(tmp, 0, keepdim=True)], dim=0)
    milloss = -torch.mean(torch.sum(labels * F.log_softmax(instance_logits, dim=1), dim=1), dim=0)
    return milloss


def CLAS2(logits, labels, lengths, device):
    instance_logits = torch.zeros(0).to(device)
    labels = 1 - labels[:, 0].reshape(labels.shape[0])
    labels = labels.to(device)
    logits = torch.sigmoid(logits).reshape(logits.shape[0], logits.shape[1])
    for i in range(logits.shape[0]):
        tmp, _ = torch.topk(logits[i, 0:lengths[i]], k=int(lengths[i] / 16 + 1), largest=True)
        tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat([instance_logits, tmp], dim=0)
    clsloss = F.binary_cross_entropy(instance_logits, labels)
    return clsloss


def center_crop_224(frame):
    """公式 crop.py の image_crop(type=0) 相当（入力は RGB）"""
    import cv2
    img = cv2.resize(frame, dsize=(340, 256))
    return img[16:240, 58:282, :]


class VadCLIP(VADModel):
    name = "vadclip"
    SNIPPET = 16

    @staticmethod
    def add_args(p):
        g = p.add_argument_group("VadCLIP")
        g.add_argument("--vadclip-anomaly-text", default="abnormal", help="異常クラスのテキスト（normal と 2 クラス）")
        g.add_argument("--vadclip-supervision", default="weak", choices=["weak", "frame"],
                       help="weak: 公式どおり動画単位ラベル / frame: フレームラベルの BCE を追加")
        g.add_argument("--vadclip-score", default="prob1", choices=["prob1", "prob2", "mean"],
                       help="prob1: 2値分類ブランチ（公式 AUC1）/ prob2: 整合ブランチ 1-P(normal)")
        g.add_argument("--vadclip-epochs", type=int, default=10)
        g.add_argument("--vadclip-batch-size", type=int, default=16, help="公式は 64（異常動画数が少ないため既定を下げている）")
        g.add_argument("--vadclip-lr", type=float, default=2e-5)
        g.add_argument("--vadclip-milestones", default="4,8")
        g.add_argument("--vadclip-visual-length", type=int, default=256)
        g.add_argument("--vadclip-attn-window", type=int, default=8)
        g.add_argument("--vadclip-prompt-prefix", type=int, default=10)
        g.add_argument("--vadclip-prompt-postfix", type=int, default=10)

    def __init__(self, args, device):
        super().__init__(args, device)
        self.prompt_text = ["normal", args.vadclip_anomaly_text]
        self._clip = None
        self.model = None

    @property
    def trainable(self):
        return True

    def cache_config(self):
        return {"clip": "ViT-B/16", "snippet": self.SNIPPET, "crop": "center"}

    # ---------- 特徴抽出（凍結 CLIP）----------
    def _clip_model(self):
        if self._clip is None:
            from third_party.vadclip import clip
            model, preprocess = clip.load("ViT-B/16", self.device, jit=False)
            model.eval()
            self._clip = (model, preprocess)
        return self._clip

    @torch.no_grad()
    def extract(self, video_path):
        model, preprocess = self._clip_model()
        imgs, n = [], 0
        for _, frames in iter_chunks(video_path, self.SNIPPET):
            imgs.append(preprocess(Image.fromarray(center_crop_224(frames[len(frames) // 2]))))
            n += len(frames)
        feats = []
        for i in range(0, len(imgs), 64):
            x = torch.stack(imgs[i:i + 64]).to(self.device)
            feats.append(model.encode_image(x).float().cpu())
        feat = torch.cat(feats).numpy().astype(np.float32)
        return {"feat": feat, "n_frames": n}

    # ---------- 学習 ----------
    def _new_model(self):
        from third_party.vadclip.model import CLIPVAD
        a = self.args
        m = CLIPVAD(len(self.prompt_text), 512, a.vadclip_visual_length, 512, 1, 2, a.vadclip_attn_window,
                    a.vadclip_prompt_prefix, a.vadclip_prompt_postfix, self.device)
        return m.to(self.device)

    def _snippet_labels(self, label, n_frames, T):
        """フレームラベルをスニペット単位（T 個）へ平均"""
        L = len(label)
        centers = (np.arange(T) * self.SNIPPET + self.SNIPPET / 2)
        idx = np.clip((centers / max(n_frames, 1) * L).astype(int), 0, L - 1)
        return label[idx].astype(np.float32)

    def fit(self, items):
        a = self.args
        torch.manual_seed(a.seed)
        np.random.seed(a.seed)
        rng = random.Random(a.seed)
        self.model = self._new_model()
        vl = a.vadclip_visual_length

        def prep(ext, label):
            feat, length = process_feat(ext["feat"], vl)
            if a.vadclip_supervision == "frame":
                sl = self._snippet_labels(label, ext["n_frames"], len(ext["feat"]))[:, None]
                sl, _ = process_feat(sl, vl)
                sl = sl[:, 0]
            else:
                sl = np.zeros(vl, np.float32)
            return feat, length, sl

        normal = [prep(e, l) for e, l, v in items if v == 0]
        anomaly = [prep(e, l) for e, l, v in items if v == 1]
        if not normal or not anomaly:
            raise ValueError("VadCLIP の学習には正常・異常の両方の動画が必要です")
        bs = max(1, min(a.vadclip_batch_size, len(normal), len(anomaly)))
        opt = torch.optim.AdamW(self.model.parameters(), lr=a.vadclip_lr)
        sched = MultiStepLR(opt, [int(x) for x in a.vadclip_milestones.split(",")], 0.1)
        C = len(self.prompt_text)

        for e in range(a.vadclip_epochs):
            self.model.train()
            rng.shuffle(normal)
            rng.shuffle(anomaly)
            n_iter = min(len(normal) // bs, len(anomaly) // bs)
            tot = 0.0
            for i in range(n_iter):
                batch = normal[i * bs:(i + 1) * bs] + anomaly[i * bs:(i + 1) * bs]
                visual = torch.tensor(np.stack([b[0] for b in batch])).to(self.device)
                lengths = torch.tensor([b[1] for b in batch]).to(self.device)
                text_labels = torch.zeros(len(batch), C)
                text_labels[:bs, 0] = 1
                text_labels[bs:, 1] = 1
                text_labels = text_labels.to(self.device)

                text_features, logits1, logits2 = self.model(visual, None, self.prompt_text, lengths)
                loss1 = CLAS2(logits1, text_labels, lengths, self.device)
                loss2 = CLASM(logits2, text_labels, lengths, self.device)
                loss3 = torch.zeros(1).to(self.device)
                tn = text_features[0] / text_features[0].norm(dim=-1, keepdim=True)
                for j in range(1, text_features.shape[0]):
                    ta = text_features[j] / text_features[j].norm(dim=-1, keepdim=True)
                    loss3 += torch.abs(tn @ ta)
                loss3 = loss3 / (C - 1) * 1e-1   # 公式は 14 クラスで /13
                loss = loss1 + loss2 + loss3
                if a.vadclip_supervision == "frame":
                    y = torch.tensor(np.stack([b[2] for b in batch])).to(self.device)
                    mask = torch.arange(visual.shape[1], device=self.device)[None] < lengths[:, None]
                    bce = F.binary_cross_entropy_with_logits(logits1[..., 0], y, reduction="none")
                    loss = loss + (bce * mask).sum() / mask.sum()

                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item()
            sched.step()
            print(f"    [vadclip] epoch {e + 1}/{a.vadclip_epochs}  loss {tot / max(n_iter, 1):.4f}  ({n_iter} iter)")

    # ---------- 推論（公式 ucf_test.py と同じ分割方法）----------
    @torch.no_grad()
    def predict(self, extracted):
        maxlen = self.args.vadclip_visual_length
        self.model.eval()
        visual, length = process_split(extracted["feat"], maxlen)
        len_cur = length
        visual = torch.tensor(visual)
        if len_cur < maxlen:
            visual = visual.unsqueeze(0)
        visual = visual.to(self.device)
        lengths = torch.zeros(int(length / maxlen) + 1)
        for j in range(int(length / maxlen) + 1):
            if j == 0 and length < maxlen:
                lengths[j] = length
            elif j == 0 and length > maxlen:
                lengths[j] = maxlen
                length -= maxlen
            elif length > maxlen:
                lengths[j] = maxlen
                length -= maxlen
            else:
                lengths[j] = length
        lengths = lengths.to(int)
        padding_mask = get_batch_mask(lengths, maxlen).to(self.device)
        _, logits1, logits2 = self.model(visual, padding_mask, self.prompt_text, lengths)
        logits1 = logits1.reshape(logits1.shape[0] * logits1.shape[1], logits1.shape[2])
        logits2 = logits2.reshape(logits2.shape[0] * logits2.shape[1], logits2.shape[2])
        prob2 = (1 - logits2[0:len_cur].softmax(dim=-1)[:, 0].squeeze(-1)).float().cpu().numpy()
        prob1 = torch.sigmoid(logits1[0:len_cur].squeeze(-1)).float().cpu().numpy()
        s = {"prob1": prob1, "prob2": prob2, "mean": (prob1 + prob2) / 2}[self.args.vadclip_score]
        s = np.repeat(np.atleast_1d(s), self.SNIPPET)          # 公式と同様に 16 フレームへ展開
        n = extracted["n_frames"]
        if len(s) < n:
            s = np.pad(s, (0, n - len(s)), mode="edge")
        return s[:n].astype(np.float64)
