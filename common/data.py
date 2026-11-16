"""データ読み込み（knn_cross_validation.py と同じ分割・本数制限・ラベル規則）

フレーム単位評価の方針:
  - ラベル npy は長さ L のフレーム単位 0/1 列（異常動画のみ存在）。
  - 正常動画はラベルが無いので、長さ = 動画フレーム数 // LABEL_UNIT_FRAMES の 0 列を作る。
  - ラベル i は動画時刻 (i + 0.5) / L * duration に対応すると見なす（長さが動画と一致しない場合も比例対応）。
  - モデルはサンプリングしたフレーム位置でスコアを出し、それをラベルの全フレームへ線形補間する。
"""
import os
import pathlib
import random
import re

import cv2
import numpy as np

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg", ".m4v")


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r"(\d+)", text)]


def read_txt(path):
    with open(path, encoding="utf-8") as f:
        return [pathlib.Path(os.path.basename(line.strip())).stem for line in f if line.strip()]


def load_splits(split_dir):
    """splits_*.txt を自然順に読み込み、各 fold を "<stem>.npy" のリストにする"""
    split_dir = pathlib.Path(split_dir)
    paths = sorted([str(p.resolve()) for p in split_dir.glob("splits_*.txt") if p.is_file()], key=natural_keys)
    if not paths:
        raise FileNotFoundError(f"splits_*.txt が見つかりません: {split_dir}")
    return [[f"{n}.npy" for n in read_txt(p)] for p in paths]


def limit_data(files, file_to_label, max_normal=None, max_anomaly=None, seed=0):
    """knn_cross_validation.py と同一の本数制限"""
    normal = [p for p in files if not file_to_label[p]]
    anomaly = [p for p in files if file_to_label[p]]
    keep = set()
    if max_normal is None:
        keep.update(normal)
    else:
        n = min(max_normal, len(normal))
        keep.update(random.Random(seed).sample(normal, n))

    if max_anomaly is None:
        keep.update(anomaly)
    else:
        n = min(max_anomaly, len(anomaly))
        keep.update(random.Random(f"{seed}_anomaly").sample(anomaly, n))

    return [p for p in files if p in keep]


def fold_split(splits, fold_idx):
    """fold_idx をテスト、(fold_idx+1)%n を閾値用、残りを学習に使う"""
    test_all = splits[fold_idx]
    cal_idx = (fold_idx + 1) % len(splits)
    cal_all = splits[cal_idx]
    train_all = [p for j, fold in enumerate(splits) if j not in (fold_idx, cal_idx) for p in fold]
    return train_all, cal_all, test_all


def index_videos(video_dir):
    """動画ディレクトリを再帰的に探索し、stem -> パス の辞書を作る"""
    index = {}
    for p in pathlib.Path(video_dir).rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            index.setdefault(p.stem, str(p))
    return index


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"動画を開けません: {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    cap.release()
    return max(n, 1), fps


def _resize(frame, max_side):
    if max_side is None:
        return frame
    h, w = frame.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        frame = cv2.resize(frame, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    return frame


def iter_chunks(path, chunk_len, max_side=None):
    """動画を先頭から chunk_len フレームずつ RGB で返す（メモリに全フレームを載せない）

    yield: (開始フレーム番号, [frame, ...])  最後のチャンクは chunk_len 未満のことがある
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"動画を開けません: {path}")
    buf, start, idx = [], 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        buf.append(_resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), max_side))
        idx += 1
        if len(buf) == chunk_len:
            yield start, buf
            buf, start = [], idx
    cap.release()
    if buf:
        yield start, buf


def load_label(label_dir, file_name, is_anomaly, n_frames, label_unit_frames=1):
    if is_anomaly:
        return np.load(pathlib.Path(label_dir) / file_name).reshape(-1).astype(np.int64)
    return np.zeros(max(n_frames // label_unit_frames, 1), dtype=np.int64)


def label_positions(n_labels, n_frames):
    """ラベル i の中心に対応する動画フレーム番号（連続値）。n_labels == n_frames なら i そのもの"""
    return (np.arange(n_labels) + 0.5) * n_frames / n_labels - 0.5


def scores_to_labels(sample_frames, sample_scores, n_labels, n_frames):
    """サンプル位置のスコアをラベルの全フレームへ線形補間する（フレーム単位スコア）"""
    sample_frames = np.asarray(sample_frames, dtype=np.float64)
    sample_scores = np.asarray(sample_scores, dtype=np.float64)
    if len(sample_frames) == 1:
        return np.full(n_labels, sample_scores[0])
    return np.interp(label_positions(n_labels, n_frames), sample_frames, sample_scores)


class Dataset:
    """動画・ラベル・分割をまとめて扱う"""

    def __init__(self, data_root, video_dir, label_unit_frames=1):
        self.data_root = pathlib.Path(data_root)
        self.label_dir = self.data_root / "label"
        self.splits = load_splits(self.data_root / "splits")
        self.label_unit_frames = label_unit_frames
        label_files = {p.name for p in self.label_dir.glob("*.npy") if p.is_file()}
        vids = index_videos(video_dir)

        all_files = [p for fold in self.splits for p in fold]
        missing = [p for p in all_files if pathlib.Path(p).stem not in vids]
        if missing:
            print(f"[warn] 動画が見つからないファイル {len(missing)} 本を除外します（例: {missing[:3]}）")
        missing = set(missing)
        self.splits = [[p for p in fold if p not in missing] for fold in self.splits]
        self.video_path = {p: vids[pathlib.Path(p).stem] for fold in self.splits for p in fold}
        # 教師データ（ラベル npy）が存在すれば異常動画
        self.file_to_label = {p: int(p in label_files) for p in self.video_path}
        self._info = {}

    def info(self, file_name):
        if file_name not in self._info:
            self._info[file_name] = video_info(self.video_path[file_name])
        return self._info[file_name]

    def label(self, file_name, n_frames=None):
        """フレーム単位ラベル。n_frames は実際に読めたフレーム数（正常動画のラベル長に使う）"""
        if n_frames is None:
            n_frames, _ = self.info(file_name)
        lab = load_label(self.label_dir, file_name, self.file_to_label[file_name], n_frames, self.label_unit_frames)
        expected = n_frames // self.label_unit_frames
        if self.file_to_label[file_name] and abs(len(lab) - expected) > max(2, 0.02 * expected):
            print(f"[warn] {file_name}: ラベル長 {len(lab)} / 動画フレーム数 {n_frames} (unit {self.label_unit_frames})  比例対応で処理します")
        return lab
