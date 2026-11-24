import os
import pathlib
import random
import re

import numpy as np
import sklearn.metrics

DATA_DIR = pathlib.Path('/home/jovyan/work/toyota2026')
LABEL_DIR = DATA_DIR / 'data/label'
SPLIT_DIR = DATA_DIR / 'data/splits'
VIDEO_DIR = DATA_DIR / 'tracking_src/output_crop_video'
VIDEO_EXT = '_crop.mp4'

ROOT = pathlib.Path(__file__).resolve().parent
WORK_DIR = ROOT / 'work'
FRAME_DIR = WORK_DIR / 'frames'

MAX_NORMAL_TEST = 100
MAX_ANOMALY_TEST = 50
MAX_CAL_VIDEOS = 50
SEED = 0
K = 5
CONTAMINATION = 0.006


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def read_txt(path):
    with open(path, encoding="utf-8") as f:
        return [pathlib.Path(os.path.basename(line.strip())).stem for line in f if line.strip()]

def evaluation(label, prediction, threshold):
    y_true = np.asarray(label).astype(np.int64).reshape(-1)
    score = np.asarray(prediction).astype(np.float64).reshape(-1)
    y_pred = (score >= threshold).astype(np.int64)

    accuracy = float((y_true == y_pred).mean())
    f1 = sklearn.metrics.f1_score(y_true, y_pred, zero_division=0)
    if len(np.unique(y_true)) < 2:
        return accuracy, float("nan"), float("nan"), f1
    roc_auc = sklearn.metrics.roc_auc_score(y_true, score)
    pr_auc = sklearn.metrics.average_precision_score(y_true, score)

    return accuracy, roc_auc, pr_auc, f1

def limit_data(files, file_to_label, max_normal=None, max_anomaly=None, seed=0):
    normal = [p for p in files if not file_to_label[p]]
    anomaly = [p for p in files if file_to_label[p]]
    keep = set()
    if max_normal is None: keep.update(normal)
    else:
        n = min(max_normal, len(normal))
        keep.update(random.Random(seed).sample(normal, n))

    if max_anomaly is None: keep.update(anomaly)
    else:
        n = min(max_anomaly, len(anomaly))
        keep.update(random.Random(f"{seed}_anomaly").sample(anomaly, n))

    return [p for p in files if p in keep]


def folds():
    split_files = sorted([str(p.resolve()) for p in SPLIT_DIR.glob("splits_*.txt") if p.is_file()], key=natural_keys)
    splits = [[f"{n}.npy" for n in read_txt(p)] for p in split_files]
    labeled = {p.name for p in LABEL_DIR.glob('*.npy')}
    file_to_label = {p: int(p in labeled) for fold in splits for p in fold}
    out = []
    for fold_idx in range(K):
        test_all = splits[fold_idx]
        cal_all = splits[(fold_idx + 1) % len(splits)]
        test_files = limit_data(test_all, file_to_label, MAX_NORMAL_TEST, MAX_ANOMALY_TEST, SEED + 100 + fold_idx)
        cal_files = random.Random(SEED + fold_idx).sample(cal_all, min(MAX_CAL_VIDEOS, len(cal_all)))
        out.append((test_files, cal_files))
    return out, file_to_label


def video_names():
    return sorted({pathlib.Path(f).stem for test, cal in folds()[0] for f in test + cal})


def video_path(name):
    return VIDEO_DIR / f"{name}{VIDEO_EXT}"


def extract_frames(name):
    import cv2
    out = FRAME_DIR / name
    if not out.exists():
        tmp = FRAME_DIR / f"{name}.tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path(name)))
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(str(tmp / f"{i:06d}.jpg"), frame)
            i += 1
        cap.release()
        tmp.rename(out)
    return len(list(out.glob('*.jpg')))


def load_score_and_label(score_dir, file_name, is_anomaly):
    score = np.load(pathlib.Path(score_dir) / file_name).astype(np.float64).reshape(-1)
    if is_anomaly:
        lab = np.load(LABEL_DIR / file_name).reshape(-1).astype(np.int64)
    else:
        lab = np.zeros(len(score), dtype=np.int64)
    t = min(len(score), len(lab))
    if len(score) != len(lab):
        print(f"[warn] {file_name}: スコア {len(score)} / ラベル {len(lab)}")
    return score[:t], lab[:t]


def cross_validate(score_dir, fixed_threshold=None):
    fold_list, file_to_label = folds()
    fold_scores = {"accuracy":[], "ROC_AUC":[], "PR_AUC":[], "F1":[]}
    for fold_idx, (test_files, cal_files) in enumerate(fold_list):
        if fixed_threshold is None:
            cal_scores = np.concatenate([load_score_and_label(score_dir, f, False)[0] for f in cal_files])
            threshold = float(np.percentile(cal_scores, 100 * (1 - CONTAMINATION)))
        else:
            threshold = fixed_threshold

        results = [load_score_and_label(score_dir, f, bool(file_to_label[f])) for f in test_files]
        score = np.concatenate([r[0] for r in results])
        y_test = np.concatenate([r[1] for r in results])

        accuracy, roc_auc, pr_auc, f1 = evaluation(y_test, score, threshold=threshold)
        fold_scores["accuracy"].append(accuracy)
        fold_scores["ROC_AUC"].append(roc_auc)
        fold_scores["PR_AUC"].append(pr_auc)
        fold_scores["F1"].append(f1)
        print(f"fold {fold_idx}: accuracy {accuracy:.4f}  ROC-AUC {roc_auc:.3f}  PR-AUC {pr_auc:.3f}  F1 {f1:.3f}  "
              f"(テスト {len(test_files)} 本 / 正常 {int((y_test == 0).sum())} / 異常 {int((y_test == 1).sum())} フレーム, "
              f"閾値用 {len(cal_files)} 本, 閾値 {threshold:.3f})")

    for name, v in fold_scores.items():
        print(f"{name}: {np.mean(v):.4f} ± {np.std(v):.4f}")
