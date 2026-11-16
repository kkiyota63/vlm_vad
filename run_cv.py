"""VLM ベース映像異常検知の k 分割交差検証（フレーム単位評価）

データの分割・本数制限・閾値決め・評価指標は knn_cross_validation.py と同一:
  - fold i をテスト、fold (i+1)%n を閾値用 (cal)、残りを学習に使う
  - limit_data のシード: 学習 SEED+fold, テスト SEED+100+fold, cal は random.Random(SEED+fold).sample
  - 閾値 = cal 動画の全フレームスコアの 100*(1-CONTAMINATION) パーセンタイル
  - accuracy / ROC-AUC / PR-AUC / F1 をフレーム単位で計算、誤検出フレームを wrong_fold{i}.csv に保存

使い方:
  python run_cv.py --model vadclip    --data-root /path/to/data --video-dir /path/to/videos
  python run_cv.py --model corevad    ...
  python run_cv.py --model anyanomaly --aa-keywords "accident,falling" ...
"""
import argparse
import json
import pathlib
import pickle
import random
import sys
from time import time

import numpy as np
import pandas as pd
import torch

from common.data import Dataset, fold_split, limit_data, scores_to_labels
from common.eval import evaluation, threshold_from_cal, wrong_frames
from models import MODEL_NAMES, get_model_class


def none_if_neg(v):
    v = int(v)
    return None if v < 0 else v


def parse_args(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--model", choices=MODEL_NAMES, required=True)
    known, _ = pre.parse_known_args(argv)

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=MODEL_NAMES, required=True)
    p.add_argument("--data-root", default="/home/jovyan/work/toyota2026/data", help="label/ と splits/ を含むディレクトリ")
    p.add_argument("--video-dir", default=None, help="生動画のディレクトリ（再帰探索）。既定: <data-root>/videos")
    p.add_argument("--label-unit-frames", type=int, default=1, help="ラベル 1 要素あたりの動画フレーム数（正常動画のラベル長に使用）")
    # knn_cross_validation.py と同じハイパーパラメータ（-1 で全て）
    p.add_argument("--max-normal-train", type=none_if_neg, default=500)
    p.add_argument("--max-anomaly-train", type=none_if_neg, default=100)
    p.add_argument("--max-normal-test", type=none_if_neg, default=100)
    p.add_argument("--max-anomaly-test", type=none_if_neg, default=50)
    p.add_argument("--max-cal-videos", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=5, help="分割数（splits_*.txt の数が上限）")
    p.add_argument("--folds", default=None, help="実行する fold 番号（例: 0,1）。既定は全 fold")
    p.add_argument("--contamination", type=float, default=0.006)
    p.add_argument("--out-dir", default="outputs")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--device", default="auto", help="auto / cuda / cuda:1 / mps / cpu")
    p.add_argument("--quick", action="store_true", help="動作確認用: 本数を最小限に制限")
    get_model_class(known.model).add_args(p)
    a = p.parse_args(argv)
    if a.video_dir is None:
        a.video_dir = str(pathlib.Path(a.data_root) / "videos")
    if a.quick:
        a.max_normal_train, a.max_anomaly_train = 2, 2
        a.max_normal_test, a.max_anomaly_test = 2, 1
        a.max_cal_videos = 2
    return a


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(argv=None):
    args = parse_args(argv)
    device = pick_device(args.device)
    print(f"model: {args.model}  device: {device}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ds = Dataset(args.data_root, args.video_dir, args.label_unit_frames)
    model = get_model_class(args.model)(args, device)
    cache_dir = pathlib.Path(args.cache_dir) / args.model / model.cache_key()
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "config.json").write_text(json.dumps(model.cache_config(), indent=2, ensure_ascii=False))
    out_dir = pathlib.Path(args.out_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False))
    print(f"cache: {cache_dir}\noutput: {out_dir}")

    def get_ext(f):
        """VLM 推論・特徴抽出の結果（動画ごとにキャッシュ。fold をまたいで再利用）"""
        path = cache_dir / (pathlib.Path(f).stem + ".pkl")
        if path.exists():
            with open(path, "rb") as fp:
                return pickle.load(fp)
        t = time()
        ext = model.extract(ds.video_path[f])
        with open(path, "wb") as fp:
            pickle.dump(ext, fp)
        print(f"    extract {f}: {ext['n_frames']} frames, {time() - t:.1f}s")
        return ext

    def frame_scores(f):
        """動画 f のフレーム単位スコアとラベル（ラベル長に対応付け）"""
        ext = get_ext(f)
        n = ext["n_frames"]
        lab = ds.label(f, n)
        s = model.predict(ext)
        return scores_to_labels(np.arange(n), s, len(lab), n), lab

    splits = ds.splits
    K = min(args.k, len(splits))
    folds = range(K) if args.folds is None else [int(x) for x in args.folds.split(",")]
    fold_scores = {"accuracy": [], "ROC_AUC": [], "PR_AUC": [], "F1": []}
    rows = []
    for fold_idx in folds:
        t0 = time()
        train_all, cal_all, test_all = fold_split(splits, fold_idx)
        train_files = limit_data(train_all, ds.file_to_label, args.max_normal_train, args.max_anomaly_train, args.seed + fold_idx)
        test_files = limit_data(test_all, ds.file_to_label, args.max_normal_test, args.max_anomaly_test, args.seed + 100 + fold_idx)
        rng = random.Random(args.seed + fold_idx)
        cal_files = rng.sample(cal_all, min(args.max_cal_videos, len(cal_all)))
        print(f"fold {fold_idx}: train {len(train_files)} / cal {len(cal_files)} / test {len(test_files)} videos")

        # --- 学習（学習しないモデルは学習 fold の推論自体を省略）---
        if model.trainable:
            items = []
            for f in train_files:
                ext = get_ext(f)
                items.append((ext, ds.label(f, ext["n_frames"]), ds.file_to_label[f]))
            model.fit(items)

        # --- 閾値決め ---
        cal = np.concatenate([frame_scores(f)[0] for f in cal_files])
        threshold = threshold_from_cal(cal, args.contamination)

        # --- テスト（フレーム単位）---
        ss, ys, row_files, row_idx = [], [], [], []
        per_video = {}
        for f in test_files:
            s, y = frame_scores(f)
            ss.append(s)
            ys.append(y)
            row_files.extend([f] * len(s))
            row_idx.extend(range(len(s)))
            per_video[f] = {"score": s.astype(np.float32), "label": y.astype(np.int8)}
        score = np.concatenate(ss)
        y_test = np.concatenate(ys)
        with open(out_dir / f"scores_fold{fold_idx}.pkl", "wb") as fp:
            pickle.dump(per_video, fp)
        wrong_frames(fold_idx, score, y_test, row_files, row_idx, threshold).to_csv(out_dir / f"wrong_fold{fold_idx}.csv", index=False)

        # --- 評価 ---
        accuracy, roc_auc, pr_auc, f1 = evaluation(y_test, score, threshold=threshold)
        for k_, v in zip(fold_scores, (accuracy, roc_auc, pr_auc, f1)):
            fold_scores[k_].append(v)
        n_nor, n_ano = int((y_test == 0).sum()), int((y_test == 1).sum())
        print(f"fold {fold_idx}: accuracy {accuracy:.4f}  ROC-AUC {roc_auc:.3f}  PR-AUC {pr_auc:.3f}  F1 {f1:.3f}  "
              f"(テスト 正常 {n_nor} / 異常 {n_ano} フレーム, 閾値 {threshold:.4f}, {time() - t0:.1f}s)")
        rows.append({"fold": fold_idx, "accuracy": accuracy, "ROC_AUC": roc_auc, "PR_AUC": pr_auc, "F1": f1,
                     "threshold": threshold, "test_normal_frames": n_nor, "test_anomaly_frames": n_ano})

    # --- 全 fold の平均 ---
    for name, v in fold_scores.items():
        print(f"{name}: {np.mean(v):.4f} ± {np.std(v):.4f}")
    pd.DataFrame(rows).to_csv(out_dir / "results.csv", index=False)


if __name__ == "__main__":
    t0 = time()
    main(sys.argv[1:])
    print(f"total time: {time() - t0:.2f}s")
