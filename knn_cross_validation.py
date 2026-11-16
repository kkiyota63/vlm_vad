from cuml.neighbors import NearestNeighbors, KNeighborsClassifier
from cuml import metrics
import sklearn
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pathlib
import os
import re
import cupy as cp
from time import time
import random
import pandas as pd

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def read_txt(path):
    with open(path, encoding="utf-8") as f:
        return [pathlib.Path(os.path.basename(line.strip())).stem for line in f if line.strip()]

def evaluation(label, prediction, threshold):
    to_np = lambda a: a.get() if hasattr(a, "get") else np.asarray(a)
    y_true = to_np(label).astype(np.int64).reshape(-1)
    score = to_np(prediction).astype(np.float64).reshape(-1)
    y_pred = (score >= threshold).astype(np.int64)                        # 閾値で 0/1 に判定

    accuracy = float((y_true == y_pred).mean())                           # 正解率
    f1 = sklearn.metrics.f1_score(y_true, y_pred, zero_division=0)        # F1-score
    if len(np.unique(y_true)) < 2:                                        # 01しか出力がない場合
        return accuracy, float("nan"), float("nan"), f1
    roc_auc = sklearn.metrics.roc_auc_score(y_true, score)                # ROC-AUC
    pr_auc = sklearn.metrics.average_precision_score(y_true, score)       # PR-AUC
    
    return accuracy, roc_auc, pr_auc, f1

def load_single_file_jepa(feat_path, label_path, file_path, is_anomaly, stride, attentive_probes, ROI = [5, 6, 9, 10]):    
    if attentive_probes:
        feat = np.load(pathlib.Path(feat_path) / file_path).astype(np.float32)                          # 単純読み込み
    else:
        feat = np.load(pathlib.Path(feat_path) / file_path).astype(np.float32)[:, ROI].mean(axis=1)     # 中央2x2の平均プーリング
        # feat = np.load(pathlib.Path(feat_path) / file_path).astype(np.float32).mean(axis=1)           # 全面平均プーリング
    if is_anomaly:
        lab = np.load(pathlib.Path(label_path) / file_path).reshape(-1).astype(np.int64)
    else:
        lab = np.zeros(len(feat), dtype=np.int64)
    t = min(len(feat), len(lab))
    if len(feat) != len(lab):       # 特徴量の長さとラベルの長さが合わない場合
        print(f"[warn] {file_path}: 特徴 {len(feat)} / ラベル {len(lab)}")
    return feat[:t:stride], lab[:t:stride]

def load_single_file_i3d(feat_path, label_path, file_path, is_anomaly, stride):    
    feat = np.load(pathlib.Path(feat_path) / file_path).astype(np.float32).T
    if is_anomaly:
        lab = np.load(pathlib.Path(label_path) / file_path).reshape(-1)[:-10].astype(np.int64)
    else:
        lab = np.zeros(len(feat), dtype=np.int64)
    t = min(len(feat), len(lab))
    if len(feat) != len(lab):       # 特徴量の長さとラベルの長さが合わない場合
        print(f"[warn] {file_path}: 特徴 {len(feat)} / ラベル {len(lab)}")
    return feat[:t:stride], lab[:t:stride]

def load_single_file_video_mae(feat_path, label_path, file_path, is_anomaly, stride):    
    feat = np.load(pathlib.Path(feat_path) / file_path).astype(np.float32)
    if is_anomaly:
        lab = np.load(pathlib.Path(label_path) / file_path).reshape(-1).astype(np.int64)
    else:
        lab = np.zeros(len(feat), dtype=np.int64)
    t = min(len(feat), len(lab))
    if len(feat) != len(lab):       # 特徴量の長さとラベルの長さが合わない場合
        print(f"[warn] {file_path}: 特徴 {len(feat)} / ラベル {len(lab)}")
    return feat[:t:stride], lab[:t:stride]

def load_features_and_labels(video_paths, labels_for_split, feat_path, label_path, method, attentive_probes, stride=1):
    flags = cp.asnumpy(labels_for_split)
    if method == 'V-JEPA2.1':
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = [
                executor.submit(load_single_file_jepa, feat_path, label_path, p, bool(f), stride, attentive_probes)
                for p, f in zip(video_paths, flags)
            ]
            results = [f.result() for f in futures]
    elif method == "VideoMAEv2":
        with ThreadPoolExecutor(max_workers=32) as executor:
                    futures = [
                        executor.submit(load_single_file_video_mae, feat_path, label_path, p, bool(f), 1)
                        for p, f in zip(video_paths, flags)
                    ]
                    results = [f.result() for f in futures]
    else:
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = [
                executor.submit(load_single_file_i3d, feat_path, label_path, p, bool(f), 1)
                for p, f in zip(video_paths, flags)
            ]
            results = [f.result() for f in futures]
    X_fold = cp.asarray(np.concatenate([r[0] for r in results], axis=0))
    y_fold = cp.asarray(np.concatenate([r[1] for r in results], axis=0))
    return X_fold, y_fold

def limit_data(files, file_to_label, max_normal=None, max_anomaly=None, seed=0):
    normal = [p for p in files if not file_to_label[p]]
    anomaly = [p for p in files if file_to_label[p]]
    keep = set()
    if max_normal is None: keep.update(normal)  # 全部の場合
    else:                                       # 制限する場合
        n = min(max_normal, len(normal))
        keep.update(random.Random(seed).sample(normal, n))

    if max_anomaly is None: keep.update(anomaly)
    else:
        n = min(max_anomaly, len(anomaly))
        keep.update(random.Random(f"{seed}_anomaly").sample(anomaly, n))

    return [p for p in files if p in keep]

def main():
    METHOD = 'I3D'              # 特徴量手法　'V-JEPA2.1' or 'VideoMAEv2' or 'I3D'
    ATTENTIVE_PROBES = False    # V-JEPA2.1のAttentive Probesを適用した特徴量の場合、Ture
    MAX_NORMAL_TRAIN = 500      # 学習に使用する正常動画の本数（全て：None）
    MAX_ANOMALY_TRAIN = 100     # 学習に使用する異常動画の本数（全て：None）
    MAX_NORMAL_TEST = 100       # テストに使用する正常動画の本数（全て：None）
    MAX_ANOMALY_TEST = 50       # テストに使用する異常動画の本数（全て：None）
    MAX_CAL_VIDEOS = 50         # 教師なし学習の閾値を決めるための動画本数
    QUERY_BATCH = 50000         # テストフレームを小分けにして計算（GPUメモリ対策）
    MEM_STRIDE = 2              # 正常ラベルのフレーム間引き。2は重複ペアの除去（V-JEPA2.1用、I3D実行時は未使用）
    SEED = 0                    # ランダムシード
    K_NORMAL = 5                # 正常ラベルの近傍数
    K_ANOMALY = 1               # 異常ラベルの近傍数
    K = 5                       # k分割検証の分割数
    SUPERVISED_FLG = True       # True:教師あり学習、False:教師なし学習
    CONTAMINATION = 0.006       # 教師なし学習：異常フレームの割合（accとF1の閾値に使用）

    # --- データセット読み込み ---
    target_dir = '/home/jovyan/work/toyota2026/data/'
    #feat_path = pathlib.Path(os.path.join(target_dir, METHOD, 'feature_grid_4'))
    feat_path = pathlib.Path("/home/jovyan/work/toyota2026/tracking_src/output_i3d_feature")
    label_path = pathlib.Path(os.path.join(target_dir, 'label'))
    split_path = pathlib.Path(os.path.join(target_dir, 'splits'))
    file_name_list = [str(os.path.basename(p.resolve())) for p in feat_path.glob('*.npy') if p.is_file()]           # 全特徴量
    label_file_name_list = [str(os.path.basename(p.resolve())) for p in label_path.glob('*.npy') if p.is_file()]    # 教師データ
    anomaly_file_list = []
    normally_file_list = []
    for name in file_name_list:     # 正常/異常ファイルを分ける 
        if name in label_file_name_list:
            anomaly_file_list.append(name)
        else:
            normally_file_list.append(name)

    # ファイルに対する正常/異常ラベル付与(教師データとして使用)
    file_to_label = {file: 1 for file in anomaly_file_list}
    file_to_label.update({file: 0 for file in normally_file_list})

    # --- split*.txt読み込み ---
    splits = []
    split_files_path = sorted([str(p.resolve()) for p in split_path.glob("splits_*.txt") if p.is_file()], key=natural_keys)
    for file_path in split_files_path: 
        names = read_txt(file_path)
        splits.append([f"{n}.npy" for n in names])

    # --- k分割交差検証 ---
    fold_scores = {"accuracy":[], "ROC_AUC":[], "PR_AUC":[], "F1":[]}
    for fold_idx in range(K):
        test_all = splits[fold_idx]             # テスト用
        cal_idx = (fold_idx + 1) % len(splits)
        cal_all = splits[cal_idx]               # 閾値用
        train_all = [p for j, fold in enumerate(splits) if j not in (fold_idx, cal_idx) for p in fold]

        # --- データ本数の制限 ---
        train_files = limit_data(train_all, file_to_label, MAX_NORMAL_TRAIN, MAX_ANOMALY_TRAIN, SEED + fold_idx)
        test_files = limit_data(test_all, file_to_label, MAX_NORMAL_TEST, MAX_ANOMALY_TEST, SEED + 100 + fold_idx)
        y_train = cp.array([file_to_label[p] for p in train_files], dtype=cp.int64)
        y_test = cp.array([file_to_label[p] for p in test_files], dtype=cp.int64)
        rng = random.Random(SEED + fold_idx)
        cal_files = rng.sample(cal_all, min(MAX_CAL_VIDEOS, len(cal_all)))

        # --- 学習 ---
        X_train_fold, y_train_fold = load_features_and_labels(train_files, y_train, feat_path, label_path, METHOD, ATTENTIVE_PROBES, stride=MEM_STRIDE)
        if SUPERVISED_FLG:  # 教師あり
            # clf = KNeighborsClassifier(n_neighbors=K_NORMAL, weights="distance").fit(X_train_fold, y_train_fold)  # ラベル識別
            nn_nor = NearestNeighbors(n_neighbors=K_NORMAL).fit(X_train_fold[y_train_fold == 0])    # 正常ラベルのフレームだけ学習
            nn_ano = NearestNeighbors(n_neighbors=K_ANOMALY).fit(X_train_fold[y_train_fold == 1])   # 異常ラベルのフレームだけ学習
            n_mem_nor, n_mem_ano = int((y_train_fold == 0).sum()), int((y_train_fold == 1).sum())
        else:               # 教師なし
            nn_all = NearestNeighbors(n_neighbors=K_NORMAL).fit(X_train_fold)                          # 全フレームを学習
            n_mem = len(X_train_fold)
        del X_train_fold, y_train_fold

        # --- 閾値決め ---
        X_cal_fold, _ = load_features_and_labels(cal_files, cp.zeros(len(cal_files), dtype=cp.int64), feat_path, label_path, METHOD, ATTENTIVE_PROBES, stride=MEM_STRIDE)
        cal_scores = []
        for i in range(0, len(X_cal_fold), QUERY_BATCH):
            xb = X_cal_fold[i:i + QUERY_BATCH]
            if SUPERVISED_FLG:      # 教師あり
                d_n = nn_nor.kneighbors(xb)[0][:, -1]
                d_a = nn_ano.kneighbors(xb)[0][:, -1]
                s = d_n / (d_n + d_a)
            else:                   # 教師なし
                s = nn_all.kneighbors(xb)[0][:, -1]      # k番目に近い特徴までの距離
            cal_scores.append(cp.asarray(s))
        threshold = float(cp.percentile(cp.concatenate(cal_scores), 100 * (1 - CONTAMINATION)))
        del X_cal_fold

        # --- テスト ---
        #X_test_fold, y_test_fold = load_features_and_labels(test_files, y_test, feat_path, label_path, METHOD, ATTENTIVE_PROBES)
        Xs, ys, test_row_files, test_row_idx = [], [], [], []
        for j, f in enumerate(test_files):
            Xf, yf = load_features_and_labels([f], y_test[j:j+1], feat_path, label_path, METHOD, ATTENTIVE_PROBES)
            Xs.append(Xf)
            ys.append(yf)
            test_row_files.extend([f] * len(Xf))   # 各行のファイル名
            test_row_idx.extend(range(len(Xf)))    # ファイル内の行番号
        xp = cp.get_array_module(Xs[0])
        X_test_fold = xp.concatenate(Xs)
        y_test_fold = xp.concatenate(ys)
        
        scores = []
        if SUPERVISED_FLG:  # 教師あり
            n_test_nor, n_test_ano = int((y_test_fold == 0).sum()), int((y_test_fold == 1).sum())
            for i in range(0, len(X_test_fold), QUERY_BATCH):
                xb = X_test_fold[i:i + QUERY_BATCH]
                d_n = nn_nor.kneighbors(xb)[0][:, -1]
                d_a = nn_ano.kneighbors(xb)[0][:, -1]
                scores.append(cp.asarray(d_n / (d_n + d_a)))
        else:               # 教師なし
            for i in range(0, len(X_test_fold), QUERY_BATCH):
                d_all = nn_all.kneighbors(X_test_fold[i:i + QUERY_BATCH])[0][:, -1]
                scores.append(cp.asarray(d_all))
        score = cp.concatenate(scores)

        # ---誤検出したフレームを調査（追加の実装）---
        score_np = cp.asnumpy(score)
        y_np = cp.asnumpy(y_test_fold) if isinstance(y_test_fold, cp.ndarray) else np.asarray(y_test_fold)

        pred = (score_np > threshold).astype(int)

        wrong = np.where(pred != y_np)[0]
        df_wrong = pd.DataFrame({
            "fold": fold_idx,
            "file": [test_row_files[i] for i in wrong],   # ← 変更
            "frame": [test_row_idx[i] for i in wrong],    # ← 追加
            "label": y_np[wrong],
            "pred": pred[wrong],
            "score": score_np[wrong],
        })
        df_wrong.to_csv(f"wrong_fold{fold_idx}.csv", index=False)


        # --- 評価 ---
        accuracy, roc_auc, pr_auc, f1 = evaluation(y_test_fold, score, threshold=threshold)
        fold_scores["accuracy"].append(accuracy)
        fold_scores["ROC_AUC"].append(roc_auc)
        fold_scores["PR_AUC"].append(pr_auc)
        fold_scores["F1"].append(f1)
        if SUPERVISED_FLG:  # 教師あり
            print(f"fold {fold_idx}: accuracy {accuracy:.4f}  ROC-AUC {roc_auc:.3f}  PR-AUC {pr_auc:.3f}  F1 {f1:.3f}  "
                f"(学習 正常 {n_mem_nor} / 異常 {n_mem_ano}, テスト 正常 {n_test_nor} / 異常 {n_test_ano} フレーム)")
        else:               # 教師なし
            print(f"fold {fold_idx}: accuracy {accuracy:.4f}  ROC-AUC {roc_auc:.3f}  PR-AUC {pr_auc:.3f}  F1 {f1:.3f}  "
                 f"(メモリ {len(train_files)} 本 / {n_mem} フレーム, 閾値用 {len(cal_files)} 本, 閾値 {threshold:.3f})")

    # --- 全 fold の平均 ---
    for name, v in fold_scores.items():
        print(f"{name}: {np.mean(v):.4f} ± {np.std(v):.4f}")

if __name__ == "__main__":
    t0 = time()
    main()
    t1 = time()
    print(f"total time: {t1 - t0:.2f}s")
