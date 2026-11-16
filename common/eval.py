"""評価（knn_cross_validation.py と同一の指標・閾値の扱い）"""
import numpy as np
import pandas as pd
import sklearn.metrics


def evaluation(label, prediction, threshold):
    y_true = np.asarray(label).astype(np.int64).reshape(-1)
    score = np.asarray(prediction).astype(np.float64).reshape(-1)
    y_pred = (score >= threshold).astype(np.int64)                        # 閾値で 0/1 に判定

    accuracy = float((y_true == y_pred).mean())                           # 正解率
    f1 = sklearn.metrics.f1_score(y_true, y_pred, zero_division=0)        # F1-score
    if len(np.unique(y_true)) < 2:                                        # 片方のクラスしか無い場合
        return accuracy, float("nan"), float("nan"), f1
    roc_auc = sklearn.metrics.roc_auc_score(y_true, score)                # ROC-AUC
    pr_auc = sklearn.metrics.average_precision_score(y_true, score)       # PR-AUC
    return accuracy, roc_auc, pr_auc, f1


def threshold_from_cal(cal_scores, contamination):
    """閾値用動画のスコアの 100*(1-contamination) パーセンタイル"""
    return float(np.percentile(np.asarray(cal_scores), 100 * (1 - contamination)))


def wrong_frames(fold_idx, score, y, row_files, row_idx, threshold):
    """誤検出したフレームの一覧（knn_cross_validation.py と同じく score > threshold で判定）"""
    pred = (score > threshold).astype(int)
    wrong = np.where(pred != y)[0]
    return pd.DataFrame({
        "fold": fold_idx,
        "file": [row_files[i] for i in wrong],
        "frame": [row_idx[i] for i in wrong],
        "label": y[wrong],
        "pred": pred[wrong],
        "score": score[wrong],
    })
