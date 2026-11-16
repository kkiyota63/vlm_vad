"""全モデル共通のインタフェース

  extract(video_path) -> dict
      重い処理（VLM 推論・CLIP 特徴抽出）をここで行う。run_cv.py が動画ごとにキャッシュする。
      必ず "n_frames"（実際に読めたフレーム数）を含める。
  fit(items)
      items = [(extracted, label, video_label)]  label はフレーム単位ラベル, video_label は 0/1
      学習しないモデルは何もしない。
  predict(extracted) -> 動画の全フレーム (n_frames,) の異常スコア
      run_cv.py がこれをラベル長へ対応付けてフレーム単位で評価する。
"""
import hashlib
import json


class VADModel:
    name = "base"

    def __init__(self, args, device):
        self.args = args
        self.device = device

    @staticmethod
    def add_args(parser):
        pass

    @property
    def trainable(self):
        """学習 fold の動画を使うか（False なら学習 fold の推論をスキップ）"""
        return False

    def cache_config(self):
        """extract の結果に影響する設定（キャッシュの区別に使う）"""
        return {}

    def cache_key(self):
        s = json.dumps(self.cache_config(), sort_keys=True)
        return hashlib.md5(s.encode()).hexdigest()[:8]

    def extract(self, video_path):
        raise NotImplementedError

    def fit(self, items):
        pass

    def predict(self, extracted):
        raise NotImplementedError
