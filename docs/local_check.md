# ローカル動作確認の記録（2026-11-16, MacBook M2 Pro / 16GB, CUDA なし）

ダミーデータ（`scripts/make_dummy_data.py`: 20 本、5 分割、異常動画は中央 1/3 に赤い物体が出現）で、
`--quick`（学習 正常2/異常2、テスト 正常2/異常1、cal 2 本）により run_cv.py を最後まで実行した。
結果は動作確認用のもので、性能評価ではない。

| モデル | 設定 | デバイス | 結果 | 所要時間 |
|---|---|---|---|---|
| vadclip（弱教師あり） | 全 5 fold, 3 epoch | CPU | 完走。ROC-AUC 0.874 ± 0.021 | 39 s |
| vadclip（`--vadclip-supervision frame`） | fold 0, 2 epoch | CPU | 完走。ROC-AUC 0.886 | 3 s |
| corevad | fold 0, **InternVL2-1B**, max_new_tokens 64 | CPU | 完走。ROC-AUC 0.855。応答のパース（"Anomalous/Normal scenes." + 説明）を確認 | 330 s |
| anyanomaly | fold 0, **Qwen2.5-VL-3B**, transformers バックエンド | MPS (bf16) | 完走。ROC-AUC 1.000（異常区間のみ 0.9〜1.0 と採点） | 420 s（DL 含む） |
| anyanomaly（`--aa-tune`） | fold 0, 学習 fold で sigma・重みを探索 | MPS (bf16) | 完走。ROC-AUC 1.000 | 130 s |

## 未確認（GPU サーバーで初めて動くもの）

- CUDA 上での実行全般（bf16、`--use-flash-attn`）
- **本番サイズの VLM**: InternVL2-8B（CoReVAD）と Qwen2.5-VL-7B（AnyAnomaly）。同じ API の小型モデルでのみ確認した
- **AnyAnomaly の vLLM バックエンド**（`--aa-backend vllm`, 公式実装と同じ経路）。Mac では vLLM が動かないため未確認。動かない場合は `--aa-backend hf` を使う
- `scripts/setup_corevad.sh` と `setup_anyanomaly.sh`: 構文チェックのみ（同じ requirements を uv で手動インストールして確認済み）。`setup_vadclip.sh` は実行して確認済み
- 実データ（生動画のパス・ラベル長と動画フレーム数の対応）
