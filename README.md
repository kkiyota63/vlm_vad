# ゼロショット映像異常検知 3 手法の比較
| 手法 | 公式リポジトリ | コミット | 実行スクリプト |
|---|---|---|---|
| CoReVAD | https://github.com/Muk-00/CoReVAD | `76ea2a0` | `run_corevad.py` |
| LAVAD | https://github.com/lucazanella/lavad | `1ad46c6` | `run_lavad.py` |
| AnyAnomaly(MiniCPM-V 版) | https://github.com/SkiddieAhn/Paper-AnyAnomaly | `ec7e9fc` | `run_anyanomaly.py` |

## 構成

```
common.py            分割・ラベル読み込み・limit_data・evaluation(knn_cross_validation.py を踏襲)
run_<name>.py        公式コードを呼んで動画ごとのスコアを保存し、common.cross_validate で評価
envs/<name>/         手法ごとの uv プロジェクト(pyproject.toml, uv.lock。Linux x86_64 専用)
third_party/<name>/  公式コード(無改変。git 管理外)
work/                中間生成物と動画ごとのスコア(git 管理外)
```

動画ごとのフレーム単位スコアは `work/` 以下に `<動画名>.npy` として保存され、再実行時は算出済みの動画をスキップする。

## GPU サーバーでのセットアップ

```bash
uv sync --project envs/corevad
uv sync --project envs/lavad
uv sync --project envs/anyanomaly
```

## 実行
```bash
uv run --project envs/corevad python run_corevad.py
uv run --project envs/lavad python run_lavad.py
uv run --project envs/anyanomaly python run_anyanomaly.py
```

### CoReVAD
- 重み: `OpenGVLab/InternVL2-8B` と CLIP ViT-B/16 が自動ダウンロードされる。
- 公式コードは 30 フレームを 1 区間として処理する。

### LAVAD
- 重み: BLIP-2 の 5 モデルと ImageBind は自動ダウンロード。**Llama 2 は手動**で、Meta の利用申請後に
  `third_party/lavad/libs/llama/download.sh` で `llama-2-13b-chat` を取得し、
  `third_party/lavad/libs/llama/llama-2-13b-chat/` と `third_party/lavad/libs/llama/tokenizer.model` に置く。
- `llama-2-13b-chat` はチェックポイントが 2 分割のため **GPU が 2 枚必要**(公式は A100 × 2)。1 枚の場合は
  `run_lavad.py` の `LLM_MODEL` を `llama-2-7b-chat` に、`LLM_NPROC` を 1 に変える。
- `run_lavad.py` の `FPS` を動画のフレームレートに合わせる。
- torch 1.13.0(CUDA 11.7)固定のため、新しい世代の GPU では動かない可能性がある。

### AnyAnomaly
- 重み: `openbmb/MiniCPM-Llama3-V-2_5`(Hugging Face で利用規約への同意が必要な場合がある)と CLIP ViT-B/32。
- **`run_anyanomaly.py` の `ANOMALY_TYPE` に検出したい異常を英語で設定する(必須)**。この手法は異常をテキストで指定する。
- 公式のグリッドサーチ(正解ラベルで重みと平滑化幅を選ぶ)は使わず、公式の固定値(α=0.6, β=0.3, γ=0.1, σ=15)を使う。

## 公式実装との差分

公式コードには手を入れていない。ラッパー側での違いは次のとおり。

- CoReVAD: CLIP 特徴の抽出を 512 フレームずつに分けて呼ぶ(メモリ対策。結果は同じ)。
- LAVAD: キャプション生成を 16 フレーム間隔にしている(公式スクリプトは全フレーム。後段は 16 の倍数フレームしか参照しない)。
- AnyAnomaly: ガウシアン平滑化を動画ごとに行う(公式は全動画を連結して実施)。24 フレームに満たない末尾は直前の値で埋める。
  環境は MiniCPM-V 公式 requirements のうち推論に必要なものだけを入れている。
- 評価: 3 手法とも `common.py` で行い、各公式の評価スクリプトは使わない。
