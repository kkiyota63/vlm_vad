# VLM_VAD: VLM ベース映像異常検知の比較

3 つの手法（**VadCLIP / AnyAnomaly / CoReVAD**）を、`knn_cross_validation.py` と同じデータ分割・評価方法
（k 分割交差検証、**フレーム単位**の accuracy / ROC-AUC / PR-AUC / F1）で比較するためのコード。
いずれも **VLM/CLIP 本体はファインチューニングしない**。サーベイと選定理由は [docs/survey.md](docs/survey.md) を参照。

| モデル | 方式 | 本番の既定モデル | 学習 fold の使い方 |
|---|---|---|---|
| `vadclip` | 凍結 CLIP ViT-B/16 + LGT-Adapter + プロンプト学習（AAAI 2024） | CLIP ViT-B/16 | アダプタを学習（既定は動画単位ラベルの弱教師あり） |
| `anyanomaly` | Context-aware VQA でキーワードを 0〜1 採点（WACV 2026） | Qwen2.5-VL-7B-Instruct + CLIP ViT-B/32 | 使わない（`--aa-tune` で平滑化と重みだけ探索） |
| `corevad` | VLM の生成文 + CLIP による補正（ICPR 2026） | InternVL2-8B + CLIP ViT-B/16 | 使わない |

## ディレクトリ構成

```
run_cv.py               # k 分割交差検証のメインスクリプト（--model で切り替え）
common/data.py          # split 読み込み・本数制限・ラベル・動画読み込み（knn_cross_validation.py と同じ規則）
common/eval.py          # evaluation()・閾値・誤検出 CSV
models/{vadclip,anyanomaly,corevad}.py   # 各手法の移植（ファイル冒頭に公式実装との対応と差分を記載）
third_party/vadclip/    # VadCLIP 公式コードの抜粋（Apache-2.0）
envs/<model>/requirements.txt            # モデルごとの依存（transformers のバージョンが衝突するため環境を分ける）
scripts/setup_<model>.sh                 # 環境作成＋重みダウンロード
scripts/make_dummy_data.py, smoke_test.sh  # ダミーデータでの動作確認
docs/survey.md          # サーベイ
knn_cross_validation.py # 比較元のベースライン（kNN）
```

## GPU サーバーでの実行手順

```bash
git clone <このリポジトリ> VLM_VAD && cd VLM_VAD

# 1. 環境構築（モデルごとに別の venv。uv があれば uv、無ければ python3 -m venv）
bash scripts/setup_vadclip.sh
FLASH_ATTN=1 bash scripts/setup_corevad.sh        # flash-attn が入らない場合は FLASH_ATTN を外す
WITH_VLLM=1  bash scripts/setup_anyanomaly.sh     # vLLM を使わない場合は WITH_VLLM を外す
#   ※ PyPI の torch は CUDA 12 系のビルド。サーバーの CUDA に合わない場合は
#     envs/<model>/requirements.txt の torch を --index-url 指定で入れ直す。
#   ※ 重みの保存先は HF_HOME（Hugging Face）と ~/.cache/clip（CLIP）

# 2. 実行（データは knn_cross_validation.py と同じ場所。生動画のディレクトリを指定する）
DATA=/home/jovyan/work/toyota2026/data
VIDEOS=/path/to/videos            # ★ 未確定。動画ファイル名の stem が splits_*.txt / label/*.npy と一致すること

.venvs/vadclip/bin/python    run_cv.py --model vadclip    --data-root $DATA --video-dir $VIDEOS
.venvs/corevad/bin/python    run_cv.py --model corevad    --data-root $DATA --video-dir $VIDEOS --use-flash-attn
.venvs/anyanomaly/bin/python run_cv.py --model anyanomaly --data-root $DATA --video-dir $VIDEOS --aa-keywords "accident,falling"
```

- 結果: `outputs/<model>/results.csv`（fold ごとの指標）、`wrong_fold{i}.csv`（誤検出フレーム）、`scores_fold{i}.pkl`（テスト動画ごとのフレームスコアとラベル）
- VLM 推論の結果は `cache/<model>/<設定ハッシュ>/<動画名>.pkl` に保存し、fold や再実行の間で再利用する（中断しても続きから再開できる）
- 長時間になるため `nohup ... &` や tmux での実行を推奨。`--folds 0,1` で一部の fold だけ実行できる。複数 GPU なら `--device cuda:1` や `CUDA_VISIBLE_DEVICES` で分けて並列に回せる（キャッシュは共有される）

### 未確定・要設定の項目

| 項目 | 引数 | 説明 |
|---|---|---|
| 生動画の場所 | `--video-dir` | 既定は `<data-root>/videos`。再帰的に探索し、拡張子 mp4/avi/mov/mkv/webm/mpg/m4v を stem で対応付ける |
| ラベルの粒度 | `--label-unit-frames` | ラベル 1 要素が何フレームに当たるか（既定 1 = フレーム単位）。正常動画のラベル長（= フレーム数 / この値）に使う |
| AnyAnomaly の異常テキスト | `--aa-keywords` | 検出したい異常イベントをカンマ区切りで指定する（例 `"car accident,person falling"`）。既定値は汎用的な仮の値 |
| VadCLIP の異常テキスト | `--vadclip-anomaly-text` | 異常クラスのテキスト（既定 `abnormal`） |

## 共通の評価方法（knn_cross_validation.py との対応）

- 分割: `splits_*.txt` を自然順に並べ、fold i をテスト、fold (i+1)%n を閾値決め（cal）、残りを学習に使う
- 本数制限: `limit_data` は同一の関数・シード（学習 `SEED+fold`、テスト `SEED+100+fold`、cal `random.Random(SEED+fold).sample`）。既定値も同じ（学習 正常500/異常100、テスト 正常100/異常50、cal 50 本）
- 異常/正常の判定: `label/<動画名>.npy` があれば異常動画（フレーム単位 0/1）、無ければ全フレーム 0 の正常動画
- 閾値: cal 動画の全フレームスコアの `100*(1-CONTAMINATION)` パーセンタイル（`--contamination 0.006`）
- 指標: `evaluation()` は同一（閾値 `>=` で accuracy・F1、スコアで ROC-AUC・PR-AUC）。`wrong_fold{i}.csv` は元コードと同じく `>` で判定
- knn との違い: cal に `MEM_STRIDE` の間引きは適用しない（全フレームを使う）

### フレーム単位スコアの作り方

各モデルは動画の全フレームにスコアを出す（セグメント単位で推論する手法は、公式実装と同様にセグメント内の全フレームへ展開する）。

| モデル | 推論の単位 | フレームへの展開（公式と同じ） |
|---|---|---|
| VadCLIP | 16 フレームのスニペット | スコアを 16 フレームに繰り返す |
| AnyAnomaly | 24 フレームのセグメント | 繰り返したあと Gaussian 平滑化（sigma=15） |
| CoReVAD | 30 フレームのセグメント | Gaussian 平滑化 → 30 フレームに繰り返す → 位置重み付け |

ラベル長 L と動画フレーム数 n が異なる場合は、ラベル i の中心時刻 `(i+0.5)*n/L - 0.5` でフレームスコアを線形補間する（L = n のときはフレームスコアそのもの）。
ラベル長と動画フレーム数が 2% 以上ずれると `[warn]` を表示する。

## 公式実装との主な違い（詳細は各 models/*.py の冒頭）

- **VadCLIP**: 特徴は 10-crop ではなく中央クロップのみ。公式はテスト AUC が最良のチェックポイントを選ぶが、ここでは最終エポックを使う（リーク防止）。クラスは normal / 異常の 2 つ。バッチサイズの既定は 16（公式 64。異常動画数が少ないため）。`--vadclip-supervision frame` でフレームラベルの BCE を追加できる
- **AnyAnomaly**: 公式は評価時にテストデータで sigma と重みをグリッド探索するが、ここでは公式の既定値（α=0.6, β=0.3, γ=0.1, σ=15）を使う。`--aa-tune` を付けると学習 fold で探索する。min-max 正規化は単調変換で AUC に影響しないため省略。最後の端数セグメントも採点する。動画は decord ではなく OpenCV で読む
- **CoReVAD**: 動画は decord ではなく OpenCV で読む。応答のパースはコードブロックの有無に頑健にした。位置重み付け（動画中央を重視）は公式どおり有効（`--corevad-no-pos-weight` で無効化）

## ローカル（Mac）での動作確認

```bash
python scripts/make_dummy_data.py     # tests/dummy に 20 本のダミー動画・ラベル・splits を生成
bash scripts/smoke_test.sh            # 3 モデルを 1 fold だけ実行（--quick）
```

ローカルでは計算量とメモリのため、本番とは次の点を変えている。

| モデル | 本番 | ローカルでの確認 |
|---|---|---|
| VadCLIP | 本番と同じ | CPU、エポック数 2〜3 |
| CoReVAD | InternVL2-8B, max_new_tokens 1024 | **InternVL2-1B**（同じ chat API）、max_new_tokens 64、CPU |
| AnyAnomaly | Qwen2.5-VL-7B（vLLM） | **Qwen2.5-VL-3B**（公式スクリプトの既定）、transformers バックエンド、MPS(bf16) |

確認結果は [docs/local_check.md](docs/local_check.md) を参照。
