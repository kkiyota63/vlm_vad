# タスク: 映像異常検知(VAD)用VLMのサーベイ・3モデルの環境構築・評価パイプライン実装

## 背景
- 作業ディレクトリ: /Users/kiyotakoki/Documents/TA/VLM_VAD（git未初期化。現状のファイルは knn_cross_validation.py のみ）
- 目的: 複数の映像異常検知VLMを同じ条件で比較するためのプロジェクトを作ること
- ローカルPC: Apple M2 Pro / メモリ16GB / CUDAなし。ここでは「動作するかの確認」だけを行う
- 本番: GitHub に push し、GPUサーバーで pull して実行する
- GPUサーバー上のデータ（knn_cross_validation.py と同じ構成）:
  - ラベル: /home/jovyan/work/toyota2026/data/label/*.npy（異常動画のみ存在する。フレームごとの0/1ラベル。ファイルがない動画は正常として扱い、全フレーム0とする）
  - 分割: /home/jovyan/work/toyota2026/data/splits/splits_*.txt（各行が動画パス。stem + ".npy" をキーとして使う）
  - 生動画のディレクトリは未確定。設定値（VIDEO_DIR）として外に出し、拡張子 .mp4/.avi などを stem で探索できるようにする

## 手順
### 1. サーベイ
VLM/MLLMベースの映像異常検知手法を調査し、比較表を docs/survey.md にまとめる。
- 候補の例: LAVAD, Holmes-VAD, Holmes-VAU, VadCLIP, VERA, AnomalyRuler, HiProbe-VAD, EventVAD, Vad-R1, 汎用VLM（Qwen2.5-VL / Qwen3-VL / InternVL3）のゼロショット利用 など。最新の手法もWeb検索で確認すること
- 表の列: 会議・年 / 学習の有無（zero-shot・weakly・fine-tune） / ベースモデル / 公開重みの有無 / 必要VRAM / フレーム単位のスコアを出せるか / ライセンス

### 2. 3モデルを選ぶ
選定の基準:
- (a) 公開コードと公開重みがあり、すぐに使えること
- (b) フレーム単位またはセグメント単位の異常スコアを出せること（ROC-AUC/PR-AUC の計算に必須）
- (c) 単一GPU（目安24〜80GB）で動くこと
- (d) 学習方式がばらけること（例: zero-shot VLM、VAD特化でfine-tune済みのMLLM、CLIP系の教師あり軽量ヘッド）

選定理由を survey.md に書く。

### 3. 実装
構成の例:
```
common/data.py      # split読込, limit_data, ラベル読込, 動画→フレーム/セグメントのサンプリング
common/eval.py      # evaluation(), 誤検出CSVの出力
models/<name>.py    # 共通インタフェース: fit(train_videos, labels)（不要なら何もしない）/ score(video) -> フレームごとのスコア
run_cv.py           # --model <name> で k分割交差検証を実行
envs/<name>/        # モデルごとの requirements.txt または environment.yml（依存が衝突するので環境は分ける）
scripts/setup_<name>.sh   # 環境作成と重みダウンロード
```

データロードと評価は knn_cross_validation.py と完全に同じにすること:
- fold i をテストに、fold (i+1)%K を閾値決め（cal）に、残りを学習に使う
- limit_data のシード規則も踏襲する（学習は SEED+fold、テストは SEED+100+fold、cal は random.Random(SEED+fold).sample）
- 閾値は cal スコアの 100*(1-CONTAMINATION) パーセンタイルとする
- evaluation() で accuracy / ROC-AUC / PR-AUC / F1 を計算する
- fold ごとに wrong_fold{i}.csv を出力し、最後に全foldの mean ± std を表示する
- ハイパーパラメータは knn_cross_validation.py と同様に main() 冒頭の定数、または argparse で指定できるようにする

ラベルの粒度とVLMのサンプリング粒度が違う点への対処:
- 長さ L のラベル列に、サンプリングした時刻 t を index = floor(t / duration * L) で対応付ける
- 評価はラベル粒度で行う。セグメントのスコアはラベル長に補間・展開してから評価する
- この方針を README に明記する

その他:
- VLMの推論結果（セグメントごとのスコアや生成テキスト）は cache/<model>/<video>.npy などにキャッシュし、再実行時は推論をスキップする
- デバイスは cuda → mps → cpu の順に自動で選ぶ。cuml/cupy には依存しない（numpy/sklearn で実装する）

### 4. ローカルでの動作確認
- scripts/make_dummy_data.py を作る。数秒のダミー動画数本と、label/・splits_*.txt を tests/dummy/ に生成するスクリプト
- 各モデルについて --dummy --max-videos 2 のような軽量モードで run_cv.py を最後まで通す
- 重いモデルは小さい代替（同系列の最小サイズのモデルなど）で通してよい。ただし、本番用の設定と何が違うかを README に明記する
- ダウンロードが合計で約15GBを超えそうな場合は、ダウンロード前に確認を取ること

### 5. GitHub への push
- .gitignore を作る（cache/, outputs/, *.npy, 動画ファイル, 重み, tests/dummy の生成物）
- git init → commit したら、gh で private リポジトリ（名前: VLM_VAD）を作って push する。リポジトリを作る前に、名前と公開範囲をユーザーに確認すること
- README.md には、GPUサーバーでの手順（clone → setup_<name>.sh → 実行コマンド → 結果の場所）と、未確定の設定値（VIDEO_DIR など）を書く

## 制約
- ファイル名はすべて英語（ASCII）にする
- 回答とコードコメントは日本語にする
- 動作確認ができなかった部分（例: CUDA専用のカーネルでMacでは動かないもの）は、「未確認」と明記して報告する。確認できたと偽らないこと
