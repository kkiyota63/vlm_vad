#!/usr/bin/env bash
# ダミーデータで 3 モデルを 1 fold だけ通す動作確認（小型 VLM で代替）
# 既定デバイス: vadclip/corevad は cpu、anyanomaly は auto（Mac では mps）。DEVICE=cuda などで上書き可
#   bash scripts/smoke_test.sh [vadclip|corevad|anyanomaly ...]
set -euo pipefail
cd "$(dirname "$0")/.."
MODELS=${*:-vadclip corevad anyanomaly}
[ -d tests/dummy ] || .venvs/vadclip/bin/python scripts/make_dummy_data.py
COMMON=(--data-root tests/dummy --quick --folds 0 --out-dir tests/outputs --cache-dir tests/cache)
for m in $MODELS; do
  echo "===== $m ====="
  case $m in
    vadclip)    .venvs/vadclip/bin/python run_cv.py --model vadclip "${COMMON[@]}" --device "${DEVICE:-cpu}" --vadclip-epochs 2 ;;
    corevad)    .venvs/corevad/bin/python run_cv.py --model corevad "${COMMON[@]}" --device "${DEVICE:-cpu}" --corevad-vlm OpenGVLab/InternVL2-1B --corevad-max-new-tokens 64 ;;
    anyanomaly) .venvs/anyanomaly/bin/python run_cv.py --model anyanomaly "${COMMON[@]}" --device "${DEVICE:-auto}" --aa-vlm Qwen/Qwen2.5-VL-3B-Instruct --aa-keywords "red fire" ;;
  esac
done
