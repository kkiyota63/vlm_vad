#!/usr/bin/env bash
# CoReVAD の環境構築と重みのダウンロード
#   bash scripts/setup_corevad.sh
#   VLM=OpenGVLab/InternVL2-8B FLASH_ATTN=1 bash scripts/setup_corevad.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ENV=${ENV:-.venvs/corevad}
VLM=${VLM:-OpenGVLab/InternVL2-8B}
source scripts/_mkenv.sh corevad "$ENV"
if [ "${FLASH_ATTN:-0}" = "1" ]; then
  "${PIP_INSTALL[@]}" flash-attn --no-build-isolation
fi
"$ENV/bin/python" - <<PY
from huggingface_hub import snapshot_download
import clip
snapshot_download("$VLM")
clip.load("ViT-B/16", device="cpu", jit=False)
print("weights OK: $VLM, CLIP ViT-B/16")
PY
echo "完了: source $ENV/bin/activate"
