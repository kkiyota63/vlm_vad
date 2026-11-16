#!/usr/bin/env bash
# AnyAnomaly の環境構築と重みのダウンロード
#   bash scripts/setup_anyanomaly.sh                 # transformers バックエンドのみ
#   WITH_VLLM=1 bash scripts/setup_anyanomaly.sh     # 公式と同じ vLLM バックエンドも入れる（CUDA）
#   VLM=Qwen/Qwen2.5-VL-3B-Instruct bash scripts/setup_anyanomaly.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ENV=${ENV:-.venvs/anyanomaly}
VLM=${VLM:-Qwen/Qwen2.5-VL-7B-Instruct}
source scripts/_mkenv.sh anyanomaly "$ENV"
if [ "${WITH_VLLM:-0}" = "1" ]; then
  "${PIP_INSTALL[@]}" -r envs/anyanomaly/requirements-vllm.txt
fi
"$ENV/bin/python" - <<PY
from huggingface_hub import snapshot_download
import clip
snapshot_download("$VLM")
clip.load("ViT-B/32", device="cpu", jit=False)
print("weights OK: $VLM, CLIP ViT-B/32")
PY
echo "完了: source $ENV/bin/activate"
