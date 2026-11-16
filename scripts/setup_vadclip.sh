#!/usr/bin/env bash
# VadCLIP の環境構築と重みのダウンロード
#   bash scripts/setup_vadclip.sh            # .venvs/vadclip に作成
#   ENV=/path/to/venv bash scripts/setup_vadclip.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ENV=${ENV:-.venvs/vadclip}
source scripts/_mkenv.sh vadclip "$ENV"
# CLIP ViT-B/16（~/.cache/clip に保存）
"$ENV/bin/python" -c "from third_party.vadclip import clip; clip.load('ViT-B/16', 'cpu', jit=False); print('CLIP ViT-B/16 OK')"
echo "完了: source $ENV/bin/activate"
