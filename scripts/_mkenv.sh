#!/usr/bin/env bash
# 共通: 仮想環境を作成して requirements を入れる
#   source scripts/_mkenv.sh <model名> <venvパス>
set -euo pipefail
NAME=$1
ENV=$2
PY=${PYTHON:-python3.11}
if command -v uv >/dev/null 2>&1; then
  uv venv "$ENV" --python 3.11
  VIRTUAL_ENV="$ENV" uv pip install -r "envs/$NAME/requirements.txt"
  PIP_INSTALL=(env VIRTUAL_ENV="$ENV" uv pip install)
else
  command -v "$PY" >/dev/null 2>&1 || PY=python3
  "$PY" -m venv "$ENV"
  "$ENV/bin/pip" install -U pip
  "$ENV/bin/pip" install -r "envs/$NAME/requirements.txt"
  PIP_INSTALL=("$ENV/bin/pip" install)
fi
