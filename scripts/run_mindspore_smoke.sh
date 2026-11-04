#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/smoke}"

mkdir -p "$OUTPUT_DIR"
echo "MindSpore smoke output: $OUTPUT_DIR"

"$PYTHON_BIN" main_mindspore.py \
  --config exps/smoke/ldeprompt_synthetic.json \
  --device cpu \
  --device-target cpu \
  --ms-mode pynative \
  --exp smoke \
  --output-dir "$OUTPUT_DIR" \
  "$@"
