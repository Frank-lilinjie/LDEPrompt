#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
DEVICE_ID="${DEVICE_ID:-0}"
CONFIG="${CONFIG:-exps/ascend/cifar/ldeprompt.json}"
EXP="${EXP:-ascend910b}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/${EXP}}"
MS_MODE="${MS_MODE:-pynative}"
BACKBONE_CHECKPOINT="${BACKBONE_CHECKPOINT:-}"
DATASET_ROOT="${DATASET_ROOT:-}"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  for candidate in \
    /usr/local/Ascend/ascend-toolkit/set_env.sh \
    /usr/local/Ascend/ascend-toolkit/latest/set_env.sh; do
    if [[ -f "$candidate" ]]; then
      # shellcheck disable=SC1090
      source "$candidate"
      break
    fi
  done
elif [[ -f "${ASCEND_HOME_PATH}/set_env.sh" ]]; then
  # shellcheck disable=SC1090
  source "${ASCEND_HOME_PATH}/set_env.sh"
fi

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  echo "ASCEND_HOME_PATH is not set. Source the CANN set_env.sh first." >&2
  echo "Example: source /usr/local/Ascend/ascend-toolkit/set_env.sh" >&2
  exit 2
fi

if ! command -v npu-smi >/dev/null 2>&1; then
  echo "npu-smi was not found. Check the 910B driver and CANN installation." >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

echo "Ascend environment: $ASCEND_HOME_PATH"
echo "Device: npu:${DEVICE_ID} | mode: ${MS_MODE}"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_DIR"
npu-smi info

ARGS=(
  main_mindspore.py
  --config "$CONFIG"
  --device "npu:${DEVICE_ID}"
  --device-target ascend
  --ms-mode "$MS_MODE"
  --exp "$EXP"
  --output-dir "$OUTPUT_DIR"
)

if [[ -n "$BACKBONE_CHECKPOINT" ]]; then
  if [[ ! -f "$BACKBONE_CHECKPOINT" ]]; then
    echo "Backbone checkpoint not found: $BACKBONE_CHECKPOINT" >&2
    exit 2
  fi
  ARGS+=(--backbone-checkpoint "$BACKBONE_CHECKPOINT")
fi

if [[ -n "$DATASET_ROOT" ]]; then
  ARGS+=(--dataset-root "$DATASET_ROOT")
fi

"$PYTHON_BIN" "${ARGS[@]}" "$@"
