#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
cd "${PROJECT_ROOT}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/kaggle/working}"
DATASET_ROOT="${DATASET_ROOT:-${OUTPUT_ROOT}}"
STAGE1_CHECKPOINT_PATH="${STAGE1_CHECKPOINT_PATH:-${OUTPUT_ROOT}/checkpoints/latest.ckpt}"
RUN_ID="${RUN_ID:-stage2-smoke-$(date -u +%Y%m%dT%H%M%SZ)}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train_stage2.yaml}"
DATA_CONFIG="${DATA_CONFIG:-configs/data_stage2.yaml}"
HARDWARE_CONFIG="${HARDWARE_CONFIG:-configs/hardware_p100.yaml}"

echo "================================================================================"
echo "Stage 2 Smoke Training AutoResume"
echo "================================================================================"
echo "Output root: ${OUTPUT_ROOT}"
echo "Stage 1 checkpoint: ${STAGE1_CHECKPOINT_PATH}"
echo "Run id: ${RUN_ID}"

echo "Launching Stage 2 smoke trainer with explicit resume fallback..."
ARGS=(
  --config "${TRAIN_CONFIG}"
  --hardware "${HARDWARE_CONFIG}"
  --data-config "${DATA_CONFIG}"
  --output-root "${OUTPUT_ROOT}"
  --stage1-checkpoint "${STAGE1_CHECKPOINT_PATH}"
  --run-id "${RUN_ID}"
  --device "${DEVICE:-auto}"
)

if [[ -n "${RESUME_CHECKPOINT_PATH:-}" ]]; then
  ARGS+=(--resume-from "${RESUME_CHECKPOINT_PATH}")
fi

python scripts/train_stage2.py "${ARGS[@]}" "$@"

echo "Stage 2 smoke training completed."
