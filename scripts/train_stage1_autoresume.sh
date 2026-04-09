#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
cd "${PROJECT_ROOT}"

DATASET_SLUG="${DATASET_SLUG:-balraj98/modelnet40-princeton-3d-object-dataset}"
DATASET_NAME="${DATASET_SLUG##*/}"
DATASET_ROOT="${DATASET_ROOT:-/kaggle/input/${DATASET_NAME}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/kaggle/working}"

TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train_stage1.yaml}"
DATA_CONFIG="${DATA_CONFIG:-configs/data_stage1.yaml}"
HARDWARE_CONFIG="${HARDWARE_CONFIG:-configs/hardware_t4x2.yaml}"
DEVICE="${DEVICE:-auto}"

echo "================================================================================"
echo "Stage 1 Shape SC-VAE AutoResume Runner"
echo "================================================================================"
echo "Dataset root: ${DATASET_ROOT}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Train config: ${TRAIN_CONFIG}"
echo "Hardware config: ${HARDWARE_CONFIG}"

echo "Running Stage 1 training with contract smoke + autoresume enabled..."
python scripts/train_stage1.py \
  --config "${TRAIN_CONFIG}" \
  --hardware "${HARDWARE_CONFIG}" \
  --data-config "${DATA_CONFIG}" \
  --dataset-root "${DATASET_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --autoresume \
  --contract-smoke \
  --device "${DEVICE}" \
  "$@"

echo "AutoResume run completed."
