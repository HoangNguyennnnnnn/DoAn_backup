#!/usr/bin/env bash

set -Eeuo pipefail

echo "================================================================================"
echo "Kaggle Bootstrap: Environment Setup + Runtime Guards"
echo "================================================================================"

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
cd "${PROJECT_ROOT}"

echo "Project root: ${PROJECT_ROOT}"

DATASET_SLUG="${DATASET_SLUG:-balraj98/modelnet40-princeton-3d-object-dataset}"
DATASET_NAME="${DATASET_SLUG##*/}"
DATASET_ROOT="${DATASET_ROOT:-/kaggle/input/${DATASET_NAME}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/kaggle/working}"

HARDWARE_CONFIG="${HARDWARE_CONFIG:-configs/hardware_p100.yaml}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train_stage1.yaml}"
DATA_CONFIG="${DATA_CONFIG:-configs/data_stage1.yaml}"
MIN_WORKING_GB="${MIN_WORKING_GB:-10}"

echo "Dataset slug: ${DATASET_SLUG}"
echo "Dataset root: ${DATASET_ROOT}"
echo "Output root: ${OUTPUT_ROOT}"

echo "--------------------------------------------------------------------------------"
echo "Step 1/3: Install dependencies (deterministic order, pinned versions)"
echo "--------------------------------------------------------------------------------"

if [[ "${BOOTSTRAP_SKIP_INSTALL:-0}" == "1" ]]; then
  echo "BOOTSTRAP_SKIP_INSTALL=1 detected. Skipping package installation."
else
  python -m pip install --quiet --no-input --upgrade \
    "pip==24.3.1" \
    "setuptools==75.6.0" \
    "wheel==0.45.1"

  python -m pip install --quiet --no-input \
    "numpy==1.26.4" \
    "scipy==1.13.1" \
    "pyyaml==6.0.2" \
    "tqdm==4.66.5" \
    "tensorboard==2.17.1" \
    "trimesh==4.5.2"
fi

echo "--------------------------------------------------------------------------------"
echo "Step 2/3: Runtime guards (fail-fast checks + metadata capture)"
echo "--------------------------------------------------------------------------------"

python -m src.utils.runtime_guards \
  --dataset-slug "${DATASET_SLUG}" \
  --dataset-root "${DATASET_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --hardware-config "${HARDWARE_CONFIG}" \
  --train-config "${TRAIN_CONFIG}" \
  --data-config "${DATA_CONFIG}" \
  --min-working-gb "${MIN_WORKING_GB}"

echo "--------------------------------------------------------------------------------"
echo "Step 3/3: Kaggle workflow guidance"
echo "--------------------------------------------------------------------------------"

echo "Data workflow:"
echo "- Data is already hosted on Kaggle; use Attach Dataset in notebook UI."
echo "- Do not manually upload raw datasets into notebook storage."

echo "Artifact publication options after training:"
echo "- Download artifacts from /kaggle/working (checkpoints, logs, metrics)."
echo "- Or package artifacts as a new Kaggle Dataset version for sharing."

echo
echo "Next commands (copy/paste):"
echo "1) Stage 1 data prep/check"
echo "python -m src.utils.runtime_guards --check-only --dataset-slug \"${DATASET_SLUG}\" --dataset-root \"${DATASET_ROOT}\" --output-root \"${OUTPUT_ROOT}\" --hardware-config \"${HARDWARE_CONFIG}\" --train-config \"${TRAIN_CONFIG}\" --data-config \"${DATA_CONFIG}\""
echo
echo "2) Stage 1 training start"
echo "python scripts/train_stage1.py --config \"${TRAIN_CONFIG}\" --hardware \"${HARDWARE_CONFIG}\" --dataset-root \"${DATASET_ROOT}\" --output-root \"${OUTPUT_ROOT}\""
echo
echo "3) Resume from latest checkpoint"
echo "python scripts/train_stage1.py --config \"${TRAIN_CONFIG}\" --hardware \"${HARDWARE_CONFIG}\" --dataset-root \"${DATASET_ROOT}\" --output-root \"${OUTPUT_ROOT}\" --resume-from \"${OUTPUT_ROOT}/checkpoints/latest.ckpt\""

echo
echo "Bootstrap completed successfully. Safe to rerun after interruption."
