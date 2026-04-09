#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
cd "${PROJECT_ROOT}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/kaggle/working}"
EXPORT_ROOT="${EXPORT_ROOT:-${OUTPUT_ROOT}/exports}"
RUN_ID="${RUN_ID:-auto}"
INCLUDE_WEIGHTS="${INCLUDE_WEIGHTS:-1}"
PUBLIC_RELEASE="${PUBLIC_RELEASE:-0}"
LICENSE_CHAIN_REVIEW="${LICENSE_CHAIN_REVIEW:-pending}"
STAGE1_CHECKPOINT_DIR="${STAGE1_CHECKPOINT_DIR:-${OUTPUT_ROOT}/checkpoints}"
STAGE2_CHECKPOINT_DIR="${STAGE2_CHECKPOINT_DIR:-${OUTPUT_ROOT}/checkpoints}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"
RUNS_DIR="${RUNS_DIR:-${OUTPUT_ROOT}/runs}"
CONFIG_DIR="${CONFIG_DIR:-${PROJECT_ROOT}/configs}"

TIMESTAMP_UTC="$(date -u +%Y%m%dT%H%M%SZ)"
BUNDLE_ID="${RUN_ID}-${TIMESTAMP_UTC}"
DEST_DIR="${EXPORT_ROOT}/${BUNDLE_ID}"
MANIFEST_PATH="${DEST_DIR}/export_manifest.json"
ARCHIVE_PATH="${EXPORT_ROOT}/${BUNDLE_ID}.tar.gz"

mkdir -p "${DEST_DIR}"
mkdir -p "${EXPORT_ROOT}"

if [[ "${PUBLIC_RELEASE}" == "1" ]]; then
  if [[ "${LICENSE_CHAIN_REVIEW}" != "approved" ]]; then
    echo "ERROR: PUBLIC_RELEASE=1 requires LICENSE_CHAIN_REVIEW=approved before weights can be packaged."
    echo "ERROR: Publish code/config/logs only until license-chain review passes."
    exit 2
  fi
fi

copy_if_exists() {
  local source_path="$1"
  local relative_path="$2"

  if [[ -e "${source_path}" ]]; then
    mkdir -p "${DEST_DIR}/${relative_path%/*}"
    cp -a "${source_path}" "${DEST_DIR}/${relative_path}"
  fi
}

write_manifest() {
  cat > "${MANIFEST_PATH}" <<EOF
{
  "bundle_id": "${BUNDLE_ID}",
  "timestamp_utc": "${TIMESTAMP_UTC}",
  "project_root": "${PROJECT_ROOT}",
  "output_root": "${OUTPUT_ROOT}",
  "include_weights": ${INCLUDE_WEIGHTS},
  "public_release": ${PUBLIC_RELEASE},
  "license_chain_review": "${LICENSE_CHAIN_REVIEW}",
  "stage1_checkpoint_dir": "${STAGE1_CHECKPOINT_DIR}",
  "stage2_checkpoint_dir": "${STAGE2_CHECKPOINT_DIR}",
  "log_dir": "${LOG_DIR}",
  "runs_dir": "${RUNS_DIR}",
  "config_dir": "${CONFIG_DIR}",
  "recovery_precedence": ["latest_step.ckpt", "interrupt.ckpt", "latest.ckpt", "best.ckpt"],
  "export_precedence": ["best.ckpt", "latest.ckpt"],
  "notes": "Bundle contains both recovery checkpoints and evaluation checkpoints by default."
}
EOF
}

copy_configs() {
  if [[ -d "${CONFIG_DIR}" ]]; then
    mkdir -p "${DEST_DIR}/configs"
    cp -a "${CONFIG_DIR}/." "${DEST_DIR}/configs/"
  fi
}

copy_logs() {
  if [[ -d "${LOG_DIR}" ]]; then
    mkdir -p "${DEST_DIR}/logs"
    cp -a "${LOG_DIR}/." "${DEST_DIR}/logs/"
  fi
}

copy_runs() {
  if [[ -d "${RUNS_DIR}" ]]; then
    mkdir -p "${DEST_DIR}/runs"
    cp -a "${RUNS_DIR}/." "${DEST_DIR}/runs/"
  fi
}

copy_checkpoints() {
  mkdir -p "${DEST_DIR}/checkpoints"

  local stage1_files=("latest_step.ckpt" "interrupt.ckpt" "latest.ckpt" "best.ckpt")
  local stage2_files=("latest_step.ckpt" "interrupt.ckpt" "latest.ckpt" "best.ckpt" "smoke_latest.ckpt")

  if [[ "${INCLUDE_WEIGHTS}" == "1" ]]; then
    for file_name in "${stage1_files[@]}"; do
      copy_if_exists "${STAGE1_CHECKPOINT_DIR}/${file_name}" "checkpoints/stage1/${file_name}"
    done
    for file_name in "${stage2_files[@]}"; do
      copy_if_exists "${STAGE2_CHECKPOINT_DIR}/${file_name}" "checkpoints/stage2/${file_name}"
    done
  else
    copy_if_exists "${STAGE1_CHECKPOINT_DIR}/latest_step.ckpt" "checkpoints/stage1/latest_step.ckpt"
    copy_if_exists "${STAGE1_CHECKPOINT_DIR}/interrupt.ckpt" "checkpoints/stage1/interrupt.ckpt"
    copy_if_exists "${STAGE2_CHECKPOINT_DIR}/latest_step.ckpt" "checkpoints/stage2/latest_step.ckpt"
    copy_if_exists "${STAGE2_CHECKPOINT_DIR}/interrupt.ckpt" "checkpoints/stage2/interrupt.ckpt"
  fi
}

copy_reports() {
  if [[ -d "${LOG_DIR}" ]]; then
    mkdir -p "${DEST_DIR}/reports"
    find "${LOG_DIR}" -maxdepth 1 -type f \( \
      -name '*summary.json' -o \
      -name '*integrity.json' -o \
      -name '*decode_sanity_report.json' -o \
      -name '*decode_sanity_report.md' -o \
      -name '*smoke_metrics.jsonl' -o \
      -name '*training_metrics.jsonl' \
    \) -exec cp -a {} "${DEST_DIR}/reports/" \;
  fi
}

copy_manifest_metadata() {
  if [[ -d "${RUNS_DIR}" ]]; then
    while IFS= read -r -d '' file_path; do
      relative_path="${file_path#${RUNS_DIR}/}"
      mkdir -p "${DEST_DIR}/run_metadata/$(dirname "${relative_path}")"
      cp -a "${file_path}" "${DEST_DIR}/run_metadata/${relative_path}"
    done < <(find "${RUNS_DIR}" -path '*/metadata/*' -type f -print0)
  fi
}

copy_configs
copy_logs
copy_runs
copy_checkpoints
copy_reports
copy_manifest_metadata
write_manifest

if command -v tar >/dev/null 2>&1; then
  tar -czf "${ARCHIVE_PATH}" -C "${EXPORT_ROOT}" "${BUNDLE_ID}"
  echo "Archive written: ${ARCHIVE_PATH}"
else
  echo "tar not available; bundle directory written at: ${DEST_DIR}"
fi

echo "Export complete."
echo "Bundle directory: ${DEST_DIR}"
echo "Manifest: ${MANIFEST_PATH}"
echo "Recovery checkpoints included by default: latest_step, interrupt, latest, best"
echo "Export/evaluation checkpoints included by default: best, latest"
echo "Publication gate: set PUBLIC_RELEASE=1 only after LICENSE_CHAIN_REVIEW=approved"
