# Kaggle Execution Runbook

## Overview

This project already has two ready-to-run Kaggle notebooks:

1. [notebooks/kaggle_stage1_train.ipynb](../notebooks/kaggle_stage1_train.ipynb): Stage 1 training + decode sanity.
2. [notebooks/kaggle_stage2_smoke.ipynb](../notebooks/kaggle_stage2_smoke.ipynb): Stage 2 smoke flow on top of Stage 1 artifacts.

Use the notebooks directly instead of manually reconstructing the pipeline in ad-hoc cells.
Both notebooks support auto-bootstrap and auto-detection so the normal flow is: attach dataset, run all.

## Quick Start

1. Create a Kaggle notebook session with GPU enabled.
2. Attach dataset in Kaggle UI (Add data):
   - `balraj98/modelnet40-princeton-3d-object-dataset`
3. Run [notebooks/kaggle_stage1_train.ipynb](../notebooks/kaggle_stage1_train.ipynb) from top to bottom.
4. Confirm Stage 1 outputs exist under `/kaggle/working/checkpoints` and `/kaggle/working/logs`.
5. Run [notebooks/kaggle_stage2_smoke.ipynb](../notebooks/kaggle_stage2_smoke.ipynb) from top to bottom.

Notes:

- If project files are missing under `/kaggle/working`, setup cells auto-clone from `REPO_URL`.
- If `HARDWARE_PROFILE="auto"`, hardware config is selected automatically.

## Notebook 1: Stage 1

Open [notebooks/kaggle_stage1_train.ipynb](../notebooks/kaggle_stage1_train.ipynb) and set parameters in the first code cell:

- `HARDWARE_PROFILE`: `auto`, `p100`, or `t4x2`
- `RUN_MODE`: `first` or `resume`
- `REPO_URL`: GitHub repo URL used when auto-clone is needed
- `DATASET_SLUG`: keep the default unless you intentionally switch dataset
- `DATASET_ROOT_OVERRIDE`: leave empty normally; set this only if Kaggle mount folder name differs from slug tail
- `RESUME_CHECKPOINT_PATH`: optional custom resume checkpoint; leave empty for default

What this notebook does:

1. Resolves Kaggle dataset path under `/kaggle/input`.
2. Runs bootstrap/runtime guards (`scripts/kaggle_bootstrap.sh`).
3. Runs canonical Stage 1 trainer (`scripts/train_stage1.py`).
4. Runs decode sanity (`src/inference/generate_mesh.py`).
5. Lists generated artifacts.

Expected outputs after success:

- `/kaggle/working/checkpoints/latest_step.ckpt`
- `/kaggle/working/checkpoints/latest.ckpt`
- `/kaggle/working/checkpoints/best.ckpt`
- `/kaggle/working/logs/stage1_training_metrics.jsonl`
- `/kaggle/working/logs/stage1_decode_sanity_report.json`
- `/kaggle/working/logs/stage1_decode_sanity_report.md`

## Notebook 2: Stage 2 Smoke

Open [notebooks/kaggle_stage2_smoke.ipynb](../notebooks/kaggle_stage2_smoke.ipynb) and set parameters in the first code cell:

- `HARDWARE_PROFILE`: `auto`, `p100`, or `t4x2`
- `RUN_MODE`: `first` or `resume`
- `REPO_URL`: GitHub repo URL used when auto-clone is needed
- `DATASET_SLUG`: should match Stage 1 source dataset
- `DATASET_ROOT_OVERRIDE`: optional explicit `/kaggle/input/...` path
- `STAGE1_CHECKPOINT`: leave empty for auto-select (`best -> latest -> latest_step -> interrupt`)
- `RESUME_CHECKPOINT_PATH`: optional resume checkpoint override

What this notebook does:

1. Runs bootstrap/runtime checks.
2. Builds/validates Stage 2 latent manifests.
3. Runs canonical Stage 2 smoke trainer (`scripts/train_stage2.py`).
4. Checks checkpoint integrity and smoke reports.
5. Lists exportable artifacts.

Expected outputs after success:

- `/kaggle/working/logs/stage2_smoke_summary.json`
- `/kaggle/working/logs/stage2_checkpoint_integrity.json`
- `/kaggle/working/logs/stage2_smoke_metrics.jsonl`
- `/kaggle/working/cache/stage2_latents/stage2-latent-v1/manifests/latent_manifest_train.jsonl`
- `/kaggle/working/cache/stage2_latents/stage2-latent-v1/manifests/latent_manifest_test.jsonl`

## Resume Rules (Session Interrupted)

For both Stage 1 and Stage 2, recovery checkpoint precedence is:

1. `latest_step.ckpt`
2. `interrupt.ckpt`
3. `latest.ckpt`
4. `best.ckpt`

If resuming via notebooks:

1. Set `RUN_MODE = "resume"`.
2. Leave `RESUME_CHECKPOINT_PATH` empty to use default (`/kaggle/working/checkpoints/latest_step.ckpt`), or set explicit path.
3. Re-run notebook from top so environment/bootstrap are re-established.

## Common Issues

### Dataset root missing

Symptom:

- Runtime error saying dataset root is missing.

Fix:

1. Re-check Kaggle Add data has the dataset attached.
2. Re-run from the first parameter/setup cells.
3. In Stage 1 notebook, set `DATASET_ROOT_OVERRIDE` to the actual mounted folder under `/kaggle/input` if needed.

### Stage 2 cannot find Stage 1 checkpoint

Fix:

1. Ensure Stage 1 finished and wrote checkpoints in `/kaggle/working/checkpoints`.
2. Leave `STAGE1_CHECKPOINT` empty for auto-select or set an existing file (`best.ckpt` preferred).

### P100 incompatible with current PyTorch build

Symptom:

- Warning/error mentions GPU capability `sm_60` is not supported by current PyTorch build.

Fix:

1. In Kaggle notebook settings, switch accelerator from `P100` to `T4` (or `2xT4` if available).
2. Re-run notebook from the first parameter cell.
3. Keep `HARDWARE_PROFILE="auto"` unless you need manual override.

### Missing project source files

Symptom:

- Errors like `scripts/train_stage1.py` or `scripts/train_stage2.py` not found.

Fix:

1. Keep `REPO_URL` pointing to your GitHub repo.
2. Re-run the setup cell; notebook will auto-clone into `/kaggle/working/DoAn_backup`.

## Benchmarking Guidance

For reproducible comparisons:

1. Run the same Stage 1 then Stage 2 flow on P100.
2. Repeat on T4x2.
3. Record outcomes in [docs/BENCHMARKING.md](BENCHMARKING.md).
