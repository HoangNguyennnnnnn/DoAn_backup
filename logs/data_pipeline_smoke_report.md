# Data Pipeline Smoke Report

- status: failed
- adapter_load_seconds: None
- first_batch_seconds: None

## Checks
- error: Dataset root does not exist. Attach the Kaggle dataset and set paths.dataset_root to /kaggle/input/<dataset-slug>. Missing path: \kaggle\input\modelnet40-princeton-3d-object-dataset
- recovery: Attach Kaggle dataset and verify paths.dataset_root and paths.output_root environment values.

## Cache Behavior
- unavailable (smoke did not complete baseline stage)

## Diagnostics
- None

## Reproducible Command

python scripts/data_pipeline_smoke.py --data-config configs/data_stage1.yaml --sample-limit 16 --refresh-sample-limit 8 --batch-size 4 --seed 42 --schema-version ovoxel-v1 --refresh-overwrite
