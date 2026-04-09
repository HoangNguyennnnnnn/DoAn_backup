# Stage 1 Decode Sanity Report

- device: cpu
- checkpoint_dir: outputs/checkpoints
- pass_count: 0
- fail_count: 1
- all_passed: False

## Checkpoint Results

- None (local runtime execution blocked before checkpoint evaluation)

## Reconstruction Trend

- metrics_path: outputs/logs/stage1_training_metrics.jsonl
- count: 0
- first: None
- last: None
- min: None
- trend: unavailable

## Note

- Local Windows runtime was non-authoritative due environment-level Numpy instability.
- Run src/inference/generate_mesh.py in Kaggle/Linux runtime to validate best/latest checkpoint decode behavior and export mesh artifacts.
