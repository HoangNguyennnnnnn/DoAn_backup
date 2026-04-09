No mesh samples were exported in this local run.
Reason: local Windows runtime instability (Numpy toolchain warning and early process exit).
Action: run `python -m src.inference.generate_mesh --config configs/train_stage1.yaml --data-config configs/data_stage1.yaml --output-root /kaggle/working --device auto --num-samples 4` inside Kaggle/Linux after checkpoints are produced.
