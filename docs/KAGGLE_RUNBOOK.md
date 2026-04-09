# Kaggle Execution Runbook

## Overview

This guide provides step-by-step instructions for running the 3D Shape Generation project on Kaggle. All commands assume execution within a Kaggle notebook environment.

## Fast Path

If you want the minimum sequence that should work on a Kaggle GPU session, run this:

1. Create a Kaggle notebook with GPU enabled.
2. Attach `balraj98/modelnet40-princeton-3d-object-dataset` through `Add data`.
3. Clone or upload the repository into `/kaggle/working`.
4. Run [notebooks/kaggle_stage1_train.ipynb](../notebooks/kaggle_stage1_train.ipynb) from top to bottom.
5. Confirm Stage 1 checkpoints and logs were written under `/kaggle/working`.
6. Run [notebooks/kaggle_stage2_smoke.ipynb](../notebooks/kaggle_stage2_smoke.ipynb) from top to bottom.
7. If the notebook session dies, resume from `latest_step.ckpt` first, then `interrupt.ckpt`, then `latest.ckpt`, then `best.ckpt`.
8. For comparisons, repeat the same flow on P100 and T4x2 and record the results in [docs/BENCHMARKING.md](BENCHMARKING.md).

---

## Part 1: Initial Setup (One-Time)

### Step 1.1: Create Kaggle Notebook

1. Go to https://www.kaggle.com/code/
2. Click `+ New Notebook`
3. Select GPU session (P100 or T4x2 recommended)

### Step 1.2: Attach Datasets

1. Click `Add data` in the notebook editor
2. Search for and add: `balraj98/modelnet40-princeton-3d-object-dataset`
3. Confirm the dataset appears in `/kaggle/input/modelnet40-princeton-3d-object-dataset/`

### Step 1.3: Clone Repository

In the first notebook cell:

```python
!cd /kaggle/working && git clone https://github.com/yourusername/3d-shape-kaggle.git
%cd /kaggle/working/3d-shape-kaggle
```

Or upload as ZIP and extract:

```python
!unzip -q /kaggle/input/yourfile.zip -d /kaggle/working/
%cd /kaggle/working/extracted-project
```

### Step 1.4: Verify Structure

```python
import os
import yaml

# Check directory structure
print("📁 Repo structure:")
for item in os.listdir("."):
    if os.path.isdir(item):
        print(f"  📂 {item}/")
    else:
        print(f"  📄 {item}")

# Verify configs
print("\n✅ Config files:")
with open("configs/hardware_p100.yaml") as f:
    config = yaml.safe_load(f)
    print(f"  Hardware: {config['hardware']['profile_name']}")
```

---

## Part 2: Environment Setup

### Step 2.1: Set Environment Variables

```python
import os

# Set Kaggle paths (these are standard on Kaggle)
os.environ["DATASET_ROOT"] = "/kaggle/input/modelnet40-princeton-3d-object-dataset"
os.environ["OUTPUT_ROOT"] = "/kaggle/working"
os.environ["BATCH_SIZE"] = "8"
os.environ["NUM_WORKERS"] = "4"
os.environ["GRAD_ACC_STEPS"] = "2"
os.environ["RESUME_CHECKPOINT_PATH"] = ""  # Empty for fresh training

print(f"✅ DATASET_ROOT: {os.environ['DATASET_ROOT']}")
print(f"✅ OUTPUT_ROOT: {os.environ['OUTPUT_ROOT']}")
```

### Step 2.2: Install Dependencies

```python
# Install required packages
!pip install -q \
    torch torchvision \
    tensorboard \
    pyyaml \
    numpy \
    tqdm \
    trimesh \
    numpy-stl

print("✅ Dependencies installed")
```

### Step 2.3: Verify Dataset Access

```python
# Verify ModelNet40 dataset is accessible
dataset_root = os.environ["DATASET_ROOT"]
off_files = []
for root, dirs, files in os.walk(dataset_root):
    off_files.extend([f for f in files if f.endswith(".off")])
    if len(off_files) > 5:
        break

print(f"✅ Found {len(off_files)} OFF files in dataset")
print(f"  Sample: {off_files[:3]}")
```

---

## Part 3: Select Hardware Profile

### Step 3.1: Detect Session Hardware

```python
import torch

# Detect GPU
device_count = torch.cuda.device_count()
device_name = torch.cuda.get_device_name(0) if device_count > 0 else "CPU"
vram = torch.cuda.get_device_properties(0).total_memory / (1024**3) if device_count > 0 else 0

print(f"🖥️ Detected Hardware:")
print(f"  GPU Count: {device_count}")
print(f"  Primary GPU: {device_name}")
print(f"  VRAM: {vram:.1f} GB")

# Auto-select config
if "P100" in device_name:
    os.environ["HARDWARE_CONFIG"] = "configs/hardware_p100.yaml"
    print("  → Selected: configs/hardware_p100.yaml")
elif "T4" in device_name and device_count >= 2:
    os.environ["HARDWARE_CONFIG"] = "configs/hardware_t4x2.yaml"
    print("  → Selected: configs/hardware_t4x2.yaml")
elif "T4" in device_name:
    os.environ["HARDWARE_CONFIG"] = "configs/hardware_p100.yaml"  # Fallback similar config
    print("  → Selected: configs/hardware_p100.yaml (T4 single, fallback profile)")
else:
    os.environ["HARDWARE_CONFIG"] = "configs/hardware_p100.yaml"
    print("  → Selected: configs/hardware_p100.yaml (default fallback)")
```

### Step 3.2: Load Hardware Config

```python
import yaml

hw_config_path = os.environ["HARDWARE_CONFIG"]
with open(hw_config_path) as f:
    hw_config = yaml.safe_load(f)

print(f"✅ Hardware Config Loaded:")
print(f"  Profile: {hw_config['hardware']['profile_name']}")
print(f"  Batch Size: {hw_config['training']['batch_size']}")
print(f"  Mixed Precision: {hw_config['training']['mixed_precision']}")
print(f"  Num Workers: {hw_config['training']['num_workers']}")
```

---

## Part 4: Stage 1 - Shape SC-VAE Training

### Step 4.1: Load Training Config

```python
with open("configs/train_stage1.yaml") as f:
    train_config = yaml.safe_load(f)

print(f"✅ Stage 1 Training Config:")
print(f"  Model: {train_config['model']['name']}")
print(f"  Epochs: {train_config['training']['max_epochs']}")
print(f"  KL Weight: {train_config['loss']['kl_weight']}")
print(f"  Learning Rate: {train_config['optimizer']['lr']}")
```

### Step 4.2: Run Training

```python
# Option A: Run training script (when scripts/ are implemented)
# !python scripts/train_stage1.py \
#   --config configs/train_stage1.yaml \
#   --hardware configs/hardware_p100.yaml \
#   --dataset-root $DATASET_ROOT \
#   --output-root $OUTPUT_ROOT

# Option B: For now, placeholder shows expected flow
print("🚀 Stage 1 Training Started...")
print("  Loading data from:", os.environ["DATASET_ROOT"])
print("  Outputs to:", os.environ["OUTPUT_ROOT"])
print("  Checkpoints to: /kaggle/working/checkpoints")

# Training would proceed here (implementation in src/)
# Model trains for specified epochs with checkpoint saves
# Tensorboard logs written to OUTPUT_ROOT/logs/
```

### Step 4.3: Monitor Training

```python
# Display tensorboard logs (if running locally, use tensorboard --logdir)
# For Kaggle, check logs in:
logs_dir = f"{os.environ['OUTPUT_ROOT']}/logs"
checkpoints_dir = f"{os.environ['OUTPUT_ROOT']}/checkpoints"

print("📊 Expected output locations:")
print(f"  Tensorboard: {logs_dir}")
print(f"  Checkpoints: {checkpoints_dir}")
print(f"  Cache: {os.environ['OUTPUT_ROOT']}/cache/")
```

### Step 4.4: Evaluate Stage 1

```python
# After training completes, optional evaluation
# !python src/inference/generate_mesh.py \
#   --checkpoint-dir {checkpoints_dir} \
#   --output-root {os.environ['OUTPUT_ROOT']} \
#   --num-samples 4

print("✅ Stage 1 training complete. Checkpoint saved to /kaggle/working/checkpoints/")
```

---

## Part 5: Stage 2 - Latent Generator Smoke Test

### Step 5.1: Load Stage 1 Checkpoint

```python
# Verify Stage 1 checkpoint exists. Prefer best.ckpt for export/evaluation.
stage1_best = f"{os.environ['OUTPUT_ROOT']}/checkpoints/best.ckpt"
stage1_latest = f"{os.environ['OUTPUT_ROOT']}/checkpoints/latest.ckpt"
if os.path.exists(stage1_best):
    print(f"✅ Stage 1 checkpoint found: {stage1_best}")
    os.environ["STAGE1_CHECKPOINT_PATH"] = stage1_best
elif os.path.exists(stage1_latest):
    print(f"⚠️  best.ckpt missing; falling back to: {stage1_latest}")
    os.environ["STAGE1_CHECKPOINT_PATH"] = stage1_latest
else:
    print("❌ Stage 1 checkpoint not found. Please complete Stage 1 first.")
```

### Step 5.2: Load Stage 2 Config

```python
with open("configs/train_stage2.yaml") as f:
    stage2_config = yaml.safe_load(f)

print(f"✅ Stage 2 Smoke Config:")
print(f"  Mode: {stage2_config['training']['smoke_test_mode']}")
print(f"  Status: {stage2_config['training']['status']}")
print(f"  Max Epochs: {stage2_config['training']['max_epochs']}")
```

### Step 5.3: Run Stage 2 Smoke Test

```python
# Optional: Run on same GPU or new session
print("🚀 Stage 2 Smoke Test Started...")
print("  Loading latents from:", os.environ["STAGE1_CHECKPOINT_PATH"])
print("  Testing latent generator pipeline...")

# Smoke test would run here (single epoch validation)
# Validates:
# - Stage 1 latent extraction
# - Latent generator backbone compatibility
# - End-to-end I/O pipeline
```

### Step 5.4: Validate Results

```python
print("✅ Stage 2 smoke test complete.")
print("  Expected output:")
print("    - smoke_latest.ckpt")
print("    - Stage 2 validation logs")
print("  Status: Ready for post-v1 Stage 2 expansion")
```

---

## Part 6: Data, Metrics & Checkpoints

### Step 6.1: Inspect Checkpoints

```python
import glob

checkpoints = sorted(glob.glob(f"{os.environ['OUTPUT_ROOT']}/checkpoints/*.ckpt"))
print(f"📦 Saved Checkpoints ({len(checkpoints)}):")
for ckpt in checkpoints[-5:]:  # Show last 5
    size_mb = os.path.getsize(ckpt) / (1024**2)
    print(f"  {os.path.basename(ckpt)}: {size_mb:.1f} MB")
```

### Step 6.2: View Training Metrics

```python
# Tensorboard metrics (saved to logs/)
logs_files = glob.glob(f"{os.environ['OUTPUT_ROOT']}/logs/**", recursive=True)
print(f"📊 Training Logs ({len(logs_files)} files):")
for log_file in logs_files[:10]:
    print(f"  {log_file}")
```

### Step 6.3: Resume Training (Session Interrupt)

If your Kaggle session is interrupted:

```python
# Detect the first available recovery checkpoint in precedence order.
recovery_candidates = [
    f"{os.environ['OUTPUT_ROOT']}/checkpoints/latest_step.ckpt",
    f"{os.environ['OUTPUT_ROOT']}/checkpoints/interrupt.ckpt",
    f"{os.environ['OUTPUT_ROOT']}/checkpoints/latest.ckpt",
    f"{os.environ['OUTPUT_ROOT']}/checkpoints/best.ckpt",
]
resume_ckpt = next((path for path in recovery_candidates if os.path.exists(path)), None)
if resume_ckpt:
    print(f"🔄 Resuming from checkpoint: {resume_ckpt}")
    os.environ["RESUME_CHECKPOINT_PATH"] = resume_ckpt

    # Re-run training script with resume flag
    # python scripts/train_stage1.py --config ... --resume-from $RESUME_CHECKPOINT_PATH
else:
    print("No checkpoint found. Starting fresh training.")
```

---

## Part 7: Post-Run Checklist

### Successful v1 Run Should Have:

```python
required_files = [
    "checkpoints/latest_step.ckpt",
    "checkpoints/latest.ckpt",
    "checkpoints/best.ckpt",
    "logs/stage1_training_metrics.jsonl",
    "logs/stage1_decode_sanity_report.json",
]

missing = [f for f in required_files if not os.path.exists(os.path.join(os.environ['OUTPUT_ROOT'], f))]

if missing:
    print(f"⚠️  Missing files (training may be incomplete):\n  {chr(10).join(missing)}")
else:
    print("✅ All expected files present. v1 baseline complete!")
```

### Verification Steps:

```python
print("🔍 Final Verification:")
print(f"1. Dataset access: {os.path.exists(os.environ['DATASET_ROOT'])}")
print(f"2. Latest recovery checkpoint: {os.path.exists(f\"{os.environ['OUTPUT_ROOT']}/checkpoints/latest_step.ckpt\")}")
print(f"3. Best checkpoint: {os.path.exists(f\"{os.environ['OUTPUT_ROOT']}/checkpoints/best.ckpt\")}")
print(f"4. Logs directory: {os.path.exists(f\"{os.environ['OUTPUT_ROOT']}/logs\")}")
print(f"5. Output root writable: {os.access(os.environ['OUTPUT_ROOT'], os.W_OK)}")

print("\n✅ Kaggle execution runbook complete!")
```

---

## Part 8: Restore and Resume After Session Expiration

### Stage 1 Restore Steps

1. Re-open the Kaggle notebook and reattach the dataset if needed.
2. Confirm the latest recovery checkpoint exists at `/kaggle/working/checkpoints/latest_step.ckpt`.
3. Use the Stage 1 notebook with `RUN_MODE = "resume"` or call the trainer directly:

```bash
python scripts/train_stage1.py \
    --config configs/train_stage1.yaml \
    --hardware configs/hardware_p100.yaml \
    --data-config configs/data_stage1.yaml \
    --dataset-root /kaggle/input/modelnet40-princeton-3d-object-dataset \
    --output-root /kaggle/working \
    --resume-from /kaggle/working/checkpoints/latest_step.ckpt
```

4. If `latest_step.ckpt` is unavailable, fall back to `interrupt.ckpt`, then `latest.ckpt`, then `best.ckpt`.

### Stage 2 Restore Steps

1. Re-open the Kaggle notebook and confirm latent manifests are still present under `/kaggle/working/cache/stage2_latents/stage2-latent-v1/manifests/`.
2. Confirm recovery checkpoint order before resuming: `latest_step.ckpt`, `interrupt.ckpt`, `latest.ckpt`, `best.ckpt`.
3. Run the Stage 2 smoke notebook in resume mode, or invoke the trainer directly:

```bash
python scripts/train_stage2.py \
    --config configs/train_stage2.yaml \
    --hardware configs/hardware_p100.yaml \
    --data-config configs/data_stage2.yaml \
    --output-root /kaggle/working \
    --stage1-checkpoint /kaggle/working/checkpoints/best.ckpt \
    --resume-from /kaggle/working/checkpoints/latest_step.ckpt
```

4. For export/evaluation, prefer `best.ckpt` first, then `latest.ckpt`.

### Restore Validation Checklist

Before resuming, verify:

- [ ] `/kaggle/working/checkpoints/latest_step.ckpt` exists or a documented fallback is chosen.
- [ ] `/kaggle/working/logs/` contains the latest smoke or training summary.
- [ ] `/kaggle/working/runs/<run_id>/metadata/run_metadata.json` exists.
- [ ] The dataset is still attached under `/kaggle/input/<dataset-slug>/`.
- [ ] The notebook uses the correct run mode and checkpoint target for the stage.
- [ ] For Stage 2, latent manifests are present and readable before resuming.

### Export and Publication Gate

Use `scripts/export_artifacts.sh` to package checkpoints, logs, configs, and run metadata.

- Recovery bundles should include both `latest_step.ckpt` and `best.ckpt` by default.
- Public weight publication is blocked until license-chain review is approved.
- If provenance is ambiguous, publish code/config/logs and defer weights.

---

## Part 9: Next Steps

1. **Verify Outputs**: Check `/kaggle/working/` for checkpoints, logs, and cache
2. **Public Release**: If planning to share, follow [DATA_LICENSE.md](../DATA_LICENSE.md) compliance
3. **Post-v1 Backlog**: Keep broader Stage 2 expansion, TPU work, and public-release automation outside the v1 lock

---

**Note**: This runbook assumes the current v1 Stage 1 and Stage 2 smoke contracts are in place. Any work beyond the current scope lock belongs in the post-v1 backlog.
