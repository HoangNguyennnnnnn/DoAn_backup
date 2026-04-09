# Checkpointing and Resume Strategy

## Overview

The project implements robust checkpointing to handle Kaggle session interruptions and enable training resumption without loss of progress.

---

## Checkpoint Format

Each checkpoint file (`.ckpt`) contains:

```python
{
    "epoch": int,                    # Current epoch number
    "model_state_dict": {...},       # Model weights
    "optimizer_state_dict": {...},   # Optimizer momentum/state
    "scheduler_state_dict": {...},   # Learning rate scheduler state
    "metrics": {                     # Training metrics
        "train_loss": float,
        "val_loss": float,
        "train_kl": float,
        "val_kl": float,
    }
}
```

---

## Checkpoint Files

Four checkpoint roles are maintained:

### `latest_step.ckpt`

- **Purpose**: Primary recovery checkpoint for interrupted training
- **Used for**: Resuming training after interruption
- **Policy**: Updated every `save_interval_steps` (default: 500 steps)
- **When to keep**: Always keep (required for resume)

### `interrupt.ckpt`

- **Purpose**: Snapshot captured on interruption
- **Used for**: Manual recovery after a KeyboardInterrupt or Kaggle stop
- **Policy**: Written when the trainer exits via interruption
- **When to keep**: Always keep (required for recovery)

### `latest.ckpt`

- **Purpose**: Most recent checkpoint in current training run
- **Used for**: Resume compatibility fallback
- **Policy**: Mirrored from the latest step payload
- **When to keep**: Always keep (required for resume compatibility)

### `best.ckpt`

- **Purpose**: Best validation checkpoint (lowest validation loss)
- **Used for**: Final model for inference/evaluation
- **Policy**: Overwritten only when validation loss improves
- **When to keep**: Always keep (required for evaluation)

### `epoch_*.ckpt`

- **Purpose**: Periodic snapshots at each epoch
- **Used for**: Examining training progression, rollback
- **Policy**: Saved every epoch; oldest deleted per `keep_last_n_checkpoints`
- **Default**: Keep 3 most recent epoch checkpoints

---

## Automatic Resume (Kaggle Sessions)

When a Kaggle session is interrupted:

1. **Session State**: Kaggle preserves `/kaggle/working/` directory
2. **Checkpoint Preservation**: `checkpoints/latest_step.ckpt` is the primary recovery target, with `interrupt.ckpt`, `latest.ckpt`, and `best.ckpt` as fallbacks
3. **Resume on Next Run**: Call training script with `--resume-from` or set `RESUME_CHECKPOINT_PATH`

### Resume Command

```bash
# Automatic detection of recovery checkpoint
export RESUME_CHECKPOINT_PATH="/kaggle/working/checkpoints/latest_step.ckpt"
python scripts/train_stage1.py \\
  --config configs/train_stage1.yaml \\
  --hardware configs/hardware_p100.yaml
```

### Resume with Environment Variable

```python
import os
os.environ["RESUME_CHECKPOINT_PATH"] = "/kaggle/working/checkpoints/latest_step.ckpt"
# Then run training script
```

---

## Resume Behavior

When `RESUME_CHECKPOINT_PATH` is set:

1. **Load Model Weights**: Restore model state
2. **Load Optimizer State**: Restore momentum, moving averages
3. **Load Scheduler State**: Restore learning rate schedule position
4. **Load Epoch Counter**: Resume from saved epoch number
5. **Restore Metrics**: Access previous metrics for logging/comparison

### Strict Loading

By default, `strict_model_loading: false` allows:

- Partial weight loading (if model architecture changed)
- Graceful skipping of mismatched layers
- Recovery from minor model modifications

If you need strict loading (explicit error on mismatch):

```yaml
checkpointing:
  strict_model_loading: true
```

---

## Manual Checkpointing

### Save Checkpoint

```python
from src.utils.checkpoint_utils import CheckpointManager

CheckpointManager.save_checkpoint(
    epoch=epoch,
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    metrics={"train_loss": 0.123, "val_loss": 0.456},
    checkpoint_dir="checkpoints",
    is_best=(val_loss < best_val_loss),
)
```

### Load Checkpoint

```python
metadata = CheckpointManager.load_checkpoint(
    checkpoint_path="checkpoints/best.ckpt",
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    device="cuda",
)

start_epoch = metadata["epoch"]
previous_metrics = metadata["metrics"]
```

---

## Checkpoint Cleanup

### Automatic Cleanup

Configuration in `train_stage1.yaml`:

```yaml
checkpointing:
  keep_last_n_checkpoints: 3 # Keep 3 most recent epoch checkpoints
```

Older epoch checkpoints are automatically deleted to save disk space.

### Manual Cleanup

```bash
# Remove old epoch checkpoints, keep recovery and best checkpoints
find checkpoints/ -name "epoch_*.ckpt" -not -newer checkpoints/best.ckpt -delete
```

---

## Storage Considerations

### Typical Checkpoint Size

- **Model weights**: ~200-500 MB (depending on architecture)
- **Optimizer state**: ~200-500 MB (same as model for Adam)
- **Metadata**: < 1 MB
- **Total per checkpoint**: ~400-1000 MB

### Kaggle Workspace Limits

- **Kaggle `/kaggle/working/` limit**: Typically 20-50 GB
- **Recommendation**: Keep `keep_last_n_checkpoints: 3` to stay within limits
- **Monitor**: Use `du -sh checkpoints/` to check disk usage

---

## Multi-GPU Considerations (T4x2)

### Distributed Checkpointing

For DDP (DistributedDataParallel):

```python
# Only save from rank 0 to avoid race conditions
if rank == 0:
    CheckpointManager.save_checkpoint(...)
```

### Load in Distributed Context

```python
# DDP automatically syncs after loading
checkpoint = CheckpointManager.load_checkpoint(
    checkpoint_path=checkpoint_path,
    model=model,
    device=device,
)
# All ranks now have the same model state
```

---

## Stage 1 → Stage 2 Handoff

### Extract Latents for Stage 2

1. **Save Stage 1 Checkpoint**: `checkpoints/best.ckpt` first, then `latest.ckpt` as a fallback for export/evaluation compatibility
2. **Set Environment Variable**: `export STAGE1_CHECKPOINT_PATH="/kaggle/working/checkpoints/best.ckpt"`
3. **Stage 2 Script Loads It**: Extracts latents using Stage 1 encoder
4. **Stage 2 Smoke Test**: Validates latent dimensions and I/O

```bash
# Stage 1 complete
python scripts/train_stage1.py ...  # Outputs checkpoints/best.ckpt

# Stage 2 uses Stage 1 checkpoint
export STAGE1_CHECKPOINT_PATH="/kaggle/working/checkpoints/best.ckpt"
python scripts/train_stage2.py \\
  --stage1-checkpoint $STAGE1_CHECKPOINT_PATH
```

---

## Artifact Export and Recovery

### Path Conventions

- Checkpoints: `/kaggle/working/checkpoints/`
- Logs and reports: `/kaggle/working/logs/`
- Run metadata: `/kaggle/working/runs/<run_id>/metadata/`
- Export bundles: `/kaggle/working/exports/<bundle_id>/` or `.tar.gz`

### Export Script

Use the canonical export wrapper to package checkpoints, logs, configs, reports, and run metadata:

```bash
bash scripts/export_artifacts.sh
```

By default, the export bundle includes both checkpoint roles:

- Recovery checkpoints: `latest_step.ckpt`, `interrupt.ckpt`, `latest.ckpt`, `best.ckpt`
- Export/evaluation checkpoints: `best.ckpt`, `latest.ckpt`

### Restore Checklist

Before resuming training after session expiration, verify:

1. `/kaggle/working/checkpoints/latest_step.ckpt` exists.
2. `/kaggle/working/logs/` contains the latest smoke or training summary.
3. `/kaggle/working/runs/<run_id>/metadata/run_metadata.json` exists.
4. The dataset remains attached and readable under `/kaggle/input/<dataset-slug>/`.
5. The notebook or shell command uses the correct resume target for the stage.

### Publication Gate

Checkpoint publication is conditional.

- Code/config/logs may be shared publicly after sanitization.
- Weights may be published only after license-chain and provenance review passes.
- If license terms are ambiguous, publish code and metrics only.

### Resume Precedence

Use the same recovery precedence for all restore operations unless a trainer explicitly overrides it:

1. `latest_step.ckpt`
2. `interrupt.ckpt`
3. `latest.ckpt`
4. `best.ckpt`

### Export/Evaluation Precedence

For export and evaluation, prefer:

1. `best.ckpt`
2. `latest.ckpt`

---

## Troubleshooting

### "Checkpoint not found"

```bash
# Check if checkpoint exists
ls -lah checkpoints/
```

**Solution**: Run Stage 1 training first to generate checkpoint.

---

### "Cuda out of memory when loading checkpoint"

Reduce batch size in hardware config:

```yaml
training:
  batch_size: 4 # Reduce from 8
```

---

### "Optimizer state mismatch during resume"

If resuming with different optimizer:

```python
# Don't load optimizer state
metadata = CheckpointManager.load_checkpoint(
    ...,
    optimizer=None,  # Skip optimizer state
)
```

---

### "Training metrics don't resume correctly"

Ensure logs directory exists:

```bash
mkdir -p /kaggle/working/logs
```

Log file paths are saved in checkpoint; verify they're accessible.

---

## Best Practices

1. **Always Enable Auto-Resume**: Set `autoresume_enabled: true`
2. **Regular Saves**: Use `save_interval_steps: 500` (every 500 steps)
3. **Keep Best**: Always maintain `best.ckpt` for evaluation/export, and `latest_step.ckpt` for interrupted recovery
4. **Monitor Disk Space**: Check `du -sh checkpoints/` periodically
5. **Test Resume**: Periodically test resuming from checkpoint (not just loading)
6. **Version Compatibility**: If changing model architecture, use `strict_model_loading: false`

---

## Example: Complete Resume Flow

```python
import torch
from src.utils.checkpoint_utils import CheckpointManager

# Setup
model = Model(...)
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# Check for resume
checkpoint_path = "/kaggle/working/checkpoints/latest.ckpt"
start_epoch = 0

if os.path.exists(checkpoint_path):
    print(f"Resuming from {checkpoint_path}")
    metadata = CheckpointManager.load_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device="cuda",
    )
    start_epoch = metadata["epoch"] + 1
    print(f"Resuming from epoch {start_epoch}")

# Training loop
for epoch in range(start_epoch, MAX_EPOCHS):
    # ... training code ...

    # Save checkpoint
    CheckpointManager.save_checkpoint(
        epoch=epoch,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        checkpoint_dir="/kaggle/working/checkpoints",
        is_best=(val_loss < best_val_loss),
    )
```

---

## See Also

- [KAGGLE_RUNBOOK.md](KAGGLE_RUNBOOK.md) — Step-by-step Kaggle execution
- [src/utils/checkpoint_utils.py](../src/utils/checkpoint_utils.py) — Checkpoint implementation
