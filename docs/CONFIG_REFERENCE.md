# Configuration Reference Guide

## Hardware Configurations

All hardware profiles are located in `configs/`:

### P100 Single GPU (`hardware_p100.yaml`)

```yaml
hardware:
  profile_name: "p100_single"
  gpu_type: "nvidia_p100"
  vram_gb: 16
```

**Best for**:

- Single GPU Kaggle sessions with P100
- Memory-constrained training (16GB VRAM)
- Conservative batch sizes (8-16)

**Optimizations**:

- Gradient checkpointing enabled (reduces memory)
- Mixed precision (float16)
- No FlashAttention (P100 doesn't support modern CUDA ops)

---

### T4x2 Dual GPU (`hardware_t4x2.yaml`)

```yaml
hardware:
  profile_name: "t4x2_dual"
  gpu_type: "nvidia_t4"
  gpu_count: 2
  vram_gb_total: 32
  distributed_strategy: "ddp"
```

**Best for**:

- Dual GPU Kaggle sessions
- Larger batch sizes (16-32)
- Distributed training with DistributedDataParallel

**Optimizations**:

- DDP for multi-GPU training
- FlashAttention enabled (T4 supports newer CUDA)
- No gradient checkpointing (sufficient memory)

---

### TPU Future (`hardware_tpu_future.yaml`)

```yaml
hardware:
  status: "deferred"
  accelerator_type: "tpu"
```

**Status**: Placeholder only (v1)

- Not yet implemented
- Deferred to Phase 2 or v2

---

## Training Configurations

### Stage 1 Training (`configs/train_stage1.yaml`)

**Model**:

```yaml
model:
  name: "shape_sc_vae"
  encoder: { ... }
  decoder: { ... }
  latent_interface:
    latent_dim: 128
```

**Loss Function** (configurable):

```yaml
loss:
  reconstruction_loss: "mse" # Mean Squared Error
  reconstruction_weight: 1.0
  kl_loss: "kl_divergence"
  kl_weight: 0.001 # Start conservative; tune based on convergence
```

**Optimizer**:

```yaml
optimizer:
  name: "adam"
  lr: 0.001
  betas: [0.9, 0.999]
```

**Checkpointing** (auto-resume support):

```yaml
checkpointing:
  autoresume_enabled: true
  save_interval_steps: 500
  keep_last_n_checkpoints: 3
```

---

### Stage 2 Smoke Test (`configs/train_stage2.yaml`)

**Purpose**: Single-epoch validation of latent generator pipeline

**Config**:

```yaml
training:
  status: "smoke_only_v1"
  max_epochs: 1 # Smoke test: single epoch only
  smoke_test_mode: true
```

**Note**: Full Stage 2 implementation (UNet, iMF loss, hyperparameter sweeps) is deferred to Phase 2.

---

## Data Configurations

### Stage 1 Data (`configs/data_stage1.yaml`)

**Dataset Contract**:

```yaml
dataset:
  primary:
    name: "modelnet40"
    kaggle_slug: "balraj98/modelnet40-princeton-3d-object-dataset"
    format: "OFF"
    num_categories: 40
    num_samples: 12311

  fallback:
    kaggle_slug: "${MODELNET40_FALLBACK_SLUG}" # Configurable
    override_policy: "use_if_primary_unavailable"
```

**Paths** (Kaggle-aware):

```yaml
paths:
  dataset_root: "${DATASET_ROOT}" # /kaggle/input/<dataset-slug>/
  output_root: "${OUTPUT_ROOT}" # /kaggle/working/
  cache_dir: "${OUTPUT_ROOT}/cache"
  checkpoint_dir: "${OUTPUT_ROOT}/checkpoints"
```

**Preprocessing**:

```yaml
preprocessing:
  format_conversion: "off_to_voxel"
  target_resolution: 32 # 32^3 voxel grid
  normalize_meshes: true
  center_objects: true
```

**Data Loading**:

```yaml
loading:
  batch_size: "${BATCH_SIZE}" # Set by hardware profile
  num_workers: "${NUM_WORKERS}"
  pin_memory: true
  shuffle_train: true
```

---

### Stage 2 Data (`configs/data_stage2.yaml`)

**Smoke Test Data**:

```yaml
smoke_config:
  num_train_samples: 100 # Minimal set
  num_val_samples: 20
  max_epochs_smoke: 1 # Single epoch
```

**Source**:

```yaml
latent_loading:
  source_checkpoint: "${STAGE1_CHECKPOINT_PATH}" # From Stage 1
  extract_latents: true
  latent_dim: 128
```

---

## Environment Variables

All configs support environment variable expansion using `${VAR_NAME}` syntax:

### Required for Kaggle

```bash
export DATASET_ROOT="/kaggle/input/modelnet40-princeton-3d-object-dataset"
export OUTPUT_ROOT="/kaggle/working"
```

### Hardware-Dependent

```bash
export BATCH_SIZE="8"           # Set by hardware profile
export NUM_WORKERS="4"          # Set by hardware profile
export GRAD_ACC_STEPS="2"       # Gradient accumulation (if needed)
```

### For Resume/Stage 2

```bash
export RESUME_CHECKPOINT_PATH=""  # Empty for fresh training
export STAGE1_CHECKPOINT_PATH="/kaggle/working/checkpoints/latest.ckpt"
```

### Custom Values

```bash
export MODELNET40_FALLBACK_SLUG="alternate-modelnet40-mirror"  # If needed
```

---

## Configuration Priority

1. **Environment Variables**: Override all (highest priority)
2. **Command-line Arguments**: Override config files (if provided)
3. **Config Files**: Base configuration
4. **Hardcoded Defaults**: Fallback (lowest priority)

Example:

```bash
# ENV overrides config
export BATCH_SIZE=16
python scripts/train_stage1.py \\
  --config configs/train_stage1.yaml \\
  --batch-size 32  # CLI overrides both env and config
```

---

## Common Configuration Adjustments

### For Faster Training

```yaml
# Reduce precision
dtype: "float16"

# Increase batch size (if GPU memory allows)
batch_size: 16

# Disable checkpointing overhead
checkpoint_interval_steps: 1000
```

### For Limited GPU Memory

```yaml
# Enable gradient checkpointing
enable_gradient_checkpointing: true

# Reduce batch size
batch_size: 4

# Enable gradient accumulation
gradient_accumulation_steps: 4
```

### For Extended Training

```yaml
# Increase epochs
max_epochs: 20

# More frequent validation
val_interval_steps: 100

# Keep more checkpoints
keep_best_n: 5
keep_last_n: 5
```

---

## Validation Checklist

Before running training, verify:

- [ ] Config files exist in `configs/`
- [ ] Hardware profile matches your GPU (P100 or T4x2)
- [ ] Environment variables set (especially `DATASET_ROOT`, `OUTPUT_ROOT`)
- [ ] Dataset attached in Kaggle (if using Kaggle)
- [ ] Output directories writable
- [ ] No conflicting parameter definitions (env, CLI, config)

---

## Support

If configs fail to load:

1. Check YAML syntax (valid indentation, colons)
2. Verify environment variables are set
3. Check file paths exist
4. Review config loaders in `scripts/config_loader.py`
