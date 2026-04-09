# 3D Shape Generation on Kaggle - v1 Baseline

## Quick Start

### Prerequisites

- Kaggle notebook session with GPU support (P100 or T4x2 recommended)
- Internet access for dataset attachment via Kaggle UI

### What To Run

If you only want the shortest working path, run the project in this order on Kaggle:

1. Attach the ModelNet40 dataset in the Kaggle notebook UI.
2. Open [notebooks/kaggle_stage1_train.ipynb](notebooks/kaggle_stage1_train.ipynb) and run all cells.
3. Verify Stage 1 outputs under `/kaggle/working/checkpoints` and `/kaggle/working/logs`.
4. Open [notebooks/kaggle_stage2_smoke.ipynb](notebooks/kaggle_stage2_smoke.ipynb) and run all cells.
5. If the session is interrupted, resume with `latest_step.ckpt` first, then `interrupt.ckpt`, `latest.ckpt`, and `best.ckpt`.
6. For a reproducible comparison, repeat the same notebook flow on P100 and T4x2 and record results in [docs/BENCHMARKING.md](docs/BENCHMARKING.md).
7. Export the final artifacts with `scripts/export_artifacts.sh` once the run is stable.

### Setup Flow

1. **Attach Datasets in Kaggle Notebook UI**
   - Click `Add data` and search for `balraj98/modelnet40-princeton-3d-object-dataset`
   - Confirm the dataset appears in `/kaggle/input/modelnet40-princeton-3d-object-dataset/`

2. **Clone/Upload Repository**
   - Copy this project to `/kaggle/working/` or attach as code/zip
   - Ensure config files are accessible at `configs/`

3. **Select Hardware Configuration**
   - Edit `configs/hardware_p100.yaml` for single P100
   - Edit `configs/hardware_t4x2.yaml` for T4x2 dual GPU
   - Pass config path to training script

4. **Run Kaggle Bootstrap (recommended first command)**

   ```bash
   bash scripts/kaggle_bootstrap.sh
   ```

5. **Run Stage 1 Training**

   ```bash
   python scripts/train_stage1.py --config configs/train_stage1.yaml --hardware configs/hardware_p100.yaml
   ```

6. **Run Stage 2 Smoke Test**
   ```bash
   python scripts/train_stage2.py --config configs/train_stage2.yaml --hardware configs/hardware_p100.yaml --stage1-checkpoint /kaggle/working/checkpoints/best.ckpt
   ```

### Recommended Run Matrix

- **Fresh Kaggle smoke run**: Stage 1 notebook, then Stage 2 notebook.
- **Interrupted session recovery**: reopen the notebook and resume from `latest_step.ckpt` first.
- **Hardware comparison**: run the same notebook flow on P100 and T4x2, using the same dataset attachment and the same smoke scope.
- **Artifact handoff**: after the run stabilizes, export checkpoints and logs with `scripts/export_artifacts.sh`.

---

## Architecture Overview

### v1 Mandatory Path

```
ModelNet40 Dataset
        ↓
OFF → O-Voxel (32³ voxel representation)
        ↓
Shape SC-VAE Encoder/Decoder
        ↓
SLAT-compatible Latent Interface (128-dim)
        ↓
[Checkpoint: latest_step.ckpt / interrupt.ckpt / latest.ckpt / best.ckpt]
```

### Stage 2 (Smoke-Level Validation Only)

```
Stage 1 Latents
        ↓
Latent Generator Placeholder (Backbone TBD)
        ↓
End-to-End I/O Validation
        ↓
[Single Epoch Smoke Test]
```

---

## Directory Structure

```
.
├── configs/                      # Configuration files
│   ├── hardware_p100.yaml        # P100 single GPU profile
│   ├── hardware_t4x2.yaml        # T4x2 dual GPU profile
│   ├── hardware_tpu_future.yaml  # TPU future branch marker (deferred)
│   ├── data_stage1.yaml          # Data loading for Stage 1
│   ├── data_stage2.yaml          # Data loading for Stage 2 (smoke)
│   ├── train_stage1.yaml         # Training config for Shape SC-VAE
│   └── train_stage2.yaml         # Training config for Latent Generator (smoke)
├── scripts/                      # Executable scripts
│   ├── build_latent_dataset.py    # Stage 2 latent cache builder from Stage 1 outputs
│   ├── config_loader.py          # YAML config loading utility
│   ├── data_pipeline_smoke.py    # Data pipeline smoke validation
│   ├── export_artifacts.sh       # Kaggle artifact export and publication gate
│   ├── kaggle_bootstrap.sh       # Kaggle bootstrap + runtime guard runner
│   ├── train_stage1_autoresume.sh # Stage 1 Kaggle autoresume wrapper
│   ├── train_stage1.py           # Stage 1 training entry point
│   ├── train_stage2_autoresume.sh # Stage 2 Kaggle autoresume wrapper
│   └── train_stage2.py           # Stage 2 smoke test entry point
├── src/                          # Source code modules
│   ├── models/                   # Model definitions
│   │   ├── encoder.py            # Shape SC-VAE encoder
│   │   └── decoder.py            # Shape SC-VAE decoder
│   ├── data/                     # Data loaders and preprocessing
│   │   ├── dataset_adapter.py    # Kaggle dataset adapter and split logic
│   │   ├── latent_dataset_builder.py # Stage 2 latent dataset builder
│   │   ├── modelnet40_loader.py  # ModelNet40 dataset and OFF loader
│   │   └── voxel_converter.py    # OFF → O-Voxel conversion
│   ├── inference/                # Decode and mesh export utilities
│   │   └── generate_mesh.py      # Stage 1 decode sanity runner
│   └── utils/                    # Utility functions
│       ├── checkpoint_utils.py   # Save/load checkpoint logic
│       ├── logging_utils.py      # Tensorboard logging setup
│       └── runtime_guards.py     # Kaggle runtime checks + metadata capture
├── notebooks/                    # Kaggle-style notebook templates
│   ├── kaggle_stage1_train.ipynb  # Stage 1 entry point notebook
│   └── kaggle_stage2_smoke.ipynb  # Stage 2 smoke test notebook
├── checkpoints/                  # Saved model checkpoints
│   ├── latest_step.ckpt          # Latest step checkpoint for recovery
│   ├── interrupt.ckpt            # Interrupted run checkpoint
│   ├── latest.ckpt               # Latest checkpoint for resume
│   └── best.ckpt                 # Best validation checkpoint
├── outputs/                      # Training outputs
│   ├── cache/                    # Preprocessed data cache
│   ├── logs/                     # Tensorboard logs
│   └── artifacts/                # Generated sample meshes, metrics
├── logs/                         # Training and run logs
│   └── .gitkeep
├── docs/                         # Documentation
│   ├── KAGGLE_RUNBOOK.md         # Step-by-step Kaggle execution guide
│   ├── CONFIG_REFERENCE.md       # Configuration parameters explanation
│   ├── DATA_PIPELINE.md          # Data loading and preprocessing flow
│   └── CHECKPOINTING.md          # Resume and checkpoint strategy
├── PROJECT_SCOPE.md              # v1 scope lock and architecture decisions
├── DATA_LICENSE.md               # Dataset provenance and publishability policy
└── README.md                     # This file
```

---

## Configuration Reference

### Hardware Profiles

All hardware configs live under `configs/`:

- **hardware_p100.yaml**: Batch size 8, no FlashAttention, 16GB optimized
- **hardware_t4x2.yaml**: Batch size 16, DDP enabled, T4-specific optimizations
- **hardware_tpu_future.yaml**: Placeholder for future TPU support (deferred)

### Training Configs

- **train_stage1.yaml**: Shape SC-VAE training with configurable KL weight, optimizer, checkpointing
- **train_stage2.yaml**: Smoke-level latent generator (full Stage 2 is deferred)

### Data Configs

- **data_stage1.yaml**: ModelNet40 primary + fallback mirror, O-Voxel preprocessing
- **data_stage2.yaml**: Latent loading from Stage 1 checkpoint, smoke-only mode

---

## Environment Variables & Kaggle Integration

The project uses environment-driven path resolution for Kaggle reproducibility:

```bash
# Set in Kaggle notebook or runtime
export DATASET_ROOT="/kaggle/input/modelnet40-princeton-3d-object-dataset"
export OUTPUT_ROOT="/kaggle/working"
export BATCH_SIZE="8"
export NUM_WORKERS="4"
export GRAD_ACC_STEPS="2"
export RESUME_CHECKPOINT_PATH=""  # Leave empty to train from scratch; recovery falls back to latest_step -> interrupt -> latest -> best
export STAGE1_CHECKPOINT_PATH="/kaggle/working/checkpoints/best.ckpt"  # For Stage 2 export/evaluation

# Or set via command-line:
python scripts/train_stage1.py \
  --config configs/train_stage1.yaml \
  --hardware configs/hardware_p100.yaml \
  --dataset-root /kaggle/input/modelnet40-princeton-3d-object-dataset \
  --output-root /kaggle/working
```

**CRITICAL**: Never use absolute local file paths. All paths must be either relative or environment-driven.

---

## Kaggle Attach-Only Data Contract

This project strictly uses Kaggle's dataset attachment mechanism:

1. **Input Path**: `/kaggle/input/<dataset-slug>/`
   - Consume datasets here only
   - Example: `/kaggle/input/modelnet40-princeton-3d-object-dataset/`

2. **Output Path**: `/kaggle/working/`
   - Write checkpoints, logs, cache, artifacts here only
   - Persists across session interruptions

3. **No Manual Upload**
   - Do not manually upload files to local `/tmp/` or other local paths
   - Ensures reproducibility across Kaggle runs

---

## v1 Sign-Off Criteria

Stage 1 is complete when:

- ✓ ModelNet40 dataset loads and preprocessing to O-Voxel succeeds
- ✓ Shape SC-VAE training loop runs end-to-end on Kaggle
- ✓ Checkpointing and auto-resume work across session interruptions
- ✓ Inference decode produces valid mesh outputs
- ✓ SLAT-compatible latent interface is functional

Stage 2 smoke test is complete when:

- ✓ Stage 1 latent extraction succeeds
- ✓ Latent generator backbone accepts extracted latents
- ✓ End-to-end latent I/O pipeline is validated (no shape mismatches)

---

## Deferred Work (Not v1)

- ✗ Full material/PBR branch from Paper 1
- ✗ Large-scale DiT training
- ✗ UNet + iMF full implementation and hyperparameter sweeps from Paper 2
- ✗ TPU runtime support (future branch marker only)
- ✗ ShapeNetPart integration (extension only after v1 stability gate)
- ✗ Production-grade checkpoint licensing & public release workflow

See [PROJECT_SCOPE.md](PROJECT_SCOPE.md) for detailed deferred/configurable breakdown.

---

## Data Licensing & Publishability

### Datasets

- **ModelNet40**: Academic research convenience; inherits Princeton CAD copyright
- **ShapeNetPart**: Deferred from v1; license chain check required before public checkpoint release

### Public Release Rules

- ✓ Code, configs, scripts: Always public (with dataset citations)
- ✓ Training logs, metrics: Public (after sanitization)
- ? Processed caches: Cautious (only if derived outputs don't violate upstream terms)
- ? Checkpoints: Conditional on explicit license review (defer if ambiguous)
- ✗ Raw dataset files: Not allowed by default

See [DATA_LICENSE.md](DATA_LICENSE.md) for full compliance checklist and citation requirements.

---

## Documentation Guides

- [KAGGLE_RUNBOOK.md](docs/KAGGLE_RUNBOOK.md) — Step-by-step Kaggle execution
- [CONFIG_REFERENCE.md](docs/CONFIG_REFERENCE.md) — All configuration parameters
- [DATA_PIPELINE.md](docs/DATA_PIPELINE.md) — Data loading and preprocessing details
- [CHECKPOINTING.md](docs/CHECKPOINTING.md) — Resume and checkpoint strategies

---

## Support & Troubleshooting

### Common Issues

**Dataset not found in `/kaggle/input/`**

- Ensure you attached the dataset in the Kaggle notebook UI before running training
- Verify the exact slug: `balraj98/modelnet40-princeton-3d-object-dataset`

**Out of memory during training**

- Check your hardware profile (P100 vs T4x2)
- Reduce batch size in hardware YAML
- Enable gradient checkpointing

**Checkpoint resume fails**

- Verify `RESUME_CHECKPOINT_PATH` points to valid `.ckpt` file
- Check that checkpoint was saved from same model architecture
- If unsure, clear path and train from scratch

**Session interrupt**

- Kaggle auto-saves to `/kaggle/working/`
- Set `autoresume_enabled: true` in hardware config
- Training will resume from the first available recovery checkpoint on next run (`latest_step.ckpt` -> `interrupt.ckpt` -> `latest.ckpt` -> `best.ckpt`)

---

## Benchmarking

Use the same smoke-scoped notebook flows and artifact locations when comparing `P100` and `T4x2` on Kaggle.

### Canonical Benchmark Entry Points

- [notebooks/kaggle_stage1_train.ipynb](notebooks/kaggle_stage1_train.ipynb) for Stage 1
- [notebooks/kaggle_stage2_smoke.ipynb](notebooks/kaggle_stage2_smoke.ipynb) for Stage 2 smoke
- [scripts/train_stage1_autoresume.sh](scripts/train_stage1_autoresume.sh) for Stage 1 resume runs
- [scripts/train_stage2_autoresume.sh](scripts/train_stage2_autoresume.sh) for Stage 2 resume runs

### Comparison Rules

- Keep dataset attachment and paths identical: `/kaggle/input/...` for input, `/kaggle/working` for output.
- Use the same smoke scope on both hardware profiles.
- Use `best.ckpt` for export/evaluation, but `latest_step.ckpt` for recovery.
- Keep resume precedence explicit: `latest_step.ckpt -> interrupt.ckpt -> latest.ckpt -> best.ckpt`.

### Profile Guidance

**P100**

- Use `configs/hardware_p100.yaml`.
- Recommended settings: batch size 8, 4 workers, mixed precision on, gradient checkpointing on.
- Best when you want the most conservative baseline and predictable memory headroom.

**T4x2**

- Use `configs/hardware_t4x2.yaml`.
- Recommended settings: batch size 16, 8 workers, mixed precision on, DDP enabled.
- Best when you want higher throughput and more room for iteration.

### What to Compare

- Wall-clock runtime
- Throughput proxy metrics
- Memory behavior
- Checkpoint/restart success behavior
- Interruption resilience

### Evidence Paths

- [logs/stage1_training_metrics.jsonl](logs/stage1_training_metrics.jsonl) when Stage 1 has run
- [logs/stage1_decode_sanity_report.json](logs/stage1_decode_sanity_report.json)
- [logs/stage2_smoke_summary.json](logs/stage2_smoke_summary.json)
- [logs/stage2_checkpoint_integrity.json](logs/stage2_checkpoint_integrity.json)

### Practical Summary

P100 is the conservative stability profile. T4x2 is the higher-throughput profile. Keep smoke settings identical when validating correctness, and only change the hardware profile and its matching batch/worker settings when comparing performance.

### Artifact Export

Use [scripts/export_artifacts.sh](scripts/export_artifacts.sh) to package checkpoints, logs, configs, and run metadata after a Kaggle session.

Empirical Kaggle benchmark numbers are still pending; record them in the benchmark logs once P100 and T4x2 runs are available.

---

## Next Steps

1. Validate Stage 1 and Stage 2 Kaggle runs against the current checkpoint/export contracts.
2. Capture empirical benchmark numbers in the logs once Kaggle P100 and T4x2 runs are executed.
3. Keep UNet/iMF expansion, TPU work, and broader public-release automation in the post-v1 backlog.

---

## Version

**v1.0** – Repository skeleton and config surface locked to v1 scope boundaries.
Commit: Task 1.3 - Repository Skeleton and Config Surface

**Paper References**:

- Paper 1: Native and Compact Structured Latents for 3D Generation (arXiv:2512.14692v1)
- Paper 2: Improved Mean Flows (arXiv:2512.02012v1)
