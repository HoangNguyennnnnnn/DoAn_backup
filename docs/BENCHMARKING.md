# Kaggle GPU Benchmarking Guide

## Purpose

This document defines the standardized comparison method for Kaggle GPU profiles used by the project: `P100` vs `T4x2`.

The benchmark method is intentionally smoke-scoped and aligned with the canonical notebook workflows and trainers.

## Scope

Benchmark the following flows on both hardware profiles:

1. Stage 1 run-all notebook flow
2. Stage 2 smoke notebook flow
3. Resume / interruption recovery behavior
4. Artifact and report generation in `/kaggle/working`

## Canonical Entry Points

- Stage 1 notebook: [notebooks/kaggle_stage1_train.ipynb](../notebooks/kaggle_stage1_train.ipynb)
- Stage 2 notebook: [notebooks/kaggle_stage2_smoke.ipynb](../notebooks/kaggle_stage2_smoke.ipynb)
- Stage 1 autoresume: [scripts/train_stage1_autoresume.sh](../scripts/train_stage1_autoresume.sh)
- Stage 2 autoresume: [scripts/train_stage2_autoresume.sh](../scripts/train_stage2_autoresume.sh)

## Comparison Rules

Use the same notebook flow and the same smoke scope on both hardware profiles.

- **Stage 1**: same dataset attachment, same runtime guards, same checkpoint/resume behavior, same decode sanity command.
- **Stage 2**: same manifest build command, same smoke trainer command, same resume precedence, same integrity-report inspection.
- **Export preference**: `best.ckpt` first for evaluation/export.
- **Recovery preference**: `latest_step.ckpt` first, then `interrupt.ckpt`, then `latest.ckpt`, then `best.ckpt`.

## Hardware Profiles

### P100

Use `configs/hardware_p100.yaml`.

Recommended tuning:

- `batch_size: 8`
- `num_workers: 4`
- `mixed_precision: true`
- `gradient_checkpointing: true`
- `enable_flash_attention: false`
- Resume from `latest_step.ckpt` when available.

### T4x2

Use `configs/hardware_t4x2.yaml`.

Recommended tuning:

- `batch_size: 16`
- `num_workers: 8`
- `mixed_precision: true`
- `distributed: true`
- `distributed_strategy: ddp`
- `enable_flash_attention: true`
- Resume from `latest_step.ckpt` when available.

## What to Record

For each profile and each smoke flow, record:

- wall-clock runtime
- throughput proxy metrics
- memory behavior
- checkpoint/restart success behavior
- interruption resilience observations
- artifact path discovery success

## Output Locations

Use these canonical paths for benchmark evidence:

- `/kaggle/working/checkpoints`
- `/kaggle/working/logs`
- `/kaggle/working/runs/<run_id>/metadata`
- `/kaggle/working/inference_decode_sanity` or Stage 2 smoke artifacts under `/kaggle/working/cache/stage2_latents`

## Stage 1 Benchmark Signals

Inspect:

- `stage1_training_metrics.jsonl`
- `stage1_decode_sanity_report.json`
- `stage1_decode_sanity_report.md`
- `train_summary.json`

Recommended interpretation:

- P100 should prioritize stability and lower batch size.
- T4x2 should provide higher throughput and more worker headroom.
- If a profile triggers OOM fallback, note the adjusted batch size and whether autoresume preserved progress.

## Stage 2 Benchmark Signals

Inspect:

- `stage2_smoke_metrics.jsonl`
- `stage2_smoke_summary.json`
- `stage2_checkpoint_integrity.json`
- manifest files under `/kaggle/working/cache/stage2_latents/stage2-latent-v1/manifests`

Recommended interpretation:

- P100 should remain the conservative reference profile.
- T4x2 should typically tolerate the same smoke settings with more margin.
- Resume precedence should remain exact and should not skip `latest_step.ckpt`.

## Practical Recommendations

- Use `P100` when you want the smallest stable baseline and more conservative memory behavior.
- Use `T4x2` when you want faster iteration and can afford higher batch sizes and worker counts.
- Keep Stage 2 smoke settings identical across profiles when validating correctness, not quality.
- Prefer `best.ckpt` for artifact export and evaluation, but prefer `latest_step.ckpt` for recovery.
- If you need reproducible comparisons, keep the run mode and sample caps fixed across both hardware profiles.

Empirical P100 and T4x2 benchmark numbers are still pending and should be captured from Kaggle runs before treating the benchmark as complete.

## README-Ready Summary

> Kaggle benchmark guidance: P100 is the conservative stability profile and T4x2 is the higher-throughput profile. Use identical smoke scopes, explicit resume precedence (`latest_step -> interrupt -> latest -> best`), and the same artifact locations under `/kaggle/working` to keep comparisons reproducible.
