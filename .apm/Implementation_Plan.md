# MeshLatent-Kaggle – APM Implementation Plan

**Memory Strategy:** Dynamic-MD
**Last Modification:** Synced Task 5.2 completion (artifact export/runbook implemented) and added Task 5.3 requirement to reconcile legacy doc fragments that conflict with current resume precedence and phase status.
**Project Overview:** Build a Kaggle-first 3D generation project with pipeline mesh -> OVoxel -> SLAT -> UNet, prioritize Stage 1 (shape SC-VAE) stability and resume/checkpoint reliability, then run Stage 2 smoke with improved mean-flow.

## Phase 1: Foundation and Research Lock

### Task 1.1 – Paper-to-Architecture Mapping Baseline - Agent_Research

**Objective:** Lock implementation boundaries from target papers into an executable v1 architecture spec.
**Output:** `PROJECT_SCOPE.md` with mandatory vs flexible design decisions.
**Guidance:** Must preserve paper-1 core path (OVoxel, SC-VAE, SLAT) and allow UNet/IMF adaptation. This is a sequential research task.

1. Ad-Hoc Delegation - Paper section mapping.
2. Extract mandatory components from paper 2512.14692v1 for v1 implementation.
3. Extract adaptable components from paper 2512.02012v1 for Stage 2.
4. Produce architecture decision table: fixed, configurable, deferred.
5. Write final scope lock and non-goals into `PROJECT_SCOPE.md`.

### Task 1.2 – Dataset Suitability and Distribution Analysis - Agent_Research

**Objective:** Select v1 dataset strategy based on Kaggle availability, data distribution fit, and license clarity.
**Output:** Dataset decision section in `PROJECT_SCOPE.md` and `DATA_LICENSE.md` baseline.
**Guidance:** Prefer ModelNet40 first; evaluate ShapeNetPart as extension. Include provenance and publishability constraints.

1. Ad-Hoc Delegation - Kaggle dataset audit.
2. Compare candidate datasets against pipeline needs: mesh quality, class coverage, format compatibility.
3. Analyze practical distribution fit for Stage 1 shape pretraining and Stage 2 smoke.
4. Record primary dataset, fallback dataset, and extension criteria.
5. Document licensing/provenance caveats and public release constraints.

### Task 1.3 – Repository Skeleton and Config Surface - Agent_MLOps

**Objective:** Initialize repo structure and config entrypoints aligned with Kaggle runtime constraints.
**Output:** Directory skeleton, core docs files, and base YAML config files.
**Guidance:** **Depends on: Task 1.1 Output by Agent_Research** and **Depends on: Task 1.2 Output by Agent_Research**. Keep paths relative, Kaggle attach-path aware, and hardware-profile driven.

1. Create repository structure for configs, scripts, src, notebooks, outputs, checkpoints, and logs.
2. Add `README.md`, `PROJECT_SCOPE.md`, and `DATA_LICENSE.md` placeholders linked to run flow.
3. Create base hardware configs for P100, T4x2, TPU future branch marker.
4. Add data and training config stubs for Stage 1 and Stage 2.

### Task 1.4 – Kaggle Bootstrap and Runtime Guards - Agent_MLOps

**Objective:** Provide repeatable environment setup with safe defaults for Kaggle sessions.
**Output:** `scripts/kaggle_bootstrap.sh` and runtime guard utilities.
**Guidance:** Configure for GPU-first, resilient restarts, and low-friction re-entry after timeouts.

1. Install and validate required Python packages in deterministic order.
2. Add runtime checks for GPU visibility, disk, and writable output paths.
3. Add run metadata capture hooks (config snapshot, hardware profile, commit hash if available).
4. Print clear next-step commands for data prep and training entrypoints.

## Phase 2: Data Pipeline to OVoxel and SLAT Inputs

### Task 2.1 – Kaggle Dataset Adapter and Split Loader - Agent_Data

**Objective:** Implement dataset ingestion from Kaggle datasets with reproducible split handling.
**Output:** `src/data/dataset_adapter.py` returning valid sample streams.
**Guidance:** **Depends on: Task 1.2 Output by Agent_Research**. Must support dataset-root configuration and no hard-coded local paths.

1. Implement dataset discovery and split loading for primary dataset.
2. Normalize file indexing and metadata records for downstream stages.
3. Add robust error handling for missing or malformed files.
4. Expose deterministic sampling and seed-aware iteration.

### Task 2.2 – OFF/OBJ Conversion and Cache Layer - Agent_Data

**Objective:** Create conversion and caching pipeline to reduce repeated heavy I/O.
**Output:** `src/data/off_to_obj_converter.py` and cache management utilities.
**Guidance:** Must support resumable conversion and safe overwrite policy.

1. Build recursive converter OFF -> OBJ with idempotent behavior.
2. Add cache index for converted assets and preprocessed tensors.
3. Implement incremental mode to process only missing entries.
4. Add integrity checks and conversion summary logs.

### Task 2.3 – Mesh-to-OVoxel Feature Construction - Agent_Data

**Objective:** Convert mesh assets into OVoxel representation suitable for SC-VAE input.
**Output:** `src/data/mesh_to_feature.py` producing OVoxel tensors.
**Guidance:** **Depends on: Task 2.2 Output**. Keep representation parameters configurable via YAML.

1. Implement mesh normalization and voxelization parameter interface.
2. Generate OVoxel tensors with optional precision controls.
3. Persist feature artifacts with schema version tagging.
4. Add quick visual/statistical sanity checks for generated tensors.

### Task 2.4 – OVoxel-to-SLAT Interface Contract (Shape Path) - Agent_ModelStage1

**Objective:** Define and implement the interface contract from OVoxel features to SLAT tokens for shape SC-VAE.
**Output:** Interface module and config schema entries used by Stage 1 model/training.
**Guidance:** **Depends on: Task 2.3 Output by Agent_Data**. Shape-only path for v1; material path deferred.

1. Specify tensor shape conventions and batching contract.
2. Implement encoder-side interface adapters for Stage 1 model.
3. Add config-driven token length and latent dimension controls.
4. Add assertion checks with informative failure messages.

### Task 2.5 – Data Pipeline Smoke and Throughput Baseline - Agent_Data

**Objective:** Validate that dataloader returns stable non-zero samples and acceptable startup throughput on Kaggle.
**Output:** Smoke report in logs and script command for repeatable validation.
**Guidance:** **Depends on: Task 2.1 Output** and **Depends on: Task 2.3 Output**. Include failure diagnostics for file, cache, shape mismatches, and occupancy-ratio anomalies by category. Validate cache refresh policy when source OFF files change (non-incremental + overwrite).

1. Run end-to-end data smoke from dataset loader to OVoxel batch.
2. Validate sample counts, tensor shapes, and batch collation.
3. Record startup timings and cache hit rates.
4. Emit actionable debug messages for common failure modes.

## Phase 3: Stage 1 Shape SC-VAE Training Stability

### Task 3.1 – Shape SC-VAE Model Implementation - Agent_ModelStage1

**Objective:** Implement shape SC-VAE architecture over OVoxel input and SLAT latent outputs.
**Output:** `src/models/shape_sc_vae.py` and related config bindings.
**Guidance:** **Depends on: Task 2.4 Output**. Must expose clear encode/decode APIs for training and inference, and re-verify OVoxel-to-SLAT contract smoke in Kaggle/Linux runtime before treating Stage 1 path as execution-validated.

1. Implement encoder, latent bottleneck, and decoder components for shape branch.
2. Add reconstruction loss and optional KL branch switches.
3. Expose latent token outputs compatible with Stage 2 preparation.
4. Add lightweight unit sanity checks for forward passes.

### Task 3.2 – Stage 1 Training Loop with Resume/Recovery - Agent_Train

**Objective:** Build robust Stage 1 training loop optimized for Kaggle interruptions.
**Output:** `src/train/train_stage1.py` and `scripts/train_stage1_autoresume.sh`.
**Guidance:** **Depends on: Task 3.1 Output by Agent_ModelStage1** and **Depends on: Task 2.5 Output by Agent_Data**. Must support interrupt/latest_step/best checkpoint policy and OOM backoff. Treat Kaggle/Linux revalidation of the Task 3.1 smoke path as a prerequisite for final Stage 1 execution confidence.

1. Implement train/validation steps with mixed precision controls.
2. Add checkpoint manager for interrupt, latest_step, and best states.
3. Add autoresume logic from latest available checkpoint.
4. Add OOM fallback hooks (batch downscale or accumulation increase).
5. Persist run metadata snapshot for reproducibility.

### Task 3.3 – Stage 1 Inference Decode Sanity - Agent_ModelStage1

**Objective:** Confirm Stage 1 checkpoints decode valid mesh outputs from latent representation.
**Output:** `src/inference/generate_mesh.py` shape-branch sanity mode and output examples.
**Guidance:** **Depends on: Task 3.2 Output by Agent_Train**. Validation target is functional correctness, not final visual SOTA. Treat Kaggle/Linux checkpoint and resume behavior as the authoritative environment for the Stage 1 decoder sanity path.

1. Load best/latest checkpoints and run decode pipeline.
2. Export sample meshes and run validity checks.
3. Record reconstruction trend and decode pass/fail summary.

### Task 3.4 – Kaggle Notebook Run-All for Stage 1 - Agent_MLOps

**Objective:** Provide one-notebook execution path for Stage 1 from setup to checkpoint artifacts.
**Output:** `notebooks/kaggle_stage1_train.ipynb` with run-all workflow.
**Guidance:** **Depends on: Task 3.2 Output by Agent_Train** and **Depends on: Task 3.3 Output by Agent_ModelStage1**. Include resume instructions for expired sessions and a Kaggle/Linux decode-sanity validation cell using Stage 1 checkpoints.

1. Add bootstrap, data prep, training, and artifact export cells.
2. Add clear parameter cells for hardware profile selection.
3. Add resume-from-checkpoint usage cell and expected outputs.

## Phase 4: Stage 2 UNet and Improved Mean-Flow Smoke

### Task 4.1 – Latent Dataset Builder from Stage 1 Outputs - Agent_Data

**Objective:** Build latent dataset artifacts usable by Stage 2 UNet training.
**Output:** Latent dataset generation utilities and index manifests.
**Guidance:** **Depends on: Task 3.2 Output by Agent_Train**. Keep format minimal for smoke testing.

1. Extract SLAT latent tokens from Stage 1 checkpoints or encoder pass.
2. Build indexed latent records with split metadata.
3. Validate latent tensor consistency for Stage 2 ingestion.

### Task 4.2 – UNet + Improved Mean-Flow Objective Integration - Agent_ModelStage2

**Objective:** Implement Stage 2 model path with UNet and improved mean-flow training objective.
**Output:** `src/models/latent_generator.py` and objective modules.
**Guidance:** **Depends on: Task 1.1 Output by Agent_Research** and **Depends on: Task 4.1 Output by Agent_Data**. Prioritize correctness and trainability over full optimization. Treat Kaggle/Linux as the authoritative environment for latent dataset availability and Stage 1 checkpoint-derived inputs.

1. Implement latent-space UNet architecture configurable for Kaggle VRAM.
2. Implement improved mean-flow objective path and scheduler hooks.
3. Connect context backend stubs with DINO-first interface.
4. Add forward-pass sanity checks and shape assertions.

### Task 4.3 – Stage 2 Smoke Training with Checkpoint Resume - Agent_Train

**Objective:** Execute short Stage 2 runs to validate end-to-end latent generation workflow.
**Output:** `src/train/train_stage2.py`, `scripts/train_stage2_autoresume.sh`, and smoke checkpoints.
**Guidance:** **Depends on: Task 4.2 Output by Agent_ModelStage2**. Reuse Stage 1 reliability patterns for interruptions. Preserve manifest-backed latent loading, checkpoint resume semantics, and Kaggle/Linux authority for runtime verification.

1. Implement Stage 2 training entrypoint and config wiring.
2. Add limited-epoch smoke mode (1-3 epochs equivalent).
3. Integrate checkpoint/resume and run metadata logging.
4. Produce minimal loss trend and checkpoint integrity report.

### Task 4.4 – Kaggle Notebook Stage 2 Smoke - Agent_MLOps

**Objective:** Provide notebook workflow for Stage 2 smoke execution and artifact capture.
**Output:** `notebooks/kaggle_stage2_smoke.ipynb`.
**Guidance:** **Depends on: Task 4.3 Output by Agent_Train**. Notebook should mirror Stage 1 UX pattern and make Stage 2 resume/export precedence explicit (resume: latest_step -> interrupt -> latest -> best; export/evaluation: best -> latest).

1. Add data/latent prep and smoke training cells.
2. Add resume guidance and expected outputs section.
3. Export smoke artifacts and logs to consistent paths.

## Phase 5: Stabilization, Benchmark, and Handoff

### Task 5.1 – P100 vs T4x2 Runtime Benchmarking - Agent_MLOps

**Objective:** Measure practical throughput/stability differences across Kaggle GPU profiles.
**Output:** Benchmark section in `README.md` and profile recommendations.
**Guidance:** **Depends on: Task 3.4 Output** and **Depends on: Task 4.4 Output**. Focus on reproducible, comparable settings.

1. Run standardized benchmark scenarios for Stage 1 and Stage 2 smoke.
2. Record runtime, memory behavior, and interruption resilience.
3. Summarize profile-specific tuning guidance.

### Task 5.2 – Artifact Export and Recovery Runbook - Agent_MLOps

**Objective:** Ensure users can re-run and recover project state with minimal friction.
**Output:** `scripts/export_artifacts.sh`, runbook updates, and artifact path conventions.
**Guidance:** **Depends on: Task 3.2 Output by Agent_Train** and **Depends on: Task 4.3 Output by Agent_Train**. Must include checkpoint and logs packaging plus a checkpoint publication gate that requires license-chain verification before public release.

1. Implement artifact export script for checkpoints, configs, and logs.
2. Add runbook steps for resume after Kaggle session expiration.
3. Add checklist for validating restored state.

### Task 5.3 – Documentation and Quality Gate Closure - Agent_QA

**Objective:** Close v1 with complete documentation, logs, and validation evidence.
**Output:** Finalized `README.md`, `WORK_LOG` entries, and completion checklist.
**Guidance:** **Depends on: Task 5.1 Output by Agent_MLOps** and **Depends on: Task 5.2 Output by Agent_MLOps**. Enforce required logging standards for every major run, reconcile conflicting legacy documentation snippets (resume precedence and outdated phase references), and explicitly record unresolved risk if empirical Kaggle GPU benchmark numbers are still pending.

1. Validate required docs: README, runbook, experiment log, config snapshots.
2. Verify DoD checklist against agreed v1 criteria.
3. Record unresolved risks and post-v1 backlog items.
