# MeshLatent-Kaggle – APM Memory Root

**Memory Strategy:** Dynamic-MD
**Project Overview:** Build a Kaggle-first 3D generation pipeline with mesh -> OVoxel -> SLAT -> UNet, using Kaggle-hosted data, notebook-friendly setup, and reliable train/resume/export guidance.

## Phase 01 - Foundation and Research Lock Summary

- Outcome summary: Phase 1 established a Kaggle-first v1 baseline with fixed shape path boundaries (OVoxel -> Shape SC-VAE -> SLAT), locked ModelNet40 as primary dataset with attach-only Kaggle usage, and delivered repository/config/bootstrap scaffolding with runtime guards and autoresume-oriented operational guidance.
- Involved Agents: Agent_Research, Agent_MLOps
- Phase task logs:
  - .apm/Memory/Phase_01_Foundation_and_Research_Lock/Task_1_1_Paper_to_Architecture_Mapping_Baseline.md
  - .apm/Memory/Phase_01_Foundation_and_Research_Lock/Task_1_2_Dataset_Suitability_and_Distribution_Analysis.md
  - .apm/Memory/Phase_01_Foundation_and_Research_Lock/Task_1_3_Repository_Skeleton_and_Config_Surface.md
  - .apm/Memory/Phase_01_Foundation_and_Research_Lock/Task_1_4_Kaggle_Bootstrap_and_Runtime_Guards.md

## Phase 02 - Data Pipeline to OVoxel and SLAT Inputs Summary

- Outcome summary: Phase 2 delivered a deterministic Kaggle dataset adapter, recursive OFF->OBJ conversion with cache/index policy, OVoxel feature construction with versioned tensor artifacts, and an end-to-end smoke runner with occupancy and cache diagnostics. Full throughput metrics remain to be collected in Kaggle because the local workspace lacks the required /kaggle/input mount.
- Involved Agents: Agent_Data
- Phase task logs:
  - .apm/Memory/Phase_02_Data_Pipeline_to_OVoxel_and_SLAT_Inputs/Task_2_1_Kaggle_Dataset_Adapter_and_Split_Loader.md
  - .apm/Memory/Phase_02_Data_Pipeline_to_OVoxel_and_SLAT_Inputs/Task_2_2_OFF_OBJ_Conversion_and_Cache_Layer.md
  - .apm/Memory/Phase_02_Data_Pipeline_to_OVoxel_and_SLAT_Inputs/Task_2_3_Mesh_to_OVoxel_Feature_Construction.md
  - .apm/Memory/Phase_02_Data_Pipeline_to_OVoxel_and_SLAT_Inputs/Task_2_4_OVoxel_to_SLAT_Interface_Contract_Shape_Path.md
  - .apm/Memory/Phase_02_Data_Pipeline_to_OVoxel_and_SLAT_Inputs/Task_2_5_Data_Pipeline_Smoke_and_Throughput_Baseline.md

## Phase 03 - Stage 1 Shape SC-VAE Training Stability Summary

- Outcome summary: Phase 3 delivered the canonical ShapeSCVAE wrapper, a Kaggle-first Stage 1 trainer with checkpoint/autoresume/OOM fallback behavior, a decode sanity runner with resilient reporting, and a run-all Kaggle notebook that unifies bootstrap, train, resume, and decode validation. Local Windows runtime remains non-authoritative for final runtime confidence, so Kaggle/Linux execution is the validation source of truth.
- Involved Agents: Agent_ModelStage1, Agent_Train, Agent_MLOps
- Phase task logs:
  - .apm/Memory/Phase_03_Stage_1_Shape_SC-VAE_Training_Stability/Task_3_1_Shape_SC-VAE_Model_Implementation.md
  - .apm/Memory/Phase_03_Stage_1_Shape_SC-VAE_Training_Stability/Task_3_2_Stage_1_Training_Loop_with_Resume_Recovery.md
  - .apm/Memory/Phase_03_Stage_1_Shape_SC-VAE_Training_Stability/Task_3_3_Stage_1_Inference_Decode_Sanity.md
  - .apm/Memory/Phase_03_Stage_1_Shape_SC-VAE_Training_Stability/Task_3_4_Kaggle_Notebook_Run-All_for_Stage_1.md

## Phase 04 - Stage 2 UNet and Improved Mean-Flow Smoke Summary

- Outcome summary: Phase 4 delivered Stage 2 latent dataset extraction from Stage 1 outputs, token-space latent UNet with improved mean-flow objective, smoke trainer resume/recovery semantics, and a Kaggle run-all smoke notebook. Resume and export precedence are explicit (resume: latest_step -> interrupt -> latest -> best; export/evaluation: best -> latest), and Kaggle/Linux remains the authoritative runtime for checkpoint/manifest availability.
- Involved Agents: Agent_Data, Agent_ModelStage2, Agent_Train, Agent_MLOps
- Phase task logs:
  - .apm/Memory/Phase_04_Stage_2_UNet_and_Improved_Mean-Flow_Smoke/Task_4_1_Latent_Dataset_Builder_from_Stage_1_Outputs.md
  - .apm/Memory/Phase_04_Stage_2_UNet_and_Improved_Mean-Flow_Smoke/Task_4_2_UNet_and_Improved_Mean-Flow_Objective_Integration.md
  - .apm/Memory/Phase_04_Stage_2_UNet_and_Improved_Mean-Flow_Smoke/Task_4_3_Stage_2_Smoke_Training_with_Checkpoint_Resume.md
  - .apm/Memory/Phase_04_Stage_2_UNet_and_Improved_Mean-Flow_Smoke/Task_4_4_Kaggle_Notebook_Stage_2_Smoke.md

## Phase 05 - Stabilization, Benchmark, and Handoff Summary

- Outcome summary: Phase 5 finalized the operational surface with a reproducible Kaggle benchmark guide, a deterministic artifact export and recovery workflow with publication gating, and a documentation quality-gate closure that reconciled the README, runbook, benchmark, and checkpoint guidance to the actual runtime behavior. Empirical P100/T4x2 benchmark numbers remain pending Kaggle execution and are tracked as a documented closure risk.
- Involved Agents: Agent_MLOps, Agent_QA
- Phase task logs:
  - .apm/Memory/Phase_05_Stabilization_Benchmark_and_Handoff/Task_5_1_P100_vs_T4x2_Runtime_Benchmarking.md
  - .apm/Memory/Phase_05_Stabilization_Benchmark_and_Handoff/Task_5_2_Artifact_Export_and_Recovery_Runbook.md
  - .apm/Memory/Phase_05_Stabilization_Benchmark_and_Handoff/Task_5_3_Documentation_and_Quality_Gate_Closure.md
