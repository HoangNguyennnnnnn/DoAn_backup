# PROJECT_SCOPE

## Task Reference

- Task 1.1 - Paper-to-Architecture Mapping Baseline
- Source papers:
  - Paper 1: Native and Compact Structured Latents for 3D Generation (arXiv:2512.14692v1)
  - Paper 2: Improved Mean Flows: On the Challenges of Fastforward Generative Models (arXiv:2512.02012v1)

## Scope Lock Summary

This document locks the v1 architecture to a Kaggle-first execution path focused on shape generation reliability and reproducibility. The mandatory core path for v1 is O-Voxel -> Shape SC-VAE -> SLAT latent interface, with Stage 1 fully implemented and Stage 2 limited to smoke-level compatibility hooks only. UNet and improved MeanFlow adaptation are explicitly constrained to configurable or deferred Stage 2 work and are not required for v1 completion.

## Hard Constraints (Kaggle Runtime)

- Execution target is Kaggle notebook runtime with Kaggle-hosted datasets.
- v1 must be operable on Kaggle GPU sessions (P100 and T4x2 profiles).
- Session interruption resilience is mandatory (frequent checkpoints and autoresume path).
- Data pipeline must avoid assumptions of local private storage and must use configurable dataset roots.

## Dataset Strategy Lock (Task 1.2)

### Kaggle Dataset Audit Result

- Primary candidate confirmed on Kaggle:
  - ModelNet40 mirror: `balraj98/modelnet40-princeton-3d-object-dataset`
  - Reported content: 12,311 shapes, 40 categories, OFF format, official train/test split metadata
  - Kaggle license field: `Other (specified in description)`
  - Provenance note: mirrors Princeton ModelNet release
- Extension candidate confirmed on Kaggle:
  - ShapeNetPart mirrors include `majdouline20/shapenetpart-dataset` (MIT-declared) and additional mirrors with unknown license fields
  - Provenance note: part annotations trace to Yi et al. ShapeNet part benchmark (16 categories, 31,963 models)

### v1 Dataset Decision

- Primary dataset for v1: ModelNet40 on Kaggle.
- Fallback dataset for v1 if primary mirror is unavailable/corrupted: alternate Kaggle ModelNet40 mirror from a different owner with matching split counts and OFF assets.
- Extension dataset (not default v1): ShapeNetPart for segmentation-aware ablations and Stage 2 robustness checks after Stage 1 baseline is stable.

### Distribution Fit Analysis (Pipeline-Oriented)

| Criterion                  | ModelNet40 (Primary)                                                   | ShapeNetPart (Extension)                                          | v1 Decision Impact                                       |
| -------------------------- | ---------------------------------------------------------------------- | ----------------------------------------------------------------- | -------------------------------------------------------- |
| Mesh quality               | Curated CAD object meshes, stable for geometry pretraining             | Part-annotated shapes with segmentation bias                      | ModelNet40 preferred for clean shape latent pretraining  |
| Class coverage             | 40 object classes, balanced enough for generic shape manifold learning | 16 classes, denser part-level labels but narrower category spread | Use ModelNet40 to maximize broad shape priors in Stage 1 |
| Format compatibility       | OFF meshes align with planned OFF -> OBJ preprocessing                 | Common mirrors are point-cloud/HDF5 or mixed representations      | ModelNet40 reduces preprocessing ambiguity for v1        |
| Stage 1 (shape SC-VAE) fit | High: direct support for shape-only reconstruction and latent learning | Medium: useful but introduces part-label-specific biases          | Stage 1 should train on ModelNet40 first                 |
| Stage 2 smoke fit          | Medium: sufficient for latent generator smoke validation               | Medium-High for conditional/part-aware experiments                | Keep ShapeNetPart as optional Stage 2 extension          |

### Kaggle Usage Contract (Attach, Not Upload)

The project must consume data through Kaggle notebook dataset attachment only.

- Required notebook workflow:
  1. Add dataset in Kaggle notebook UI (`Add data`) using exact dataset slug.
  2. Read from mounted input paths under `/kaggle/input/<dataset-slug>/`.
  3. Write derived caches/checkpoints only to `/kaggle/working/`.
  4. Never rely on manual local uploads for baseline reproducibility.
- Path contract examples:
  - ModelNet40 root: `/kaggle/input/modelnet40-princeton-3d-object-dataset/`
  - ShapeNetPart root (extension): `/kaggle/input/shapenetpart-dataset/` (or mirror-specific slug)

### Extension Criteria Gate (When to Move Beyond Primary)

Move from ModelNet40-only to include ShapeNetPart only if all are true:

1. Stage 1 training reaches stable checkpoint/restart behavior on Kaggle.
2. OFF -> OBJ conversion and O-Voxel feature extraction pass reproducibility checks.
3. Stage 2 smoke loop runs end-to-end at least once with valid latent I/O.
4. ShapeNetPart mirror selected has acceptable provenance notes and no unresolved blocking license ambiguity for intended artifact publication mode.

### License/Publishability Scope Rules

- For public workflows, treat Kaggle mirror license text as necessary but not sufficient; always honor upstream dataset terms and provenance.
- ModelNet40 inherits original CAD copyright constraints and is documented for academic research convenience.
- ShapeNetPart mirror licenses can be inconsistent across Kaggle owners; checkpoint publication must be reviewed against upstream ShapeNet/asset-origin restrictions.
- If license chain is unclear, publish code/config/logs and metrics first, and defer public checkpoint release.

## Paper-to-Architecture Mapping

### Paper 1 (2512.14692v1) -> v1 Core Path

- O-Voxel representation is the native structure boundary for shape assets.
- Shape SC-VAE is the required latent learning module for Stage 1.
- SLAT-compatible latent interface is required as the contract between Stage 1 and Stage 2.
- Full material/PBR branch from paper is out of v1 baseline unless needed for interface placeholders.

### Paper 2 (2512.02012v1) -> Stage 2 Adaptation Boundary

- iMF objective and conditioning strategy are treated as Stage 2 generator-level options.
- v-loss reparameterization and flexible guidance conditioning are allowed only in Stage 2 experiments.
- No commitment to replicate ImageNet-scale training setup, TPU stack, or paper-level FID targets in v1.

## Architecture Decision Table

| Area                    | Decision                                                                  | Status                  | Rationale                                                                         |
| ----------------------- | ------------------------------------------------------------------------- | ----------------------- | --------------------------------------------------------------------------------- |
| Data source             | Kaggle-hosted datasets only for training runs                             | Fixed (v1)              | Required for reproducible Kaggle execution and zero large local upload dependency |
| Primary dataset         | ModelNet40 Kaggle mirror                                                  | Fixed (v1)              | Best fit for shape-first Stage 1 and OFF-based preprocessing path                 |
| Fallback dataset        | Alternate ModelNet40 Kaggle mirror with matching counts/splits            | Configurable (v1)       | Preserves continuity if a single mirror becomes unavailable                       |
| Extension dataset       | ShapeNetPart Kaggle mirror                                                | Deferred/Configurable   | Useful for extension experiments after v1 baseline stability                      |
| 3D representation       | O-Voxel shape path as canonical internal representation                   | Fixed (v1)              | Directly preserves paper-1 core method boundary                                   |
| Stage 1 model           | Shape SC-VAE implementation with encoder/decoder and latent bottleneck    | Fixed (v1)              | Mandatory core path from paper-1 mapping                                          |
| Latent interface        | SLAT-compatible latent tokens and tensor contract                         | Fixed (v1)              | Required handoff from Stage 1 to Stage 2                                          |
| Stage 1 objective       | Reconstruction + optional KL (shape-first)                                | Configurable (v1)       | Paper supports staged loss strategy; Kaggle compute may require reduced objective |
| Stage 1 training regime | Two-stage high-res perceptual rendering supervision                       | Deferred (Stage 2+)     | Too heavy for baseline Kaggle-first v1, can be added after stable v1              |
| Material branch         | Material SC-VAE and PBR attribute generation                              | Deferred (Stage 2+)     | v1 scope is shape-first stability and baseline generation path                    |
| Stage 2 backbone        | UNet latent generator                                                     | Configurable (Stage 2)  | Allowed by plan but not mandatory in v1                                           |
| Stage 2 objective       | Improved MeanFlow adaptation (iMF-style v-loss and guidance conditioning) | Configurable (Stage 2)  | Allowed as adaptation path, intentionally excluded from v1 lock                   |
| Stage 2 execution       | Smoke run only (1-3 epochs equivalent)                                    | Fixed for Stage 2 entry | Aligns with Kaggle time limits and project plan                                   |
| Conditioning backend    | DINO-based context (dual-view default, single-view fallback)              | Configurable (v1)       | Fits object-domain data and Kaggle practical constraints                          |
| Hardware profile        | GPU-first (P100/T4x2), TPU as future branch                               | Fixed (v1)              | Matches current stack readiness and runtime reliability goals                     |

## v1 Mandatory Components

- Kaggle-compatible dataset adapter and preprocessing path into O-Voxel shape features.
- Shape SC-VAE training and checkpointing with interrupt/latest/best artifacts.
- SLAT latent export/interface contract for downstream generator consumption.
- Stage 1 inference decode sanity path for valid mesh output.
- Notebook-driven execution path for Kaggle run-all reproducibility.

## v1 Configurable Components

- SC-VAE KL weighting and reconstruction-loss weighting.
- DINO context mode (dual-view vs single-view) and embedding dimension.
- Mixed precision, batch size, gradient accumulation, and dataloader worker counts by hardware profile.
- Stage 2 smoke backend selection placeholder (UNet or alternative) without quality commitments.

## Deferred Components (Not Required for v1 Sign-Off)

- Full material generation branch and PBR-targeted training/evaluation.
- Large-scale DiT training regime from paper-1.
- Production-grade UNet + iMF integration and hyperparameter sweeps.
- CFG interval optimization and in-context conditioning tuning from paper-2.
- TPU-native training implementation and optimization.

## Non-Goals (v1)

- Reproducing paper-scale model size, dataset scale, or benchmark numbers.
- Achieving SOTA visual quality or paper-level FID/IS equivalents.
- Shipping a fully optimized multi-stage production generator in first milestone.
- Solving full texture/material realism in the initial Kaggle baseline.

## Explicit Scope Lock

v1 is complete when the project can run end-to-end in Kaggle for the shape-first path: dataset ingestion -> O-Voxel feature path -> Shape SC-VAE training with resume/checkpoints -> SLAT-compatible latent handoff -> decode sanity outputs. Any UNet-specific design or improved MeanFlow training objective beyond smoke-level hooks is outside this lock and must be treated as Stage 2 follow-up work.
