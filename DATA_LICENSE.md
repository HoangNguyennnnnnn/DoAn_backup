# DATA_LICENSE

## Purpose

This document defines baseline dataset provenance, licensing caveats, and publishability rules for the Kaggle-first v1 workflow.

## Kaggle Attachment Rule (Mandatory)

Datasets must be consumed via Kaggle notebook attachment, not manual local upload.

1. Attach dataset in Kaggle notebook UI (`Add data`).
2. Read only from `/kaggle/input/<dataset-slug>/`.
3. Persist generated artifacts to `/kaggle/working/`.
4. Keep dataset slug configurable in project config.

## Dataset Audit Summary

| Dataset Role         | Candidate         | Kaggle Signal                                                                     | Data/Profile Fit                                                                | Baseline Decision                 |
| -------------------- | ----------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | --------------------------------- |
| Primary (v1)         | ModelNet40        | `balraj98/modelnet40-princeton-3d-object-dataset` available, active usage history | 12,311 aligned CAD shapes, 40 classes, OFF meshes, clear train/test split usage | Adopt as v1 primary               |
| Fallback (v1)        | ModelNet40 mirror | Multiple mirrors exist on Kaggle (owner-dependent)                                | Same task semantics if counts/splits and structure match                        | Use if primary mirror unavailable |
| Extension (Stage 2+) | ShapeNetPart      | Kaggle mirrors available (for example `majdouline20/shapenetpart-dataset`)        | 16 categories with part annotations, useful for segmentation-aware extensions   | Defer from default v1             |

## Provenance Notes

### ModelNet40

- Original source: Princeton ModelNet project.
- Public benchmark context and OFF-format release are documented by Princeton.
- Kaggle mirror descriptions indicate the data is sourced from official ModelNet materials.

### ShapeNetPart

- Benchmark source: Yi et al. part annotation dataset built on ShapeNetCore.
- Commonly cited stats: 31,963 models, 16 categories, part-level annotations.
- Kaggle mirrors may provide repackaged variants (raw points, HDF5, or mixed layouts).

## License Baseline and Caveats

### ModelNet40

- Kaggle license field may show `Other (specified in description)`.
- Princeton ModelNet copyright statement indicates original CAD model copyright remains with source authors.
- Dataset is described as provided for academic research convenience.

### ShapeNetPart

- Mirror license fields vary by owner (MIT-declared vs unknown).
- Mirror-level license declaration does not automatically override upstream ShapeNet/source-asset restrictions.
- Treat unresolved upstream term mismatch as a publishability risk.

## Publishability Policy (v1)

| Artifact Type              | Default Policy         | Condition                                                     |
| -------------------------- | ---------------------- | ------------------------------------------------------------- |
| Code/config/scripts        | Public allowed         | Include dataset citations and no bundled raw dataset files    |
| Training logs/metrics      | Public allowed         | Remove private tokens and environment secrets                 |
| Processed cache samples    | Cautious               | Only if derived outputs do not violate upstream dataset terms |
| Full checkpoints/weights   | Conditional            | Publish only after upstream license/provenance review passes  |
| Raw dataset redistribution | Not allowed by default | Only if explicit upstream terms allow redistribution          |

## Compliance Checklist Before Public Release

1. Confirm selected Kaggle dataset slug and version in run log.
2. Record upstream source citation (ModelNet / ShapeNetPart references).
3. Verify that no raw dataset files are included in artifacts.
4. Perform license chain review for any planned checkpoint publication.
5. If ambiguity remains, publish code + metrics only and defer weights.

## Required Citations (Baseline)

- Wu et al., 3D ShapeNets (CVPR 2015) for ModelNet benchmark usage.
- Yi et al., A Scalable Active Framework for Region Annotation in 3D Shape Collections (SIGGRAPH Asia 2016) for ShapeNetPart benchmark usage.
- ShapeNetCore reference (Chang et al., 2015) when ShapeNet-derived data is used.
