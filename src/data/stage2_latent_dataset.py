"""Manifest-backed dataset for Stage 2 latent smoke training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
from torch.utils.data import Dataset


class Stage2LatentDatasetError(RuntimeError):
    """Raised when Stage 2 latent manifests or artifacts are invalid."""


def default_stage2_latent_root(output_root: str, schema_version: str) -> Path:
    return Path(output_root) / "cache" / "stage2_latents" / schema_version


def default_stage2_manifest_path(output_root: str, schema_version: str, split: str) -> Path:
    manifest_dir = default_stage2_latent_root(output_root, schema_version) / "manifests"
    if split not in {"train", "test", "both"}:
        raise Stage2LatentDatasetError(
            f"Invalid latent split: {split!r}. Expected train, test, or both."
        )
    if split == "both":
        raise Stage2LatentDatasetError(
            "A single manifest path cannot be derived for split='both'. Use two loaders or "
            "Stage2LatentDataset.from_stage2_configs()."
        )
    return manifest_dir / f"latent_manifest_{split}.jsonl"


@dataclass(frozen=True)
class Stage2LatentDatasetConfig:
    """Configuration for manifest-backed latent loading."""

    output_root: str
    schema_version: str = "stage2-latent-v1"
    split: str = "train"
    manifest_path: Optional[str] = None
    sample_limit: Optional[int] = None
    expected_token_length: Optional[int] = None
    expected_token_dim: Optional[int] = None
    expected_latent_dim: Optional[int] = None
    strict_schema: bool = True

    @staticmethod
    def from_stage2_configs(
        data_cfg: Mapping[str, Any],
        split: str,
        sample_limit: Optional[int] = None,
    ) -> "Stage2LatentDatasetConfig":
        paths = data_cfg.get("paths", {})
        latent_loading = data_cfg.get("latent_loading", {})
        output_root = str(paths.get("output_root") or "").strip() or "/kaggle/working"
        schema_version = str(latent_loading.get("schema_version") or "stage2-latent-v1").strip()
        manifest_override = latent_loading.get(f"{split}_manifest") or latent_loading.get("manifest_path")

        token_length = latent_loading.get("token_length")
        token_dim = latent_loading.get("token_dim")
        latent_dim = latent_loading.get("latent_dim")

        return Stage2LatentDatasetConfig(
            output_root=output_root,
            schema_version=schema_version,
            split=split,
            manifest_path=str(manifest_override).strip() if manifest_override else None,
            sample_limit=sample_limit,
            expected_token_length=int(token_length) if token_length is not None else None,
            expected_token_dim=int(token_dim) if token_dim is not None else None,
            expected_latent_dim=int(latent_dim) if latent_dim is not None else None,
            strict_schema=bool(latent_loading.get("strict_schema", True)),
        )


@dataclass(frozen=True)
class Stage2LatentRecord:
    """Single latent artifact manifest row."""

    sample_uid: str
    split: str
    class_id: int
    class_name: str
    source_key: str
    artifact_path: str
    token_shape: Sequence[int]
    latent_dim: int
    dtype: str
    schema_version: str
    checkpoint_path: str
    checkpoint_global_step: int


class Stage2LatentDataset(Dataset):
    """Loads Stage 2 latent artifacts from split-aware manifests."""

    def __init__(self, config: Stage2LatentDatasetConfig):
        self.config = config
        self.output_root = Path(config.output_root)
        self.latent_root = default_stage2_latent_root(config.output_root, config.schema_version)

        if config.manifest_path:
            manifest_paths = [Path(config.manifest_path)]
        elif config.split == "both":
            manifest_paths = [
                default_stage2_manifest_path(config.output_root, config.schema_version, "train"),
                default_stage2_manifest_path(config.output_root, config.schema_version, "test"),
            ]
        else:
            manifest_paths = [
                default_stage2_manifest_path(config.output_root, config.schema_version, config.split)
            ]

        records: List[Stage2LatentRecord] = []
        for manifest_path in manifest_paths:
            records.extend(self._load_manifest(manifest_path))

        if not records:
            raise Stage2LatentDatasetError(
                "No latent artifacts found in manifest(s). Ensure Task 4.1 latent extraction ran "
                f"successfully under {self.latent_root}."
            )

        if config.sample_limit is not None and config.sample_limit > 0:
            records = records[: int(config.sample_limit)]

        self.records = records

    @classmethod
    def from_stage2_configs(
        cls,
        data_cfg: Mapping[str, Any],
        split: str,
        sample_limit: Optional[int] = None,
    ) -> "Stage2LatentDataset":
        return cls(Stage2LatentDatasetConfig.from_stage2_configs(data_cfg, split, sample_limit))

    def _load_manifest(self, manifest_path: Path) -> List[Stage2LatentRecord]:
        if not manifest_path.exists():
            raise Stage2LatentDatasetError(
                "Stage 2 latent manifest is missing. Run scripts/build_latent_dataset.py first. "
                f"Missing path: {manifest_path}"
            )

        records: List[Stage2LatentRecord] = []
        for line_number, raw_line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue

            row = json.loads(line)
            token_shape = row.get("token_shape") or []
            if not isinstance(token_shape, list) or len(token_shape) != 2:
                raise Stage2LatentDatasetError(
                    f"Malformed token_shape in {manifest_path} at line {line_number}."
                )

            schema_version = str(row.get("schema_version") or "").strip()
            if self.config.strict_schema and schema_version != self.config.schema_version:
                raise Stage2LatentDatasetError(
                    "Schema version mismatch between manifest and loader config. "
                    f"Expected {self.config.schema_version!r}, received {schema_version!r} "
                    f"for {manifest_path} line {line_number}."
                )

            if self.config.expected_token_length is not None and int(token_shape[0]) != self.config.expected_token_length:
                raise Stage2LatentDatasetError(
                    "Token length mismatch between manifest and config. "
                    f"Expected {self.config.expected_token_length}, got {int(token_shape[0])}."
                )
            if self.config.expected_token_dim is not None and int(token_shape[1]) != self.config.expected_token_dim:
                raise Stage2LatentDatasetError(
                    "Token dim mismatch between manifest and config. "
                    f"Expected {self.config.expected_token_dim}, got {int(token_shape[1])}."
                )
            if self.config.expected_latent_dim is not None and int(row.get("latent_dim", -1)) != self.config.expected_latent_dim:
                raise Stage2LatentDatasetError(
                    "Latent dim mismatch between manifest and config. "
                    f"Expected {self.config.expected_latent_dim}, got {int(row.get('latent_dim', -1))}."
                )

            records.append(
                Stage2LatentRecord(
                    sample_uid=str(row.get("sample_uid", "")),
                    split=str(row.get("split", "")),
                    class_id=int(row.get("class_id", -1)),
                    class_name=str(row.get("class_name", "")),
                    source_key=str(row.get("source_key", "")),
                    artifact_path=str(row.get("artifact_path", "")),
                    token_shape=[int(token_shape[0]), int(token_shape[1])],
                    latent_dim=int(row.get("latent_dim", -1)),
                    dtype=str(row.get("dtype", "float32")),
                    schema_version=schema_version,
                    checkpoint_path=str(row.get("checkpoint_path", "")),
                    checkpoint_global_step=int(row.get("checkpoint_global_step", 0)),
                )
            )

        return records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        artifact_path = Path(record.artifact_path)
        if not artifact_path.exists():
            raise Stage2LatentDatasetError(f"Latent artifact missing: {artifact_path}")

        payload = torch.load(artifact_path, map_location="cpu")
        if not isinstance(payload, Mapping):
            raise Stage2LatentDatasetError(f"Latent artifact payload is not a mapping: {artifact_path}")

        tokens = payload.get("tokens")
        mu = payload.get("mu")
        if not isinstance(tokens, torch.Tensor) or not isinstance(mu, torch.Tensor):
            raise Stage2LatentDatasetError(f"Latent artifact missing tensor fields in {artifact_path}.")

        expected_shape = (int(record.token_shape[0]), int(record.token_shape[1]))
        if tokens.ndim != 2 or tuple(tokens.shape) != expected_shape:
            raise Stage2LatentDatasetError(
                "Latent token tensor shape mismatch. "
                f"Expected {expected_shape}, got {tuple(tokens.shape)} for {artifact_path}."
            )
        if mu.ndim != 1 or int(mu.shape[0]) != int(record.latent_dim):
            raise Stage2LatentDatasetError(
                "Latent mu tensor shape mismatch. "
                f"Expected ({record.latent_dim},), got {tuple(mu.shape)} for {artifact_path}."
            )

        return {
            "tokens": tokens.to(dtype=torch.float32),
            "mu": mu.to(dtype=torch.float32),
            "class_id": torch.tensor(int(record.class_id), dtype=torch.long),
            "class_name": record.class_name,
            "split": record.split,
            "sample_uid": record.sample_uid,
            "source_key": record.source_key,
            "artifact_path": str(artifact_path),
            "checkpoint_path": record.checkpoint_path,
            "checkpoint_global_step": torch.tensor(int(record.checkpoint_global_step), dtype=torch.long),
        }

    @staticmethod
    def collate_fn(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        tokens = torch.stack([item["tokens"] for item in batch], dim=0)
        mu = torch.stack([item["mu"] for item in batch], dim=0)
        class_id = torch.stack([item["class_id"] for item in batch], dim=0)
        checkpoint_global_step = torch.stack([item["checkpoint_global_step"] for item in batch], dim=0)
        return {
            "tokens": tokens,
            "mu": mu,
            "class_id": class_id,
            "class_name": [item["class_name"] for item in batch],
            "split": [item["split"] for item in batch],
            "sample_uid": [item["sample_uid"] for item in batch],
            "source_key": [item["source_key"] for item in batch],
            "artifact_path": [item["artifact_path"] for item in batch],
            "checkpoint_path": [item["checkpoint_path"] for item in batch],
            "checkpoint_global_step": checkpoint_global_step,
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "count": len(self.records),
            "schema_version": self.config.schema_version,
            "split": self.config.split,
            "manifest_path": self.config.manifest_path,
            "output_root": str(self.output_root),
            "latent_root": str(self.latent_root),
            "sample_limit": self.config.sample_limit,
        }