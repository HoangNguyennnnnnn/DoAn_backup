"""Build Stage 2 latent dataset artifacts from Stage 1 outputs."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import yaml

from src.data.dataset_adapter import AdapterConfig, KaggleDatasetAdapter, SampleRecord
from src.data.mesh_to_feature import MeshFeatureConfig, MeshToFeatureBuilder
from src.data.off_to_obj_converter import ConverterConfig, OffToObjConverter
from src.models import ShapeSCVAE, ShapeSCVAEError, ShapePathContractConfig


class LatentDatasetError(RuntimeError):
    """Raised when latent extraction/build pipeline fails."""


@dataclass(frozen=True)
class LatentBuildConfig:
    """Runtime config for latent dataset extraction/build."""

    dataset_root: str
    output_root: str
    checkpoint_path: Optional[str] = None
    checkpoint_preference: Tuple[str, ...] = ("latest_step", "best", "interrupt", "latest")
    split: str = "both"
    batch_size: int = 16
    device: str = "cpu"
    latent_schema_version: str = "stage2-latent-v1"
    latent_dtype: str = "float32"
    verify_hash: bool = False
    enforce_kaggle_paths: bool = False

    @staticmethod
    def from_stage1_data_config(data_cfg: Mapping[str, Any]) -> "LatentBuildConfig":
        paths = data_cfg.get("paths", {})
        dataset_root = str(paths.get("dataset_root") or "").strip()
        output_root = str(paths.get("output_root") or "").strip()
        if not dataset_root or "${" in dataset_root:
            dataset_root = "/kaggle/input/modelnet40-princeton-3d-object-dataset"
        if not output_root or "${" in output_root:
            output_root = "/kaggle/working"

        return LatentBuildConfig(
            dataset_root=dataset_root,
            output_root=output_root,
        )


@dataclass
class LatentBuildSummary:
    total_records: int = 0
    generated: int = 0
    skipped: int = 0
    failed: int = 0
    elapsed_seconds: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_records": self.total_records,
            "generated": self.generated,
            "skipped": self.skipped,
            "failed": self.failed,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


def _expand_env(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def _load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise LatentDatasetError(f"Config must be YAML object: {path}")
    return _expand_env(payload)


def _normalize_key(path_like: str) -> str:
    return path_like.replace("\\", "/")


def _sample_uid(source_key: str, schema_version: str) -> str:
    base = f"{schema_version}:{source_key}".encode("utf-8")
    return hashlib.sha1(base).hexdigest()[:16]


def _dtype_from_name(dtype_name: str) -> torch.dtype:
    name = dtype_name.lower().strip()
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise LatentDatasetError(f"Unsupported latent dtype: {dtype_name}")


def resolve_checkpoint_path(
    output_root: Path,
    explicit_path: Optional[Path],
    preference: Sequence[str],
) -> Path:
    if explicit_path is not None:
        if explicit_path.exists() and explicit_path.is_file():
            return explicit_path
        raise LatentDatasetError(f"Provided checkpoint does not exist: {explicit_path}")

    ckpt_dir = output_root / "checkpoints"
    names = {
        "latest_step": "latest_step.ckpt",
        "best": "best.ckpt",
        "interrupt": "interrupt.ckpt",
        "latest": "latest.ckpt",
    }
    for key in preference:
        filename = names.get(key)
        if not filename:
            continue
        candidate = ckpt_dir / filename
        if candidate.exists() and candidate.is_file():
            return candidate

    searched = [str((ckpt_dir / names[k])) for k in preference if k in names]
    raise LatentDatasetError(
        "No compatible Stage 1 checkpoint found using configured precedence. "
        f"Searched: {searched}."
    )


def _checkpoint_configs(
    checkpoint_payload: Mapping[str, Any],
    fallback_train_cfg_path: Optional[Path],
    fallback_data_cfg_path: Optional[Path],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    train_cfg = checkpoint_payload.get("train_config")
    data_cfg = checkpoint_payload.get("data_config")

    if isinstance(train_cfg, Mapping) and isinstance(data_cfg, Mapping):
        return dict(train_cfg), dict(data_cfg)

    if fallback_train_cfg_path is None or fallback_data_cfg_path is None:
        raise LatentDatasetError(
            "Checkpoint payload missing train_config/data_config and no fallback config paths were provided."
        )

    return _load_yaml(fallback_train_cfg_path), _load_yaml(fallback_data_cfg_path)


def _build_feature_cache(
    records: Sequence[SampleRecord],
    data_cfg: Mapping[str, Any],
    dataset_root: Path,
    output_root: Path,
) -> Mapping[str, Mapping[str, Any]]:
    conv_cfg = ConverterConfig.from_data_config(data_cfg)
    conv_cfg = ConverterConfig(
        input_root=str(dataset_root),
        output_root=str(output_root),
        cache_index_path=str(output_root / "cache" / "conversion_cache_index.json"),
        overwrite=False,
        incremental=True,
        verify_hash=False,
        enforce_kaggle_paths=False,
    )
    OffToObjConverter(conv_cfg).convert_from_records(records)

    feat_cfg = MeshFeatureConfig.from_data_config(data_cfg)
    feat_cfg = MeshFeatureConfig(
        dataset_root=str(dataset_root),
        output_root=str(output_root),
        cache_dir=str(output_root / "cache"),
        resolution=feat_cfg.resolution,
        dtype=feat_cfg.dtype,
        normalize_meshes=feat_cfg.normalize_meshes,
        center_objects=feat_cfg.center_objects,
        scale_to_unit_box=feat_cfg.scale_to_unit_box,
        samples_per_mesh=feat_cfg.samples_per_mesh,
        prefer_obj=True,
        overwrite=False,
        incremental=True,
        verify_hash=False,
        enforce_kaggle_paths=False,
        schema_version=feat_cfg.schema_version,
    )
    builder = MeshToFeatureBuilder(feat_cfg)
    builder.build_from_records(records, seed=int(data_cfg.get("seed", 42)))
    return builder.index.tensor_refs


def _read_ovoxel_tensor(
    tensor_refs: Mapping[str, Mapping[str, Any]],
    record: SampleRecord,
) -> torch.Tensor:
    key = _normalize_key(record.relative_path)
    ref = tensor_refs.get(key)
    if not isinstance(ref, Mapping):
        raise LatentDatasetError(
            f"Tensor reference missing for record key={key}. "
            "Run feature builder before latent extraction."
        )

    tensor_path = Path(str(ref.get("tensor_path", "")))
    if not tensor_path.exists():
        raise LatentDatasetError(f"OVoxel tensor file missing for key={key}: {tensor_path}")

    tensor = torch.load(tensor_path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise LatentDatasetError(f"Tensor payload is not torch.Tensor for key={key}: {tensor_path}")
    if tensor.ndim != 4:
        raise LatentDatasetError(
            f"OVoxel tensor rank mismatch for key={key}. Expected (C,R,R,R), got {tuple(tensor.shape)}"
        )
    return tensor


def _validate_contract(
    train_cfg: Mapping[str, Any],
    data_cfg: Mapping[str, Any],
) -> ShapePathContractConfig:
    contract = ShapePathContractConfig.from_stage1_configs(train_cfg, data_cfg)
    if contract.token_length <= 0 or contract.token_dim <= 0:
        raise LatentDatasetError(
            "Invalid latent token contract. token_length and token_dim must be > 0."
        )
    return contract


class LatentDatasetBuilder:
    """Extracts latent token dataset consumable by Stage 2 smoke training."""

    def __init__(self, config: LatentBuildConfig):
        self.config = config
        self.dataset_root = Path(config.dataset_root)
        self.output_root = Path(config.output_root)
        self.latent_root = self.output_root / "cache" / "stage2_latents" / config.latent_schema_version
        self.manifest_dir = self.latent_root / "manifests"

    def build(
        self,
        fallback_train_cfg_path: Optional[Path] = None,
        fallback_data_cfg_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        started = time.time()
        self.manifest_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = resolve_checkpoint_path(
            output_root=self.output_root,
            explicit_path=Path(self.config.checkpoint_path) if self.config.checkpoint_path else None,
            preference=self.config.checkpoint_preference,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" not in checkpoint:
            raise LatentDatasetError(
                f"Checkpoint payload missing model_state_dict: {checkpoint_path}"
            )

        train_cfg, data_cfg = _checkpoint_configs(
            checkpoint,
            fallback_train_cfg_path=fallback_train_cfg_path,
            fallback_data_cfg_path=fallback_data_cfg_path,
        )
        contract = _validate_contract(train_cfg, data_cfg)

        adapter_cfg = AdapterConfig.from_data_config(data_cfg)
        adapter_cfg = AdapterConfig(
            dataset_root=str(self.dataset_root),
            kaggle_slug=adapter_cfg.kaggle_slug,
            split=self.config.split,
            seed=adapter_cfg.seed,
            strict_split=adapter_cfg.strict_split,
            file_extension=adapter_cfg.file_extension,
            enforce_kaggle_input_prefix=self.config.enforce_kaggle_paths,
        )
        records = KaggleDatasetAdapter(adapter_cfg).load()
        if not records:
            raise LatentDatasetError("No sample records discovered for latent extraction.")

        tensor_refs = _build_feature_cache(
            records=records,
            data_cfg=data_cfg,
            dataset_root=self.dataset_root,
            output_root=self.output_root,
        )

        model = ShapeSCVAE.from_stage1_configs(train_cfg, data_cfg)
        try:
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        except (RuntimeError, ShapeSCVAEError) as exc:
            raise LatentDatasetError(
                f"Checkpoint/model incompatibility while loading state_dict: {exc}"
            ) from exc

        device_name = self.config.device
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)
        model = model.to(device)
        model.eval()

        dtype = _dtype_from_name(self.config.latent_dtype)
        summary = LatentBuildSummary(total_records=len(records))

        manifest_all = self.manifest_dir / "latent_manifest.jsonl"
        manifest_train = self.manifest_dir / "latent_manifest_train.jsonl"
        manifest_test = self.manifest_dir / "latent_manifest_test.jsonl"
        for path in (manifest_all, manifest_train, manifest_test):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")

        lines_all: List[str] = []
        lines_train: List[str] = []
        lines_test: List[str] = []

        for start in range(0, len(records), self.config.batch_size):
            batch_records = records[start : start + self.config.batch_size]

            voxels: List[torch.Tensor] = []
            ok_records: List[SampleRecord] = []
            for record in batch_records:
                try:
                    ovx = _read_ovoxel_tensor(tensor_refs, record)
                    voxels.append(ovx)
                    ok_records.append(record)
                except Exception:
                    summary.failed += 1

            if not voxels:
                continue

            batch = torch.stack(voxels, dim=0).to(device=device)
            try:
                with torch.no_grad():
                    encoded = model.encode(batch, sample=False)
            except Exception as exc:
                raise LatentDatasetError(
                    f"Encoder pass failed for latent extraction batch {start}:{start + len(batch_records)}: {exc}"
                ) from exc

            tokens = encoded.get("tokens")
            mu = encoded.get("mu")
            if not isinstance(tokens, torch.Tensor) or not isinstance(mu, torch.Tensor):
                raise LatentDatasetError(
                    "Encoder output missing required tensors ('tokens', 'mu')."
                )
            if tokens.ndim != 3:
                raise LatentDatasetError(
                    f"Latent token rank mismatch. Expected (B,T,D), got {tuple(tokens.shape)}"
                )
            if tokens.shape[1] != contract.token_length or tokens.shape[2] != contract.token_dim:
                raise LatentDatasetError(
                    "Latent token shape mismatch against shape contract. "
                    f"Expected (*,{contract.token_length},{contract.token_dim}), got {tuple(tokens.shape)}"
                )

            tokens = tokens.to("cpu", dtype=dtype)
            mu = mu.to("cpu", dtype=dtype)

            for idx, record in enumerate(ok_records):
                source_key = _normalize_key(record.relative_path)
                sample_uid = _sample_uid(source_key, self.config.latent_schema_version)
                split_dir = self.latent_root / record.split
                split_dir.mkdir(parents=True, exist_ok=True)

                artifact_path = split_dir / f"{sample_uid}.pt"
                payload = {
                    "tokens": tokens[idx],
                    "mu": mu[idx],
                    "class_id": int(record.class_id),
                    "class_name": str(record.class_name),
                    "split": str(record.split),
                    "sample_uid": sample_uid,
                    "source_key": source_key,
                    "schema_version": self.config.latent_schema_version,
                }
                if artifact_path.exists() and not self.config.verify_hash:
                    summary.skipped += 1
                else:
                    torch.save(payload, artifact_path)
                    if not artifact_path.exists() or artifact_path.stat().st_size <= 0:
                        raise LatentDatasetError(f"Failed to save latent artifact: {artifact_path}")
                    summary.generated += 1

                manifest_entry = {
                    "sample_uid": sample_uid,
                    "split": str(record.split),
                    "class_id": int(record.class_id),
                    "class_name": str(record.class_name),
                    "source_key": source_key,
                    "artifact_path": str(artifact_path),
                    "token_shape": [int(contract.token_length), int(contract.token_dim)],
                    "latent_dim": int(contract.latent_dim),
                    "dtype": self.config.latent_dtype,
                    "schema_version": self.config.latent_schema_version,
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_global_step": int(checkpoint.get("global_step", 0)),
                }
                line = json.dumps(manifest_entry)
                lines_all.append(line)
                if record.split == "train":
                    lines_train.append(line)
                elif record.split == "test":
                    lines_test.append(line)

        self._write_manifest(manifest_all, lines_all)
        self._write_manifest(manifest_train, lines_train)
        self._write_manifest(manifest_test, lines_test)

        self._validate_manifests(manifest_all, records)

        summary.elapsed_seconds = time.time() - started
        report = {
            "summary": summary.as_dict(),
            "checkpoint_path": str(checkpoint_path),
            "manifest_all": str(manifest_all),
            "manifest_train": str(manifest_train),
            "manifest_test": str(manifest_test),
            "token_contract": {
                "token_length": contract.token_length,
                "token_dim": contract.token_dim,
                "latent_dim": contract.latent_dim,
                "schema_version": self.config.latent_schema_version,
                "dtype": self.config.latent_dtype,
            },
        }

        report_path = self.latent_root / "latent_build_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["report_path"] = str(report_path)
        return report

    @staticmethod
    def _write_manifest(path: Path, lines: Sequence[str]) -> None:
        payload = "\n".join(lines)
        if payload:
            payload += "\n"
        path.write_text(payload, encoding="utf-8")

    @staticmethod
    def _validate_manifests(manifest_all: Path, records: Sequence[SampleRecord]) -> None:
        lines = [line for line in manifest_all.read_text(encoding="utf-8").splitlines() if line.strip()]
        seen: set[str] = set()
        split_counts: Dict[str, int] = {}

        for line in lines:
            row = json.loads(line)
            sample_uid = str(row.get("sample_uid", ""))
            if not sample_uid:
                raise LatentDatasetError("Manifest row missing sample_uid.")
            if sample_uid in seen:
                raise LatentDatasetError(f"Duplicate sample_uid in manifest: {sample_uid}")
            seen.add(sample_uid)

            split = str(row.get("split", ""))
            split_counts[split] = split_counts.get(split, 0) + 1

            artifact = Path(str(row.get("artifact_path", "")))
            if not artifact.exists():
                raise LatentDatasetError(f"Manifest artifact missing: {artifact}")

            token_shape = row.get("token_shape")
            if token_shape is None or len(token_shape) != 2:
                raise LatentDatasetError(f"Manifest token_shape malformed for sample_uid={sample_uid}")

        expected_splits = {item.split for item in records}
        missing = [split for split in expected_splits if split_counts.get(split, 0) == 0]
        if missing:
            raise LatentDatasetError(
                "Split/index inconsistency detected. Missing manifest records for splits: "
                f"{missing}"
            )


def build_latent_dataset(
    config: LatentBuildConfig,
    fallback_train_cfg_path: Optional[Path] = None,
    fallback_data_cfg_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Convenience helper for latent dataset generation."""
    builder = LatentDatasetBuilder(config)
    return builder.build(
        fallback_train_cfg_path=fallback_train_cfg_path,
        fallback_data_cfg_path=fallback_data_cfg_path,
    )
