"""Mesh to OVoxel feature construction with cache-safe artifact persistence."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import trimesh

from src.data.dataset_adapter import SampleRecord
from src.data.off_to_obj_converter import CacheIndex, ConversionError


LOGGER = logging.getLogger(__name__)


class FeatureConstructionError(RuntimeError):
    """Raised when mesh-to-feature conversion or artifact validation fails."""


@dataclass(frozen=True)
class MeshFeatureConfig:
    """Configuration for OVoxel feature generation."""

    dataset_root: str
    output_root: str
    cache_dir: str
    resolution: int = 32
    dtype: str = "float32"
    normalize_meshes: bool = True
    center_objects: bool = True
    scale_to_unit_box: bool = True
    samples_per_mesh: int = 5000
    prefer_obj: bool = True
    overwrite: bool = False
    incremental: bool = True
    verify_hash: bool = False
    enforce_kaggle_paths: bool = False
    schema_version: str = "ovoxel-v1"

    @staticmethod
    def from_data_config(data_cfg: Mapping[str, Any]) -> "MeshFeatureConfig":
        paths = data_cfg.get("paths", {})
        prep = data_cfg.get("preprocessing", {})

        dataset_root = str(paths.get("dataset_root") or "").strip()
        output_root = str(paths.get("output_root") or "").strip()
        cache_dir = str(paths.get("cache_dir") or "").strip()
        if not dataset_root or "${" in dataset_root:
            raise FeatureConstructionError(
                "paths.dataset_root must resolve to a concrete path before feature generation."
            )
        if not output_root or "${" in output_root:
            raise FeatureConstructionError(
                "paths.output_root must resolve to a concrete path before feature generation."
            )
        if not cache_dir or "${" in cache_dir:
            cache_dir = str(Path(output_root) / "cache")

        dtype = str(prep.get("voxel_dtype", "float32")).lower().strip()
        if dtype not in {"float16", "float32"}:
            raise FeatureConstructionError(
                "preprocessing.voxel_dtype must be one of: float16, float32. "
                f"Received: {dtype!r}"
            )

        return MeshFeatureConfig(
            dataset_root=dataset_root,
            output_root=output_root,
            cache_dir=cache_dir,
            resolution=int(prep.get("target_resolution", 32)),
            dtype=dtype,
            normalize_meshes=bool(prep.get("normalize_meshes", True)),
            center_objects=bool(prep.get("center_objects", True)),
            scale_to_unit_box=bool(prep.get("scale_to_unit_box", True)),
        )


@dataclass
class FeatureSummary:
    total_scanned: int = 0
    generated: int = 0
    skipped: int = 0
    failed: int = 0
    elapsed_seconds: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_scanned": self.total_scanned,
            "generated": self.generated,
            "skipped": self.skipped,
            "failed": self.failed,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _assert_kaggle_path(path: Path, role: str) -> None:
    if not str(path).startswith("/kaggle/"):
        raise FeatureConstructionError(
            f"{role} must be under /kaggle for Kaggle-compatible execution. Got: {path}"
        )


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _source_state(path: Path, verify_hash: bool) -> Dict[str, Any]:
    stat = path.stat()
    state: Dict[str, Any] = {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if verify_hash:
        state["sha256"] = _sha256_file(path)
    return state


def _torch_dtype(dtype: str) -> torch.dtype:
    if dtype == "float16":
        return torch.float16
    if dtype == "float32":
        return torch.float32
    raise FeatureConstructionError(f"Unsupported voxel dtype: {dtype}")


def _normalize_vertices(
    vertices: np.ndarray,
    center_objects: bool,
    scale_to_unit_box: bool,
) -> np.ndarray:
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise FeatureConstructionError(
            f"Invalid vertex tensor shape. Expected (N, 3), got {vertices.shape}."
        )

    v = vertices.astype(np.float64, copy=True)
    if center_objects:
        centroid = v.mean(axis=0)
        v = v - centroid

    if scale_to_unit_box:
        mins = v.min(axis=0)
        maxs = v.max(axis=0)
        extent = maxs - mins
        max_extent = float(np.max(extent))
        if max_extent <= 0.0:
            raise FeatureConstructionError(
                "Degenerate mesh after centering/scaling: zero extent on all axes."
            )
        v = v / max_extent

    return np.clip(v, -0.5, 0.5)


def _load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    if not mesh_path.exists() or not mesh_path.is_file():
        raise FeatureConstructionError(f"Mesh file does not exist: {mesh_path}")

    try:
        loaded = trimesh.load(str(mesh_path), force="mesh", process=False)
    except Exception as exc:
        raise FeatureConstructionError(
            f"Failed to parse mesh file {mesh_path}: {exc}"
        ) from exc

    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise FeatureConstructionError(f"Mesh scene has no geometry: {mesh_path}")
        mesh = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    else:
        mesh = loaded

    if not isinstance(mesh, trimesh.Trimesh):
        raise FeatureConstructionError(f"Unsupported mesh type for {mesh_path}: {type(mesh)}")
    if mesh.vertices is None or len(mesh.vertices) == 0:
        raise FeatureConstructionError(f"Mesh has no vertices: {mesh_path}")
    if mesh.faces is None or len(mesh.faces) == 0:
        raise FeatureConstructionError(f"Mesh has no faces: {mesh_path}")

    return mesh


def _mesh_to_ovoxel(
    mesh: trimesh.Trimesh,
    resolution: int,
    samples_per_mesh: int,
    normalize_meshes: bool,
    center_objects: bool,
    scale_to_unit_box: bool,
    rng_seed: int,
) -> np.ndarray:
    if resolution <= 0:
        raise FeatureConstructionError(f"resolution must be > 0, got {resolution}")

    vertices = np.asarray(mesh.vertices)
    if normalize_meshes:
        vertices = _normalize_vertices(
            vertices,
            center_objects=center_objects,
            scale_to_unit_box=scale_to_unit_box,
        )

    rng_state = np.random.get_state()
    try:
        np.random.seed(rng_seed)
        sample_count = max(samples_per_mesh, len(vertices) * 4)
        sampled, _ = trimesh.sample.sample_surface(mesh, sample_count)
    except Exception:
        sampled = vertices
    finally:
        np.random.set_state(rng_state)

    if normalize_meshes:
        sampled = _normalize_vertices(
            sampled,
            center_objects=center_objects,
            scale_to_unit_box=scale_to_unit_box,
        )

    points = np.concatenate([vertices, sampled], axis=0)
    points = np.clip(points + 0.5, 0.0, 1.0)

    grid = np.zeros((resolution, resolution, resolution), dtype=np.uint8)
    idx = np.floor(points * (resolution - 1)).astype(np.int64)
    idx = np.clip(idx, 0, resolution - 1)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = 1

    return grid


def _build_sanity_stats(tensor: torch.Tensor) -> Dict[str, Any]:
    if tensor.ndim != 4:
        raise FeatureConstructionError(
            f"Invalid tensor shape: {tuple(tensor.shape)}. Expected (1, R, R, R)."
        )

    nonzero = int(torch.count_nonzero(tensor).item())
    total = int(tensor.numel())
    occupancy = float(nonzero / total) if total else 0.0
    stats = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
        "nonzero": nonzero,
        "occupancy_ratio": occupancy,
        "checks": {
            "shape_ok": bool(tensor.shape[0] == 1 and tensor.shape[1] == tensor.shape[2] == tensor.shape[3]),
            "nonzero_ok": bool(nonzero > 0),
            "range_ok": bool(0.0 <= float(tensor.min()) and float(tensor.max()) <= 1.0),
        },
    }
    return stats


class MeshToFeatureBuilder:
    """Build OVoxel feature artifacts from mesh records."""

    def __init__(self, config: MeshFeatureConfig):
        self.config = config
        self.dataset_root = Path(config.dataset_root)
        self.output_root = Path(config.output_root)
        self.cache_dir = Path(config.cache_dir)
        self.feature_root = self.cache_dir / "features" / "ovoxel" / config.schema_version
        self.index = CacheIndex(self.cache_dir / "conversion_cache_index.json")
        self.index.load()

        if self.config.enforce_kaggle_paths:
            _assert_kaggle_path(self.dataset_root, "dataset_root")
            _assert_kaggle_path(self.output_root, "output_root")

    @classmethod
    def from_data_config(cls, data_cfg: Mapping[str, Any]) -> "MeshToFeatureBuilder":
        return cls(MeshFeatureConfig.from_data_config(data_cfg))

    def build_from_records(
        self,
        records: Sequence[Union[SampleRecord, Mapping[str, Any]]],
        seed: int = 42,
    ) -> FeatureSummary:
        started = time.time()
        summary = FeatureSummary(total_scanned=len(records))

        for item in records:
            try:
                record = self._normalize_record(item)
                source_mesh = self._resolve_source_mesh(record)
                source_state = _source_state(source_mesh, verify_hash=self.config.verify_hash)

                rel = Path(str(record["relative_path"]))
                source_key = str(rel).replace("\\", "/")
                feature_path = self.feature_root / rel.with_suffix(".pt")
                meta_path = feature_path.with_suffix(".json")

                if self._can_skip(source_key, source_state, feature_path, meta_path):
                    if source_key not in self.index.tensor_refs:
                        self.index.upsert_tensor_ref(source_key, feature_path)
                    summary.skipped += 1
                    continue

                mesh = _load_mesh(source_mesh)
                ovx = _mesh_to_ovoxel(
                    mesh=mesh,
                    resolution=self.config.resolution,
                    samples_per_mesh=self.config.samples_per_mesh,
                    normalize_meshes=self.config.normalize_meshes,
                    center_objects=self.config.center_objects,
                    scale_to_unit_box=self.config.scale_to_unit_box,
                    rng_seed=seed + int(record.get("index", 0)),
                )

                tensor = torch.from_numpy(ovx).unsqueeze(0).to(dtype=_torch_dtype(self.config.dtype))
                stats = _build_sanity_stats(tensor)
                if not all(stats["checks"].values()):
                    raise FeatureConstructionError(
                        "Sanity checks failed for generated tensor "
                        f"({source_mesh}): {stats['checks']}"
                    )

                _ensure_parent(feature_path)
                torch.save(tensor, feature_path)
                if not feature_path.exists() or feature_path.stat().st_size <= 0:
                    raise FeatureConstructionError(f"Failed to persist tensor artifact: {feature_path}")

                metadata = {
                    "schema_version": self.config.schema_version,
                    "created_unix": int(time.time()),
                    "source_key": source_key,
                    "source_mesh": str(source_mesh),
                    "source_state": source_state,
                    "feature_path": str(feature_path),
                    "dtype": self.config.dtype,
                    "resolution": self.config.resolution,
                    "sample_record": record,
                    "sanity": stats,
                }
                meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

                self.index.tensor_refs[source_key] = {
                    "tensor_path": str(feature_path),
                    "meta_path": str(meta_path),
                    "source_size": int(source_state["size"]),
                    "source_mtime_ns": int(source_state["mtime_ns"]),
                    "source_sha256": source_state.get("sha256"),
                    "schema_version": self.config.schema_version,
                    "dtype": self.config.dtype,
                    "resolution": self.config.resolution,
                    "updated_unix": int(time.time()),
                }
                summary.generated += 1
            except Exception as exc:
                summary.failed += 1
                LOGGER.error("Mesh->OVoxel build failed: %s", exc)

        self.index.save()
        consistency = self.index.validate_consistency()
        if consistency:
            raise ConversionError(
                "Cache index consistency validation failed:\n- " + "\n- ".join(consistency)
            )

        summary.elapsed_seconds = time.time() - started
        self._log_summary(summary)
        return summary

    def _normalize_record(
        self,
        item: Union[SampleRecord, Mapping[str, Any]],
    ) -> Dict[str, Any]:
        if isinstance(item, SampleRecord):
            return item.as_dict()
        if isinstance(item, Mapping):
            required = [
                "index",
                "split",
                "class_id",
                "class_name",
                "sample_id",
                "relative_path",
                "absolute_path",
                "dataset_slug",
                "provenance",
            ]
            missing = [key for key in required if key not in item]
            if missing:
                raise FeatureConstructionError(
                    "Sample record schema mismatch for Task 2.1 compatibility. Missing fields: "
                    f"{missing}"
                )
            return dict(item)
        raise FeatureConstructionError(
            "Unsupported record input type. Expected SampleRecord or mapping."
        )

    def _resolve_source_mesh(self, record: Mapping[str, Any]) -> Path:
        absolute_off = Path(str(record["absolute_path"]))
        rel = Path(str(record["relative_path"]))

        obj_candidate = self.output_root / rel.with_suffix(".obj")
        if self.config.prefer_obj and obj_candidate.exists():
            return obj_candidate
        if absolute_off.exists():
            return absolute_off
        if obj_candidate.exists():
            return obj_candidate

        raise FeatureConstructionError(
            "No readable mesh source found for record. Expected converted OBJ or source OFF. "
            f"OFF: {absolute_off}; OBJ: {obj_candidate}"
        )

    def _can_skip(
        self,
        source_key: str,
        source_state: Mapping[str, Any],
        feature_path: Path,
        meta_path: Path,
    ) -> bool:
        if self.config.overwrite:
            return False
        if not self.config.incremental:
            return False
        if not feature_path.exists() or feature_path.stat().st_size <= 0:
            return False
        if not meta_path.exists():
            return False

        ref = self.index.tensor_refs.get(source_key)
        if not isinstance(ref, dict):
            return False
        if int(ref.get("source_size", -1)) != int(source_state["size"]):
            return False
        if int(ref.get("source_mtime_ns", -1)) != int(source_state["mtime_ns"]):
            return False
        if ref.get("schema_version") != self.config.schema_version:
            return False
        if ref.get("dtype") != self.config.dtype:
            return False
        if int(ref.get("resolution", -1)) != int(self.config.resolution):
            return False
        if self.config.verify_hash and ref.get("source_sha256") != source_state.get("sha256"):
            return False

        return True

    @staticmethod
    def _log_summary(summary: FeatureSummary) -> None:
        LOGGER.info("Mesh->OVoxel summary")
        LOGGER.info("total_scanned=%d", summary.total_scanned)
        LOGGER.info("generated=%d", summary.generated)
        LOGGER.info("skipped=%d", summary.skipped)
        LOGGER.info("failed=%d", summary.failed)
        LOGGER.info("elapsed_seconds=%.3f", summary.elapsed_seconds)


def build_ovoxel_features(
    config: MeshFeatureConfig,
    records: Sequence[Union[SampleRecord, Mapping[str, Any]]],
    seed: int = 42,
) -> Dict[str, Any]:
    """Convenience entrypoint for OVoxel feature generation."""
    builder = MeshToFeatureBuilder(config)
    summary = builder.build_from_records(records=records, seed=seed)
    return summary.as_dict()
