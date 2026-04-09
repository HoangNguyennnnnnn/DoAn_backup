"""OFF/OBJ conversion and cache utilities for Kaggle-first data pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

from src.data.dataset_adapter import SampleRecord


LOGGER = logging.getLogger(__name__)


class ConversionError(RuntimeError):
    """Raised when conversion or integrity validation fails."""


@dataclass(frozen=True)
class ConverterConfig:
    """Config-driven conversion settings."""

    input_root: str
    output_root: str
    cache_index_path: str
    overwrite: bool = False
    incremental: bool = True
    verify_hash: bool = False
    enforce_kaggle_paths: bool = False

    @staticmethod
    def from_data_config(data_cfg: Mapping[str, Any]) -> "ConverterConfig":
        paths = data_cfg.get("paths", {})

        input_root = str(paths.get("dataset_root") or "").strip()
        output_root = str(paths.get("output_root") or "").strip()
        cache_dir = str(paths.get("cache_dir") or "").strip()

        if not input_root or "${" in input_root:
            raise ConversionError(
                "paths.dataset_root must be set to a concrete dataset path before conversion."
            )
        if not output_root or "${" in output_root:
            raise ConversionError(
                "paths.output_root must be set to a concrete output path before conversion."
            )
        if not cache_dir or "${" in cache_dir:
            cache_dir = str(Path(output_root) / "cache")

        cache_index_path = str(Path(cache_dir) / "conversion_cache_index.json")
        return ConverterConfig(
            input_root=input_root,
            output_root=output_root,
            cache_index_path=cache_index_path,
        )


@dataclass
class ConversionSummary:
    total_scanned: int = 0
    converted: int = 0
    skipped: int = 0
    failed: int = 0
    elapsed_seconds: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_scanned": self.total_scanned,
            "converted": self.converted,
            "skipped": self.skipped,
            "failed": self.failed,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _source_state(path: Path, verify_hash: bool = False) -> Dict[str, Any]:
    stat = path.stat()
    state: Dict[str, Any] = {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if verify_hash:
        state["sha256"] = _sha256_file(path)
    return state


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _assert_kaggle_path(path: Path, role: str) -> None:
    prefix = "/kaggle/"
    if not str(path).startswith(prefix):
        raise ConversionError(
            f"{role} must be under /kaggle for Kaggle-compatible execution. Received: {path}"
        )


def validate_off_file(path: Path) -> Tuple[int, int]:
    """Validate minimal OFF structure and return vertex/face counts."""
    if not path.exists() or not path.is_file():
        raise ConversionError(f"OFF file does not exist: {path}")

    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    if not lines:
        raise ConversionError(f"OFF file is empty: {path}")

    idx = 0
    header = lines[idx].strip()
    if header != "OFF":
        if header.startswith("OFF") and len(header.split()) == 4:
            counts_line = header[3:].strip()
        else:
            raise ConversionError(
                f"Invalid OFF header in {path}. Expected first token 'OFF'."
            )
    else:
        idx += 1
        while idx < len(lines) and (not lines[idx].strip() or lines[idx].strip().startswith("#")):
            idx += 1
        if idx >= len(lines):
            raise ConversionError(f"Missing OFF counts line in {path}.")
        counts_line = lines[idx].strip()

    tokens = counts_line.split()
    if len(tokens) < 2:
        raise ConversionError(
            f"Malformed OFF counts line in {path}. Expected '<num_vertices> <num_faces> ...'."
        )
    try:
        num_vertices = int(tokens[0])
        num_faces = int(tokens[1])
    except ValueError as exc:
        raise ConversionError(
            f"Non-integer OFF counts in {path}: {counts_line!r}"
        ) from exc

    min_expected = idx + 1 + num_vertices + num_faces
    if len(lines) < min_expected:
        raise ConversionError(
            f"OFF file appears truncated: {path}. Expected at least {min_expected} lines, got {len(lines)}."
        )

    return num_vertices, num_faces


def _parse_off_geometry(path: Path) -> Tuple[List[Tuple[float, float, float]], List[List[int]]]:
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    if not lines:
        raise ConversionError(f"OFF file is empty: {path}")

    pos = 0
    first = lines[pos].strip()
    if first == "OFF":
        pos += 1
    elif first.startswith("OFF"):
        lines.insert(1, first[3:].strip())
        lines[0] = "OFF"
        pos = 1
    else:
        raise ConversionError(f"Invalid OFF header in {path}.")

    while pos < len(lines) and (not lines[pos].strip() or lines[pos].strip().startswith("#")):
        pos += 1
    if pos >= len(lines):
        raise ConversionError(f"Missing OFF counts line in {path}.")

    parts = lines[pos].split()
    if len(parts) < 2:
        raise ConversionError(f"Malformed OFF counts line in {path}: {lines[pos]!r}")

    try:
        vertex_count = int(parts[0])
        face_count = int(parts[1])
    except ValueError as exc:
        raise ConversionError(f"Non-integer OFF counts in {path}: {lines[pos]!r}") from exc

    pos += 1
    vertices: List[Tuple[float, float, float]] = []
    faces: List[List[int]] = []

    for _ in range(vertex_count):
        while pos < len(lines) and (not lines[pos].strip() or lines[pos].strip().startswith("#")):
            pos += 1
        if pos >= len(lines):
            raise ConversionError(f"OFF vertex block is incomplete in {path}.")
        coords = lines[pos].split()
        if len(coords) < 3:
            raise ConversionError(f"Malformed OFF vertex row in {path}: {lines[pos]!r}")
        try:
            x, y, z = float(coords[0]), float(coords[1]), float(coords[2])
        except ValueError as exc:
            raise ConversionError(f"Non-numeric OFF vertex row in {path}: {lines[pos]!r}") from exc
        vertices.append((x, y, z))
        pos += 1

    for _ in range(face_count):
        while pos < len(lines) and (not lines[pos].strip() or lines[pos].strip().startswith("#")):
            pos += 1
        if pos >= len(lines):
            raise ConversionError(f"OFF face block is incomplete in {path}.")
        row = lines[pos].split()
        if not row:
            raise ConversionError(f"Malformed OFF face row in {path}: {lines[pos]!r}")
        try:
            n = int(row[0])
        except ValueError as exc:
            raise ConversionError(f"Non-integer OFF face size in {path}: {lines[pos]!r}") from exc
        if len(row) < n + 1:
            raise ConversionError(f"OFF face row has insufficient indices in {path}: {lines[pos]!r}")
        try:
            indices = [int(row[i]) for i in range(1, n + 1)]
        except ValueError as exc:
            raise ConversionError(f"Non-integer OFF face indices in {path}: {lines[pos]!r}") from exc
        if any(index < 0 or index >= vertex_count for index in indices):
            raise ConversionError(f"OFF face index out of range in {path}: {lines[pos]!r}")
        if n >= 3:
            faces.append(indices)
        pos += 1

    return vertices, faces


def convert_off_to_obj(off_path: Path, obj_path: Path, overwrite: bool = False) -> None:
    """Convert one OFF file to OBJ and validate output integrity."""
    validate_off_file(off_path)

    if obj_path.exists() and not overwrite:
        raise ConversionError(
            "OBJ output already exists. Re-run with overwrite=True to replace existing output. "
            f"Existing file: {obj_path}"
        )

    vertices, faces = _parse_off_geometry(off_path)
    if not vertices:
        raise ConversionError(f"No vertices parsed from OFF file: {off_path}")
    if not faces:
        raise ConversionError(f"No valid faces parsed from OFF file: {off_path}")

    _ensure_parent(obj_path)

    with obj_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# Converted from OFF: {off_path.name}\n")
        for x, y, z in vertices:
            handle.write(f"v {x:.9g} {y:.9g} {z:.9g}\n")
        for face in faces:
            one_based = [str(idx + 1) for idx in face]
            handle.write(f"f {' '.join(one_based)}\n")

    if not obj_path.exists():
        raise ConversionError(f"OBJ output was not created: {obj_path}")
    if obj_path.stat().st_size <= 0:
        raise ConversionError(f"OBJ output is empty after conversion: {obj_path}")


class CacheIndex:
    """Persistent index for converted meshes and preprocessed tensor references."""

    def __init__(self, index_path: Union[str, Path]):
        self.index_path = Path(index_path)
        self.payload: Dict[str, Any] = {
            "version": 1,
            "converted_assets": {},
            "tensor_refs": {},
        }

    def load(self) -> None:
        if not self.index_path.exists():
            return
        raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ConversionError(
                f"Cache index is malformed (expected object): {self.index_path}"
            )
        raw.setdefault("version", 1)
        raw.setdefault("converted_assets", {})
        raw.setdefault("tensor_refs", {})
        if not isinstance(raw["converted_assets"], dict) or not isinstance(raw["tensor_refs"], dict):
            raise ConversionError(
                f"Cache index missing expected map fields in {self.index_path}"
            )
        self.payload = raw

    def save(self) -> None:
        _ensure_parent(self.index_path)
        self.index_path.write_text(json.dumps(self.payload, indent=2), encoding="utf-8")

    @property
    def converted_assets(self) -> MutableMapping[str, Any]:
        return self.payload["converted_assets"]

    @property
    def tensor_refs(self) -> MutableMapping[str, Any]:
        return self.payload["tensor_refs"]

    def get_converted(self, source_key: str) -> Optional[Dict[str, Any]]:
        value = self.converted_assets.get(source_key)
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ConversionError(
                f"Cache index entry is malformed for key={source_key!r} in {self.index_path}"
            )
        return value

    def upsert_converted(
        self,
        source_key: str,
        off_path: Path,
        obj_path: Path,
        source_state: Mapping[str, Any],
    ) -> None:
        self.converted_assets[source_key] = {
            "off_path": str(off_path),
            "obj_path": str(obj_path),
            "source_size": int(source_state["size"]),
            "source_mtime_ns": int(source_state["mtime_ns"]),
            "source_sha256": source_state.get("sha256"),
            "obj_size": int(obj_path.stat().st_size),
            "updated_unix": int(time.time()),
        }

    def upsert_tensor_ref(self, source_key: str, tensor_path: Union[str, Path]) -> None:
        self.tensor_refs[source_key] = {
            "tensor_path": str(tensor_path),
            "updated_unix": int(time.time()),
        }

    def validate_consistency(self) -> List[str]:
        problems: List[str] = []
        for key, meta in self.converted_assets.items():
            if not isinstance(meta, dict):
                problems.append(f"converted_assets[{key!r}] must be an object")
                continue
            obj_path = Path(str(meta.get("obj_path", "")))
            off_path = Path(str(meta.get("off_path", "")))
            if not off_path.exists():
                problems.append(f"source OFF missing for key={key}: {off_path}")
            if not obj_path.exists():
                problems.append(f"converted OBJ missing for key={key}: {obj_path}")
            elif obj_path.stat().st_size <= 0:
                problems.append(f"converted OBJ empty for key={key}: {obj_path}")
        for key, meta in self.tensor_refs.items():
            if not isinstance(meta, dict):
                problems.append(f"tensor_refs[{key!r}] must be an object")
                continue
            tensor_path = Path(str(meta.get("tensor_path", "")))
            if not tensor_path.exists():
                problems.append(f"tensor reference missing for key={key}: {tensor_path}")
            elif tensor_path.stat().st_size <= 0:
                problems.append(f"tensor reference empty for key={key}: {tensor_path}")
        return problems


class OffToObjConverter:
    """Recursive OFF to OBJ converter with idempotent cache-backed behavior."""

    def __init__(self, config: ConverterConfig):
        self.config = config
        self.input_root = Path(config.input_root)
        self.output_root = Path(config.output_root)
        self.index = CacheIndex(config.cache_index_path)
        self.index.load()

        if self.config.enforce_kaggle_paths:
            _assert_kaggle_path(self.input_root, "input_root")
            _assert_kaggle_path(self.output_root, "output_root")

    @classmethod
    def from_data_config(cls, data_cfg: Mapping[str, Any]) -> "OffToObjConverter":
        return cls(ConverterConfig.from_data_config(data_cfg))

    def convert_recursive(self) -> ConversionSummary:
        entries = [
            (off_path, off_path.relative_to(self.input_root))
            for off_path in sorted(self.input_root.rglob("*.off"))
        ]
        return self._convert_entries(entries)

    def convert_from_records(
        self,
        records: Sequence[Union[SampleRecord, Mapping[str, Any]]],
    ) -> ConversionSummary:
        entries: List[Tuple[Path, Path]] = []
        for item in records:
            if isinstance(item, SampleRecord):
                off_path = Path(item.absolute_path)
                rel = Path(item.relative_path)
            elif isinstance(item, Mapping):
                missing = [field for field in ("absolute_path", "relative_path") if field not in item]
                if missing:
                    raise ConversionError(
                        "Sample record schema mismatch. Required fields are missing: "
                        f"{missing}. Record keys: {sorted(item.keys())}"
                    )
                off_path = Path(str(item["absolute_path"]))
                rel = Path(str(item["relative_path"]))
            else:
                raise ConversionError(
                    "Unsupported record type. Expected SampleRecord or mapping with "
                    "absolute_path/relative_path fields."
                )

            if off_path.suffix.lower() != ".off":
                raise ConversionError(
                    f"Record does not reference OFF source: {off_path}. "
                    "Ensure Task 2.1 sample records are passed directly."
                )
            entries.append((off_path, rel))

        return self._convert_entries(entries)

    def _convert_entries(self, entries: Sequence[Tuple[Path, Path]]) -> ConversionSummary:
        if not self.input_root.exists():
            raise ConversionError(
                f"input_root does not exist: {self.input_root}. Attach dataset and verify config paths."
            )

        started = time.time()
        summary = ConversionSummary(total_scanned=len(entries))

        for off_path, rel in entries:
            try:
                source_key = str(rel).replace("\\", "/")
                obj_rel = rel.with_suffix(".obj")
                obj_path = self.output_root / obj_rel

                if not off_path.exists():
                    raise ConversionError(
                        "Source OFF file is missing (partial conversion state likely). "
                        f"Expected source: {off_path}"
                    )

                source_state = _source_state(off_path, verify_hash=self.config.verify_hash)
                cache_hit, reason = self._can_skip(source_key, obj_path, source_state)
                if cache_hit:
                    cached = self.index.get_converted(source_key)
                    if cached is None and obj_path.exists() and obj_path.stat().st_size > 0:
                        self.index.upsert_converted(
                            source_key=source_key,
                            off_path=off_path,
                            obj_path=obj_path,
                            source_state=source_state,
                        )
                    summary.skipped += 1
                    LOGGER.debug("Skip OFF->OBJ (%s): %s", reason, off_path)
                    continue

                convert_off_to_obj(off_path, obj_path, overwrite=self.config.overwrite)
                if not obj_path.exists() or obj_path.stat().st_size <= 0:
                    raise ConversionError(
                        f"OBJ output failed integrity check after conversion: {obj_path}"
                    )

                self.index.upsert_converted(
                    source_key=source_key,
                    off_path=off_path,
                    obj_path=obj_path,
                    source_state=source_state,
                )
                summary.converted += 1
            except Exception as exc:
                summary.failed += 1
                LOGGER.error("OFF->OBJ conversion failed for %s: %s", off_path, exc)

        self.index.save()
        consistency = self.index.validate_consistency()
        if consistency:
            raise ConversionError(
                "Cache index consistency validation failed after conversion:\n- "
                + "\n- ".join(consistency)
            )

        summary.elapsed_seconds = time.time() - started
        self._log_summary(summary)
        return summary

    def _can_skip(
        self,
        source_key: str,
        obj_path: Path,
        source_state: Mapping[str, Any],
    ) -> Tuple[bool, str]:
        cached = self.index.get_converted(source_key)

        if self.config.incremental:
            if obj_path.exists() and obj_path.stat().st_size > 0:
                return True, "incremental_existing_obj"
            if cached is not None and Path(str(cached.get("obj_path", ""))).exists():
                return True, "incremental_existing_cache"

        if self.config.overwrite:
            return False, "overwrite_enabled"

        if cached is None:
            return False, "full_scan_no_cache"

        if not obj_path.exists() or obj_path.stat().st_size <= 0:
            return False, "obj_missing_or_empty"

        if int(cached.get("source_size", -1)) != int(source_state["size"]):
            return False, "source_size_changed"
        if int(cached.get("source_mtime_ns", -1)) != int(source_state["mtime_ns"]):
            return False, "source_mtime_changed"

        if self.config.verify_hash:
            if cached.get("source_sha256") != source_state.get("sha256"):
                return False, "source_hash_changed"

        return True, "unchanged"

    @staticmethod
    def _log_summary(summary: ConversionSummary) -> None:
        LOGGER.info("OFF->OBJ conversion summary")
        LOGGER.info("total_scanned=%d", summary.total_scanned)
        LOGGER.info("converted=%d", summary.converted)
        LOGGER.info("skipped=%d", summary.skipped)
        LOGGER.info("failed=%d", summary.failed)
        LOGGER.info("elapsed_seconds=%.3f", summary.elapsed_seconds)


def run_off_to_obj_conversion(
    config: ConverterConfig,
    records: Optional[Sequence[Union[SampleRecord, Mapping[str, Any]]]] = None,
) -> Dict[str, Any]:
    """Convenience entrypoint for conversion runs and tests."""
    converter = OffToObjConverter(config)
    if records is None:
        summary = converter.convert_recursive()
    else:
        summary = converter.convert_from_records(records)
    return summary.as_dict()
