"""Dataset adapter for Kaggle-attached 3D datasets.

This module provides deterministic discovery and split loading for ModelNet40
mirrors mounted under /kaggle/input/<dataset-slug>/ while keeping dataset paths
fully config-driven.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple


class DatasetAdapterError(RuntimeError):
    """Raised when dataset discovery or split loading fails."""


@dataclass(frozen=True)
class SampleRecord:
    """Normalized sample entry used by downstream pipeline stages."""

    index: int
    split: str
    class_id: int
    class_name: str
    sample_id: str
    relative_path: str
    absolute_path: str
    dataset_slug: str
    provenance: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "split": self.split,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "sample_id": self.sample_id,
            "relative_path": self.relative_path,
            "absolute_path": self.absolute_path,
            "dataset_slug": self.dataset_slug,
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class AdapterConfig:
    """Configuration surface for dataset discovery and split behavior."""

    dataset_root: str
    kaggle_slug: str
    split: str = "both"
    seed: int = 42
    strict_split: bool = True
    file_extension: str = ".off"
    enforce_kaggle_input_prefix: bool = False

    @staticmethod
    def from_data_config(data_cfg: Mapping[str, Any]) -> "AdapterConfig":
        dataset = data_cfg.get("dataset", {})
        paths = data_cfg.get("paths", {})
        primary = dataset.get("primary", {})

        kaggle_slug = str(primary.get("kaggle_slug") or "").strip()
        if not kaggle_slug:
            raise DatasetAdapterError(
                "dataset.primary.kaggle_slug is missing in data config. "
                "Set it to the Kaggle dataset slug, for example "
                "'balraj98/modelnet40-princeton-3d-object-dataset'."
            )

        dataset_root = str(paths.get("dataset_root") or "").strip()
        if not dataset_root or "${" in dataset_root:
            dataset_root = default_dataset_root_for_slug(kaggle_slug)

        split = str(primary.get("split") or "both").strip().lower()
        if split not in {"train", "test", "both"}:
            raise DatasetAdapterError(
                "dataset.primary.split must be one of: train, test, both. "
                f"Received: {split!r}"
            )

        return AdapterConfig(
            dataset_root=dataset_root,
            kaggle_slug=kaggle_slug,
            split=split,
        )


def default_dataset_root_for_slug(kaggle_slug: str) -> str:
    """Map owner/slug to Kaggle mounted path contract."""
    dataset_name = kaggle_slug.split("/")[-1].strip()
    if not dataset_name:
        raise DatasetAdapterError(
            "Invalid kaggle_slug format. Expected owner/name, got empty name."
        )
    return str(Path("/kaggle/input") / dataset_name)


def _normalize_sample_key(value: str) -> str:
    return value.replace("\\", "/").strip().lower()


def _parse_split_manifest(path: Path) -> List[str]:
    keys: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw:
            continue
        text = raw.replace("\\", "/")
        text = text.rsplit(".", 1)[0]
        keys.append(_normalize_sample_key(text))
    return keys


def _path_indicates_split(path: Path) -> Optional[str]:
    parts = [_normalize_sample_key(item) for item in path.parts]
    if "train" in parts:
        return "train"
    if "test" in parts:
        return "test"
    return None


def _find_modelnet_content_root(dataset_root: Path) -> Path:
    candidates = [
        dataset_root,
        dataset_root / "ModelNet40",
        dataset_root / "modelnet40",
        dataset_root / "ModelNet40_Aligned",
    ]
    for candidate in candidates:
        if not candidate.exists() or not candidate.is_dir():
            continue
        class_dirs = [
            item
            for item in candidate.iterdir()
            if item.is_dir() and item.name.lower() not in {"__macosx", ".ipynb_checkpoints"}
        ]
        valid = 0
        for class_dir in class_dirs:
            train_dir = class_dir / "train"
            test_dir = class_dir / "test"
            if train_dir.is_dir() or test_dir.is_dir():
                valid += 1
        if valid >= 5:
            return candidate
    raise DatasetAdapterError(
        "Unable to detect ModelNet-style class layout under dataset_root. "
        f"Checked: {[str(path) for path in candidates]}. "
        "Expected class directories containing train/ and test/ folders."
    )


def _discover_manifest_files(dataset_root: Path) -> Dict[str, Path]:
    manifest_map: Dict[str, Path] = {}
    preferred_names = {
        "train": ["modelnet40_train.txt", "train.txt", "split_train.txt"],
        "test": ["modelnet40_test.txt", "test.txt", "split_test.txt"],
    }

    for split_name, names in preferred_names.items():
        for name in names:
            for base in (dataset_root, dataset_root / "ModelNet40", dataset_root / "modelnet40"):
                candidate = base / name
                if candidate.exists() and candidate.is_file():
                    manifest_map[split_name] = candidate
                    break
            if split_name in manifest_map:
                break

    return manifest_map


class KaggleDatasetAdapter:
    """Discover dataset assets and provide deterministic split-aware streams."""

    def __init__(self, config: AdapterConfig):
        self.config = config
        self.dataset_root = Path(config.dataset_root)
        self.content_root: Optional[Path] = None
        self.class_to_id: Dict[str, int] = {}
        self._records: List[SampleRecord] = []

    @classmethod
    def from_data_config(cls, data_cfg: Mapping[str, Any]) -> "KaggleDatasetAdapter":
        return cls(AdapterConfig.from_data_config(data_cfg))

    def load(self) -> List[SampleRecord]:
        if self.config.enforce_kaggle_input_prefix and not str(self.dataset_root).startswith(
            "/kaggle/input"
        ):
            raise DatasetAdapterError(
                "dataset_root must start with /kaggle/input for Kaggle attach workflow. "
                f"Received: {self.dataset_root}"
            )

        if not self.dataset_root.exists():
            raise DatasetAdapterError(
                "Dataset root does not exist. Attach the Kaggle dataset and set paths.dataset_root "
                "to /kaggle/input/<dataset-slug>. "
                f"Missing path: {self.dataset_root}"
            )

        self.content_root = _find_modelnet_content_root(self.dataset_root)
        manifest_files = _discover_manifest_files(self.dataset_root)
        manifest_keys: Dict[str, set[str]] = {}
        for split_name, path in manifest_files.items():
            manifest_keys[split_name] = set(_parse_split_manifest(path))

        extension = self.config.file_extension.lower()
        files = sorted(self.content_root.rglob(f"*{extension}"))
        if not files:
            raise DatasetAdapterError(
                f"No '{extension}' files found under content root {self.content_root}. "
                "Check that the Kaggle dataset mirror contains OFF assets."
            )

        records: List[Tuple[str, str, Path]] = []
        for file_path in files:
            rel = file_path.relative_to(self.content_root)
            class_name = rel.parts[0] if len(rel.parts) >= 2 else "unknown"
            sample_id = file_path.stem
            records.append((class_name, sample_id, file_path))

        class_names = sorted({entry[0] for entry in records})
        self.class_to_id = {name: index for index, name in enumerate(class_names)}

        normalized_records: List[SampleRecord] = []
        split_counts = {"train": 0, "test": 0}

        for class_name, sample_id, file_path in sorted(
            records,
            key=lambda item: (item[0].lower(), item[1].lower(), str(item[2]).lower()),
        ):
            rel = file_path.relative_to(self.content_root)
            rel_no_ext = _normalize_sample_key(str(rel.with_suffix("")))
            class_key = _normalize_sample_key(f"{class_name}/{sample_id}")
            split = _path_indicates_split(rel)

            if "train" in manifest_keys and (rel_no_ext in manifest_keys["train"] or class_key in manifest_keys["train"]):
                split = "train"
            elif "test" in manifest_keys and (rel_no_ext in manifest_keys["test"] or class_key in manifest_keys["test"]):
                split = "test"

            if split is None:
                if self.config.strict_split:
                    raise DatasetAdapterError(
                        "Unable to determine split for sample. "
                        f"File: {file_path}. Expected train/test folder or split manifest entries."
                    )
                split = "train"

            if self.config.split != "both" and split != self.config.split:
                continue

            idx = len(normalized_records)
            split_counts[split] = split_counts.get(split, 0) + 1
            normalized_records.append(
                SampleRecord(
                    index=idx,
                    split=split,
                    class_id=self.class_to_id[class_name],
                    class_name=class_name,
                    sample_id=sample_id,
                    relative_path=str(rel).replace("\\", "/"),
                    absolute_path=str(file_path),
                    dataset_slug=self.config.kaggle_slug,
                    provenance="princeton_modelnet40_via_kaggle_mirror",
                )
            )

        if not normalized_records:
            raise DatasetAdapterError(
                "Dataset discovery produced zero usable samples after split filtering. "
                f"Requested split: {self.config.split}."
            )

        if self.config.split in {"train", "both"} and split_counts.get("train", 0) == 0:
            raise DatasetAdapterError(
                "No training samples were discovered. Verify train/ directories or train split manifest."
            )

        if self.config.split in {"test", "both"} and split_counts.get("test", 0) == 0:
            raise DatasetAdapterError(
                "No test samples were discovered. Verify test/ directories or test split manifest."
            )

        self._records = normalized_records
        return list(self._records)

    def records(self) -> List[SampleRecord]:
        if not self._records:
            raise DatasetAdapterError("Adapter has not been loaded. Call load() first.")
        return list(self._records)

    def iter_samples(
        self,
        split: str = "both",
        shuffle: bool = False,
        seed: Optional[int] = None,
    ) -> Iterator[SampleRecord]:
        if not self._records:
            raise DatasetAdapterError("Adapter has not been loaded. Call load() first.")

        selected = self._filter_split(split)
        if shuffle:
            rng = random.Random(self.config.seed if seed is None else seed)
            rng.shuffle(selected)

        for sample in selected:
            yield sample

    def _filter_split(self, split: str) -> List[SampleRecord]:
        split = split.lower().strip()
        if split not in {"train", "test", "both"}:
            raise DatasetAdapterError(
                f"Invalid split: {split!r}. Expected one of train, test, both."
            )
        if split == "both":
            return list(self._records)
        return [item for item in self._records if item.split == split]

    def summary(self) -> Dict[str, Any]:
        if not self._records:
            raise DatasetAdapterError("Adapter has not been loaded. Call load() first.")

        train_count = sum(1 for item in self._records if item.split == "train")
        test_count = sum(1 for item in self._records if item.split == "test")
        return {
            "dataset_root": str(self.dataset_root),
            "content_root": str(self.content_root) if self.content_root else None,
            "dataset_slug": self.config.kaggle_slug,
            "total_samples": len(self._records),
            "train_samples": train_count,
            "test_samples": test_count,
            "num_classes": len(self.class_to_id),
            "seed": self.config.seed,
        }


def build_sample_stream(
    config: AdapterConfig,
    split: str = "both",
    shuffle: bool = False,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Convenience helper returning normalized sample dictionaries."""
    adapter = KaggleDatasetAdapter(config)
    adapter.load()
    return [sample.as_dict() for sample in adapter.iter_samples(split=split, shuffle=shuffle, seed=seed)]
