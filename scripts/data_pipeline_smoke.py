"""End-to-end data pipeline smoke and throughput baseline for Kaggle runs."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))



def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def load_config(path: Path) -> Dict[str, Any]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise RuntimeError(f"Config must be a YAML object: {path}")
    return _expand_env(payload)


def _float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return fallback


def _diagnostic_hint(error_text: str) -> str:
    text = error_text.lower()
    if "no module named 'src'" in text:
        return "Run the script from repository root or keep the default workspace-root bootstrap in scripts/data_pipeline_smoke.py."
    if "no module named" in text:
        return "Install required Python dependencies in the active environment (torch, trimesh, numpy, pyyaml), then rerun smoke script."
    if "does not exist" in text or "missing" in text:
        return "Attach Kaggle dataset and verify paths.dataset_root and paths.output_root environment values."
    if "cache index" in text or "consistency" in text:
        return "Delete stale cache index entries or run refresh mode with --refresh-overwrite to rebuild artifacts."
    if "shape" in text or "tensor" in text:
        return "Inspect generated .pt files and metadata sidecars; confirm resolution and schema version are consistent."
    if "occupancy" in text or "nonzero" in text:
        return "Inspect category-specific mesh quality and increase sampling density for sparse meshes."
    return "Inspect report diagnostics and rerun with --verbose for detailed stage-level traces."


def _split_counts(records: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for rec in records:
        split = str(rec.get("split", "unknown"))
        counts[split] = counts.get(split, 0) + 1
    return counts


def _collect_batch(
    builder: Any,
    records: Sequence[Mapping[str, Any]],
    batch_size: int,
) -> Dict[str, Any]:
    import torch

    tensors: List[torch.Tensor] = []
    occupancies: List[float] = []
    missing: List[str] = []

    for rec in records[:batch_size]:
        rel = Path(str(rec["relative_path"]))
        key = str(rel).replace("\\", "/")
        entry = builder.index.tensor_refs.get(key)
        if not isinstance(entry, dict):
            missing.append(f"Missing tensor ref for key={key}")
            continue
        tensor_path = Path(str(entry.get("tensor_path", "")))
        if not tensor_path.exists():
            missing.append(f"Tensor path missing for key={key}: {tensor_path}")
            continue
        tensor = torch.load(tensor_path, map_location="cpu")
        if not isinstance(tensor, torch.Tensor):
            missing.append(f"Saved artifact is not tensor for key={key}: {tensor_path}")
            continue
        if tensor.ndim != 4:
            missing.append(f"Invalid tensor rank for key={key}: {tuple(tensor.shape)}")
            continue
        if tensor.shape[0] != 1:
            missing.append(f"Invalid channel shape for key={key}: {tuple(tensor.shape)}")
            continue

        nonzero = int(torch.count_nonzero(tensor).item())
        total = int(tensor.numel())
        occupancies.append(nonzero / total if total else 0.0)
        tensors.append(tensor)

    if not tensors:
        raise RuntimeError("Batch collation failed: no valid tensors loaded.")

    batch = torch.stack(tensors, dim=0)
    return {
        "batch_shape": list(batch.shape),
        "batch_dtype": str(batch.dtype).replace("torch.", ""),
        "batch_min": float(batch.min().item()),
        "batch_max": float(batch.max().item()),
        "batch_nonzero": int(torch.count_nonzero(batch).item()),
        "batch_total": int(batch.numel()),
        "batch_nonzero_ratio": float(torch.count_nonzero(batch).item() / batch.numel()),
        "sample_occupancies": occupancies,
        "load_warnings": missing,
    }


def _occupancy_by_category(
    builder: Any,
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    import torch

    per_cat: Dict[str, List[float]] = {}
    errors: List[str] = []

    for rec in records:
        cat = str(rec.get("class_name", "unknown"))
        rel = Path(str(rec["relative_path"]))
        key = str(rel).replace("\\", "/")
        entry = builder.index.tensor_refs.get(key)
        if not isinstance(entry, dict):
            errors.append(f"Missing tensor ref for occupancy analysis key={key}")
            continue

        tensor_path = Path(str(entry.get("tensor_path", "")))
        if not tensor_path.exists():
            errors.append(f"Missing tensor file for occupancy key={key}: {tensor_path}")
            continue

        tensor = torch.load(tensor_path, map_location="cpu")
        if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
            errors.append(f"Invalid tensor in occupancy analysis key={key}: {tensor_path}")
            continue

        ratio = float(torch.count_nonzero(tensor).item() / tensor.numel())
        per_cat.setdefault(cat, []).append(ratio)

    category_mean = {
        cat: float(sum(vals) / len(vals)) for cat, vals in per_cat.items() if vals
    }

    means = sorted(category_mean.values())
    outliers: List[str] = []
    bounds = {"lower": None, "upper": None}
    if len(means) >= 4:
        q1 = statistics.quantiles(means, n=4, method="inclusive")[0]
        q3 = statistics.quantiles(means, n=4, method="inclusive")[2]
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        bounds = {"lower": lower, "upper": upper}
        for cat, mean in category_mean.items():
            if mean < lower or mean > upper:
                outliers.append(cat)

    return {
        "category_mean_occupancy": category_mean,
        "outlier_categories": outliers,
        "outlier_bounds": bounds,
        "analysis_errors": errors,
    }


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def run_smoke(args: argparse.Namespace) -> Dict[str, Any]:
    from src.data.dataset_adapter import AdapterConfig, KaggleDatasetAdapter
    from src.data.mesh_to_feature import MeshFeatureConfig, MeshToFeatureBuilder
    from src.data.off_to_obj_converter import ConverterConfig, OffToObjConverter

    report: Dict[str, Any] = {
        "status": "ok",
        "diagnostics": [],
        "timings": {},
        "cache_behavior": {},
    }

    config = load_config(Path(args.data_config))

    adapter_cfg = AdapterConfig.from_data_config(config)
    adapter_cfg = AdapterConfig(
        dataset_root=args.dataset_root or adapter_cfg.dataset_root,
        kaggle_slug=adapter_cfg.kaggle_slug,
        split=adapter_cfg.split,
        seed=args.seed,
        strict_split=adapter_cfg.strict_split,
        file_extension=adapter_cfg.file_extension,
        enforce_kaggle_input_prefix=args.enforce_kaggle,
    )

    t0 = time.perf_counter()
    adapter = KaggleDatasetAdapter(adapter_cfg)
    records = [rec.as_dict() for rec in adapter.load()]
    report["timings"]["adapter_load_seconds"] = round(time.perf_counter() - t0, 4)

    if not records:
        raise RuntimeError("Dataset adapter returned zero records.")

    split_counts = _split_counts(records)
    report["sample_counts"] = {
        "total": len(records),
        "splits": split_counts,
    }

    subset = records[: max(1, min(args.sample_limit, len(records)))]

    conv_base = ConverterConfig.from_data_config(config)
    conv_base = ConverterConfig(
        input_root=args.dataset_root or conv_base.input_root,
        output_root=args.output_root or conv_base.output_root,
        cache_index_path=conv_base.cache_index_path,
        overwrite=False,
        incremental=True,
        verify_hash=args.verify_hash,
        enforce_kaggle_paths=args.enforce_kaggle,
    )

    t1 = time.perf_counter()
    base_conv_summary = OffToObjConverter(conv_base).convert_from_records(subset).as_dict()
    report["timings"]["baseline_conversion_stage_seconds"] = round(time.perf_counter() - t1, 4)

    feat_base = MeshFeatureConfig.from_data_config(config)
    feat_base = MeshFeatureConfig(
        dataset_root=args.dataset_root or feat_base.dataset_root,
        output_root=args.output_root or feat_base.output_root,
        cache_dir=feat_base.cache_dir,
        resolution=feat_base.resolution,
        dtype=args.dtype or feat_base.dtype,
        normalize_meshes=feat_base.normalize_meshes,
        center_objects=feat_base.center_objects,
        scale_to_unit_box=feat_base.scale_to_unit_box,
        samples_per_mesh=args.samples_per_mesh,
        prefer_obj=True,
        overwrite=False,
        incremental=True,
        verify_hash=args.verify_hash,
        enforce_kaggle_paths=args.enforce_kaggle,
        schema_version=args.schema_version,
    )

    t2 = time.perf_counter()
    builder = MeshToFeatureBuilder(feat_base)
    base_feat_summary = builder.build_from_records(subset, seed=args.seed).as_dict()
    report["timings"]["baseline_feature_stage_seconds"] = round(time.perf_counter() - t2, 4)

    t3 = time.perf_counter()
    batch_info = _collect_batch(builder, subset, args.batch_size)
    report["timings"]["first_batch_seconds"] = round(time.perf_counter() - t3, 4)
    report["batch_contract"] = batch_info

    occupancy_info = _occupancy_by_category(builder, subset)
    report["occupancy"] = occupancy_info
    if occupancy_info.get("outlier_categories"):
        report["diagnostics"].append(
            {
                "level": "warning",
                "code": "occupancy_outlier",
                "message": "Category occupancy outliers detected.",
                "categories": occupancy_info["outlier_categories"],
                "action": "Inspect source meshes for outlier categories and adjust sampling or normalization settings.",
            }
        )

    report["cache_behavior"]["baseline"] = {
        "conversion": base_conv_summary,
        "features": base_feat_summary,
        "conversion_cache_hit_rate": (
            float(base_conv_summary["skipped"]) / float(base_conv_summary["total_scanned"])
            if base_conv_summary["total_scanned"]
            else 0.0
        ),
        "feature_cache_hit_rate": (
            float(base_feat_summary["skipped"]) / float(base_feat_summary["total_scanned"])
            if base_feat_summary["total_scanned"]
            else 0.0
        ),
    }

    conv_refresh = ConverterConfig(
        input_root=conv_base.input_root,
        output_root=conv_base.output_root,
        cache_index_path=conv_base.cache_index_path,
        overwrite=args.refresh_overwrite,
        incremental=False,
        verify_hash=args.verify_hash,
        enforce_kaggle_paths=args.enforce_kaggle,
    )
    feat_refresh = MeshFeatureConfig(
        dataset_root=feat_base.dataset_root,
        output_root=feat_base.output_root,
        cache_dir=feat_base.cache_dir,
        resolution=feat_base.resolution,
        dtype=feat_base.dtype,
        normalize_meshes=feat_base.normalize_meshes,
        center_objects=feat_base.center_objects,
        scale_to_unit_box=feat_base.scale_to_unit_box,
        samples_per_mesh=feat_base.samples_per_mesh,
        prefer_obj=feat_base.prefer_obj,
        overwrite=args.refresh_overwrite,
        incremental=False,
        verify_hash=feat_base.verify_hash,
        enforce_kaggle_paths=feat_base.enforce_kaggle_paths,
        schema_version=feat_base.schema_version,
    )

    refresh_subset = subset[: max(1, min(args.refresh_sample_limit, len(subset)))]
    t4 = time.perf_counter()
    refresh_conv_summary = OffToObjConverter(conv_refresh).convert_from_records(refresh_subset).as_dict()
    refresh_builder = MeshToFeatureBuilder(feat_refresh)
    refresh_feat_summary = refresh_builder.build_from_records(refresh_subset, seed=args.seed).as_dict()
    report["timings"]["refresh_stage_seconds"] = round(time.perf_counter() - t4, 4)
    report["cache_behavior"]["refresh"] = {
        "conversion": refresh_conv_summary,
        "features": refresh_feat_summary,
        "mode": "non_incremental_overwrite",
    }

    if refresh_conv_summary["failed"] > 0 or refresh_feat_summary["failed"] > 0:
        report["diagnostics"].append(
            {
                "level": "error",
                "code": "refresh_failed",
                "message": "Refresh mode encountered failures.",
                "action": "Check malformed source meshes and clear stale cache entries before rerun.",
            }
        )

    checks = {
        "split_consistency": bool(sum(split_counts.values()) == len(records)),
        "batch_shape_ok": bool(len(batch_info["batch_shape"]) == 5 and batch_info["batch_shape"][1] == 1),
        "batch_nonzero_ok": bool(batch_info["batch_nonzero"] > 0),
        "batch_range_ok": bool(0.0 <= batch_info["batch_min"] and batch_info["batch_max"] <= 1.0),
    }
    report["checks"] = checks
    if not all(checks.values()):
        report["status"] = "failed"
        report["diagnostics"].append(
            {
                "level": "error",
                "code": "contract_check_failed",
                "message": "One or more tensor contract checks failed.",
                "checks": checks,
                "action": "Review batch_contract and occupancy diagnostics; then rerun refresh mode.",
            }
        )

    report["reproducible_command"] = (
        f"python scripts/data_pipeline_smoke.py --data-config {args.data_config} "
        f"--sample-limit {args.sample_limit} --batch-size {args.batch_size} "
        f"--seed {args.seed} --schema-version {args.schema_version}"
    )

    return report


def write_reports(report: Mapping[str, Any], json_path: Path, md_path: Path) -> None:
    _ensure_dir(json_path)
    _ensure_dir(md_path)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Data Pipeline Smoke Report",
        "",
        f"- status: {report.get('status')}",
        f"- adapter_load_seconds: {report.get('timings', {}).get('adapter_load_seconds')}",
        f"- first_batch_seconds: {report.get('timings', {}).get('first_batch_seconds')}",
        "",
        "## Checks",
    ]
    checks = report.get("checks", {})
    for key, value in checks.items():
        lines.append(f"- {key}: {value}")

    if report.get("status") != "ok":
        lines.extend(
            [
                f"- error: {report.get('error')}",
                f"- recovery: {report.get('recovery')}",
            ]
        )

    lines.extend(["", "## Cache Behavior"])
    cache = report.get("cache_behavior", {})
    baseline = cache.get("baseline", {})
    if baseline:
        lines.append(
            f"- baseline conversion cache hit rate: {baseline.get('conversion_cache_hit_rate', 0.0):.4f}"
        )
        lines.append(
            f"- baseline feature cache hit rate: {baseline.get('feature_cache_hit_rate', 0.0):.4f}"
        )
    else:
        lines.append("- unavailable (smoke did not complete baseline stage)")

    lines.extend(["", "## Diagnostics"])
    diagnostics = report.get("diagnostics", [])
    if diagnostics:
        for diag in diagnostics:
            lines.append(
                f"- [{diag.get('level', 'info')}] {diag.get('code', 'diag')}: {diag.get('message')}"
            )
            action = diag.get("action")
            if action:
                lines.append(f"- recovery: {action}")
    else:
        lines.append("- None")

    lines.extend(["", "## Reproducible Command", "", f"{report.get('reproducible_command', '')}"])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Data pipeline smoke and throughput baseline")
    parser.add_argument("--data-config", type=str, default="configs/data_stage1.yaml")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--sample-limit", type=int, default=128)
    parser.add_argument("--refresh-sample-limit", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--samples-per-mesh", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--schema-version", type=str, default="ovoxel-v1")
    parser.add_argument("--verify-hash", action="store_true")
    parser.add_argument("--enforce-kaggle", action="store_true")
    parser.add_argument("--refresh-overwrite", action="store_true")
    parser.add_argument(
        "--report-json",
        type=str,
        default="logs/data_pipeline_smoke_report.json",
    )
    parser.add_argument(
        "--report-markdown",
        type=str,
        default="logs/data_pipeline_smoke_report.md",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.verbose:
        import logging

        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    try:
        report = run_smoke(args)
    except (ModuleNotFoundError, RuntimeError, ValueError) as exc:
        message = str(exc)
        report = {
            "status": "failed",
            "error": message,
            "recovery": _diagnostic_hint(message),
            "reproducible_command": (
                f"python scripts/data_pipeline_smoke.py --data-config {args.data_config} "
                f"--sample-limit {args.sample_limit} --refresh-sample-limit {args.refresh_sample_limit} "
                f"--batch-size {args.batch_size} --seed {args.seed} "
                f"--schema-version {args.schema_version} --refresh-overwrite"
            ),
        }

    json_path = Path(args.report_json)
    md_path = Path(args.report_markdown)
    write_reports(report, json_path, md_path)

    print(json.dumps(report, indent=2))
    return 0 if report.get("status") == "ok" else 2


if __name__ == "__main__":
    sys.exit(main())
