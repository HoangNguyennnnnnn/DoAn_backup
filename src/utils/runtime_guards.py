"""
Runtime guard utilities for Kaggle sessions.

This module validates runtime prerequisites and captures reproducibility metadata
before Stage 1 training starts.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class GuardContext:
    dataset_slug: str
    dataset_root: Path
    output_root: Path
    hardware_config: Path
    train_config: Path
    data_config: Path
    min_working_gb: float
    run_id: str


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return payload


def _expand_env(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return os.path.expandvars(value)


def _default_dataset_root_from_slug(dataset_slug: str) -> Path:
    # Kaggle mounts attached datasets under /kaggle/input/<dataset-name>.
    dataset_name = dataset_slug.split("/")[-1]
    return Path("/kaggle/input") / dataset_name


def _resolve_dataset_slug(data_cfg: Dict[str, Any], cli_slug: Optional[str]) -> str:
    if cli_slug:
        return cli_slug

    slug = (
        data_cfg.get("dataset", {})
        .get("primary", {})
        .get("kaggle_slug", "")
    )
    if slug:
        return str(slug)

    return "balraj98/modelnet40-princeton-3d-object-dataset"


def _resolve_dataset_root(
    data_cfg: Dict[str, Any],
    dataset_slug: str,
    cli_root: Optional[str],
) -> Path:
    if cli_root:
        root = Path(_expand_env(cli_root)).resolve()
        return root

    cfg_root = (
        data_cfg.get("paths", {}).get("dataset_root")
        or data_cfg.get("dataset", {}).get("root")
    )
    if cfg_root:
        expanded = _expand_env(str(cfg_root))
        if expanded and "${" not in expanded:
            return Path(expanded).resolve()

    return _default_dataset_root_from_slug(dataset_slug)


def _resolve_output_root(cli_output: Optional[str]) -> Path:
    candidate = cli_output or os.environ.get("OUTPUT_ROOT") or "/kaggle/working"
    return Path(_expand_env(candidate)).resolve()


def _git_commit_hash() -> str:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return value or "N/A"
    except Exception:
        return "N/A"


def _check_gpu() -> Dict[str, Any]:
    details: Dict[str, Any] = {
        "cuda_available": False,
        "gpu_count": 0,
        "gpu_names": [],
        "backend": "none",
    }

    try:
        import torch

        details["backend"] = "torch"
        details["cuda_available"] = bool(torch.cuda.is_available())
        if details["cuda_available"]:
            count = int(torch.cuda.device_count())
            details["gpu_count"] = count
            details["gpu_names"] = [
                torch.cuda.get_device_name(index) for index in range(count)
            ]
            return details
    except Exception:
        pass

    # Fallback to nvidia-smi for visibility checks when torch import is unavailable.
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            stderr=subprocess.STDOUT,
            text=True,
        )
        names = [line.strip() for line in output.splitlines() if line.strip()]
        details["backend"] = "nvidia-smi"
        details["cuda_available"] = bool(names)
        details["gpu_count"] = len(names)
        details["gpu_names"] = names
    except Exception:
        details["backend"] = "none"

    return details


def _check_disk_space(path: Path, min_gb: float) -> Dict[str, Any]:
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024 ** 3)
    total_gb = usage.total / (1024 ** 3)
    return {
        "path": str(path),
        "free_gb": round(free_gb, 2),
        "total_gb": round(total_gb, 2),
        "required_min_gb": min_gb,
        "ok": free_gb >= min_gb,
    }


def _ensure_writable_dirs(output_root: Path) -> List[str]:
    targets = [output_root, output_root / "checkpoints", output_root / "logs"]
    checked: List[str] = []
    for target in targets:
        target.mkdir(parents=True, exist_ok=True)
        probe = target / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        checked.append(str(target))
    return checked


def _capture_metadata(context: GuardContext) -> Path:
    run_dir = context.output_root / "runs" / context.run_id
    metadata_dir = run_dir / "metadata"
    snapshot_dir = metadata_dir / "config_snapshot"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    for cfg in [context.hardware_config, context.train_config, context.data_config]:
        if cfg.exists():
            shutil.copy2(cfg, snapshot_dir / cfg.name)

    hw_cfg = _load_yaml(context.hardware_config)
    hardware_profile = (
        hw_cfg.get("hardware", {}).get("profile_name")
        or hw_cfg.get("hardware", {}).get("accelerator_type")
        or "unknown"
    )

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": context.run_id,
        "hardware_profile_name": hardware_profile,
        "dataset_slug": context.dataset_slug,
        "dataset_root": str(context.dataset_root),
        "output_root": str(context.output_root),
        "config_snapshot_path": str(snapshot_dir),
        "git_commit_hash": _git_commit_hash(),
        "kaggle_attach_workflow": "required",
    }

    metadata_path = metadata_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def _print_header(title: str) -> None:
    print("=" * 80)
    print(title)
    print("=" * 80)


def _print_next_steps(context: GuardContext) -> None:
    latest_ckpt = context.output_root / "checkpoints" / "latest.ckpt"

    print("\nNEXT STEPS (copy/paste):")
    print("1) Stage 1 data prep/check")
    print(
        "python -m src.utils.runtime_guards "
        "--check-only "
        f"--dataset-slug {context.dataset_slug} "
        f"--dataset-root {context.dataset_root} "
        f"--output-root {context.output_root} "
        f"--hardware-config {context.hardware_config} "
        f"--train-config {context.train_config} "
        f"--data-config {context.data_config}"
    )

    print("\n2) Stage 1 training start")
    print(
        "python scripts/train_stage1.py "
        f"--config {context.train_config} "
        f"--hardware {context.hardware_config} "
        f"--dataset-root {context.dataset_root} "
        f"--output-root {context.output_root}"
    )

    print("\n3) Resume from latest checkpoint")
    print(
        "python scripts/train_stage1.py "
        f"--config {context.train_config} "
        f"--hardware {context.hardware_config} "
        f"--dataset-root {context.dataset_root} "
        f"--output-root {context.output_root} "
        f"--resume-from {latest_ckpt}"
    )

    print("\nKAGGLE USER FLOW GUIDANCE:")
    print("- Data is already on Kaggle. Use the Attach Dataset workflow in notebook UI.")
    print("- Do not manually upload large datasets into notebook storage.")
    print("- After training, publish artifacts by either:")
    print("  a) downloading files from /kaggle/working, or")
    print("  b) packaging checkpoints/logs as a new Kaggle Dataset version.")


def _build_context(args: argparse.Namespace) -> GuardContext:
    data_cfg = _load_yaml(Path(args.data_config))

    dataset_slug = _resolve_dataset_slug(data_cfg, args.dataset_slug)
    dataset_root = _resolve_dataset_root(data_cfg, dataset_slug, args.dataset_root)
    output_root = _resolve_output_root(args.output_root)

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.run_id is None:
        run_id = f"{run_id}-{uuid.uuid4().hex[:8]}"

    return GuardContext(
        dataset_slug=dataset_slug,
        dataset_root=dataset_root,
        output_root=output_root,
        hardware_config=Path(args.hardware_config).resolve(),
        train_config=Path(args.train_config).resolve(),
        data_config=Path(args.data_config).resolve(),
        min_working_gb=float(args.min_working_gb),
        run_id=run_id,
    )


def run_guards(context: GuardContext, check_only: bool = False) -> int:
    _print_header("Kaggle Runtime Guards")

    if not Path("/kaggle").exists():
        print("ERROR: /kaggle is not available. This bootstrap supports Kaggle runtime only.")
        return 2

    if not str(context.dataset_root).startswith("/kaggle/input"):
        print(
            "ERROR: Dataset root must be under /kaggle/input for attach-only workflow. "
            f"Got: {context.dataset_root}"
        )
        return 2

    if not context.dataset_root.exists():
        print(
            "ERROR: Dataset root is missing. Attach dataset in Kaggle UI first.\n"
            f"Expected path: {context.dataset_root}\n"
            f"Dataset slug: {context.dataset_slug}"
        )
        return 2

    gpu = _check_gpu()
    if not gpu["cuda_available"] or gpu["gpu_count"] < 1:
        print("ERROR: No visible GPU/CUDA device detected. Enable GPU accelerator in Kaggle settings.")
        return 2

    disk = _check_disk_space(Path("/kaggle/working"), context.min_working_gb)
    if not disk["ok"]:
        print(
            "ERROR: Insufficient free space in /kaggle/working. "
            f"Required >= {context.min_working_gb:.1f} GB, available {disk['free_gb']:.2f} GB."
        )
        return 2

    writable = _ensure_writable_dirs(context.output_root)

    print(f"GPU backend: {gpu['backend']}")
    print(f"GPU count: {gpu['gpu_count']}")
    print("GPU names: " + ", ".join(gpu["gpu_names"]))
    print(f"Disk free (/kaggle/working): {disk['free_gb']} GB")
    print(f"Writable dirs: {', '.join(writable)}")
    print(f"Dataset root found: {context.dataset_root}")

    metadata_path: Optional[Path] = None
    if not check_only:
        metadata_path = _capture_metadata(context)
        print(f"Metadata saved: {metadata_path}")

    _print_next_steps(context)

    if check_only:
        print("\nStatus: READY (checks only).")
    else:
        print("\nStatus: READY (checks + metadata capture complete).")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kaggle runtime guards and metadata capture")
    parser.add_argument(
        "--dataset-slug",
        type=str,
        default=os.environ.get("DATASET_SLUG"),
        help="Kaggle dataset slug owner/name",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=os.environ.get("DATASET_ROOT"),
        help="Mounted dataset root under /kaggle/input",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=os.environ.get("OUTPUT_ROOT", "/kaggle/working"),
        help="Output root under /kaggle/working",
    )
    parser.add_argument(
        "--hardware-config",
        type=str,
        default=os.environ.get("HARDWARE_CONFIG", "configs/hardware_p100.yaml"),
        help="Hardware config YAML path",
    )
    parser.add_argument(
        "--train-config",
        type=str,
        default=os.environ.get("TRAIN_CONFIG", "configs/train_stage1.yaml"),
        help="Training config YAML path",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default=os.environ.get("DATA_CONFIG", "configs/data_stage1.yaml"),
        help="Data config YAML path",
    )
    parser.add_argument(
        "--min-working-gb",
        type=float,
        default=float(os.environ.get("MIN_WORKING_GB", "10")),
        help="Minimum free GB required in /kaggle/working",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=os.environ.get("RUN_ID"),
        help="Optional fixed run id for deterministic reruns",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run guards without writing metadata files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    context = _build_context(args)
    return run_guards(context=context, check_only=bool(args.check_only))


if __name__ == "__main__":
    sys.exit(main())
