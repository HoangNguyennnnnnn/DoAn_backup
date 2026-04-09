"""CLI for Stage 2 latent dataset build from Stage 1 outputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import yaml


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.data.latent_dataset_builder import (
    LatentBuildConfig,
    LatentDatasetError,
    build_latent_dataset,
)


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise RuntimeError(f"Config must be YAML object: {path}")
    return _expand_env(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage 2 latent dataset from Stage 1 outputs")
    parser.add_argument("--train-config", type=str, default="configs/train_stage1.yaml")
    parser.add_argument("--data-config", type=str, default="configs/data_stage1.yaml")
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--checkpoint-preference", type=str, default="latest_step,best,interrupt,latest")
    parser.add_argument("--split", type=str, default="both", choices=["train", "test", "both"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--latent-schema-version", type=str, default="stage2-latent-v1")
    parser.add_argument("--latent-dtype", type=str, default="float32", choices=["float16", "float32"])
    parser.add_argument("--verify-hash", action="store_true")
    parser.add_argument("--enforce-kaggle", action="store_true")
    parser.add_argument("--report-path", type=str, default="logs/latent_dataset_build_report.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_cfg = load_yaml(Path(args.data_config))

    base_cfg = LatentBuildConfig.from_stage1_data_config(data_cfg)
    cfg = LatentBuildConfig(
        dataset_root=args.dataset_root or base_cfg.dataset_root,
        output_root=args.output_root or base_cfg.output_root,
        checkpoint_path=args.checkpoint_path,
        checkpoint_preference=tuple(
            [item.strip() for item in args.checkpoint_preference.split(",") if item.strip()]
        ),
        split=args.split,
        batch_size=max(1, int(args.batch_size)),
        device=args.device,
        latent_schema_version=args.latent_schema_version,
        latent_dtype=args.latent_dtype,
        verify_hash=bool(args.verify_hash),
        enforce_kaggle_paths=bool(args.enforce_kaggle),
    )

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = build_latent_dataset(
            config=cfg,
            fallback_train_cfg_path=Path(args.train_config),
            fallback_data_cfg_path=Path(args.data_config),
        )
        payload = {
            "status": "ok",
            "result": result,
            "reproducible_command": (
                f"python scripts/build_latent_dataset.py --train-config {args.train_config} "
                f"--data-config {args.data_config} --split {args.split} --batch-size {args.batch_size} "
                f"--latent-schema-version {args.latent_schema_version} --latent-dtype {args.latent_dtype}"
            ),
        }
    except (LatentDatasetError, RuntimeError, ValueError, FileNotFoundError) as exc:
        payload = {
            "status": "failed",
            "error": str(exc),
            "recovery": (
                "Verify checkpoint availability under /kaggle/working/checkpoints and ensure "
                "dataset_root/output_root paths are valid Kaggle mounts."
            ),
            "reproducible_command": (
                f"python scripts/build_latent_dataset.py --train-config {args.train_config} "
                f"--data-config {args.data_config} --checkpoint-preference {args.checkpoint_preference} "
                f"--split {args.split} --batch-size {args.batch_size}"
            ),
        }

    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
