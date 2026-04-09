"""Stage 1 decode sanity runner for Shape SC-VAE checkpoints.

Kaggle-first utility that loads best/latest checkpoints, decodes sample latent vectors,
exports mesh artifacts, and writes functional pass/fail summaries.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import trimesh
import yaml

from src.models import ShapeSCVAE


class DecodeSanityError(RuntimeError):
    """Raised when checkpoint decode sanity cannot be executed safely."""


@dataclass
class CheckpointDecodeResult:
    checkpoint_name: str
    checkpoint_path: str
    loaded: bool
    decode_ok: bool
    export_ok: bool
    mesh_valid_count: int
    mesh_total_count: int
    message: str
    output_dir: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_name": self.checkpoint_name,
            "checkpoint_path": self.checkpoint_path,
            "loaded": self.loaded,
            "decode_ok": self.decode_ok,
            "export_ok": self.export_ok,
            "mesh_valid_count": self.mesh_valid_count,
            "mesh_total_count": self.mesh_total_count,
            "message": self.message,
            "output_dir": self.output_dir,
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
        raise DecodeSanityError(f"Config must be a YAML object: {path}")
    return _expand_env(payload)


def _resolve_output_root(args: argparse.Namespace, data_cfg: Mapping[str, Any]) -> Path:
    return Path(
        args.output_root
        or str(data_cfg.get("paths", {}).get("output_root", ""))
        or os.environ.get("OUTPUT_ROOT", "")
        or "outputs"
    ).resolve()


def _resolve_checkpoint_dir(
    args: argparse.Namespace,
    train_cfg: Mapping[str, Any],
    output_root: Path,
) -> Path:
    if args.checkpoint_dir:
        return Path(args.checkpoint_dir).resolve()

    ckpt_cfg = train_cfg.get("checkpointing", {})
    configured = str(ckpt_cfg.get("autoresume_checkpoint_path", "")).strip()
    if configured:
        return Path(configured).resolve()

    return (output_root / "checkpoints").resolve()


def _checkpoint_candidates(checkpoint_dir: Path) -> List[tuple[str, Path]]:
    ordered_names = ["best.ckpt", "latest.ckpt", "latest_step.ckpt", "interrupt.ckpt"]
    candidates: List[tuple[str, Path]] = []
    for name in ordered_names:
        path = checkpoint_dir / name
        if path.exists() and path.is_file():
            label = name.replace(".ckpt", "")
            candidates.append((label, path))
    return candidates


def _load_training_metrics(log_dir: Path) -> Dict[str, Any]:
    metrics_path = log_dir / "stage1_training_metrics.jsonl"
    if not metrics_path.exists():
        return {
            "metrics_path": str(metrics_path),
            "available": False,
            "val_total_loss": {
                "count": 0,
                "first": None,
                "last": None,
                "min": None,
                "trend": "unavailable",
            },
        }

    val_losses: List[float] = []
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "val_total_loss" in payload and isinstance(payload["val_total_loss"], (int, float)):
            val_losses.append(float(payload["val_total_loss"]))

    if not val_losses:
        trend = "unavailable"
        first_val = None
        last_val = None
        min_val = None
    else:
        first_val = float(val_losses[0])
        last_val = float(val_losses[-1])
        min_val = float(min(val_losses))
        if math.isclose(first_val, last_val, rel_tol=1e-6, abs_tol=1e-8):
            trend = "flat"
        elif last_val < first_val:
            trend = "improving"
        else:
            trend = "worsening"

    return {
        "metrics_path": str(metrics_path),
        "available": bool(val_losses),
        "val_total_loss": {
            "count": len(val_losses),
            "first": first_val,
            "last": last_val,
            "min": min_val,
            "trend": trend,
        },
    }


def _voxel_to_mesh(voxel: torch.Tensor, threshold: float) -> trimesh.Trimesh:
    if voxel.ndim != 3:
        raise DecodeSanityError(f"Expected voxel shape (R,R,R), received {tuple(voxel.shape)}")

    occupancy = (voxel.detach().cpu().numpy() >= threshold)
    if occupancy.sum() == 0:
        raise DecodeSanityError("Decoded voxel has zero occupied cells after thresholding.")

    grid = trimesh.voxel.VoxelGrid(encoding=occupancy)
    mesh = grid.as_boxes()
    if not isinstance(mesh, trimesh.Trimesh):
        raise DecodeSanityError("Voxel-to-mesh conversion did not produce a Trimesh object.")

    if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
        raise DecodeSanityError("Generated mesh is empty (no vertices or faces).")

    return mesh


def _export_and_validate_mesh(mesh: trimesh.Trimesh, output_path: Path) -> Dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(output_path))

    reloaded = trimesh.load(str(output_path), force="mesh", process=False)
    if not isinstance(reloaded, trimesh.Trimesh):
        raise DecodeSanityError(f"Exported artifact is not reloadable as mesh: {output_path}")
    if reloaded.vertices.shape[0] == 0 or reloaded.faces.shape[0] == 0:
        raise DecodeSanityError(f"Reloaded mesh has empty geometry: {output_path}")

    return {
        "mesh_path": str(output_path),
        "vertices": int(reloaded.vertices.shape[0]),
        "faces": int(reloaded.faces.shape[0]),
        "watertight": bool(reloaded.is_watertight),
    }


def _decode_from_checkpoint(
    checkpoint_name: str,
    checkpoint_path: Path,
    train_cfg: Mapping[str, Any],
    data_cfg: Mapping[str, Any],
    device: torch.device,
    num_samples: int,
    threshold: float,
    export_root: Path,
) -> tuple[CheckpointDecodeResult, List[Dict[str, Any]]]:
    outputs: List[Dict[str, Any]] = []

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as exc:
        return (
            CheckpointDecodeResult(
                checkpoint_name=checkpoint_name,
                checkpoint_path=str(checkpoint_path),
                loaded=False,
                decode_ok=False,
                export_ok=False,
                mesh_valid_count=0,
                mesh_total_count=0,
                message=f"checkpoint load failed: {exc}",
                output_dir=str(export_root / checkpoint_name),
            ),
            outputs,
        )

    model = ShapeSCVAE.from_stage1_configs(train_cfg, data_cfg).to(device)

    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    except Exception as exc:
        return (
            CheckpointDecodeResult(
                checkpoint_name=checkpoint_name,
                checkpoint_path=str(checkpoint_path),
                loaded=False,
                decode_ok=False,
                export_ok=False,
                mesh_valid_count=0,
                mesh_total_count=0,
                message=f"state_dict load failed: {exc}",
                output_dir=str(export_root / checkpoint_name),
            ),
            outputs,
        )

    model.eval()

    sample_out_dir = export_root / checkpoint_name
    valid_count = 0

    with torch.no_grad():
        latent = torch.randn(num_samples, model.config.latent_dim, device=device)
        recon = model.decode(latent)

        if recon.ndim != 5:
            raise DecodeSanityError(
                f"Decoded output rank mismatch for {checkpoint_name}. Expected 5D, got {tuple(recon.shape)}"
            )

        expected_res = model.config.input_resolution
        if tuple(recon.shape[-3:]) != (expected_res, expected_res, expected_res):
            raise DecodeSanityError(
                "Decoded output resolution mismatch. "
                f"Expected {(expected_res, expected_res, expected_res)}, got {tuple(recon.shape[-3:])}."
            )

        for idx in range(num_samples):
            voxel = recon[idx, 0].detach().cpu()
            mesh = _voxel_to_mesh(voxel, threshold=threshold)
            mesh_info = _export_and_validate_mesh(mesh, sample_out_dir / f"sample_{idx:03d}.obj")
            outputs.append(mesh_info)
            valid_count += 1

    result = CheckpointDecodeResult(
        checkpoint_name=checkpoint_name,
        checkpoint_path=str(checkpoint_path),
        loaded=True,
        decode_ok=True,
        export_ok=(valid_count == num_samples),
        mesh_valid_count=valid_count,
        mesh_total_count=num_samples,
        message="ok",
        output_dir=str(sample_out_dir),
    )
    return result, outputs


def _write_reports(summary: Dict[str, Any], log_dir: Path) -> Dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    report_json = log_dir / "stage1_decode_sanity_report.json"
    report_md = log_dir / "stage1_decode_sanity_report.md"

    report_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    checkpoint_results = summary.get("checkpoints", [])
    trend = summary.get("reconstruction_trend", {})
    trend_info = trend.get("val_total_loss", {})

    md_lines = [
        "# Stage 1 Decode Sanity Report",
        "",
        f"- device: {summary['runtime']['device']}",
        f"- checkpoint_dir: {summary['runtime']['checkpoint_dir']}",
        f"- pass_count: {summary['decode_summary']['pass_count']}",
        f"- fail_count: {summary['decode_summary']['fail_count']}",
        f"- all_passed: {summary['decode_summary']['all_passed']}",
        "",
        "## Checkpoint Results",
    ]

    if checkpoint_results:
        for item in checkpoint_results:
            md_lines.append(
                f"- {item['checkpoint_name']}: loaded={item['loaded']}, decode_ok={item['decode_ok']}, "
                f"export_ok={item['export_ok']}, valid={item['mesh_valid_count']}/{item['mesh_total_count']}, "
                f"message={item['message']}"
            )
    else:
        md_lines.append("- None")

    md_lines.extend(
        [
            "",
            "## Reconstruction Trend",
            f"- metrics_path: {trend.get('metrics_path')}",
            f"- count: {trend_info.get('count')}",
            f"- first: {trend_info.get('first')}",
            f"- last: {trend_info.get('last')}",
            f"- min: {trend_info.get('min')}",
            f"- trend: {trend_info.get('trend')}",
            "",
            "## Note",
            "- Kaggle/Linux checkpoint behavior is authoritative for final Stage 1 validation.",
        ]
    )
    report_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    summary["report_json"] = str(report_json)
    summary["report_md"] = str(report_md)
    return summary


def run_decode_sanity(args: argparse.Namespace) -> Dict[str, Any]:
    train_cfg = _load_yaml(Path(args.config))
    data_cfg = _load_yaml(Path(args.data_config))

    output_root = _resolve_output_root(args, data_cfg)
    checkpoint_dir = _resolve_checkpoint_dir(args, train_cfg, output_root)
    log_dir = output_root / "logs"
    export_root = output_root / "inference_decode_sanity"

    candidates: List[tuple[str, Path]] = []
    preflight_errors: List[str] = []
    if not checkpoint_dir.exists():
        preflight_errors.append(
            f"Checkpoint directory does not exist: {checkpoint_dir}. Use Kaggle/Linux output checkpoints or pass --checkpoint-dir."
        )
    else:
        candidates = _checkpoint_candidates(checkpoint_dir)
        if not candidates:
            preflight_errors.append(
                f"No checkpoints found in {checkpoint_dir}. Expected best.ckpt and/or latest.ckpt."
            )

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    checkpoint_results: List[CheckpointDecodeResult] = []
    exported_meshes: Dict[str, List[Dict[str, Any]]] = {}

    for name, path in candidates:
        try:
            result, mesh_infos = _decode_from_checkpoint(
                checkpoint_name=name,
                checkpoint_path=path,
                train_cfg=train_cfg,
                data_cfg=data_cfg,
                device=device,
                num_samples=max(1, int(args.num_samples)),
                threshold=float(args.threshold),
                export_root=export_root,
            )
        except Exception as exc:
            result = CheckpointDecodeResult(
                checkpoint_name=name,
                checkpoint_path=str(path),
                loaded=False,
                decode_ok=False,
                export_ok=False,
                mesh_valid_count=0,
                mesh_total_count=max(1, int(args.num_samples)),
                message=str(exc),
                output_dir=str(export_root / name),
            )
            mesh_infos = []

        checkpoint_results.append(result)
        exported_meshes[name] = mesh_infos

    trend = _load_training_metrics(log_dir)

    passed = [r for r in checkpoint_results if r.decode_ok and r.export_ok]
    failed = [r for r in checkpoint_results if not (r.decode_ok and r.export_ok)]

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "device": str(device),
            "output_root": str(output_root),
            "checkpoint_dir": str(checkpoint_dir),
            "kaggle_first_note": "Kaggle/Linux checkpoint outputs are authoritative for final validation.",
        },
        "decode_summary": {
            "checkpoint_count": len(checkpoint_results),
            "pass_count": len(passed),
            "fail_count": len(failed) + (1 if preflight_errors else 0),
            "all_passed": len(failed) == 0 and not preflight_errors,
        },
        "checkpoints": [item.as_dict() for item in checkpoint_results],
        "reconstruction_trend": trend,
        "exported_meshes": exported_meshes,
        "preflight_errors": preflight_errors,
    }
    return _write_reports(summary, log_dir)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 ShapeSCVAE decode sanity runner")
    parser.add_argument("--config", type=str, default="configs/train_stage1.yaml")
    parser.add_argument("--data-config", type=str, default="configs/data_stage1.yaml")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=os.environ.get("OUTPUT_ROOT"))
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = run_decode_sanity(args)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
