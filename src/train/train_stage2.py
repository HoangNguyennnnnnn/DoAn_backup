"""Stage 2 smoke trainer for latent UNet and improved mean-flow objective."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.stage2_latent_dataset import Stage2LatentDataset
from src.models.latent_generator import build_latent_generator
from src.models.mean_flow_objective import ImprovedMeanFlowObjective


class Stage2TrainingError(RuntimeError):
    """Raised when the Stage 2 smoke trainer fails."""


class Stage2CheckpointManager:
    """Manage smoke checkpoint persistence and autoresume precedence."""

    def __init__(self, checkpoint_dir: Path, keep_last_n: int = 1):
        self.checkpoint_dir = checkpoint_dir
        self.keep_last_n = max(1, int(keep_last_n))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, filename: str, payload: Dict[str, Any]) -> Path:
        path = self.checkpoint_dir / filename
        torch.save(payload, path)
        return path

    def save_latest(self, payload: Dict[str, Any]) -> Path:
        path = self._save("latest.ckpt", payload)
        self._save("latest_step.ckpt", payload)
        self._prune_step_checkpoints()
        return path

    def save_best(self, payload: Dict[str, Any]) -> Path:
        return self._save("best.ckpt", payload)

    def save_interrupt(self, payload: Dict[str, Any]) -> Path:
        return self._save("interrupt.ckpt", payload)

    def save_smoke_latest(self, payload: Dict[str, Any]) -> Path:
        return self._save("smoke_latest.ckpt", payload)

    def find_resume_checkpoint(self) -> Optional[Path]:
        for candidate in ("latest_step.ckpt", "interrupt.ckpt", "latest.ckpt", "best.ckpt"):
            path = self.checkpoint_dir / candidate
            if path.exists():
                return path
        return None

    def _prune_step_checkpoints(self) -> None:
        steps = sorted(self.checkpoint_dir.glob("step_*.ckpt"))
        if len(steps) <= self.keep_last_n:
            return
        for stale in steps[: len(steps) - self.keep_last_n]:
            stale.unlink(missing_ok=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _record_metrics(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload)) + "\n")


def _expand_env(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def _load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise Stage2TrainingError(f"Config must be a YAML object: {path}")
    return _expand_env(payload)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_name)


def _autocast_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _make_optimizer(model: torch.nn.Module, train_cfg: Mapping[str, Any]) -> torch.optim.Optimizer:
    opt_cfg = train_cfg.get("optimizer", {})
    name = str(opt_cfg.get("name", "adam")).lower().strip()
    lr = float(opt_cfg.get("lr", 5e-4))
    betas = tuple(float(v) for v in opt_cfg.get("betas", [0.9, 0.999]))
    eps = float(opt_cfg.get("epsilon", 1e-8))
    wd = float(opt_cfg.get("weight_decay", 0.0))

    if name != "adam":
        raise Stage2TrainingError(f"Unsupported optimizer: {name}")
    return torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=wd)


def _make_scheduler(optimizer: torch.optim.Optimizer, train_cfg: Mapping[str, Any]) -> Optional[Any]:
    sched_cfg = train_cfg.get("scheduler", {})
    name = str(sched_cfg.get("name", "constant")).lower().strip()
    kwargs = dict(sched_cfg.get("scheduler_kwargs", {}))

    if not name or name == "constant":
        return None
    if name == "cosine_annealing":
        t_max = int(kwargs.get("T_max", 1))
        eta_min = float(kwargs.get("eta_min", 1e-6))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    raise Stage2TrainingError(f"Unsupported scheduler: {name}")


def _stage2_dataset_limits(data_cfg: Mapping[str, Any]) -> Tuple[int, int]:
    smoke_cfg = data_cfg.get("smoke_config", {})
    train_limit = int(smoke_cfg.get("num_train_samples", 0))
    val_limit = int(smoke_cfg.get("num_val_samples", 0))
    return train_limit, val_limit


def _build_dataloader(
    data_cfg: Mapping[str, Any],
    split: str,
    sample_limit: Optional[int],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = Stage2LatentDataset.from_stage2_configs(data_cfg, split=split, sample_limit=sample_limit)
    loading_cfg = data_cfg.get("loading", {})
    num_workers = int(loading_cfg.get("num_workers", 0))
    pin_memory = bool(loading_cfg.get("pin_memory", True))
    drop_last = bool(loading_cfg.get("drop_last_batch", False))
    return DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=Stage2LatentDataset.collate_fn,
    )


def _resolve_output_root(args: argparse.Namespace, data_cfg: Mapping[str, Any]) -> Path:
    if args.output_root:
        return Path(args.output_root)
    paths = data_cfg.get("paths", {})
    output_root = str(paths.get("output_root") or "").strip()
    if output_root and "${" not in output_root:
        return Path(output_root)
    return Path(os.environ.get("OUTPUT_ROOT", "/kaggle/working"))


def _resolve_resume_precedence(checkpoint_dir: Path) -> Dict[str, Optional[Path]]:
    candidates = {
        "latest_step": checkpoint_dir / "latest_step.ckpt",
        "interrupt": checkpoint_dir / "interrupt.ckpt",
        "latest": checkpoint_dir / "latest.ckpt",
        "best": checkpoint_dir / "best.ckpt",
    }
    return {name: path if path.exists() else None for name, path in candidates.items()}


def _save_checkpoint(checkpoint_dir: Path, filename: str, payload: Dict[str, Any]) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / filename
    torch.save(payload, path)
    return path


def _run_validation(
    model: torch.nn.Module,
    objective: ImprovedMeanFlowObjective,
    loader: DataLoader,
    device: torch.device,
    mixed_precision: bool,
    global_step: int,
) -> Dict[str, float]:
    model.eval()
    totals = {"total_loss": 0.0, "flow_loss": 0.0, "v_loss": 0.0}
    batches = 0

    with torch.no_grad():
        for batch in loader:
            tokens = batch["tokens"].to(device, non_blocking=True)
            class_labels = batch["class_id"].to(device, non_blocking=True)
            with torch.amp.autocast(
                device_type="cuda",
                dtype=_autocast_dtype(),
                enabled=bool(mixed_precision and device.type == "cuda"),
            ):
                losses = objective.compute_loss(
                    model=model,
                    clean_latents=tokens,
                    class_labels=class_labels,
                    step=global_step,
                )

            for key in totals:
                totals[key] += float(losses[key].detach().cpu().item())
            batches += 1

    if batches == 0:
        return {key: math.inf for key in totals}

    return {key: value / batches for key, value in totals.items()}


def _save_run_metadata(
    run_dir: Path,
    train_cfg: Mapping[str, Any],
    hardware_cfg: Mapping[str, Any],
    data_cfg: Mapping[str, Any],
    args: argparse.Namespace,
) -> Path:
    metadata_dir = run_dir / "metadata"
    snapshot_dir = metadata_dir / "config_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    for source in (args.config, args.hardware, args.data_config):
        source_path = Path(source)
        if source_path.exists():
            shutil.copy2(source_path, snapshot_dir / source_path.name)

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_dir.name,
        "output_root": str(run_dir.parent.parent),
        "run_dir": str(run_dir),
        "train_config": str(Path(args.config).resolve()),
        "hardware_config": str(Path(args.hardware).resolve()),
        "data_config": str(Path(args.data_config).resolve()),
        "stage1_checkpoint": args.stage1_checkpoint,
        "resume_from": args.resume_from,
        "resume_precedence": ["latest_step.ckpt", "interrupt.ckpt", "latest.ckpt", "best.ckpt"],
        "validation_export_precedence": ["best.ckpt", "latest.ckpt"],
        "smoke_epochs": int(train_cfg.get("training", {}).get("max_epochs", 1)),
        "schema_version": data_cfg.get("latent_loading", {}).get("schema_version", "stage2-latent-v1"),
        "hardware_profile": hardware_cfg.get("hardware", {}).get("profile_name", "unknown"),
    }
    metadata_path = metadata_dir / "run_metadata.json"
    _write_json(metadata_path, metadata)
    return metadata_path


def train(args: argparse.Namespace) -> Dict[str, Any]:
    train_cfg = _load_yaml(Path(args.config))
    hardware_cfg = _load_yaml(Path(args.hardware))
    data_cfg = _load_yaml(Path(args.data_config))
    resume_cfg = train_cfg.get("resume", {})

    seed = int(train_cfg.get("seed", 42))
    _set_seed(seed)

    output_root = _resolve_output_root(args, data_cfg)
    checkpoint_dir = output_root / "checkpoints"
    log_dir = output_root / "logs"
    run_dir = output_root / "runs" / (args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metadata").mkdir(parents=True, exist_ok=True)

    smoke_cfg = data_cfg.get("smoke_config", {})
    train_limit, val_limit = _stage2_dataset_limits(data_cfg)

    train_batch_size = int(smoke_cfg.get("batch_size_smoke", data_cfg.get("loading", {}).get("batch_size", 8)))
    val_batch_size = max(1, int(train_batch_size))

    train_loader = _build_dataloader(
        data_cfg=data_cfg,
        split="train",
        sample_limit=train_limit or None,
        batch_size=train_batch_size,
        shuffle=bool(data_cfg.get("loading", {}).get("shuffle_train", True)),
    )
    val_loader = _build_dataloader(
        data_cfg=data_cfg,
        split="test",
        sample_limit=val_limit or None,
        batch_size=val_batch_size,
        shuffle=False,
    )

    device = _resolve_device(args.device)
    model = build_latent_generator(train_cfg, data_cfg).to(device)
    objective = ImprovedMeanFlowObjective.from_stage2_configs(train_cfg, data_cfg)
    optimizer = _make_optimizer(model, train_cfg)
    scheduler = _make_scheduler(optimizer, train_cfg)
    checkpointing_cfg = train_cfg.get("checkpointing", {})
    checkpoint_manager = Stage2CheckpointManager(
        checkpoint_dir=checkpoint_dir,
        keep_last_n=int(checkpointing_cfg.get("keep_last_n", 1)),
    )

    loop_cfg = train_cfg.get("loop", {})
    mixed_precision = bool(loop_cfg.get("mixed_precision", False))
    max_epochs = int(train_cfg.get("training", {}).get("max_epochs", 1))
    grad_clip = float(loop_cfg.get("gradient_clipping_max_norm", 1.0))

    smoke_report = model.forward_sanity_check(batch_size=2, device=str(device))
    metadata_path = _save_run_metadata(run_dir, train_cfg, hardware_cfg, data_cfg, args)

    train_log_path = log_dir / "stage2_smoke_metrics.jsonl"
    summary_path = log_dir / "stage2_smoke_summary.json"
    integrity_path = log_dir / "stage2_checkpoint_integrity.json"
    best_val = float("inf")
    global_step = 0

    resume_state: Dict[str, Any] = {}
    resume_path: Optional[Path] = None
    if args.resume_from:
        resume_path = Path(args.resume_from)
    else:
        config_resume = str(resume_cfg.get("resume_from_checkpoint") or "").strip()
        if config_resume and "${" not in config_resume:
            resume_path = Path(config_resume)
    if resume_path is None and bool(checkpointing_cfg.get("autoresume_enabled", True)):
        resume_path = checkpoint_manager.find_resume_checkpoint()

    if resume_path is not None and resume_path.exists():
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=bool(train_cfg.get("resume", {}).get("strict_model_loading", False)))
        if bool(resume_cfg.get("resume_optimizer_state", True)) and checkpoint.get("optimizer_state_dict") is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and bool(resume_cfg.get("resume_optimizer_state", True)) and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        resume_state = {
            "resume_path": str(resume_path),
            "epoch": int(checkpoint.get("epoch", 0)),
            "global_step": int(checkpoint.get("global_step", 0)),
            "best_val_loss": float(checkpoint.get("best_val_loss", float("inf"))),
            "checkpoint_keys": sorted(list(checkpoint.keys())),
        }
        best_val = resume_state["best_val_loss"]
        global_step = resume_state["global_step"]
        start_epoch = max(0, resume_state["epoch"] if bool(resume_cfg.get("resume_epoch", True)) else 0)
    else:
        start_epoch = 0

    for epoch in range(start_epoch, max_epochs):
        model.train()
        for batch in train_loader:
            tokens = batch["tokens"].to(device, non_blocking=True)
            class_labels = batch["class_id"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type="cuda",
                dtype=_autocast_dtype(),
                enabled=bool(mixed_precision and device.type == "cuda"),
            ):
                losses = objective.compute_loss(
                    model=model,
                    clean_latents=tokens,
                    class_labels=class_labels,
                    step=global_step,
                )
                loss = losses["total_loss"]

            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            global_step += 1
            payload = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "epoch": epoch,
                "global_step": global_step,
                "train_total_loss": float(losses["total_loss"].detach().cpu().item()),
                "train_flow_loss": float(losses["flow_loss"].detach().cpu().item()),
                "train_v_loss": float(losses["v_loss"].detach().cpu().item()),
                "guidance_dropout_prob": float(losses["guidance_dropout_prob"].detach().cpu().item()),
            }
            with train_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")

            if global_step % max(1, int(train_cfg.get("loop", {}).get("log_interval_steps", 10))) == 0:
                print(json.dumps(payload))

        if scheduler is not None:
            scheduler.step()

        val_metrics = _run_validation(
            model=model,
            objective=objective,
            loader=val_loader,
            device=device,
            mixed_precision=mixed_precision,
            global_step=global_step,
        )
        if val_metrics["total_loss"] < best_val:
            best_val = val_metrics["total_loss"]
            best_payload = {
                "epoch": epoch,
                "global_step": global_step,
                "best_val_loss": best_val,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_config": dict(train_cfg),
                "data_config": dict(data_cfg),
                "hardware_config": dict(hardware_cfg),
                "objective_config": objective.config.__dict__,
                "smoke_report": smoke_report,
                "checkpoint_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "checkpoint_kind": "best",
            }
            if scheduler is not None:
                best_payload["scheduler_state_dict"] = scheduler.state_dict()
            best_path = checkpoint_manager.save_best(best_payload)

    latest_payload = {
        "epoch": max_epochs,
        "global_step": global_step,
        "best_val_loss": best_val,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_config": dict(train_cfg),
        "data_config": dict(data_cfg),
        "hardware_config": dict(hardware_cfg),
        "objective_config": objective.config.__dict__,
        "smoke_report": smoke_report,
        "checkpoint_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_kind": "latest",
    }
    if scheduler is not None:
        latest_payload["scheduler_state_dict"] = scheduler.state_dict()

    latest_path = checkpoint_manager.save_latest(latest_payload)
    smoke_path = checkpoint_manager.save_smoke_latest(latest_payload)

    integrity_report = {
        "resume_precedence": ["latest_step.ckpt", "interrupt.ckpt", "latest.ckpt", "best.ckpt"],
        "validation_export_precedence": ["best.ckpt", "latest.ckpt"],
        "latest_exists": (checkpoint_dir / "latest.ckpt").exists(),
        "latest_step_exists": (checkpoint_dir / "latest_step.ckpt").exists(),
        "best_exists": (checkpoint_dir / "best.ckpt").exists(),
        "interrupt_exists": (checkpoint_dir / "interrupt.ckpt").exists(),
        "smoke_latest_exists": (checkpoint_dir / "smoke_latest.ckpt").exists(),
        "resume_state": resume_state or None,
        "latest_path": str(latest_path),
        "best_path": str(checkpoint_dir / "best.ckpt") if (checkpoint_dir / "best.ckpt").exists() else None,
        "smoke_path": str(smoke_path),
    }
    _write_json(integrity_path, integrity_report)

    summary = {
        "status": "ok",
        "stage1_checkpoint": args.stage1_checkpoint,
        "checkpoint_path": str(latest_path),
        "smoke_checkpoint_path": str(smoke_path),
        "best_checkpoint_path": str(checkpoint_dir / "best.ckpt") if (checkpoint_dir / "best.ckpt").exists() else None,
        "train_log_path": str(train_log_path),
        "integrity_report_path": str(integrity_path),
        "metadata_path": str(metadata_path),
        "best_val_loss": float(best_val),
        "global_step": global_step,
        "train_loader_size": len(train_loader.dataset),
        "val_loader_size": len(val_loader.dataset),
        "smoke_report": smoke_report,
        "train_config": train_cfg.get("training", {}),
        "resume_state": resume_state or None,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2 latent UNet smoke trainer")
    parser.add_argument("--config", type=str, default="configs/train_stage2.yaml")
    parser.add_argument("--hardware", type=str, default="configs/hardware_p100.yaml")
    parser.add_argument("--data-config", type=str, default="configs/data_stage2.yaml")
    parser.add_argument("--stage1-checkpoint", type=str, default=os.environ.get("STAGE1_CHECKPOINT_PATH"))
    parser.add_argument("--output-root", type=str, default=os.environ.get("OUTPUT_ROOT"))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--resume-from", type=str, default=os.environ.get("RESUME_CHECKPOINT_PATH"))
    parser.add_argument("--run-id", type=str, default=os.environ.get("RUN_ID"))
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = train(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())