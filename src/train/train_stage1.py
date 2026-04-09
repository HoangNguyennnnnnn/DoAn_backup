"""Stage 1 Shape SC-VAE trainer with Kaggle-first resume and recovery behavior."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from src.data.dataset_adapter import AdapterConfig, KaggleDatasetAdapter, SampleRecord
from src.data.mesh_to_feature import MeshFeatureConfig, MeshToFeatureBuilder
from src.data.off_to_obj_converter import ConverterConfig, OffToObjConverter
from src.models import ShapeSCVAE, run_shape_sc_vae_sanity


class Stage1TrainingError(RuntimeError):
    """Raised when Stage 1 trainer cannot continue safely."""


class EpochRestartRequested(RuntimeError):
    """Signals that current epoch should restart after OOM fallback adjustments."""


@dataclass
class RuntimePaths:
    dataset_root: Path
    output_root: Path
    checkpoint_dir: Path
    log_dir: Path
    run_dir: Path
    metadata_dir: Path


class OVoxelRecordDataset(Dataset):
    """Loads cached OVoxel tensors from cache index by sample records."""

    def __init__(
        self,
        records: Sequence[SampleRecord],
        tensor_refs: Mapping[str, Mapping[str, Any]],
    ):
        self.records = list(records)
        self.tensor_refs = tensor_refs

        if not self.records:
            raise Stage1TrainingError("Dataset split has zero records.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        key = str(record.relative_path).replace("\\", "/")
        ref = self.tensor_refs.get(key)
        if not isinstance(ref, Mapping):
            raise Stage1TrainingError(
                f"Tensor reference missing for key={key}. "
                "Run pipeline smoke / feature build before training."
            )

        tensor_path = Path(str(ref.get("tensor_path", "")))
        if not tensor_path.exists():
            raise Stage1TrainingError(f"Tensor file missing for key={key}: {tensor_path}")

        voxel = torch.load(tensor_path, map_location="cpu")
        if not isinstance(voxel, torch.Tensor) or voxel.ndim != 4:
            raise Stage1TrainingError(
                f"Invalid tensor payload for key={key}. Expected shape (C,R,R,R)."
            )

        label = torch.tensor(int(record.class_id), dtype=torch.long)
        return voxel, label


class OOMFallbackManager:
    """Applies configurable backoff when CUDA OOM occurs."""

    def __init__(
        self,
        batch_size: int,
        grad_accum_steps: int,
        policy: Mapping[str, Any],
    ):
        self.batch_size = int(batch_size)
        self.grad_accum_steps = int(grad_accum_steps)
        self.policy = dict(policy)

        self.min_batch_size = max(1, int(self.policy.get("min_batch_size", 1)))
        self.max_grad_accum_steps = max(
            self.grad_accum_steps,
            int(self.policy.get("max_gradient_accumulation_steps", 32)),
        )
        self.enabled = bool(self.policy.get("enabled", True))
        self.order = list(self.policy.get("order", ["batch_size", "grad_accumulation"]))

    def apply(self) -> Optional[str]:
        if not self.enabled:
            return None

        for action in self.order:
            if action == "batch_size" and self.batch_size > self.min_batch_size:
                next_bs = max(self.min_batch_size, self.batch_size // 2)
                if next_bs < self.batch_size:
                    self.batch_size = next_bs
                    return f"batch_size->{self.batch_size}"

            if action == "grad_accumulation" and self.grad_accum_steps < self.max_grad_accum_steps:
                self.grad_accum_steps = min(self.max_grad_accum_steps, self.grad_accum_steps * 2)
                return f"grad_accumulation->{self.grad_accum_steps}"

        return None


class Stage1CheckpointManager:
    """Persists latest_step, best, and interrupt checkpoints with retention policy."""

    def __init__(self, checkpoint_dir: Path, keep_last_n: int = 3):
        self.checkpoint_dir = checkpoint_dir
        self.keep_last_n = max(1, int(keep_last_n))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, name: str, payload: Mapping[str, Any]) -> Path:
        path = self.checkpoint_dir / name
        torch.save(dict(payload), path)
        return path

    def save_latest_step(self, payload: Mapping[str, Any], global_step: int) -> Path:
        step_name = f"step_{global_step:09d}.ckpt"
        step_path = self._save(step_name, payload)
        self._save("latest_step.ckpt", payload)
        self._save("latest.ckpt", payload)
        self._prune_step_checkpoints()
        return step_path

    def save_best(self, payload: Mapping[str, Any]) -> Path:
        return self._save("best.ckpt", payload)

    def save_interrupt(self, payload: Mapping[str, Any]) -> Path:
        return self._save("interrupt.ckpt", payload)

    def find_autoresume_candidate(self) -> Optional[Path]:
        for name in ("latest_step.ckpt", "interrupt.ckpt", "latest.ckpt", "best.ckpt"):
            path = self.checkpoint_dir / name
            if path.exists():
                return path
        return None

    def _prune_step_checkpoints(self) -> None:
        steps = sorted(self.checkpoint_dir.glob("step_*.ckpt"))
        if len(steps) <= self.keep_last_n:
            return

        remove_count = len(steps) - self.keep_last_n
        for stale in steps[:remove_count]:
            stale.unlink(missing_ok=True)


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
        raise Stage1TrainingError(f"Config must be a YAML object: {path}")
    return _expand_env(payload)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _is_oom_error(exc: RuntimeError) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda oom" in text


def _resolve_runtime_paths(args: argparse.Namespace, data_cfg: Mapping[str, Any]) -> RuntimePaths:
    dataset_root = Path(
        args.dataset_root
        or str(data_cfg.get("paths", {}).get("dataset_root", ""))
        or os.environ.get("DATASET_ROOT", "")
        or "/kaggle/input/modelnet40-princeton-3d-object-dataset"
    ).resolve()

    output_root = Path(
        args.output_root
        or str(data_cfg.get("paths", {}).get("output_root", ""))
        or os.environ.get("OUTPUT_ROOT", "")
        or "/kaggle/working"
    ).resolve()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / "runs" / run_id

    checkpoint_dir = output_root / "checkpoints"
    log_dir = output_root / "logs"
    metadata_dir = run_dir / "metadata"

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    return RuntimePaths(
        dataset_root=dataset_root,
        output_root=output_root,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        run_dir=run_dir,
        metadata_dir=metadata_dir,
    )


def _write_run_metadata(
    runtime: RuntimePaths,
    args: argparse.Namespace,
    train_cfg: Mapping[str, Any],
    hardware_cfg: Mapping[str, Any],
    data_cfg: Mapping[str, Any],
) -> Path:
    snapshot_dir = runtime.metadata_dir / "config_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, snapshot_dir / Path(args.config).name)
    shutil.copy2(args.hardware, snapshot_dir / Path(args.hardware).name)
    shutil.copy2(args.data_config, snapshot_dir / Path(args.data_config).name)

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": runtime.run_dir.name,
        "dataset_root": str(runtime.dataset_root),
        "output_root": str(runtime.output_root),
        "train_config": str(Path(args.config).resolve()),
        "hardware_config": str(Path(args.hardware).resolve()),
        "data_config": str(Path(args.data_config).resolve()),
        "device": args.device,
        "autoresume": bool(args.autoresume),
        "contract_smoke": bool(args.contract_smoke),
        "oom_policy": train_cfg.get("oom_fallback", {}),
        "seed": int(train_cfg.get("seed", 42)),
        "hardware_profile": hardware_cfg.get("hardware", {}).get("profile_name", "unknown"),
    }

    metadata_path = runtime.metadata_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def _resolve_dataloader_knobs(
    data_cfg: Mapping[str, Any],
    hardware_cfg: Mapping[str, Any],
) -> Dict[str, int]:
    load_cfg = data_cfg.get("loading", {})
    hw_train = hardware_cfg.get("training", {})

    batch_size = int(hw_train.get("batch_size") or load_cfg.get("batch_size") or 8)
    num_workers = int(hw_train.get("num_workers") or load_cfg.get("num_workers") or 2)
    prefetch_factor = int(hw_train.get("prefetch_factor") or load_cfg.get("prefetch_factor") or 2)

    return {
        "batch_size": max(1, batch_size),
        "num_workers": max(0, num_workers),
        "prefetch_factor": max(2, prefetch_factor),
    }


def _build_records_and_cache(
    data_cfg: Mapping[str, Any],
    runtime: RuntimePaths,
    seed: int,
) -> Tuple[List[SampleRecord], Mapping[str, Mapping[str, Any]], Dict[str, Any]]:
    adapter_cfg = AdapterConfig.from_data_config(data_cfg)
    adapter_cfg = AdapterConfig(
        dataset_root=str(runtime.dataset_root),
        kaggle_slug=adapter_cfg.kaggle_slug,
        split="both",
        seed=seed,
        strict_split=adapter_cfg.strict_split,
        file_extension=adapter_cfg.file_extension,
        enforce_kaggle_input_prefix=False,
    )

    adapter = KaggleDatasetAdapter(adapter_cfg)
    records = adapter.load()

    conv_cfg = ConverterConfig.from_data_config(data_cfg)
    conv_cfg = ConverterConfig(
        input_root=str(runtime.dataset_root),
        output_root=str(runtime.output_root),
        cache_index_path=str(Path(runtime.output_root) / "cache" / "conversion_cache_index.json"),
        overwrite=False,
        incremental=True,
        verify_hash=bool(data_cfg.get("checks", {}).get("verify_hash", False)),
        enforce_kaggle_paths=False,
    )
    conv_summary = OffToObjConverter(conv_cfg).convert_from_records(records).as_dict()

    feat_cfg = MeshFeatureConfig.from_data_config(data_cfg)
    feat_cfg = MeshFeatureConfig(
        dataset_root=str(runtime.dataset_root),
        output_root=str(runtime.output_root),
        cache_dir=str(Path(runtime.output_root) / "cache"),
        resolution=feat_cfg.resolution,
        dtype=feat_cfg.dtype,
        normalize_meshes=feat_cfg.normalize_meshes,
        center_objects=feat_cfg.center_objects,
        scale_to_unit_box=feat_cfg.scale_to_unit_box,
        samples_per_mesh=feat_cfg.samples_per_mesh,
        prefer_obj=True,
        overwrite=False,
        incremental=True,
        verify_hash=feat_cfg.verify_hash,
        enforce_kaggle_paths=False,
        schema_version=feat_cfg.schema_version,
    )
    builder = MeshToFeatureBuilder(feat_cfg)
    feat_summary = builder.build_from_records(records, seed=seed).as_dict()

    pipeline_summary = {
        "dataset": adapter.summary(),
        "off_to_obj": conv_summary,
        "ovoxel": feat_summary,
    }

    return records, builder.index.tensor_refs, pipeline_summary


def _split_records(records: Sequence[SampleRecord]) -> Tuple[List[SampleRecord], List[SampleRecord]]:
    train_records = [item for item in records if item.split == "train"]
    val_records = [item for item in records if item.split == "test"]

    if not train_records:
        raise Stage1TrainingError("No train split records available.")
    if not val_records:
        ratio = 0.2
        split_point = max(1, int(len(train_records) * (1.0 - ratio)))
        val_records = train_records[split_point:]
        train_records = train_records[:split_point]

    return train_records, val_records


def _build_loaders(
    train_records: Sequence[SampleRecord],
    val_records: Sequence[SampleRecord],
    tensor_refs: Mapping[str, Mapping[str, Any]],
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = OVoxelRecordDataset(train_records, tensor_refs)
    val_ds = OVoxelRecordDataset(val_records, tensor_refs)

    kwargs: Dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(batch_size, 1),
        shuffle=False,
        drop_last=False,
        **kwargs,
    )
    return train_loader, val_loader


def _make_optimizer(model: torch.nn.Module, train_cfg: Mapping[str, Any]) -> torch.optim.Optimizer:
    opt_cfg = train_cfg.get("optimizer", {})
    name = str(opt_cfg.get("name", "adam")).lower()
    lr = float(opt_cfg.get("lr", 1e-3))
    betas = tuple(float(v) for v in opt_cfg.get("betas", [0.9, 0.999]))
    eps = float(opt_cfg.get("epsilon", 1e-8))
    wd = float(opt_cfg.get("weight_decay", 0.0))

    if name != "adam":
        raise Stage1TrainingError(f"Unsupported optimizer: {name}")

    return torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=wd)


def _make_scheduler(optimizer: torch.optim.Optimizer, train_cfg: Mapping[str, Any]) -> Optional[Any]:
    sched_cfg = train_cfg.get("scheduler", {})
    name = str(sched_cfg.get("name", "")).lower().strip()
    kwargs = dict(sched_cfg.get("scheduler_kwargs", {}))

    if not name:
        return None
    if name == "cosine_annealing":
        t_max = int(kwargs.get("T_max", 10))
        eta_min = float(kwargs.get("eta_min", 1e-6))
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)

    raise Stage1TrainingError(f"Unsupported scheduler: {name}")


def _autocast_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _evaluate(
    model: ShapeSCVAE,
    val_loader: DataLoader,
    device: torch.device,
    mixed_precision: bool,
) -> Dict[str, float]:
    model.eval()
    totals = {
        "total_loss": 0.0,
        "reconstruction_loss": 0.0,
        "kl_loss": 0.0,
    }
    count = 0

    with torch.no_grad():
        for batch, _ in val_loader:
            batch = batch.to(device, non_blocking=True)
            with torch.amp.autocast(
                device_type="cuda",
                dtype=_autocast_dtype(),
                enabled=bool(mixed_precision and device.type == "cuda"),
            ):
                outputs = model(batch, sample=False)
                losses = model.compute_losses(batch, outputs)

            for key in totals:
                totals[key] += float(losses[key].detach().cpu().item())
            count += 1

    if count == 0:
        return {k: math.inf for k in totals}

    return {k: v / count for k, v in totals.items()}


def _checkpoint_payload(
    model: ShapeSCVAE,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    train_cfg: Mapping[str, Any],
    data_cfg: Mapping[str, Any],
    hardware_cfg: Mapping[str, Any],
    runtime: RuntimePaths,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": float(best_val_loss),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "train_config": dict(train_cfg),
        "data_config": dict(data_cfg),
        "hardware_config": dict(hardware_cfg),
        "run_root": str(runtime.output_root),
        "checkpoint_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    return payload


def _load_resume(
    checkpoint_path: Path,
    model: ShapeSCVAE,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    scaler: torch.amp.GradScaler,
    device: torch.device,
    strict_model_loading: bool,
) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=bool(strict_model_loading))

    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return {
        "epoch": int(checkpoint.get("epoch", 0)),
        "global_step": int(checkpoint.get("global_step", 0)),
        "best_val_loss": float(checkpoint.get("best_val_loss", float("inf"))),
        "path": str(checkpoint_path),
    }


def train(args: argparse.Namespace) -> Dict[str, Any]:
    train_cfg = _load_yaml(Path(args.config))
    hardware_cfg = _load_yaml(Path(args.hardware))
    data_cfg = _load_yaml(Path(args.data_config))

    runtime = _resolve_runtime_paths(args, data_cfg)
    metadata_path = _write_run_metadata(runtime, args, train_cfg, hardware_cfg, data_cfg)

    seed = int(train_cfg.get("seed", 42))
    _set_seed(seed)

    loop_cfg = train_cfg.get("loop", {})
    mixed_precision = bool(loop_cfg.get("mixed_precision", True))
    grad_accum_steps_cfg = int(loop_cfg.get("gradient_accumulation_steps", 1))
    grad_clip = float(loop_cfg.get("gradient_clipping_max_norm", 1.0))
    log_interval = int(loop_cfg.get("log_interval_steps", 50))
    val_interval = int(loop_cfg.get("val_interval_steps", 500))
    ckpt_interval = int(loop_cfg.get("checkpoint_interval_steps", 500))
    max_epochs = int(train_cfg.get("training", {}).get("max_epochs", 1))

    dataloader_knobs = _resolve_dataloader_knobs(data_cfg, hardware_cfg)
    oom_policy = dict(train_cfg.get("oom_fallback", {}))
    fallback = OOMFallbackManager(
        batch_size=dataloader_knobs["batch_size"],
        grad_accum_steps=grad_accum_steps_cfg,
        policy=oom_policy,
    )

    if args.contract_smoke:
        smoke_report = run_shape_sc_vae_sanity(train_cfg, data_cfg, device="cpu")
        print("[contract_smoke] passed")
        print(json.dumps(smoke_report, indent=2))

    records, tensor_refs, pipeline_summary = _build_records_and_cache(data_cfg, runtime, seed)
    train_records, val_records = _split_records(records)

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    model = ShapeSCVAE.from_stage1_configs(train_cfg, data_cfg).to(device)
    optimizer = _make_optimizer(model, train_cfg)
    scheduler = _make_scheduler(optimizer, train_cfg)
    scaler = torch.amp.GradScaler(enabled=bool(mixed_precision and device.type == "cuda"))

    ckpt_cfg = train_cfg.get("checkpointing", {})
    keep_last_n = int(
        ckpt_cfg.get("keep_last_n", ckpt_cfg.get("keep_last_n_checkpoints", 3))
    )
    checkpoint_manager = Stage1CheckpointManager(runtime.checkpoint_dir, keep_last_n=keep_last_n)

    resume_cfg = train_cfg.get("resume", {})
    strict_model_loading = bool(resume_cfg.get("strict_model_loading", False))

    start_epoch = 0
    global_step = 0
    best_val = float("inf")

    resume_path: Optional[Path] = None
    if args.resume_from:
        resume_path = Path(args.resume_from)
    elif args.autoresume or bool(ckpt_cfg.get("autoresume_enabled", True)):
        resume_path = checkpoint_manager.find_autoresume_candidate()

    if resume_path and resume_path.exists():
        resume_state = _load_resume(
            resume_path,
            model,
            optimizer,
            scheduler,
            scaler,
            device,
            strict_model_loading=strict_model_loading,
        )
        start_epoch = resume_state["epoch"]
        global_step = resume_state["global_step"]
        best_val = resume_state["best_val_loss"]
        print(f"[resume] restored from {resume_state['path']}")
        print(
            f"[resume] start_epoch={start_epoch} global_step={global_step} best_val={best_val:.6f}"
        )

    def rebuild_loaders() -> Tuple[DataLoader, DataLoader]:
        return _build_loaders(
            train_records=train_records,
            val_records=val_records,
            tensor_refs=tensor_refs,
            batch_size=fallback.batch_size,
            num_workers=dataloader_knobs["num_workers"],
            prefetch_factor=dataloader_knobs["prefetch_factor"],
        )

    train_loader, val_loader = rebuild_loaders()

    train_log_path = runtime.log_dir / "stage1_training_metrics.jsonl"
    train_log_path.parent.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "run_id": runtime.run_dir.name,
        "metadata_path": str(metadata_path),
        "train_log_path": str(train_log_path),
        "pipeline_summary": pipeline_summary,
        "resume_path": str(resume_path) if resume_path else None,
    }

    try:
        for epoch in range(start_epoch, max_epochs):
            epoch_completed = False
            while not epoch_completed:
                try:
                    model.train()
                    running = {
                        "total_loss": 0.0,
                        "reconstruction_loss": 0.0,
                        "kl_loss": 0.0,
                        "count": 0,
                    }

                    optimizer.zero_grad(set_to_none=True)

                    for step_index, (batch, _) in enumerate(train_loader):
                        batch = batch.to(device, non_blocking=True)

                        try:
                            with torch.amp.autocast(
                                device_type="cuda",
                                dtype=_autocast_dtype(),
                                enabled=bool(mixed_precision and device.type == "cuda"),
                            ):
                                outputs = model(batch, sample=True)
                                losses = model.compute_losses(batch, outputs)
                                loss_for_backward = losses["total_loss"] / fallback.grad_accum_steps

                            scaler.scale(loss_for_backward).backward()

                        except RuntimeError as exc:
                            if not _is_oom_error(exc):
                                raise

                            if device.type == "cuda":
                                torch.cuda.empty_cache()

                            action = fallback.apply()
                            if action is None:
                                raise Stage1TrainingError(
                                    "OOM recovery exhausted. No additional fallback actions available."
                                ) from exc

                            print(f"[oom] applied fallback {action}")
                            if action.startswith("batch_size"):
                                train_loader, val_loader = rebuild_loaders()
                                optimizer.zero_grad(set_to_none=True)
                                raise EpochRestartRequested(action)
                            continue

                        if (step_index + 1) % fallback.grad_accum_steps == 0:
                            scaler.unscale_(optimizer)
                            if grad_clip > 0:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad(set_to_none=True)

                            global_step += 1

                            for metric_key in ("total_loss", "reconstruction_loss", "kl_loss"):
                                running[metric_key] += float(losses[metric_key].detach().cpu().item())
                            running["count"] += 1

                            if global_step % max(1, log_interval) == 0 and running["count"] > 0:
                                payload = {
                                    "ts": datetime.now(timezone.utc).isoformat(),
                                    "epoch": epoch,
                                    "global_step": global_step,
                                    "batch_size": fallback.batch_size,
                                    "grad_accum_steps": fallback.grad_accum_steps,
                                    "lr": float(optimizer.param_groups[0]["lr"]),
                                    "train_total_loss": running["total_loss"] / running["count"],
                                    "train_reconstruction_loss": running["reconstruction_loss"] / running["count"],
                                    "train_kl_loss": running["kl_loss"] / running["count"],
                                }
                                with train_log_path.open("a", encoding="utf-8") as handle:
                                    handle.write(json.dumps(payload) + "\n")
                                print(json.dumps(payload))

                            if global_step % max(1, ckpt_interval) == 0:
                                payload = _checkpoint_payload(
                                    model,
                                    optimizer,
                                    scheduler,
                                    scaler,
                                    epoch,
                                    global_step,
                                    best_val,
                                    train_cfg,
                                    data_cfg,
                                    hardware_cfg,
                                    runtime,
                                )
                                checkpoint_manager.save_latest_step(payload, global_step)

                            if global_step % max(1, val_interval) == 0:
                                val_metrics = _evaluate(model, val_loader, device, mixed_precision)
                                metric_payload = {
                                    "ts": datetime.now(timezone.utc).isoformat(),
                                    "epoch": epoch,
                                    "global_step": global_step,
                                    **{f"val_{k}": float(v) for k, v in val_metrics.items()},
                                }
                                with train_log_path.open("a", encoding="utf-8") as handle:
                                    handle.write(json.dumps(metric_payload) + "\n")
                                print(json.dumps(metric_payload))

                                if val_metrics["total_loss"] < best_val:
                                    best_val = val_metrics["total_loss"]
                                    payload = _checkpoint_payload(
                                        model,
                                        optimizer,
                                        scheduler,
                                        scaler,
                                        epoch,
                                        global_step,
                                        best_val,
                                        train_cfg,
                                        data_cfg,
                                        hardware_cfg,
                                        runtime,
                                    )
                                    checkpoint_manager.save_best(payload)

                    epoch_completed = True

                except EpochRestartRequested:
                    print(
                        "[oom] epoch restart requested with "
                        f"batch_size={fallback.batch_size}, grad_accum={fallback.grad_accum_steps}"
                    )
                    continue

            if scheduler is not None:
                scheduler.step()

        final_payload = _checkpoint_payload(
            model,
            optimizer,
            scheduler,
            scaler,
            max_epochs,
            global_step,
            best_val,
            train_cfg,
            data_cfg,
            hardware_cfg,
            runtime,
        )
        checkpoint_manager.save_latest_step(final_payload, global_step)

    except KeyboardInterrupt:
        interrupt_payload = _checkpoint_payload(
            model,
            optimizer,
            scheduler,
            scaler,
            start_epoch,
            global_step,
            best_val,
            train_cfg,
            data_cfg,
            hardware_cfg,
            runtime,
        )
        path = checkpoint_manager.save_interrupt(interrupt_payload)
        print(f"[interrupt] checkpoint saved to {path}")
        raise

    summary["global_step"] = global_step
    summary["best_val_loss"] = float(best_val)
    summary["checkpoint_dir"] = str(runtime.checkpoint_dir)
    summary["batch_size_final"] = fallback.batch_size
    summary["grad_accum_final"] = fallback.grad_accum_steps

    summary_path = runtime.metadata_dir / "train_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[summary] {summary_path}")
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 Shape SC-VAE trainer")
    parser.add_argument("--config", type=str, default="configs/train_stage1.yaml")
    parser.add_argument("--hardware", type=str, default="configs/hardware_p100.yaml")
    parser.add_argument("--data-config", type=str, default="configs/data_stage1.yaml")
    parser.add_argument("--dataset-root", type=str, default=os.environ.get("DATASET_ROOT"))
    parser.add_argument("--output-root", type=str, default=os.environ.get("OUTPUT_ROOT"))
    parser.add_argument("--resume-from", type=str, default=os.environ.get("RESUME_CHECKPOINT_PATH"))
    parser.add_argument("--autoresume", action="store_true", help="Resume from latest checkpoint if available")
    parser.add_argument("--contract-smoke", action="store_true", help="Run shape contract smoke before training")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--run-id", type=str, default=os.environ.get("RUN_ID"))
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.contract_smoke:
        args.contract_smoke = True
    if not args.autoresume:
        args.autoresume = True

    started = time.time()
    summary = train(args)
    elapsed = time.time() - started

    print("=" * 80)
    print("Stage 1 training completed")
    print("=" * 80)
    print(json.dumps({"elapsed_seconds": round(elapsed, 2), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
