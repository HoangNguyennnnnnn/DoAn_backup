"""Improved mean-flow objective for Stage 2 latent smoke training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F


class MeanFlowObjectiveError(ValueError):
    """Raised when mean-flow objective inputs are invalid."""


@dataclass(frozen=True)
class MeanFlowObjectiveConfig:
    """Config surface for the improved mean-flow smoke objective."""

    flow_loss_weight: float = 1.0
    v_loss_weight: float = 0.25
    v_loss_enabled: bool = True
    guidance_conditioning_enabled: bool = True
    guidance_dropout_start: float = 0.0
    guidance_dropout_final: float = 0.1
    guidance_warmup_steps: int = 0
    time_weight_mode: str = "midpoint"
    clamp_timesteps: bool = True
    timestep_eps: float = 1e-4

    @staticmethod
    def from_stage2_configs(train_cfg: Mapping[str, Any], data_cfg: Mapping[str, Any]) -> "MeanFlowObjectiveConfig":
        loss_cfg = train_cfg.get("loss", {})
        scheduler_cfg = train_cfg.get("scheduler", {})

        return MeanFlowObjectiveConfig(
            flow_loss_weight=float(loss_cfg.get("flow_loss_weight", 1.0)),
            v_loss_weight=float(loss_cfg.get("v_loss_weight", 0.25)),
            v_loss_enabled=bool(loss_cfg.get("v_loss_enabled", True)),
            guidance_conditioning_enabled=bool(loss_cfg.get("guidance_conditioning_enabled", True)),
            guidance_dropout_start=float(loss_cfg.get("guidance_dropout_start", 0.0)),
            guidance_dropout_final=float(loss_cfg.get("guidance_dropout_prob", loss_cfg.get("guidance_dropout_final", 0.1))),
            guidance_warmup_steps=int(scheduler_cfg.get("objective_warmup_steps", 0)),
            time_weight_mode=str(loss_cfg.get("time_weight_mode", "midpoint")),
            clamp_timesteps=bool(loss_cfg.get("clamp_timesteps", True)),
            timestep_eps=float(loss_cfg.get("timestep_eps", 1e-4)),
        )


@dataclass(frozen=True)
class MeanFlowScheduleState:
    flow_loss_weight: float
    v_loss_weight: float
    guidance_dropout_prob: float


class ImprovedMeanFlowObjective:
    """Flow-matching objective with a lightweight v-loss branch and guidance dropout."""

    def __init__(self, config: MeanFlowObjectiveConfig):
        self.config = config

    @classmethod
    def from_stage2_configs(
        cls,
        train_cfg: Mapping[str, Any],
        data_cfg: Mapping[str, Any],
    ) -> "ImprovedMeanFlowObjective":
        return cls(MeanFlowObjectiveConfig.from_stage2_configs(train_cfg, data_cfg))

    def schedule(self, step: int) -> MeanFlowScheduleState:
        if self.config.guidance_warmup_steps <= 0:
            dropout_prob = self.config.guidance_dropout_final
        else:
            progress = max(0.0, min(1.0, float(step) / float(self.config.guidance_warmup_steps)))
            dropout_prob = self.config.guidance_dropout_start + (
                (self.config.guidance_dropout_final - self.config.guidance_dropout_start) * progress
            )

        return MeanFlowScheduleState(
            flow_loss_weight=self.config.flow_loss_weight,
            v_loss_weight=self.config.v_loss_weight,
            guidance_dropout_prob=dropout_prob,
        )

    def _apply_guidance_dropout(
        self,
        class_labels: Optional[torch.Tensor],
        context_embeddings: Optional[torch.Tensor],
        guidance_dropout_prob: float,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.config.guidance_conditioning_enabled or guidance_dropout_prob <= 0.0:
            return class_labels, context_embeddings

        if class_labels is None and context_embeddings is None:
            return class_labels, context_embeddings

        reference = context_embeddings if context_embeddings is not None else class_labels
        if reference is None:
            return class_labels, context_embeddings

        batch_size = reference.shape[0]
        drop_mask = torch.rand(batch_size, device=reference.device) < guidance_dropout_prob
        if not drop_mask.any():
            return class_labels, context_embeddings

        if class_labels is not None:
            dropped_labels = class_labels.clone()
            dropped_labels[drop_mask] = -1
            class_labels = dropped_labels

        if context_embeddings is not None:
            dropped_context = context_embeddings.clone()
            dropped_context[drop_mask] = 0
            context_embeddings = dropped_context

        return class_labels, context_embeddings

    def compute_loss(
        self,
        model: torch.nn.Module,
        clean_latents: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        context_embeddings: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        step: int = 0,
    ) -> Dict[str, torch.Tensor]:
        if clean_latents.ndim != 3:
            raise MeanFlowObjectiveError(
                "Clean latents must have shape (B, T, D). "
                f"Received {tuple(clean_latents.shape)}."
            )

        batch_size = clean_latents.shape[0]
        device = clean_latents.device
        dtype = clean_latents.dtype

        if noise is None:
            noise = torch.randn_like(clean_latents)
        if timesteps is None:
            timesteps = torch.rand(batch_size, device=device, dtype=dtype)
        elif timesteps.ndim == 0:
            timesteps = timesteps.unsqueeze(0)
        elif timesteps.ndim == 2 and timesteps.shape[1] == 1:
            timesteps = timesteps.squeeze(-1)

        if timesteps.ndim != 1 or timesteps.shape[0] != batch_size:
            raise MeanFlowObjectiveError(
                "Timesteps must have shape (B,) for mean-flow training. "
                f"Received {tuple(timesteps.shape)} for batch_size={batch_size}."
            )

        schedule = self.schedule(step)
        class_labels, context_embeddings = self._apply_guidance_dropout(
            class_labels=class_labels,
            context_embeddings=context_embeddings,
            guidance_dropout_prob=schedule.guidance_dropout_prob,
        )

        if self.config.clamp_timesteps:
            timesteps = timesteps.clamp(self.config.timestep_eps, 1.0 - self.config.timestep_eps)

        noisy_latents = (1.0 - timesteps.view(-1, 1, 1)) * clean_latents + timesteps.view(-1, 1, 1) * noise
        velocity_target = noise - clean_latents
        prediction = model(
            noisy_latents,
            timesteps,
            class_labels=class_labels,
            context_embeddings=context_embeddings,
        )

        if prediction.shape != velocity_target.shape:
            raise MeanFlowObjectiveError(
                "Model prediction shape mismatch against flow target. "
                f"prediction={tuple(prediction.shape)}, target={tuple(velocity_target.shape)}."
            )

        per_sample_flow = F.mse_loss(prediction, velocity_target, reduction="none").flatten(1).mean(dim=1)
        if self.config.time_weight_mode == "midpoint":
            weights = 1.0 + 2.0 * timesteps * (1.0 - timesteps)
        elif self.config.time_weight_mode == "edge":
            weights = 1.0 + 2.0 * torch.abs(timesteps - 0.5)
        else:
            weights = torch.ones_like(timesteps)
        flow_loss = (per_sample_flow * weights).mean()

        if self.config.v_loss_enabled:
            per_sample_v = F.smooth_l1_loss(prediction, velocity_target, reduction="none").flatten(1).mean(dim=1)
            v_loss = (per_sample_v * weights).mean()
        else:
            v_loss = torch.zeros_like(flow_loss)

        total_loss = schedule.flow_loss_weight * flow_loss + schedule.v_loss_weight * v_loss

        return {
            "total_loss": total_loss,
            "flow_loss": flow_loss,
            "v_loss": v_loss,
            "flow_weight": torch.tensor(schedule.flow_loss_weight, device=device, dtype=dtype),
            "v_loss_weight": torch.tensor(schedule.v_loss_weight, device=device, dtype=dtype),
            "guidance_dropout_prob": torch.tensor(schedule.guidance_dropout_prob, device=device, dtype=dtype),
            "mean_timestep": timesteps.mean(),
        }