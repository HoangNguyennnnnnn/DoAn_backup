"""Stage 2 latent-space UNet generator for smoke-level training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentGeneratorError(ValueError):
    """Raised when Stage 2 latent generator contracts are invalid."""


def _group_count(channels: int, max_groups: int = 8) -> int:
    return max(1, math.gcd(max_groups, channels))


def _sinusoidal_time_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    if timesteps.ndim == 0:
        timesteps = timesteps.unsqueeze(0)
    timesteps = timesteps.float().view(-1, 1)

    half_dim = max(1, embedding_dim // 2)
    exponent = torch.arange(half_dim, device=timesteps.device, dtype=timesteps.dtype)
    exponent = exponent * (-math.log(10000.0) / max(1, half_dim - 1))
    frequencies = torch.exp(exponent).view(1, -1)
    angles = timesteps * frequencies
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if emb.shape[-1] < embedding_dim:
        emb = F.pad(emb, (0, embedding_dim - emb.shape[-1]))
    return emb


@dataclass(frozen=True)
class LatentGeneratorConfig:
    """Config surface for the Stage 2 latent UNet."""

    token_length: int = 8
    token_dim: int = 16
    latent_dim: int = 128
    base_channels: int = 32
    channel_multipliers: Tuple[int, ...] = (1, 2, 4)
    time_embed_dim: int = 128
    context_dim: int = 64
    dino_context_dim: int = 768
    num_classes: int = 40
    dropout: float = 0.0
    context_backend: str = "dino_first"
    allow_flat_input: bool = True
    schema_version: str = "stage2-latent-v1"

    @staticmethod
    def from_stage2_configs(
        train_cfg: Mapping[str, Any],
        data_cfg: Mapping[str, Any],
    ) -> "LatentGeneratorConfig":
        model_cfg = train_cfg.get("model", {})
        latent_cfg = model_cfg.get("latent_contract", {})
        unet_cfg = model_cfg.get("unet", {})
        conditioning_cfg = model_cfg.get("conditioning", {})
        latent_loading = data_cfg.get("latent_loading", {})

        token_length = int(latent_cfg.get("token_length", latent_loading.get("token_length", 8)))
        token_dim = latent_cfg.get("token_dim", latent_loading.get("token_dim"))
        latent_dim = int(latent_cfg.get("latent_dim", latent_loading.get("latent_dim", token_length * int(token_dim or 16))))
        if token_dim is None:
            if token_length <= 0 or latent_dim % token_length != 0:
                raise LatentGeneratorError(
                    "token_dim is missing and latent_dim is not divisible by token_length. "
                    f"Received latent_dim={latent_dim}, token_length={token_length}."
                )
            token_dim = latent_dim // token_length

        if int(token_length) * int(token_dim) != latent_dim:
            raise LatentGeneratorError(
                "Stage 2 latent contract mismatch. token_length * token_dim must equal latent_dim. "
                f"Got token_length={token_length}, token_dim={token_dim}, latent_dim={latent_dim}."
            )

        channel_multipliers_raw = unet_cfg.get("channel_multipliers", [1, 2, 4])
        channel_multipliers = tuple(int(value) for value in channel_multipliers_raw)

        return LatentGeneratorConfig(
            token_length=token_length,
            token_dim=int(token_dim),
            latent_dim=latent_dim,
            base_channels=int(unet_cfg.get("base_channels", 32)),
            channel_multipliers=channel_multipliers,
            time_embed_dim=int(unet_cfg.get("time_embed_dim", 128)),
            context_dim=int(conditioning_cfg.get("context_dim", 64)),
            dino_context_dim=int(conditioning_cfg.get("dino_context_dim", 768)),
            num_classes=int(conditioning_cfg.get("num_classes", 40)),
            dropout=float(unet_cfg.get("dropout", 0.0)),
            context_backend=str(conditioning_cfg.get("context_backend", "dino_first")),
            allow_flat_input=bool(latent_cfg.get("allow_flat_input", True)),
            schema_version=str(latent_loading.get("schema_version", "stage2-latent-v1")),
        )


class LatentContextAdapter(nn.Module):
    """DINO-first conditioning adapter with class-label fallback."""

    def __init__(
        self,
        context_dim: int,
        num_classes: int,
        dino_context_dim: int,
        context_backend: str = "dino_first",
    ):
        super().__init__()
        self.context_dim = int(context_dim)
        self.num_classes = int(num_classes)
        self.dino_context_dim = int(dino_context_dim)
        self.context_backend = str(context_backend).lower().strip()

        self.class_embedding = nn.Embedding(self.num_classes, self.context_dim)
        self.dino_projection = nn.Linear(self.dino_context_dim, self.context_dim)

    def forward(
        self,
        batch_size: int,
        class_labels: Optional[torch.Tensor] = None,
        context_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if context_embeddings is not None:
            if context_embeddings.ndim != 2:
                raise LatentGeneratorError(
                    "Context embeddings must have shape (B, C). "
                    f"Received {tuple(context_embeddings.shape)}."
                )
            if context_embeddings.shape[0] != batch_size:
                raise LatentGeneratorError(
                    "Context batch size mismatch. "
                    f"Expected {batch_size}, received {context_embeddings.shape[0]}."
                )
            if context_embeddings.shape[1] != self.dino_context_dim:
                raise LatentGeneratorError(
                    "DINO context width mismatch. "
                    f"Expected {self.dino_context_dim}, received {context_embeddings.shape[1]}."
                )
            return self.dino_projection(context_embeddings.float())

        if class_labels is not None:
            if class_labels.ndim != 1:
                raise LatentGeneratorError(
                    "Class labels must have shape (B,). "
                    f"Received {tuple(class_labels.shape)}."
                )
            if class_labels.shape[0] != batch_size:
                raise LatentGeneratorError(
                    "Class label batch size mismatch. "
                    f"Expected {batch_size}, received {class_labels.shape[0]}."
                )

            mask = class_labels < 0
            safe_labels = class_labels.clamp_min(0)
            if safe_labels.max().item() >= self.num_classes:
                raise LatentGeneratorError(
                    "Class label out of range for latent context adapter. "
                    f"Received max label {int(safe_labels.max().item())}, num_classes={self.num_classes}."
                )
            embeddings = self.class_embedding(safe_labels)
            if mask.any():
                embeddings = embeddings.masked_fill(mask.unsqueeze(-1), 0.0)
            return embeddings

        return torch.zeros(batch_size, self.context_dim, device=self.class_embedding.weight.device)


class LatentResBlock1D(nn.Module):
    """Residual 1D block with timestep/context conditioning."""

    def __init__(self, in_channels: int, out_channels: int, embedding_dim: int, dropout: float):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.embedding_dim = int(embedding_dim)
        self.dropout = float(dropout)

        self.norm1 = nn.GroupNorm(_group_count(self.in_channels), self.in_channels)
        self.conv1 = nn.Conv1d(self.in_channels, self.out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_group_count(self.out_channels), self.out_channels)
        self.conv2 = nn.Conv1d(self.out_channels, self.out_channels, kernel_size=3, padding=1)
        self.emb_proj = nn.Linear(self.embedding_dim, self.out_channels * 2)
        self.skip = (
            nn.Conv1d(self.in_channels, self.out_channels, kernel_size=1)
            if self.in_channels != self.out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale_shift = self.emb_proj(F.silu(embedding)).unsqueeze(-1)
        scale, shift = torch.chunk(scale_shift, 2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        h = F.silu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.conv2(h)
        return h + self.skip(x)


class LatentUNet1D(nn.Module):
    """UNet over latent token sequences with DINO-first conditioning hooks."""

    def __init__(self, config: LatentGeneratorConfig):
        super().__init__()
        self.config = config

        if config.token_length <= 0 or config.token_dim <= 0:
            raise LatentGeneratorError(
                "token_length and token_dim must be positive for LatentUNet1D."
            )
        if config.latent_dim != config.token_length * config.token_dim:
            raise LatentGeneratorError(
                "LatentUNet1D requires token_length * token_dim == latent_dim. "
                f"Got token_length={config.token_length}, token_dim={config.token_dim}, latent_dim={config.latent_dim}."
            )
        if not config.channel_multipliers:
            raise LatentGeneratorError("channel_multipliers must not be empty.")

        channel_sizes = [config.base_channels * max(1, mult) for mult in config.channel_multipliers]
        self.channel_sizes = channel_sizes
        self.embedding_dim = config.time_embed_dim + config.context_dim

        self.input_proj = nn.Conv1d(config.token_dim, channel_sizes[0], kernel_size=3, padding=1)
        self.time_embedding = nn.Sequential(
            nn.Linear(config.time_embed_dim, config.time_embed_dim * 2),
            nn.SiLU(),
            nn.Linear(config.time_embed_dim * 2, config.time_embed_dim),
        )
        self.context_adapter = LatentContextAdapter(
            context_dim=config.context_dim,
            num_classes=config.num_classes,
            dino_context_dim=config.dino_context_dim,
            context_backend=config.context_backend,
        )

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current_channels = channel_sizes[0]
        for index, target_channels in enumerate(channel_sizes):
            self.down_blocks.append(
                LatentResBlock1D(current_channels, target_channels, self.embedding_dim, config.dropout)
            )
            current_channels = target_channels
            if index < len(channel_sizes) - 1:
                next_channels = channel_sizes[index + 1]
                self.downsamples.append(
                    nn.Conv1d(current_channels, next_channels, kernel_size=4, stride=2, padding=1)
                )
                current_channels = next_channels

        self.mid_block = LatentResBlock1D(channel_sizes[-1], channel_sizes[-1], self.embedding_dim, config.dropout)

        self.upsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for index in range(len(channel_sizes) - 1, 0, -1):
            current = channel_sizes[index]
            target = channel_sizes[index - 1]
            self.upsamples.append(
                nn.ConvTranspose1d(current, target, kernel_size=4, stride=2, padding=1)
            )
            self.up_blocks.append(
                LatentResBlock1D(target * 2, target, self.embedding_dim, config.dropout)
            )

        self.output_norm = nn.GroupNorm(_group_count(channel_sizes[0]), channel_sizes[0])
        self.output_proj = nn.Conv1d(channel_sizes[0], config.token_dim, kernel_size=3, padding=1)

    @classmethod
    def from_stage2_configs(
        cls,
        train_cfg: Mapping[str, Any],
        data_cfg: Mapping[str, Any],
    ) -> "LatentUNet1D":
        return cls(LatentGeneratorConfig.from_stage2_configs(train_cfg, data_cfg))

    def _normalize_latent_input(self, latent_tokens: torch.Tensor) -> torch.Tensor:
        if not isinstance(latent_tokens, torch.Tensor):
            raise LatentGeneratorError(
                f"Latent input must be a tensor, received {type(latent_tokens)!r}."
            )
        if latent_tokens.ndim == 2 and self.config.allow_flat_input:
            if latent_tokens.shape[1] != self.config.latent_dim:
                raise LatentGeneratorError(
                    "Flat latent input width mismatch. "
                    f"Expected {self.config.latent_dim}, received {latent_tokens.shape[1]}."
                )
            latent_tokens = latent_tokens.view(latent_tokens.shape[0], self.config.token_length, self.config.token_dim)
        if latent_tokens.ndim != 3:
            raise LatentGeneratorError(
                "Latent input must have shape (B, T, D) or flat (B, latent_dim). "
                f"Received {tuple(latent_tokens.shape)}."
            )
        if latent_tokens.shape[1] != self.config.token_length or latent_tokens.shape[2] != self.config.token_dim:
            raise LatentGeneratorError(
                "Latent token shape mismatch. "
                f"Expected (B, {self.config.token_length}, {self.config.token_dim}), "
                f"received {tuple(latent_tokens.shape)}."
            )
        return latent_tokens.float()

    def _normalize_timesteps(self, timesteps: torch.Tensor, batch_size: int) -> torch.Tensor:
        if timesteps.ndim == 0:
            timesteps = timesteps.unsqueeze(0)
        if timesteps.ndim == 2 and timesteps.shape[1] == 1:
            timesteps = timesteps.squeeze(-1)
        if timesteps.ndim != 1:
            raise LatentGeneratorError(
                f"Timesteps must have shape (B,) or scalar. Received {tuple(timesteps.shape)}."
            )
        if timesteps.shape[0] == 1 and batch_size > 1:
            timesteps = timesteps.expand(batch_size)
        if timesteps.shape[0] != batch_size:
            raise LatentGeneratorError(
                "Timestep batch size mismatch. "
                f"Expected {batch_size}, received {timesteps.shape[0]}."
            )
        return timesteps.float()

    def forward(
        self,
        latent_tokens: torch.Tensor,
        timesteps: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        context_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        latent_tokens = self._normalize_latent_input(latent_tokens)
        batch_size = latent_tokens.shape[0]
        timesteps = self._normalize_timesteps(timesteps, batch_size)

        time_embedding = _sinusoidal_time_embedding(timesteps.to(latent_tokens.device), self.config.time_embed_dim)
        time_embedding = self.time_embedding(time_embedding)
        context_embedding = self.context_adapter(
            batch_size=batch_size,
            class_labels=class_labels.to(latent_tokens.device) if class_labels is not None else None,
            context_embeddings=context_embeddings.to(latent_tokens.device) if context_embeddings is not None else None,
        )
        embedding = torch.cat([time_embedding, context_embedding], dim=-1)

        x = self.input_proj(latent_tokens.transpose(1, 2))
        skips = []
        for index, down_block in enumerate(self.down_blocks):
            x = down_block(x, embedding)
            skips.append(x)
            if index < len(self.downsamples):
                x = self.downsamples[index](x)

        x = self.mid_block(x, embedding)

        for upsample, up_block, skip in zip(self.upsamples, self.up_blocks, reversed(skips[:-1])):
            x = upsample(x)
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = up_block(x, embedding)

        x = self.output_proj(F.silu(self.output_norm(x)))
        output = x.transpose(1, 2)

        if output.shape != latent_tokens.shape:
            raise LatentGeneratorError(
                "Latent UNet output shape mismatch. "
                f"Expected {tuple(latent_tokens.shape)}, received {tuple(output.shape)}."
            )

        return output

    def forward_sanity_check(
        self,
        batch_size: int = 2,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        tokens = torch.zeros(
            batch_size,
            self.config.token_length,
            self.config.token_dim,
            dtype=torch.float32,
            device=device,
        )
        timesteps = torch.linspace(0.0, 1.0, batch_size, device=device)
        class_labels = torch.zeros(batch_size, dtype=torch.long, device=device)

        with torch.no_grad():
            output = self.forward(tokens, timesteps, class_labels=class_labels)

        return {
            "input_shape": tuple(tokens.shape),
            "output_shape": tuple(output.shape),
            "token_length": self.config.token_length,
            "token_dim": self.config.token_dim,
            "latent_dim": self.config.latent_dim,
            "context_backend": self.config.context_backend,
        }


def build_latent_generator(
    train_cfg: Mapping[str, Any],
    data_cfg: Mapping[str, Any],
) -> LatentUNet1D:
    return LatentUNet1D.from_stage2_configs(train_cfg, data_cfg)