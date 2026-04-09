"""OVoxel-to-SLAT interface contract for Stage 1 shape path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn


class ShapeInterfaceError(ValueError):
    """Raised when OVoxel/SLAT shape-path contract validation fails."""


def _torch_dtype_from_name(dtype_name: str) -> torch.dtype:
    name = str(dtype_name).strip().lower()
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ShapeInterfaceError(
        "Unsupported voxel dtype in interface config. "
        f"Expected one of ['float16', 'float32'], received: {dtype_name!r}."
    )


@dataclass(frozen=True)
class ShapePathContractConfig:
    """Configurable OVoxel-to-SLAT shape-path contract."""

    input_channels: int
    input_resolution: int
    voxel_dtype: str
    latent_dim: int
    token_length: int
    token_dim: int
    schema_version: str = "ovoxel-v1"
    tensor_layout: str = "B,C,R,R,R"

    @staticmethod
    def from_stage1_configs(
        train_cfg: Mapping[str, Any],
        data_cfg: Mapping[str, Any],
    ) -> "ShapePathContractConfig":
        model_cfg = train_cfg.get("model", {})
        encoder_cfg = model_cfg.get("encoder", {})
        latent_cfg = model_cfg.get("latent_interface", {})
        shape_cfg = latent_cfg.get("shape_path", {})

        prep_cfg = data_cfg.get("preprocessing", {})

        input_channels = int(encoder_cfg.get("input_channels", 1))
        input_resolution = int(encoder_cfg.get("input_resolution", 32))
        latent_dim = int(encoder_cfg.get("latent_dim", latent_cfg.get("latent_dim", 128)))
        token_length = int(shape_cfg.get("token_length", 8))

        token_dim_raw = shape_cfg.get("token_dim")
        if token_dim_raw is None:
            if token_length <= 0:
                raise ShapeInterfaceError(
                    "model.latent_interface.shape_path.token_length must be > 0."
                )
            if latent_dim % token_length != 0:
                raise ShapeInterfaceError(
                    "Invalid token contract: latent_dim is not divisible by token_length and "
                    "token_dim is not provided. Set model.latent_interface.shape_path.token_dim "
                    "explicitly or adjust latent/token lengths. "
                    f"Got latent_dim={latent_dim}, token_length={token_length}."
                )
            token_dim = latent_dim // token_length
        else:
            token_dim = int(token_dim_raw)

        return ShapePathContractConfig(
            input_channels=input_channels,
            input_resolution=input_resolution,
            voxel_dtype=str(prep_cfg.get("voxel_dtype", "float32")).lower().strip(),
            latent_dim=latent_dim,
            token_length=token_length,
            token_dim=token_dim,
            schema_version=str(prep_cfg.get("ovoxel_schema_version", "ovoxel-v1")),
            tensor_layout=str(prep_cfg.get("ovoxel_tensor_layout", "B,C,R,R,R")),
        )


class OVoxelToSLATShapeAdapter(nn.Module):
    """Normalizes OVoxel tensors and converts latent vectors to SLAT shape tokens."""

    def __init__(self, config: ShapePathContractConfig):
        super().__init__()
        self.config = config

        if config.input_channels <= 0:
            raise ShapeInterfaceError(
                f"input_channels must be > 0, received {config.input_channels}."
            )
        if config.input_resolution <= 0:
            raise ShapeInterfaceError(
                f"input_resolution must be > 0, received {config.input_resolution}."
            )
        if config.latent_dim <= 0:
            raise ShapeInterfaceError(
                f"latent_dim must be > 0, received {config.latent_dim}."
            )
        if config.token_length <= 0:
            raise ShapeInterfaceError(
                f"token_length must be > 0, received {config.token_length}."
            )
        if config.token_dim <= 0:
            raise ShapeInterfaceError(
                f"token_dim must be > 0, received {config.token_dim}."
            )

        self.expected_dtype = _torch_dtype_from_name(config.voxel_dtype)
        self.token_feature_dim = config.token_length * config.token_dim
        self.requires_projection = self.token_feature_dim != config.latent_dim
        self.latent_projection = (
            nn.Linear(config.latent_dim, self.token_feature_dim)
            if self.requires_projection
            else nn.Identity()
        )

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize tensor layout to (B, C, R, R, R) and enforce contract checks."""
        if not isinstance(x, torch.Tensor):
            raise ShapeInterfaceError(
                f"OVoxel input must be torch.Tensor, received: {type(x)}."
            )

        if x.ndim == 3:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.ndim == 4:
            if x.shape[0] == self.config.input_channels:
                x = x.unsqueeze(0)
            else:
                x = x.unsqueeze(1)
        elif x.ndim != 5:
            raise ShapeInterfaceError(
                "OVoxel input rank mismatch. Expected one of [3D, 4D, 5D] that can be normalized "
                f"to (B,C,R,R,R); received shape: {tuple(x.shape)}."
            )

        if x.shape[1] != self.config.input_channels:
            raise ShapeInterfaceError(
                "OVoxel channel mismatch after normalization. "
                f"Expected C={self.config.input_channels}, received C={x.shape[1]}. "
                "Check producer artifact layout and model.encoder.input_channels."
            )

        expected_res = self.config.input_resolution
        spatial = tuple(x.shape[-3:])
        if spatial != (expected_res, expected_res, expected_res):
            raise ShapeInterfaceError(
                "OVoxel resolution mismatch. "
                f"Expected (R,R,R)=({expected_res},{expected_res},{expected_res}), got {spatial}. "
                "Align preprocessing.target_resolution and model.encoder.input_resolution."
            )

        if x.dtype != self.expected_dtype:
            raise ShapeInterfaceError(
                "OVoxel dtype mismatch. "
                f"Expected {self.expected_dtype}, received {x.dtype}. "
                "Align preprocessing.voxel_dtype and upstream tensor persistence dtype."
            )

        if not torch.is_floating_point(x):
            raise ShapeInterfaceError(
                "OVoxel tensor must be floating-point with occupancy values in [0, 1]. "
                f"Received dtype: {x.dtype}."
            )

        return x

    def tokens_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Convert latent vectors (B, latent_dim) to tokens (B, token_length, token_dim)."""
        if z.ndim != 2:
            raise ShapeInterfaceError(
                "Latent tensor rank mismatch. Expected (B, latent_dim), "
                f"received shape: {tuple(z.shape)}."
            )
        if z.shape[1] != self.config.latent_dim:
            raise ShapeInterfaceError(
                "Latent dimension mismatch. "
                f"Expected latent_dim={self.config.latent_dim}, received {z.shape[1]}."
            )

        projected = self.latent_projection(z)
        if projected.shape[1] != self.token_feature_dim:
            raise ShapeInterfaceError(
                "Token projection mismatch. "
                f"Expected projected width {self.token_feature_dim}, got {projected.shape[1]}."
            )

        return projected.reshape(z.shape[0], self.config.token_length, self.config.token_dim)

    def contract_summary(self) -> Dict[str, Any]:
        return {
            "layout": "B,C,R,R,R",
            "input_channels": self.config.input_channels,
            "input_resolution": self.config.input_resolution,
            "input_dtype": str(self.expected_dtype).replace("torch.", ""),
            "latent_dim": self.config.latent_dim,
            "token_length": self.config.token_length,
            "token_dim": self.config.token_dim,
            "token_projection": self.requires_projection,
            "schema_version": self.config.schema_version,
        }


def validate_shape_path_smoke(
    adapter: OVoxelToSLATShapeAdapter,
    ovx: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
) -> Dict[str, Tuple[int, ...]]:
    """Validate shape-path contract for train/inference entrypoint compatibility."""
    normalized = adapter.normalize_input(ovx)

    if mu.shape != log_var.shape:
        raise ShapeInterfaceError(
            "Encoder output mismatch: mu and log_var must have identical shapes. "
            f"Got mu={tuple(mu.shape)}, log_var={tuple(log_var.shape)}."
        )

    tokens = adapter.tokens_from_latent(mu)
    if tokens.shape[0] != normalized.shape[0]:
        raise ShapeInterfaceError(
            "Batch size mismatch between OVoxel input and SLAT tokens. "
            f"input_batch={normalized.shape[0]}, token_batch={tokens.shape[0]}."
        )

    return {
        "input_shape": tuple(normalized.shape),
        "mu_shape": tuple(mu.shape),
        "log_var_shape": tuple(log_var.shape),
        "token_shape": tuple(tokens.shape),
    }
