"""Shape SC-VAE model wrapper for Stage 1 shape-only training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.decoder import ShapeDecoder
from src.models.encoder import ShapeEncoder
from src.models.shape_interface import ShapePathContractConfig


class ShapeSCVAEError(ValueError):
    """Raised when Shape SC-VAE configuration or runtime contracts are invalid."""


@dataclass(frozen=True)
class ShapeSCVAEConfig:
    """Minimal config surface for Shape SC-VAE model construction."""

    input_channels: int
    input_resolution: int
    voxel_dtype: str
    latent_dim: int
    encoder_hidden_channels: list[int]
    decoder_hidden_channels: list[int]
    output_channels: int
    token_length: int
    token_dim: int
    reconstruction_loss: str = "mse"
    reconstruction_weight: float = 1.0
    use_kl_loss: bool = True
    kl_weight: float = 0.001

    @staticmethod
    def from_stage1_configs(
        train_cfg: Mapping[str, Any],
        data_cfg: Mapping[str, Any],
    ) -> "ShapeSCVAEConfig":
        model_cfg = train_cfg.get("model", {})
        encoder_cfg = model_cfg.get("encoder", {})
        decoder_cfg = model_cfg.get("decoder", {})
        loss_cfg = train_cfg.get("loss", {})
        contract = ShapePathContractConfig.from_stage1_configs(train_cfg, data_cfg)

        return ShapeSCVAEConfig(
            input_channels=contract.input_channels,
            input_resolution=contract.input_resolution,
            voxel_dtype=contract.voxel_dtype,
            latent_dim=contract.latent_dim,
            encoder_hidden_channels=[int(c) for c in encoder_cfg.get("hidden_channels", [64, 128, 256])],
            decoder_hidden_channels=[int(c) for c in decoder_cfg.get("hidden_channels", [256, 128, 64])],
            output_channels=int(decoder_cfg.get("output_channels", 1)),
            token_length=contract.token_length,
            token_dim=contract.token_dim,
            reconstruction_loss=str(loss_cfg.get("reconstruction_loss", "mse")).lower().strip(),
            reconstruction_weight=float(loss_cfg.get("reconstruction_weight", 1.0)),
            use_kl_loss=bool(loss_cfg.get("use_kl_loss", True)),
            kl_weight=float(loss_cfg.get("kl_weight", 0.001)),
        )


class ShapeSCVAE(nn.Module):
    """Stage 1 shape-only VAE with explicit OVoxel-to-SLAT token contract exposure."""

    def __init__(self, config: ShapeSCVAEConfig):
        super().__init__()
        self.config = config

        if config.reconstruction_loss not in {"mse", "bce"}:
            raise ShapeSCVAEError(
                "loss.reconstruction_loss must be one of ['mse', 'bce']. "
                f"Received: {config.reconstruction_loss!r}."
            )
        if config.reconstruction_weight <= 0:
            raise ShapeSCVAEError(
                "loss.reconstruction_weight must be > 0. "
                f"Received: {config.reconstruction_weight}."
            )
        if config.use_kl_loss and config.kl_weight < 0:
            raise ShapeSCVAEError(
                "loss.kl_weight must be >= 0 when KL branch is enabled. "
                f"Received: {config.kl_weight}."
            )

        self.encoder = ShapeEncoder(
            input_channels=config.input_channels,
            input_resolution=config.input_resolution,
            latent_dim=config.latent_dim,
            hidden_channels=config.encoder_hidden_channels,
            voxel_dtype=config.voxel_dtype,
            token_length=config.token_length,
            token_dim=config.token_dim,
        )
        self.decoder = ShapeDecoder(
            latent_dim=config.latent_dim,
            output_resolution=config.input_resolution,
            hidden_channels=config.decoder_hidden_channels,
            output_channels=config.output_channels,
        )

    @classmethod
    def from_stage1_configs(
        cls,
        train_cfg: Mapping[str, Any],
        data_cfg: Mapping[str, Any],
    ) -> "ShapeSCVAE":
        return cls(ShapeSCVAEConfig.from_stage1_configs(train_cfg, data_cfg))

    def encode(self, x: torch.Tensor, sample: bool = True) -> Dict[str, torch.Tensor]:
        """Encode OVoxel input and expose latent stats and SLAT shape tokens."""
        return self.encoder.encode_to_shape_tokens(x, sample=sample)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors into voxel reconstructions."""
        return self.decoder(latent)

    def forward(self, x: torch.Tensor, sample: bool = True) -> Dict[str, torch.Tensor]:
        """Full shape-path forward pass returning recon + tokenized latent outputs."""
        latent_pack = self.encode(x, sample=sample)
        recon = self.decode(latent_pack["latent"])
        return {
            "recon": recon,
            "tokens": latent_pack["tokens"],
            "mu": latent_pack["mu"],
            "log_var": latent_pack["log_var"],
            "latent": latent_pack["latent"],
        }

    def reconstruction_loss(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss according to configured loss type."""
        if recon.shape != target.shape:
            raise ShapeSCVAEError(
                "Reconstruction target mismatch. "
                f"recon shape={tuple(recon.shape)}, target shape={tuple(target.shape)}."
            )
        if self.config.reconstruction_loss == "mse":
            return F.mse_loss(recon, target)
        return F.binary_cross_entropy(recon, target)

    @staticmethod
    def kl_divergence(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Compute mean KL divergence term for VAE training."""
        if mu.shape != log_var.shape:
            raise ShapeSCVAEError(
                "KL input mismatch: mu and log_var must have identical shapes. "
                f"Got mu={tuple(mu.shape)}, log_var={tuple(log_var.shape)}."
            )
        kl_per_sample = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        return torch.mean(kl_per_sample)

    def compute_losses(
        self,
        batch: torch.Tensor,
        outputs: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute reconstruction and optional KL losses from a forward pass."""
        recon_loss = self.reconstruction_loss(outputs["recon"], batch)
        losses: Dict[str, torch.Tensor] = {
            "reconstruction_loss": recon_loss,
            "kl_loss": torch.zeros_like(recon_loss),
        }

        total = self.config.reconstruction_weight * recon_loss
        if self.config.use_kl_loss:
            kl = self.kl_divergence(outputs["mu"], outputs["log_var"])
            losses["kl_loss"] = kl
            total = total + self.config.kl_weight * kl

        losses["total_loss"] = total
        return losses

    def forward_sanity_check(
        self,
        batch_size: int = 2,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """Lightweight shape-only sanity check for train/inference entrypoints."""
        dtype = torch.float16 if self.config.voxel_dtype == "float16" else torch.float32
        batch = torch.zeros(
            batch_size,
            self.config.input_channels,
            self.config.input_resolution,
            self.config.input_resolution,
            self.config.input_resolution,
            dtype=dtype,
            device=device,
        )

        with torch.no_grad():
            output = self.forward(batch, sample=False)
            losses = self.compute_losses(batch, output)

        if output["tokens"].shape[1] != self.config.token_length:
            raise ShapeSCVAEError(
                "Token length mismatch in sanity check. "
                f"Expected {self.config.token_length}, got {output['tokens'].shape[1]}."
            )
        if output["tokens"].shape[2] != self.config.token_dim:
            raise ShapeSCVAEError(
                "Token dimension mismatch in sanity check. "
                f"Expected {self.config.token_dim}, got {output['tokens'].shape[2]}."
            )

        return {
            "input_shape": tuple(batch.shape),
            "recon_shape": tuple(output["recon"].shape),
            "token_shape": tuple(output["tokens"].shape),
            "mu_shape": tuple(output["mu"].shape),
            "log_var_shape": tuple(output["log_var"].shape),
            "losses": {k: float(v.detach().cpu().item()) for k, v in losses.items()},
            "contract": self.encoder.contract_summary(),
        }


def run_shape_sc_vae_sanity(
    train_cfg: Mapping[str, Any],
    data_cfg: Mapping[str, Any],
    device: str = "cpu",
) -> Dict[str, Any]:
    """Config-driven sanity entrypoint for script-level smoke checks."""
    model = ShapeSCVAE.from_stage1_configs(train_cfg, data_cfg)
    return model.forward_sanity_check(device=device)
