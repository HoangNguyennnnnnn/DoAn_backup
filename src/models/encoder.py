"""
Shape SC-VAE Encoder

Implements the encoder component of the Shape SC-VAE for voxel-based shape encoding.
Maps 3D voxel grids to latent space.

Architecture:
- Input: 32^3 voxel grid (float)
- Backbone: 3D CNN with progressively increasing channels
- Output: Latent mean and log-variance (for VAE reparameterization)

Expected to be called as part of the training loop in scripts/train_stage1.py
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from src.models.shape_interface import (
    OVoxelToSLATShapeAdapter,
    ShapeInterfaceError,
    ShapePathContractConfig,
    validate_shape_path_smoke,
)


class ShapeEncoder(nn.Module):
    """
    Voxel-to-latent encoder for Shape SC-VAE.
    
    Attributes:
        input_channels: Number of input channels (1 for binary voxel)
        latent_dim: Dimensionality of latent space
        hidden_channels: List of hidden channel sizes for progressive downsampling
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        input_resolution: int = 32,
        latent_dim: int = 128,
        hidden_channels: Optional[list[int]] = None,
        voxel_dtype: str = "float32",
        token_length: int = 8,
        token_dim: Optional[int] = None,
    ):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [64, 128, 256]
        if token_dim is None:
            if latent_dim % token_length != 0:
                raise ShapeInterfaceError(
                    "token_dim is not provided and latent_dim is not divisible by token_length. "
                    f"Received latent_dim={latent_dim}, token_length={token_length}."
                )
            token_dim = latent_dim // token_length
        
        self.input_channels = input_channels
        self.input_resolution = input_resolution
        self.latent_dim = latent_dim
        self.hidden_channels = hidden_channels
        self.token_length = token_length
        self.token_dim = token_dim
        
        layers: list[nn.Module] = []
        in_ch = input_channels
        for out_ch in hidden_channels:
            layers.extend(
                [
                    nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm3d(out_ch),
                    nn.ReLU(inplace=True),
                ]
            )
            in_ch = out_ch
        self.encoder_layers = nn.Sequential(*layers)

        with torch.no_grad():
            probe = torch.zeros(
                1,
                input_channels,
                input_resolution,
                input_resolution,
                input_resolution,
            )
            encoded_probe = self.encoder_layers(probe)
            self._encoded_shape = tuple(encoded_probe.shape[1:])
            encoded_features = int(encoded_probe.numel())

        self.fc_mu = nn.Linear(encoded_features, latent_dim)
        self.fc_log_var = nn.Linear(encoded_features, latent_dim)

        contract = ShapePathContractConfig(
            input_channels=input_channels,
            input_resolution=input_resolution,
            voxel_dtype=voxel_dtype,
            latent_dim=latent_dim,
            token_length=token_length,
            token_dim=token_dim,
        )
        self.shape_adapter = OVoxelToSLATShapeAdapter(contract)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode voxel grid to latent distribution.
        
        Args:
            x: Input voxel grid (B, C, H, W, D) where H=W=D=32
            
        Returns:
            (mu, log_var): Latent mean and log-variance for VAE reparameterization
        """
        x = self.shape_adapter.normalize_input(x)
        features = self.encoder_layers(x)
        flattened = torch.flatten(features, start_dim=1)
        mu = self.fc_mu(flattened)
        log_var = self.fc_log_var(flattened)
        return mu, log_var

    def encode_to_shape_tokens(
        self,
        x: torch.Tensor,
        sample: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Return latent stats plus SLAT shape tokens for downstream stages."""
        mu, log_var = self.forward(x)
        z = LatentSampler.sample(mu, log_var) if sample else mu
        tokens = self.shape_adapter.tokens_from_latent(z)
        return {
            "tokens": tokens,
            "mu": mu,
            "log_var": log_var,
            "latent": z,
        }

    def contract_summary(self) -> Dict[str, Any]:
        """Expose active OVoxel-to-SLAT contract for logging and diagnostics."""
        return self.shape_adapter.contract_summary()

    def validate_shape_contract(self, x: torch.Tensor) -> Dict[str, tuple[int, ...]]:
        """Quick contract smoke-check for train/inference entrypoints."""
        mu, log_var = self.forward(x)
        return validate_shape_path_smoke(self.shape_adapter, x, mu, log_var)


# Placeholder for future integrations
class LatentSampler(nn.Module):
    """Helper for VAE reparameterization trick."""
    
    @staticmethod
    def sample(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: sample from N(mu, sigma^2) = mu + sigma * N(0,1)
        
        Args:
            mu: Mean of latent distribution
            log_var: Log-variance of latent distribution
            
        Returns:
            z: Sampled latent vector
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
