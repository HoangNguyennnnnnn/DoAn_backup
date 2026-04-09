"""
Shape SC-VAE Decoder

Implements the decoder component of the Shape SC-VAE for voxel generation.
Maps latent vectors back to 3D voxel grids.

Architecture:
- Input: Latent vector (128-dim)
- Backbone: 3D transposed CNN with progressively decreasing channels
- Output: 32^3 voxel grid (sigmoid activation for [0, 1] range)

Expected to be called as part of the training loop in scripts/train_stage1.py
"""

import torch
import torch.nn as nn


class ShapeDecoder(nn.Module):
    """
    Latent-to-voxel decoder for Shape SC-VAE.
    
    Attributes:
        latent_dim: Dimensionality of input latent space
        output_resolution: Size of output voxel grid (32 for 32^3)
        hidden_channels: List of hidden channel sizes for progressive upsampling
        output_channels: Number of output channels (1 for binary voxel)
    """
    
    def __init__(
        self,
        latent_dim: int = 128,
        output_resolution: int = 32,
        hidden_channels: list[int] = None,
        output_channels: int = 1,
    ):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = [256, 128, 64]
        
        self.latent_dim = latent_dim
        self.output_resolution = output_resolution
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels

        upsample_factor = 2 ** len(hidden_channels)
        if output_resolution % upsample_factor != 0:
            raise ValueError(
                "output_resolution must be divisible by 2^len(hidden_channels). "
                f"Got output_resolution={output_resolution}, hidden_layers={len(hidden_channels)}."
            )

        self.base_resolution = output_resolution // upsample_factor
        if self.base_resolution <= 0:
            raise ValueError(
                f"Invalid base resolution derived from output_resolution={output_resolution}."
            )

        base_channels = hidden_channels[0]
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, base_channels * (self.base_resolution ** 3)),
            nn.ReLU(inplace=True),
        )

        blocks: list[nn.Module] = []
        for idx in range(len(hidden_channels) - 1):
            blocks.extend(
                [
                    nn.ConvTranspose3d(
                        hidden_channels[idx],
                        hidden_channels[idx + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm3d(hidden_channels[idx + 1]),
                    nn.ReLU(inplace=True),
                ]
            )
        self.decoder_layers = nn.Sequential(*blocks)

        self.output_layer = nn.ConvTranspose3d(
            hidden_channels[-1],
            output_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to voxel grid.
        
        Args:
            z: Latent vector (B, latent_dim)
            
        Returns:
            x_recon: Reconstructed voxel grid (B, C, 32, 32, 32)
        """
        if z.ndim != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(
                "Latent input must have shape (B, latent_dim). "
                f"Expected latent_dim={self.latent_dim}, received {tuple(z.shape)}."
            )

        batch = z.shape[0]
        x = self.fc(z)
        x = x.view(
            batch,
            self.hidden_channels[0],
            self.base_resolution,
            self.base_resolution,
            self.base_resolution,
        )
        x = self.decoder_layers(x)
        x = self.output_layer(x)
        return torch.sigmoid(x)


class ReconstructionLoss(nn.Module):
    """Reconstruction loss for voxel grids (typically MSE)."""
    
    def forward(
        self,
        recon: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute reconstruction loss between reconstructed and target voxels.
        
        Args:
            recon: Reconstructed voxel grid (B, C, 32, 32, 32)
            target: Target voxel grid (B, C, 32, 32, 32)
            
        Returns:
            loss: Scalar loss value
        """
        # TODO: Implement reconstruction loss (typically MSE or BCE)
        # For continuous voxels: torch.nn.MSELoss
        # For binary voxels: torch.nn.BCELoss
        return torch.nn.functional.mse_loss(recon, target)
