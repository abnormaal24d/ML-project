"""Scratch-trained image codec for latent diffusion image generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from torch import nn

if TYPE_CHECKING:
    from config.multimodal.generation_settings import ImageCodecSettings


@dataclass(frozen=True, slots=True)
class ImageCodecOutput:
    """Output from the image codec encoder."""

    latents: torch.Tensor  # [B, C, H, W]
    embedding: torch.Tensor  # [B, D] global embedding for contrastive losses


class ScratchImageCodecEncoder(nn.Module):
    """Encode images into a spatial latent grid."""

    def __init__(self, *, config: "ImageCodecSettings") -> None:
        super().__init__()
        self.latent_channels = int(config.latent_channels)
        self.downsample_factor = int(config.downsample_factor)
        self.hidden_dim = int(config.hidden_dim)
        self.input_resolution = int(config.input_resolution)
        self.downsample_blocks = int(config.upsample_blocks)
        self.latent_resolution = (
            self.input_resolution // self.downsample_factor
        )

        stages: list[nn.Module] = []
        current_dim = 3
        for index in range(self.downsample_blocks):
            output_dim = (
                self.hidden_dim * 2
                if index == self.downsample_blocks - 1
                else max(
                    self.hidden_dim
                    // (2 ** max(self.downsample_blocks - index - 2, 0)),
                    64,
                )
            )
            stages.extend(
                (
                    nn.Conv2d(
                        current_dim,
                        output_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm2d(output_dim),
                    nn.GELU(),
                )
            )
            current_dim = output_dim
        stages.append(
            nn.Conv2d(current_dim, self.latent_channels, kernel_size=1)
        )
        self.encoder = nn.Sequential(*stages)

        # Global average pooling for global embedding
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_projection = nn.Linear(config.latent_channels, 256)

    def forward(self, images: torch.Tensor) -> ImageCodecOutput:
        """Encode images into latent grid and global embedding.

        Args:
            images: [B, 3, H, W] input images in range [0, 1] or [-1, 1]

        Returns:
            ImageCodecOutput with latents [B, C, H, W] and global embedding [B, 256]
        """
        if images.ndim != 4:
            raise ValueError("image codec inputs must have shape [B, 3, H, W]")
        expected_shape = (
            3,
            self.input_resolution,
            self.input_resolution,
        )
        if tuple(images.shape[1:]) != expected_shape:
            raise ValueError(
                "image codec input shape must match the configured contract: "
                f"expected=[B, {expected_shape[0]}, {expected_shape[1]}, "
                f"{expected_shape[2]}], got={tuple(images.shape)}"
            )

        # Normalize to [-1, 1] if needed
        if images.min() >= 0 and images.max() <= 1:
            images = images * 2 - 1

        latents = cast(
            torch.Tensor, self.encoder(images)
        )  # [B, C, H/factor, W/factor]
        global_emb = self.global_pool(latents).flatten(1)  # [B, C]
        global_emb = self.global_projection(global_emb)  # [B, 256]

        return ImageCodecOutput(latents=latents, embedding=global_emb)


class ScratchImageCodecDecoder(nn.Module):
    """Decode latent grid back to RGB images."""

    def __init__(self, *, config: "ImageCodecSettings") -> None:
        super().__init__()
        self.latent_channels = int(config.latent_channels)
        self.output_channels = int(config.output_channels)
        self.upsample_block_count = int(config.upsample_blocks)
        self.latent_resolution = int(config.input_resolution) // int(
            config.downsample_factor
        )

        # Initial projection from latent channels
        self.input_projection = nn.Conv2d(
            config.latent_channels, config.hidden_dim * 2, kernel_size=1
        )

        # Upsampling blocks: 14x14 -> 28x28 -> 56x56 -> 112x112 -> 224x224
        self.upsample_blocks = nn.ModuleList()
        current_dim = config.hidden_dim * 2
        for i in range(self.upsample_block_count):
            out_dim = max(config.hidden_dim // (2**i), 64)
            self.upsample_blocks.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(current_dim, out_dim, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_dim),
                    nn.GELU(),
                    nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_dim),
                    nn.GELU(),
                )
            )
            current_dim = out_dim

        # Final RGB output
        self.final_conv = nn.Sequential(
            nn.Conv2d(
                current_dim, config.output_channels, kernel_size=3, padding=1
            ),
            nn.Tanh(),  # Output in [-1, 1]
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent grid back to RGB images.

        Args:
            latents: [B, C, H, W] latent grid

        Returns:
            images: [B, 3, H, W] RGB images in range [-1, 1]
        """
        if latents.ndim != 4:
            raise ValueError(
                "image codec latents must have shape [B, C, H, W]"
            )
        expected_shape = (
            self.latent_channels,
            self.latent_resolution,
            self.latent_resolution,
        )
        if tuple(latents.shape[1:]) != expected_shape:
            raise ValueError(
                "image codec latent shape must match the configured contract: "
                f"expected=[B, {expected_shape[0]}, {expected_shape[1]}, "
                f"{expected_shape[2]}], got={tuple(latents.shape)}"
            )

        x = self.input_projection(latents)

        for upsample_block in self.upsample_blocks:
            x = upsample_block(x)

        images = cast(torch.Tensor, self.final_conv(x))
        return images


class ScratchImageCodec(nn.Module):
    """Complete image codec with encoder and decoder."""

    def __init__(self, *, config: "ImageCodecSettings") -> None:
        super().__init__()
        self.encoder = ScratchImageCodecEncoder(config=config)
        self.decoder = ScratchImageCodecDecoder(config=config)
        self.config = config

    def encode(self, images: torch.Tensor) -> ImageCodecOutput:
        """Encode images to latent grid."""
        return cast(ImageCodecOutput, self.encoder(images))

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to RGB images."""
        return cast(torch.Tensor, self.decoder(latents))

    def forward(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, ImageCodecOutput]:
        """Full encode-decode cycle for reconstruction loss.

        Args:
            images: [B, 3, H, W] input images

        Returns:
            reconstructed: [B, 3, H, W] reconstructed images
            codec_output: ImageCodecOutput with latents and global embedding
        """
        codec_output = self.encode(images)
        reconstructed = self.decode(codec_output.latents)
        return reconstructed, codec_output

    def reconstruct_loss(
        self, images: torch.Tensor, reconstructed: torch.Tensor
    ) -> torch.Tensor:
        """Compute reconstruction loss (L1 + MSE)."""
        l1_loss = nn.functional.l1_loss(reconstructed, images)
        mse_loss = nn.functional.mse_loss(reconstructed, images)
        return l1_loss + mse_loss
