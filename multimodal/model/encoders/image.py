"""Image modality encoders (moved)."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from torch import nn

from multimodal.model.encoders.primitives import (
    FeedForwardEncoder,
)

if TYPE_CHECKING:
    from config.multimodal.encoder_settings import EncoderSettings


class AutoVisionEncoder(nn.Module):
    """Encode feature vectors or raw image pixels."""

    def __init__(
        self,
        config: EncoderSettings,
        *,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.feature_encoder = FeedForwardEncoder(config)
        self.raw_encoder = ScratchVisionEncoder(
            config,
            gradient_checkpointing=gradient_checkpointing,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        output_keys: set[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        if x.ndim == 4:
            return cast(
                "dict[str, torch.Tensor]",
                self.raw_encoder(x, output_keys=output_keys),
            )
        embedding = cast("torch.Tensor", self.feature_encoder(x))
        return {
            "embedding": embedding,
            "tokens": embedding.unsqueeze(1),
            "attention_mask": torch.ones(
                embedding.shape[0],
                1,
                dtype=torch.bool,
                device=embedding.device,
            ),
            "spatial_positions": torch.zeros(
                embedding.shape[0],
                1,
                2,
                dtype=embedding.dtype,
                device=embedding.device,
            ),
        }


class ScratchVisionEncoder(nn.Module):
    """CNN image encoder plus lightweight pixel reconstruction head."""

    def __init__(
        self,
        config: EncoderSettings,
        *,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.gradient_checkpointing = bool(gradient_checkpointing)
        hidden = config.hidden_dim
        self.stem = nn.Sequential(
            nn.Conv2d(3, hidden // 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(hidden // 2),
            nn.GELU(),
            nn.Conv2d(hidden // 2, hidden, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.projection = nn.Linear(hidden, config.output_dim)
        self.reconstruction_head = nn.Sequential(
            nn.Conv2d(hidden, hidden // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden // 2, 3, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        pixels: torch.Tensor,
        *,
        output_keys: set[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        features = self.stem(pixels)
        pooled = F.adaptive_avg_pool2d(features, output_size=1).flatten(1)
        region_tokens = features.flatten(2).transpose(1, 2)
        projected_tokens = self.projection(region_tokens)
        grid_height, grid_width = features.shape[-2:]
        y = torch.linspace(0.0, 1.0, grid_height, device=features.device)
        x = torch.linspace(0.0, 1.0, grid_width, device=features.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        spatial_positions = torch.stack((xx, yy), dim=-1).reshape(1, -1, 2)
        spatial_positions = spatial_positions.expand(features.shape[0], -1, -1)
        outputs = {
            "embedding": self.projection(pooled),
            "tokens": projected_tokens,
            "attention_mask": torch.ones(
                projected_tokens.shape[:2],
                dtype=torch.bool,
                device=projected_tokens.device,
            ),
            "spatial_positions": spatial_positions.to(
                dtype=projected_tokens.dtype
            ),
        }
        if output_keys is None or "image_reconstruction" in output_keys:
            outputs["image_reconstruction"] = self.reconstruction_head(
                F.interpolate(
                    features,
                    size=pixels.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            )
        if output_keys is None or "image_region_features" in output_keys:
            outputs["image_region_features"] = projected_tokens
        return outputs
