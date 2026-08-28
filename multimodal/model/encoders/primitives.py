"""Shared neural-network primitives used by multiple modality encoders."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

if TYPE_CHECKING:
    from config.multimodal.encoder_settings import EncoderSettings


class FeedForwardEncoder(nn.Module):
    """Project feature vectors into the configured encoder output space."""

    def __init__(self, config: EncoderSettings) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast("torch.Tensor", self.net(x))


def maybe_checkpoint(
    function: Callable[[torch.Tensor], torch.Tensor],
    value: torch.Tensor,
    *,
    enabled: bool,
) -> torch.Tensor:
    if not enabled:
        return function(value)
    return cast(
        "torch.Tensor",
        checkpoint(function, value, use_reentrant=False),
    )
