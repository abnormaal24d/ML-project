"""Audio modality encoders (moved)."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from torch import nn

from multimodal.model.encoders.primitives import (
    FeedForwardEncoder,
    maybe_checkpoint,
)

if TYPE_CHECKING:
    from config.multimodal.encoder_settings import EncoderSettings


class AutoAudioEncoder(nn.Module):
    """Encode feature vectors or raw waveform tensors."""

    def __init__(
        self,
        config: EncoderSettings,
        *,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.feature_encoder = FeedForwardEncoder(config)
        self.raw_encoder = ScratchAudioEncoder(
            config,
            gradient_checkpointing=gradient_checkpointing,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        output_keys: set[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        if x.ndim == 3:
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
            "temporal_positions": torch.zeros(
                embedding.shape[0],
                1,
                dtype=embedding.dtype,
                device=embedding.device,
            ),
        }


class ScratchAudioEncoder(nn.Module):
    """Convolutional waveform encoder plus masked-span reconstruction head."""

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
            nn.Conv1d(1, hidden // 2, kernel_size=15, stride=4, padding=7),
            nn.GELU(),
            nn.Conv1d(hidden // 2, hidden, kernel_size=9, stride=4, padding=4),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.projection = nn.Linear(hidden, config.output_dim)
        self.reconstruction_head = nn.Sequential(
            nn.Conv1d(hidden, hidden // 2, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden // 2, 1, kernel_size=1),
            nn.Tanh(),
        )

    def forward(
        self,
        waveform: torch.Tensor,
        *,
        output_keys: set[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        features = maybe_checkpoint(
            self.stem,
            waveform,
            enabled=self.gradient_checkpointing and self.training,
        )
        pooled = F.adaptive_avg_pool1d(
            features,
            output_size=1,
        ).flatten(1)
        frame_tokens = self.projection(features.transpose(1, 2))
        temporal_positions = (
            torch.linspace(
                0.0, 1.0, frame_tokens.shape[1], device=frame_tokens.device
            )
            .unsqueeze(0)
            .expand(frame_tokens.shape[0], -1)
        )
        outputs = {
            "embedding": self.projection(pooled),
            "tokens": frame_tokens,
            "attention_mask": torch.ones(
                frame_tokens.shape[:2],
                dtype=torch.bool,
                device=frame_tokens.device,
            ),
            "temporal_positions": temporal_positions.to(
                dtype=frame_tokens.dtype
            ),
        }
        if output_keys is None or "audio_reconstruction" in output_keys:
            outputs["audio_reconstruction"] = self.reconstruction_head(
                F.interpolate(features, size=waveform.shape[-1], mode="linear")
            )
        return outputs
