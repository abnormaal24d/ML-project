"""Video modality encoders (moved)."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import nn

from multimodal.model.encoders.primitives import (
    FeedForwardEncoder,
    maybe_checkpoint,
)

if TYPE_CHECKING:
    from config.multimodal.encoder_settings import EncoderSettings


class AutoVideoEncoder(nn.Module):
    """Encode feature vectors or raw sampled video frames."""

    def __init__(
        self,
        config: EncoderSettings,
        *,
        max_frames: int,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.feature_encoder = FeedForwardEncoder(config)
        self.raw_encoder = ScratchVideoEncoder(
            config,
            max_frames=max_frames,
            gradient_checkpointing=gradient_checkpointing,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        output_keys: set[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        if x.ndim == 5:
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


class ScratchVideoEncoder(nn.Module):
    """Frame CNN with a temporal transformer trained from raw frames.

    Produces spatiotemporal tokens with spatial grid per frame
    and temporal positions. Uses 4x4 spatial grid per frame
    (16 spatial tokens per frame) for a total of max_frames * 16 tokens.
    """

    def __init__(
        self,
        config: EncoderSettings,
        *,
        max_frames: int,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if max_frames <= 0:
            raise ValueError("max_frames must be greater than zero")
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.max_frames = int(max_frames)
        self.spatial_grid_size = 4  # 4x4 spatial grid = 16 tokens per frame
        self.spatial_tokens_per_frame = (
            self.spatial_grid_size * self.spatial_grid_size
        )
        self.max_tokens = max_frames * self.spatial_tokens_per_frame
        hidden = config.hidden_dim
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(3, hidden // 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(hidden // 2),
            nn.GELU(),
            nn.Conv2d(hidden // 2, hidden, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.temporal_position_embedding = nn.Embedding(max_frames, hidden)
        # Spatial position embeddings for 4x4 grid (16 positions)
        self.spatial_position_embedding = nn.Embedding(
            self.spatial_tokens_per_frame, hidden
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.attention_heads,
            dim_feedforward=hidden * 4,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.temporal_encoder = nn.TransformerEncoder(
            layer,
            num_layers=2,
            enable_nested_tensor=False,
        )
        self.projection = nn.Linear(hidden, config.output_dim)
        self.temporal_head = nn.Linear(hidden, 2)

    def forward(
        self,
        frames: torch.Tensor,
        *,
        output_keys: set[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size, frame_count, channels, height, width = frames.shape
        if frame_count > self.max_frames:
            raise ValueError(
                f"video contains {frame_count} frames, configured maximum is "
                f"{self.max_frames}"
            )
        flat = frames.reshape(
            batch_size * frame_count, channels, height, width
        )
        # frame_encoder output: [B*T, hidden, H, W]
        frame_features = self.frame_encoder(flat)
        # Reshape to spatial tokens: [B*T, H*W, hidden]
        spatial_features = frame_features.flatten(2).transpose(1, 2)
        # Add spatial position embeddings
        spatial_pos_ids = torch.arange(
            self.spatial_tokens_per_frame,
            device=frames.device,
            dtype=torch.long,
        )
        spatial_pos_emb = self.spatial_position_embedding(spatial_pos_ids)
        spatial_features = spatial_features + spatial_pos_emb.unsqueeze(0)
        # Reshape back to [B, T*S, hidden]
        sequence = spatial_features.reshape(
            batch_size, frame_count * self.spatial_tokens_per_frame, -1
        )
        # Add temporal position embeddings
        temporal_pos_ids = torch.arange(
            frame_count, device=frames.device, dtype=torch.long
        )
        temporal_pos_emb = self.temporal_position_embedding(temporal_pos_ids)
        # Expand temporal embeddings to match spatial tokens
        temporal_pos_emb = temporal_pos_emb.repeat_interleave(
            self.spatial_tokens_per_frame, dim=0
        )
        temporal_pos_emb = temporal_pos_emb.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        sequence = sequence + temporal_pos_emb
        encoded = maybe_checkpoint(
            self.temporal_encoder,
            sequence,
            enabled=self.gradient_checkpointing and self.training,
        )
        # Project tokens
        spatiotemporal_tokens = self.projection(encoded)
        # Create temporal positions for each token
        temporal_positions = torch.linspace(
            0.0,
            1.0,
            frame_count,
            device=frames.device,
        )
        temporal_positions = temporal_positions.repeat_interleave(
            self.spatial_tokens_per_frame
        )
        temporal_positions = (
            temporal_positions.unsqueeze(0)
            .expand(batch_size, -1)
            .to(dtype=spatiotemporal_tokens.dtype)
        )
        # Create spatial positions (x, y) for each token
        grid_y = torch.arange(
            self.spatial_grid_size, device=frames.device
        ).repeat_interleave(self.spatial_grid_size)
        grid_x = torch.arange(
            self.spatial_grid_size, device=frames.device
        ).repeat(self.spatial_grid_size)
        spatial_positions = torch.stack([grid_x, grid_y], dim=1).float()
        # Normalize to [0, 1]
        spatial_positions = spatial_positions / (self.spatial_grid_size - 1)
        # Repeat for each frame
        spatial_positions = spatial_positions.repeat(frame_count, 1)
        spatial_positions = (
            spatial_positions.unsqueeze(0)
            .expand(batch_size, -1, -1)
            .to(dtype=spatiotemporal_tokens.dtype)
        )
        outputs = {
            "embedding": self.projection(encoded.mean(dim=1)),
            "tokens": spatiotemporal_tokens,
            "attention_mask": torch.ones(
                spatiotemporal_tokens.shape[:2],
                dtype=torch.bool,
                device=spatiotemporal_tokens.device,
            ),
            "temporal_positions": temporal_positions,
            "spatial_positions": spatial_positions,
        }
        if output_keys is None or "video_temporal_logits" in output_keys:
            outputs["video_temporal_logits"] = self.temporal_head(
                encoded.mean(dim=1)
            )
        return outputs
