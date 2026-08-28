"""Text modality encoders (moved).

This file is a relocation of multimodal.model.text_encoder.py to the
encoders subpackage for clearer grouping. No behavior changes.
"""

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


class AutoTextEncoder(nn.Module):
    """Encode feature vectors or scratch-tokenized text."""

    def __init__(
        self,
        config: EncoderSettings,
        *,
        vocab_size: int,
        max_tokens: int,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.feature_encoder = FeedForwardEncoder(config)
        self.raw_encoder = ScratchTextTransformer(
            config,
            vocab_size=vocab_size,
            max_tokens=max_tokens,
            gradient_checkpointing=gradient_checkpointing,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        output_keys: set[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        if x.dtype in {torch.int16, torch.int32, torch.int64, torch.long}:
            return cast(
                "dict[str, torch.Tensor]",
                self.raw_encoder(
                    x.to(dtype=torch.long),
                    output_keys=output_keys,
                ),
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


class ScratchTextTransformer(nn.Module):
    """Small transformer trained from project-local token ids."""

    def __init__(
        self,
        config: EncoderSettings,
        *,
        vocab_size: int,
        max_tokens: int,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.token_embedding = nn.Embedding(
            vocab_size,
            config.hidden_dim,
            padding_idx=0,
        )
        self.position_embedding = nn.Embedding(max_tokens, config.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=2,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.projection = nn.Linear(config.hidden_dim, config.output_dim)
        self.mlm_head = nn.Linear(config.hidden_dim, vocab_size)

    def _prepare_token_ids(self, tokens: torch.Tensor) -> torch.Tensor:
        """Normalize token shape and reject ids outside the vocabulary."""
        if not torch.is_tensor(tokens):
            raise TypeError("tokens must be a torch.Tensor")

        vocab_size = self.token_embedding.num_embeddings
        max_tokens = self.position_embedding.num_embeddings

        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")

        tokens = tokens.to(
            device=self.token_embedding.weight.device, dtype=torch.long
        )
        if tokens.ndim == 0:
            tokens = tokens.reshape(1, 1)
        elif tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        elif tokens.ndim > 2:
            tokens = tokens.reshape(tokens.shape[0], -1)
        if tokens.shape[1] > max_tokens:
            tokens = tokens[:, :max_tokens]
        if tokens.shape[1] == 0:
            tokens = torch.zeros(
                (tokens.shape[0], 1),
                dtype=torch.long,
                device=self.token_embedding.weight.device,
            )

        invalid = (tokens < 0) | (tokens >= vocab_size)
        if invalid.any():
            positions = invalid.nonzero(as_tuple=False)
            context: list[str] = []
            for batch_index, token_index in positions[:8].tolist():
                token_id = int(tokens[batch_index, token_index].item())
                context.append(
                    f"batch_index={batch_index}, token_index={token_index}, token_id={token_id}"
                )
            remaining = int(positions.shape[0]) - len(context)
            suffix = f"; plus {remaining} more" if remaining > 0 else ""
            raise ValueError(
                f"text token ids must be in [0, {vocab_size}); "
                f"invalid entries: {'; '.join(context)}{suffix}"
            )
        return tokens

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        output_keys: set[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        tokens = self._prepare_token_ids(tokens)

        batch_size, sequence_length = tokens.shape
        positions = (
            torch.arange(
                sequence_length,
                device=tokens.device,
                dtype=torch.long,
            )
            .unsqueeze(0)
            .expand(batch_size, sequence_length)
        )

        hidden = self.token_embedding(tokens) + self.position_embedding(
            positions
        )

        padding_mask = tokens.eq(0)

        encoded = maybe_checkpoint(
            lambda value: self.transformer(
                value,
                src_key_padding_mask=padding_mask,
            ),
            hidden,
            enabled=self.gradient_checkpointing and self.training,
        )

        encoded = self.norm(encoded)

        valid = tokens.ne(0).to(dtype=encoded.dtype).unsqueeze(-1)
        pooled = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)

        projected_tokens = self.projection(encoded)
        outputs: dict[str, torch.Tensor] = {
            "embedding": self.projection(pooled),
            "tokens": projected_tokens,
            "attention_mask": tokens.ne(0),
            "temporal_positions": positions.to(dtype=projected_tokens.dtype),
        }

        if output_keys is None or "text_mlm_logits" in output_keys:
            outputs["text_mlm_logits"] = self.mlm_head(encoded)

        return outputs
