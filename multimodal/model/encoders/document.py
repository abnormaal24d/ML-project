"""Layout-aware document encoder components (moved)."""

from __future__ import annotations

import torch
from torch import nn


class LayoutAwareDocumentEncoder(nn.Module):
    """Fuse token embeddings with page and bounding-box layout features."""

    def __init__(
        self,
        *,
        token_dim: int,
        hidden_dim: int,
        attention_heads: int,
        max_pages: int = 512,
    ) -> None:
        super().__init__()
        self.page_embedding = nn.Embedding(max_pages, hidden_dim)
        self.box_projection = nn.Linear(4, hidden_dim)
        self.token_projection = nn.Linear(token_dim, hidden_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=attention_heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
                activation="gelu",
            ),
            num_layers=2,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        *,
        token_embeddings: torch.Tensor,
        boxes: torch.Tensor,
        page_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        page_ids = page_ids.clamp(0, self.page_embedding.num_embeddings - 1)
        hidden = (
            self.token_projection(token_embeddings)
            + self.box_projection(boxes.to(dtype=token_embeddings.dtype))
            + self.page_embedding(page_ids)
        )
        resolved_mask = (
            attention_mask.to(dtype=torch.bool)
            if attention_mask is not None
            else torch.ones(
                hidden.shape[:2],
                dtype=torch.bool,
                device=hidden.device,
            )
        )
        # PyTorch attention is undefined when every key in a row is masked.
        # Give those rows one temporary sentinel key and zero the encoded row
        # against the original mask after the transformer.
        safe_attention_mask = resolved_mask
        inactive_rows = ~resolved_mask.any(dim=1)
        if bool(inactive_rows.any().item()):
            safe_attention_mask = resolved_mask.clone()
            safe_attention_mask[inactive_rows, 0] = True
        padding_mask = ~safe_attention_mask
        encoded = self.encoder(hidden, src_key_padding_mask=padding_mask)
        encoded = self.norm(encoded)
        encoded = encoded.masked_fill(~resolved_mask.unsqueeze(-1), 0.0)
        weights = resolved_mask.to(dtype=encoded.dtype).unsqueeze(-1)
        pooled = encoded.sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return {
            "document_tokens": encoded,
            "tokens": encoded,
            "attention_mask": resolved_mask,
            "embedding": pooled,
            "temporal_positions": page_ids.to(dtype=encoded.dtype),
            "spatial_positions": boxes[..., :2].to(dtype=encoded.dtype),
            "page_ids": page_ids,
            "boxes": boxes,
        }
