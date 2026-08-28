"""Combined fusion subsystem: module, composition, and runner.

This consolidates gated_fusion.py, fusion_composition.py and fusion_runner.py
into a single cohesive module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from multimodal.model.contracts import (
    CONDITIONING_MODALITIES,
    MODALITY_ORDER,
    MODALITY_TOKEN_IDS,
    CollatedBatch,
    ModalityTokenSequence,
)

if TYPE_CHECKING:
    from config.multimodal.model_settings import ModelSettings
    from multimodal.model.outputs.projection import ProjectedModalities


class GatedFusion(nn.Module):
    """Gate and fuse per-modality embeddings with optional top-k routing."""

    def __init__(self, *, modality_dim: int, modality_count: int = 4) -> None:
        super().__init__()
        self.modality_count = modality_count
        self.gate = nn.Sequential(nn.Linear(modality_dim, 1), nn.Sigmoid())
        self.output = nn.Sequential(
            nn.Linear(modality_dim, modality_dim),
            nn.LayerNorm(modality_dim),
            nn.GELU(),
        )

    def forward(
        self,
        modality_embeddings: torch.Tensor,
        modality_mask: torch.Tensor,
        *,
        max_active_modalities: int | None = None,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if modality_embeddings.ndim != 3:
            raise ValueError(
                "modality_embeddings must be [batch, modalities, dim]"
            )
        if modality_mask.ndim != 2:
            raise ValueError("modality_mask must be [batch, modalities]")

        batch_size, modality_count, _modality_dim = modality_embeddings.shape
        if modality_mask.shape != (batch_size, modality_count):
            raise ValueError(
                "modality_mask must match modality_embeddings batch/modality axes"
            )
        if modality_count > self.modality_count:
            raise ValueError(
                f"expected at most {self.modality_count} modalities, got {modality_count}"
            )

        routed_mask = modality_mask.to(dtype=torch.bool)
        scores = self.gate(modality_embeddings).squeeze(-1)
        if max_active_modalities is not None:
            if max_active_modalities <= 0:
                raise ValueError("max_active_modalities must be positive")
            top_k = min(int(max_active_modalities), modality_count)
            masked_scores = scores.masked_fill(
                ~routed_mask, torch.finfo(scores.dtype).min
            )
            _, indices = masked_scores.topk(k=top_k, dim=1)
            top_mask = torch.zeros_like(routed_mask)
            top_mask.scatter_(dim=1, index=indices, value=True)
            routed_mask = routed_mask & top_mask

        masked_scores = scores.masked_fill(
            ~routed_mask, torch.finfo(scores.dtype).min
        )
        weights = torch.softmax(masked_scores, dim=1)
        weights = weights * routed_mask.to(dtype=weights.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        fused = (modality_embeddings * weights.unsqueeze(-1)).sum(dim=1)
        fused = self.output(fused)
        has_route = routed_mask.any(dim=1).unsqueeze(-1)
        fused = torch.where(has_route, fused, torch.zeros_like(fused))
        if return_weights:
            return fused, weights
        return fused


class MultimodalContextAssembler(nn.Module):
    """Assemble bounded, position-aware modality sequences for decoding."""

    def __init__(self, *, config: ModelSettings) -> None:
        super().__init__()
        decoder = config.text_decoder
        self.hidden_dim = int(decoder.hidden_dim)
        self.max_context_tokens = int(decoder.max_context_tokens)
        self.budgets = {
            "text": int(decoder.max_text_context_tokens),
            "document": int(decoder.max_document_context_tokens),
            "image": int(decoder.max_image_context_tokens),
            "audio": int(decoder.max_audio_context_tokens),
            "video": int(decoder.max_video_context_tokens),
            "layout": 1,
            "mask": 1,
        }
        modality_count = len(MODALITY_TOKEN_IDS)
        self.modality_embedding = nn.Embedding(modality_count, self.hidden_dim)
        self.separator_embedding = nn.Embedding(
            modality_count, self.hidden_dim
        )
        self.temporal_projection = nn.Linear(1, self.hidden_dim, bias=False)
        self.spatial_projection = nn.Linear(2, self.hidden_dim, bias=False)
        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(
        self,
        *,
        encoded_by_modality: dict[str, dict[str, torch.Tensor]],
        batch_size: int,
        device: torch.device,
        task_types: tuple[str, ...] | None = None,
    ) -> ModalityTokenSequence:
        token_parts: list[torch.Tensor] = []
        mask_parts: list[torch.Tensor] = []
        modality_parts: list[torch.Tensor] = []
        temporal_parts: list[torch.Tensor] = []
        spatial_parts: list[torch.Tensor] = []
        separator_parts: list[torch.Tensor] = []
        ordered_names = (*MODALITY_ORDER, "layout", "mask")
        remaining = self.max_context_tokens

        # Check if any row has causal_text_pretrain task
        causal_pretrain_rows = set()
        if task_types:
            for idx, t in enumerate(task_types):
                if t == "causal_text_pretrain":
                    causal_pretrain_rows.add(idx)

        for name in ordered_names:
            encoded = encoded_by_modality.get(name)
            if encoded is None or remaining <= 1:
                continue
            content_budget = min(
                self.budgets.get(name, 0),
                max(0, remaining - 1),
            )
            if content_budget <= 0:
                continue
            tokens = encoded.get("tokens")
            attention_mask = encoded.get("attention_mask")
            if not torch.is_tensor(tokens):
                embedding = encoded.get("embedding")
                if not torch.is_tensor(embedding):
                    continue
                tokens = embedding.unsqueeze(1)
            if tokens.ndim != 3 or tokens.shape[0] != batch_size:
                raise ValueError(
                    f"{name} context tokens must be [batch, tokens, hidden]"
                )
            if tokens.shape[-1] != self.hidden_dim:
                raise ValueError(
                    f"{name} token dimension must equal decoder hidden_dim"
                )
            tokens = tokens.to(device=device)
            if not torch.is_tensor(attention_mask):
                attention_mask = torch.ones(
                    tokens.shape[:2],
                    dtype=torch.bool,
                    device=device,
                )
            attention_mask = attention_mask.to(device=device, dtype=torch.bool)

            # For causal_text_pretrain, exclude text modality entirely
            if name == "text" and causal_pretrain_rows:
                row_has_text = attention_mask.any(dim=1, keepdim=True)
                # Zero out tokens for causal_text_pretrain rows
                for row_idx in causal_pretrain_rows:
                    if row_has_text[row_idx].item():
                        tokens[row_idx] = 0
                        attention_mask[row_idx] = False

            count = min(tokens.shape[1], content_budget)
            token_indices = _context_token_indices(
                modality=name,
                token_count=int(tokens.shape[1]),
                selected_count=count,
                device=device,
            )
            tokens = tokens.index_select(1, token_indices)
            attention_mask = attention_mask.index_select(1, token_indices)
            modality_id = MODALITY_TOKEN_IDS[name]
            modality_ids = torch.full(
                (batch_size, count),
                modality_id,
                dtype=torch.long,
                device=device,
            )

            temporal = _position_tensor(
                encoded.get("temporal_positions"),
                batch_size=batch_size,
                count=count,
                width=None,
                device=device,
                dtype=tokens.dtype,
                indices=token_indices,
            )
            spatial = _position_tensor(
                encoded.get("spatial_positions"),
                batch_size=batch_size,
                count=count,
                width=2,
                device=device,
                dtype=tokens.dtype,
                indices=token_indices,
            )
            tokens = tokens + self.modality_embedding(modality_ids)
            if temporal is not None:
                tokens = tokens + self.temporal_projection(
                    temporal.unsqueeze(-1)
                )
            if spatial is not None:
                tokens = tokens + self.spatial_projection(spatial)
            tokens = self.norm(tokens)
            tokens = tokens * attention_mask.unsqueeze(-1).to(tokens.dtype)

            row_has_modality = attention_mask.any(dim=1, keepdim=True)
            separator_ids = torch.full(
                (batch_size, 1),
                modality_id,
                dtype=torch.long,
                device=device,
            )
            separator = self.modality_embedding(
                separator_ids
            ) + self.separator_embedding(separator_ids)
            separator = self.norm(separator)
            separator = separator * row_has_modality.unsqueeze(-1).to(
                separator.dtype
            )

            token_parts.extend((separator, tokens))
            mask_parts.extend((row_has_modality, attention_mask))
            modality_parts.extend((separator_ids, modality_ids))
            temporal_parts.extend(
                (
                    torch.zeros(
                        batch_size, 1, dtype=tokens.dtype, device=device
                    ),
                    temporal
                    if temporal is not None
                    else torch.zeros(
                        batch_size, count, dtype=tokens.dtype, device=device
                    ),
                )
            )
            spatial_parts.extend(
                (
                    torch.zeros(
                        batch_size, 1, 2, dtype=tokens.dtype, device=device
                    ),
                    spatial
                    if spatial is not None
                    else torch.zeros(
                        batch_size, count, 2, dtype=tokens.dtype, device=device
                    ),
                )
            )
            separator_parts.extend(
                (
                    torch.ones(batch_size, 1, dtype=torch.bool, device=device),
                    torch.zeros(
                        batch_size, count, dtype=torch.bool, device=device
                    ),
                )
            )
            remaining -= count + 1

        if not token_parts:
            result = ModalityTokenSequence(
                tokens=torch.zeros(
                    batch_size, 0, self.hidden_dim, device=device
                ),
                attention_mask=torch.zeros(
                    batch_size, 0, dtype=torch.bool, device=device
                ),
                modality_ids=torch.zeros(
                    batch_size, 0, dtype=torch.long, device=device
                ),
                temporal_positions=torch.zeros(
                    batch_size, 0, dtype=torch.float32, device=device
                ),
                spatial_positions=torch.zeros(
                    batch_size, 0, 2, dtype=torch.float32, device=device
                ),
                separator_mask=torch.zeros(
                    batch_size, 0, dtype=torch.bool, device=device
                ),
            )
        else:
            result = ModalityTokenSequence(
                tokens=torch.cat(token_parts, dim=1),
                attention_mask=torch.cat(mask_parts, dim=1),
                modality_ids=torch.cat(modality_parts, dim=1),
                temporal_positions=torch.cat(temporal_parts, dim=1),
                spatial_positions=torch.cat(spatial_parts, dim=1),
                separator_mask=torch.cat(separator_parts, dim=1),
            )
        result.validate()
        return result


def _position_tensor(
    value: object,
    *,
    batch_size: int,
    count: int,
    width: int | None,
    device: torch.device,
    dtype: torch.dtype,
    indices: torch.Tensor,
) -> torch.Tensor | None:
    if not torch.is_tensor(value):
        return None
    expected_ndim = 2 if width is None else 3
    if value.ndim != expected_ndim or value.shape[0] != batch_size:
        kind = "temporal" if width is None else "spatial"
        raise ValueError(f"{kind} positions have an invalid shape")
    if width is not None and value.shape[-1] != width:
        raise ValueError("spatial positions must have width 2")
    if value.shape[1] < count:
        raise ValueError("position sequence is shorter than context tokens")
    selected = value.to(device=device).index_select(1, indices)
    return selected.to(dtype=dtype)


def _context_token_indices(
    *,
    modality: str,
    token_count: int,
    selected_count: int,
    device: torch.device,
) -> torch.Tensor:
    """Select bounded context tokens without discarding the media tail."""

    if selected_count <= 0 or token_count <= 0:
        return torch.zeros(0, dtype=torch.long, device=device)
    if selected_count >= token_count:
        return torch.arange(token_count, dtype=torch.long, device=device)
    if modality in {"image", "audio", "video"}:
        return (
            torch.linspace(
                0,
                token_count - 1,
                steps=selected_count,
                device=device,
            )
            .round()
            .to(dtype=torch.long)
        )
    return torch.arange(selected_count, dtype=torch.long, device=device)


class FusionComposition(nn.Module):
    """Own the configured fusion backend."""

    def __init__(
        self,
        *,
        config: ModelSettings,
        enabled_modalities: tuple[str, ...],
    ) -> None:
        super().__init__()
        self.fusion = GatedFusion(
            modality_dim=config.fusion_dim,
            modality_count=(
                len(enabled_modalities)
                + len(conditioning_modalities(enabled_modalities))
            ),
        )
        self.context_assembler = MultimodalContextAssembler(config=config)


def conditioning_modalities(
    enabled_modalities: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        modality
        for modality in CONDITIONING_MODALITIES
        if modality != "document" or "document" not in enabled_modalities
    )


def build_fusion_composition(
    *,
    config: ModelSettings,
    enabled_modalities: tuple[str, ...],
) -> FusionComposition:
    return FusionComposition(
        config=config,
        enabled_modalities=enabled_modalities,
    )


@dataclass
class FusedRepresentation:
    fused: torch.Tensor
    fusion_weights: torch.Tensor
    modality_masks: list[torch.Tensor]


class FusionRunner:
    def __init__(
        self, *, config: ModelSettings, fusion: FusionComposition
    ) -> None:
        self._config = config
        self._fusion = fusion

    def context_sequence(
        self,
        *,
        projected: "ProjectedModalities",
        batch: "CollatedBatch",
        task_types: tuple[str, ...] | None = None,
    ) -> ModalityTokenSequence:
        device = _projected_device(projected=projected, batch=batch)
        return self._fusion.context_assembler.forward(
            encoded_by_modality=projected.encoded_by_modality,
            batch_size=len(batch.sample_ids),
            device=device,
            task_types=task_types,
        )

    def fuse(
        self,
        projected: "ProjectedModalities",
        batch: "CollatedBatch",
        max_active_modalities: int | None,
    ) -> FusedRepresentation:
        if projected.modality_embeddings:
            modalities = torch.stack(projected.modality_embeddings, dim=1)
            modality_mask = torch.stack(projected.modality_masks, dim=1)
            fused, fusion_weights = self._fusion.fusion(
                modalities,
                modality_mask,
                max_active_modalities=max_active_modalities,
                return_weights=True,
            )
        else:
            reference = torch.zeros(1, dtype=torch.float32)
            # Attempt to obtain a device from projected modality masks if possible
            device = None
            if projected.modality_masks:
                device = projected.modality_masks[0].device
            if device is None:
                device = reference.device
            fused = torch.zeros(
                len(batch.sample_ids),
                self._config.fusion_dim,
                dtype=torch.float32,
                device=device,
            )
            fusion_weights = torch.zeros(
                len(batch.sample_ids),
                0,
                dtype=torch.float32,
                device=device,
            )

        return FusedRepresentation(
            fused=fused,
            fusion_weights=fusion_weights,
            modality_masks=projected.modality_masks,
        )


def _projected_device(
    *, projected: "ProjectedModalities", batch: "CollatedBatch"
) -> torch.device:
    for encoded in projected.encoded_by_modality.values():
        for value in encoded.values():
            if torch.is_tensor(value):
                return value.device
    for modality_tensor in (
        batch.text,
        batch.image,
        batch.audio,
        batch.video,
        batch.document,
    ):
        if torch.is_tensor(modality_tensor):
            return modality_tensor.device
    return torch.device("cpu")
