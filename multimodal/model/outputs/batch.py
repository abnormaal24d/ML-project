"""Batch-related output helpers (renamed from context.py)."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import nn

from multimodal.model.contracts import (
    CONDITIONING_MODALITIES,
    ENCODER_OUTPUT_HEADS,
    MODALITY_ORDER,
    CollatedBatch,
)
from multimodal.model.encoders.runner import required_embedding
from multimodal.model.outputs.routing import task_route_masks

if TYPE_CHECKING:
    from collections.abc import Collection

    from config.multimodal.model_settings import ModelSettings


def head_requested(output_heads: frozenset[str], name: str) -> bool:
    return name in output_heads


def include_encoder_output(*, key: str, output_heads: frozenset[str]) -> bool:
    head_name = ENCODER_OUTPUT_HEADS.get(key)
    return head_name is not None and head_name in output_heads


def modality_embedding_outputs(
    *,
    encoded_by_modality: dict[str, dict[str, torch.Tensor]],
    enabled_modalities: tuple[str, ...],
    batch_size: int,
    fusion_dim: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    outputs: dict[str, torch.Tensor] = {}

    for modality in enabled_modalities:
        encoded = encoded_by_modality.get(modality)
        if encoded is None:
            embedding = torch.zeros(
                batch_size,
                fusion_dim,
                dtype=torch.float32,
                device=device,
            )
        else:
            embedding = required_embedding(
                encoded=encoded,
                source=f"{modality} modality embedding",
            )

        outputs.update(
            {
                f"{modality}_embedding": nn.functional.normalize(
                    embedding,
                    dim=-1,
                )
            }
        )

    for modality in CONDITIONING_MODALITIES:
        encoded = encoded_by_modality.get(modality)
        if encoded is None:
            continue
        embedding = required_embedding(
            encoded=encoded,
            source=f"{modality} conditioning embedding",
        )
        outputs.update(
            {
                f"{modality}_embedding": nn.functional.normalize(
                    embedding,
                    dim=-1,
                )
            }
        )

    return outputs


def resolve_modality_row_masks(
    *,
    batch: "CollatedBatch",
    config: "ModelSettings",
    enabled_modalities: tuple[str, ...],
    requested_modalities: Collection[str] | None,
) -> dict[str, torch.Tensor]:
    reference = reference_tensor(batch)
    batch_size = len(batch.sample_ids)

    if requested_modalities is None:
        route_masks = task_route_masks(
            batch=batch,
            config=config,
            reference=reference,
        )
    else:
        requested = {
            str(modality).strip().lower() for modality in requested_modalities
        }
        route_masks = {
            modality: torch.full(
                (batch_size,),
                modality in requested,
                dtype=torch.bool,
                device=reference.device,
            )
            for modality in MODALITY_ORDER
        }

    masks: dict[str, torch.Tensor] = {}
    for modality in enabled_modalities:
        base_mask = resolve_modality_mask(
            batch=batch, modality=modality, reference=reference
        )
        route_mask = route_masks.get(
            modality,
            torch.ones(batch_size, dtype=torch.bool, device=reference.device),
        )
        masks[modality] = base_mask & route_mask.to(base_mask.device)

    return masks


def resolve_modality_mask(
    *, batch: "CollatedBatch", modality: str, reference: torch.Tensor
) -> torch.Tensor:
    mask = getattr(batch, f"{modality}_mask", None)
    if mask is not None:
        return cast(
            "torch.Tensor", mask.to(device=reference.device, dtype=torch.bool)
        )

    modality_mask = getattr(batch, "modality_mask", None)
    if modality_mask is not None and modality in MODALITY_ORDER:
        index = MODALITY_ORDER.index(modality)
        if modality_mask.ndim == 2 and modality_mask.shape[1] > index:
            return cast(
                "torch.Tensor",
                modality_mask[:, index].to(
                    device=reference.device, dtype=torch.bool
                ),
            )
    return torch.ones(
        reference.shape[0], dtype=torch.bool, device=reference.device
    )


def optional_batch_mask(
    *, batch: "CollatedBatch", name: str, reference: torch.Tensor
) -> torch.Tensor:
    mask = getattr(batch, name, None)
    if mask is None or not torch.is_tensor(mask):
        return torch.zeros(
            reference.shape[0], dtype=torch.bool, device=reference.device
        )
    return mask.to(device=reference.device, dtype=torch.bool)


def reference_tensor(batch: "CollatedBatch") -> torch.Tensor:
    for name in MODALITY_ORDER:
        value = getattr(batch, name, None)
        if isinstance(value, torch.Tensor):
            return value

    raise ValueError("batch does not contain any modality tensors")
