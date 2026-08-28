"""Training contract validation: required losses and generation output checks."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import torch

from multimodal.tasks.registry import (
    GENERATION_OBJECTIVES,
    get_task,
    task_loss_requirements,
    task_producible_loss_terms,
)


class NonFiniteLossError(ValueError):
    """Identify the exact objective term that produced a non-finite value."""


SUPPORTED_LOSS_KEYS: frozenset[str] = frozenset(
    {
        "audio_generation",
        "audio_reconstruction",
        "contrastive",
        "hard_negative",
        "image_generation",
        "image_reconstruction",
        "label",
        "language_modeling",
        "ocr_sequence",
        "preference",
        "safety",
        "sequence",
        "text_mlm",
        "video_generation",
        "video_temporal",
    }
)


GENERATION_LOSS_TERMS_BY_MODALITY = MappingProxyType(
    {
        "text": frozenset(
            {
                "sequence",
                "language_modeling",
                "ocr_sequence",
                "preference",
            }
        ),
        "json": frozenset(
            {
                "sequence",
                "language_modeling",
                "ocr_sequence",
                "preference",
            }
        ),
        "code": frozenset({"sequence", "language_modeling", "preference"}),
        "image": frozenset({"image_generation"}),
        "audio": frozenset({"audio_generation"}),
        "video": frozenset({"video_generation"}),
    }
)


def validate_required_losses(
    *,
    losses: Mapping[str, torch.Tensor],
    coverage: Mapping[str, torch.Tensor],
    batch: object,
    model_output: Mapping[str, torch.Tensor],
    require_targets_for_generation: bool,
    training_stage: str,
) -> None:
    """Validate that every training row has a required loss and generation targets are met.

    Args:
        losses: Produced loss terms (name -> scalar tensor).
        coverage: Row coverage for each loss term (name -> bool tensor [batch_size]).
        batch: Training batch with task_types, output_modalities, etc.
        model_output: Model outputs (keys used for error messages).
        require_targets_for_generation: If True, enforce generation loss presence.
        training_stage: Current training stage string.

    Raises:
        RuntimeError: If any row lacks required loss coverage or generation
            targets are missing corresponding losses.
    """
    if require_targets_for_generation:
        requested_modalities = _requested_generation_modalities(batch)
        missing = [
            modality
            for modality in requested_modalities
            if (expected := GENERATION_LOSS_TERMS_BY_MODALITY.get(modality))
            and not expected.intersection(losses)
        ]
        if missing:
            available = sorted(model_output.keys())
            raise RuntimeError(
                f"batch requests generated outputs for {missing}, "
                f"but no corresponding generation loss was produced; "
                f"available model outputs={available}"
            )

    task_types = getattr(batch, "task_types", None)
    if not isinstance(task_types, list) or not task_types:
        return
    if training_stage == "PREFERENCE_TUNING":
        return

    for row_index, task_type in enumerate(task_types):
        definition = get_task(str(task_type))
        if definition is None:
            raise RuntimeError(f"unknown training task {task_type!r}")
        requirement_groups = task_loss_requirements(definition.name)
        if requirement_groups is None:
            raise RuntimeError(
                f"task {definition.name!r} uses objective "
                f"{definition.loss_key!r}, but that objective has no "
                "implemented training-loss contract"
            )
        for group in requirement_groups:
            if any(
                name in coverage and bool(coverage[name][row_index].item())
                for name in group
            ):
                continue
            raise RuntimeError(
                f"row {row_index} of task {definition.name!r} "
                f"requires one of {sorted(group)}, but no covering "
                f"loss term was produced; "
                f"available loss terms={sorted(losses)}"
            )


def _requested_generation_modalities(batch: object) -> set[str]:
    """Extract requested generation modalities from batch."""
    task_types = getattr(batch, "task_types", None)
    output_rows = getattr(batch, "output_modalities", None)
    if not isinstance(output_rows, list) or not all(
        isinstance(row, tuple) for row in output_rows
    ):
        raise TypeError(
            "batch.output_modalities must be a list of canonical tuples"
        )

    requested: set[str] = set()
    if not isinstance(task_types, list) or not task_types:
        return {modality for row in output_rows for modality in row}

    for index, task_type in enumerate(task_types):
        definition = get_task(str(task_type))
        if definition is None:
            if index < len(output_rows):
                requested.update(output_rows[index])
            continue
        objective = definition.loss_key
        if objective not in GENERATION_OBJECTIVES:
            continue
        modalities = (
            output_rows[index]
            if index < len(output_rows) and output_rows[index]
            else definition.output_modalities
        )
        requested.update(modalities)
    return requested


def validate_final_losses(
    losses: Mapping[str, torch.Tensor],
    loss_weights: Mapping[str, float],
    available_outputs: Mapping[str, object] | list[str],
) -> dict[str, torch.Tensor]:
    """Apply weights, check finiteness, and compute total loss.

    Args:
        losses: Raw loss terms (name -> scalar tensor).
        loss_weights: Configured weight for each loss term.
        available_outputs: Model output keys (for error message).

    Returns:
        Dict with weighted losses plus "total" key.

    Raises:
        RuntimeError: If no losses produced, weight is zero for a produced term,
            or total loss is non-finite.
        NonFiniteLossError: If any loss term is non-finite.
    """
    if not losses:
        raise RuntimeError(
            "no trainable loss terms were produced for this batch; "
            f"available model outputs={sorted(available_outputs)}"
        )
    weighted_losses: dict[str, torch.Tensor] = {}
    for name, raw_loss in losses.items():
        if raw_loss.numel() != 1:
            raise ValueError(
                f"loss term {name!r} must be scalar, got shape "
                f"{tuple(raw_loss.shape)}"
            )
        if not torch.isfinite(raw_loss).all():
            value = raw_loss.detach().cpu().reshape(-1).tolist()
            raise NonFiniteLossError(f"non-finite loss term {name!r}: {value}")
        weight = loss_weights[name]
        if weight <= 0.0:
            raise RuntimeError(
                f"loss term {name!r} was produced although its "
                "configured weight is zero"
            )
        weighted_losses[name] = raw_loss * weight
    total = torch.stack(tuple(weighted_losses.values())).sum()
    if not torch.isfinite(total).all():
        raise NonFiniteLossError("weighted total loss is non-finite")
    weighted_losses["total"] = total
    return weighted_losses


def rows_producing_loss(
    task_types: list[str],
    loss_term: str,
    *,
    device: torch.device,
) -> torch.Tensor | None:
    """Return boolean mask of rows that produce the given loss term.

    Args:
        task_types: List of task type strings for each row.
        loss_term: Concrete loss term name (e.g., "language_modeling").
        device: Target device for returned tensor.

    Returns:
        Boolean tensor of shape [batch_size] or None if task_types is not a list.
    """
    if not isinstance(task_types, list):
        return None
    return torch.tensor(
        [
            loss_term in task_producible_loss_terms(str(task_type))
            for task_type in task_types
        ],
        dtype=torch.bool,
        device=device,
    )
