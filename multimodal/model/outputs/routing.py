"""Output routing and head resolution (combines several resolver modules)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from multimodal.model.contracts import (
    ALL_OUTPUT_HEADS,
    LOGICAL_TO_PHYSICAL_MODALITIES,
    MODALITY_ORDER,
    PHYSICAL_MODALITIES,
    CollatedBatch,
)
from multimodal.tasks.registry import (
    CAUSAL_TEXT_OBJECTIVES,
    get_task,
    resolved_input_modalities,
    resolved_output_modalities,
)

if TYPE_CHECKING:
    from collections.abc import Collection

    from config.multimodal.model_settings import ModelSettings


def has_valid_targets(values: object) -> bool:
    import torch

    if values is None or not torch.is_tensor(values):
        return False

    if values.numel() == 0:
        return False

    return bool(values.ne(-100).any().item())


def has_nonzero_rows(values: object) -> bool:
    import torch

    if values is None or not torch.is_tensor(values):
        return False

    if values.numel() == 0:
        return False

    return bool(values.detach().abs().sum().gt(0).item())


TEXT_LIKE_OUTPUTS = {"text", "json", "code"}


def resolve_output_heads(
    *,
    batch: "CollatedBatch",
    config: "ModelSettings",
    requested: Collection[str] | str | None,
) -> frozenset[str]:
    normalized = normalize_output_heads(requested)
    if normalized is not None:
        return normalized

    heads = {"projection", "modality_embeddings"}
    heads.update(resolve_loss_output_heads(batch=batch, config=config))
    heads.update(resolve_generation_output_heads(batch=batch, config=config))
    return frozenset(heads)


def resolve_loss_output_heads(
    *, batch: "CollatedBatch", config: "ModelSettings"
) -> set[str]:
    heads: set[str] = set()
    output_modalities = resolve_batch_output_modalities(
        batch=batch, config=config
    )

    for output_modality in output_modalities:
        if output_modality == "class" and batch.labels is not None:
            heads.add("classifier")

    if batch.labels is not None:
        heads.add("classifier")

    if has_valid_targets(getattr(batch, "emotion_label_ids", None)):
        heads.add("emotion")

    if has_valid_targets(getattr(batch, "speaker_label_ids", None)):
        heads.add("speaker")

    if has_nonzero_rows(getattr(batch, "layout_box_targets", None)):
        heads.add("layout")

    if has_nonzero_rows(getattr(batch, "object_box_targets", None)):
        heads.add("object")

    if getattr(batch, "text_mlm_targets", None) is not None:
        heads.add("text_mlm")

    if getattr(batch, "image_reconstruction_target", None) is not None:
        heads.add("image_reconstruction")

    if getattr(batch, "audio_reconstruction_target", None) is not None:
        heads.add("audio_reconstruction")

    if getattr(batch, "video_temporal_labels", None) is not None:
        heads.add("video_temporal")

    return heads


def normalize_output_heads(
    requested: Collection[str] | str | None,
) -> frozenset[str] | None:
    if requested is None:
        return None

    values = (requested,) if isinstance(requested, str) else requested
    normalized = {str(value) for value in values}

    if "all" in normalized:
        return ALL_OUTPUT_HEADS

    unknown = normalized - ALL_OUTPUT_HEADS
    if unknown:
        raise ValueError(f"unknown output heads: {sorted(unknown)}")
    return frozenset(normalized)


def resolve_generation_output_heads(
    *, batch: "CollatedBatch", config: "ModelSettings"
) -> set[str]:
    heads: set[str] = set()
    task_types = list(getattr(batch, "task_types", None) or [])

    if not task_types:
        _append_generation_heads(
            heads=heads,
            output_modalities=resolve_batch_output_modalities(
                batch=batch, config=config
            ),
            objective=None,
        )
        return heads

    batch_modalities = list(getattr(batch, "output_modalities", None) or [])
    for index, task_type in enumerate(task_types):
        definition = get_task(task_type)
        if definition is None:
            output_modalities = (
                batch_modalities[index]
                if index < len(batch_modalities)
                else ()
            )
            _append_generation_heads(
                heads=heads,
                output_modalities=output_modalities,
                objective=None,
            )
            continue

        output_modalities = (
            batch_modalities[index]
            if index < len(batch_modalities) and batch_modalities[index]
            else task_output_modalities(
                config=config,
                task_type=definition.name,
            )
        )
        _append_generation_heads(
            heads=heads,
            output_modalities=output_modalities,
            objective=definition.loss_key,
        )
    return heads


def _append_generation_heads(
    *,
    heads: set[str],
    output_modalities: Collection[str],
    objective: str | None,
) -> None:
    modalities = {str(value).strip().lower() for value in output_modalities}
    if objective is None:
        if _requests_text_generation(modalities):
            heads.add("sequence")
        if "image" in modalities:
            heads.update(("image_latents", "generated_image"))
        if "audio" in modalities:
            heads.add("audio_token")
        if "video" in modalities:
            heads.add("video_generation")
        return

    if objective in CAUSAL_TEXT_OBJECTIVES and _requests_text_generation(
        modalities
    ):
        heads.add("sequence")
    if (
        objective in {"image_generation", "image_editing"}
        and "image" in modalities
    ):
        heads.update(("image_latents", "generated_image"))
    if (
        objective in {"audio_generation", "speech_translation"}
        and "audio" in modalities
    ):
        heads.add("audio_token")
    if objective == "video_generation" and "video" in modalities:
        heads.add("video_generation")


def _requests_text_generation(
    output_modalities: set[str] | frozenset[str],
) -> bool:
    return bool(TEXT_LIKE_OUTPUTS.intersection(output_modalities))


# Task routing helpers (moved from task_routing.py)


def resolve_batch_output_modalities(
    *, batch: "CollatedBatch", config: "ModelSettings"
) -> frozenset[str]:
    values: set[str] = set()
    saw_batch_route = False

    for modalities in getattr(batch, "output_modalities", None) or []:
        if modalities:
            saw_batch_route = True
            values.update(
                str(modality).strip().lower() for modality in modalities
            )

    for task_type in getattr(batch, "task_types", None) or []:
        task_modalities = task_output_modalities(
            config=config, task_type=task_type
        )
        if task_modalities:
            saw_batch_route = True
            values.update(task_modalities)

    if not saw_batch_route and not getattr(batch, "task_types", None):
        values.update(config.output_modalities)

    return frozenset(value for value in values if value)


def task_output_modalities(
    *, config: "ModelSettings", task_type: str
) -> tuple[str, ...]:
    definition = get_task(task_type)
    if definition is None:
        return ()
    modalities = resolved_output_modalities(
        definition.name,
        overrides=config.modality_routing.task_output_overrides,
    )
    return tuple(str(modality).strip().lower() for modality in modalities)


def task_route_masks(
    *, batch: "CollatedBatch", config: "ModelSettings", reference: torch.Tensor
) -> dict[str, torch.Tensor]:
    batch_size = len(batch.sample_ids)
    route_masks = {
        modality: torch.zeros(
            batch_size, dtype=torch.bool, device=reference.device
        )
        for modality in MODALITY_ORDER
    }
    row_has_route = torch.zeros(
        batch_size, dtype=torch.bool, device=reference.device
    )

    task_types = getattr(batch, "task_types", None) or []
    for index in range(batch_size):
        task_type = task_types[index] if index < len(task_types) else ""
        modalities = task_input_modalities(config=config, task_type=task_type)
        if not modalities:
            continue

        row_has_route[index] = True
        for modality in modalities:
            route_masks[modality][index] = True

    if not bool(row_has_route.all().item()):
        fallback = ~row_has_route
        for modality in MODALITY_ORDER:
            route_masks[modality] = route_masks[modality] | fallback

    return route_masks


def task_input_modalities(
    *, config: "ModelSettings", task_type: str
) -> tuple[str, ...]:
    definition = get_task(task_type)
    if definition is None:
        return ()
    configured = resolved_input_modalities(
        definition.name, overrides=config.modality_routing.task_input_overrides
    )
    modalities: list[str] = []
    for modality in configured:
        if modality in PHYSICAL_MODALITIES:
            modalities.append(modality)
            continue
        modalities.extend(LOGICAL_TO_PHYSICAL_MODALITIES.get(modality, ()))
    return tuple(dict.fromkeys(modalities))
