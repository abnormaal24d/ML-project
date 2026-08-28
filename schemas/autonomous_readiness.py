"""Canonical autonomous multimodal readiness contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

AUTONOMOUS_REQUIRED_MODALITIES: Final[tuple[str, ...]] = (
    "text",
    "image",
    "audio",
    "video",
)

AUTONOMOUS_REQUIRED_TASKS: Final[tuple[str, ...]] = (
    "text_pretrain",
    "image_text_pair",
    "audio_text_pair",
    "video_text_pair",
)


def missing_autonomous_tasks(
    task_counts: Mapping[str, int],
) -> tuple[str, ...]:
    """Return autonomous tasks without usable dataset/training evidence."""

    return tuple(
        task
        for task in AUTONOMOUS_REQUIRED_TASKS
        if int(task_counts.get(task, 0)) <= 0
    )


def missing_autonomous_modalities(
    modality_counts: Mapping[str, int],
) -> tuple[str, ...]:
    """Return autonomous modalities without usable evidence."""

    return tuple(
        modality
        for modality in AUTONOMOUS_REQUIRED_MODALITIES
        if int(modality_counts.get(modality, 0)) <= 0
    )
