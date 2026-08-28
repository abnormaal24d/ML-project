"""Strategy ordering rules for sample-level augmentation."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.augmentation.augmentation_settings import AugmentationSettings


def plan_augmentation_strategies(
    *,
    settings: AugmentationSettings,
    source_sample_id: str,
    modality: str | None,
    task_type: str | None,
    domain: str | None,
) -> tuple[str, ...]:
    strategies = list(
        _enabled_strategies(
            settings=settings,
            modality=modality,
            task_type=task_type,
            domain=domain,
        )
    )
    if not strategies:
        return ()
    limit = settings.text.max_variants_per_sample
    if limit >= len(strategies):
        return tuple(strategies)
    offset = _stable_offset(
        source_sample_id=source_sample_id,
        strategy_count=len(strategies),
    )
    rotated = strategies[offset:] + strategies[:offset]
    return tuple(rotated[:limit])


def _enabled_strategies(
    *,
    settings: AugmentationSettings,
    modality: str | None,
    task_type: str | None,
    domain: str | None,
) -> tuple[str, ...]:
    strategies: list[str] = []
    if settings.text.text_span_focus_enabled:
        strategies.append("text_span_focus")
    if settings.text.title_prefix_enabled:
        strategies.append("title_prefix")
    if _should_apply_context_prefix(
        settings=settings,
        modality=modality,
        task_type=task_type,
        domain=domain,
    ):
        strategies.append("context_prefix")
    return tuple(strategies)


def _should_apply_context_prefix(
    *,
    settings: AugmentationSettings,
    modality: str | None,
    task_type: str | None,
    domain: str | None,
) -> bool:
    if not settings.text.context_prefix_enabled:
        return False
    if not any(_as_opt_str(value) for value in (modality, task_type, domain)):
        return False
    allowed_task_types = settings.text.context_prefix_task_types
    if not allowed_task_types:
        return True
    if task_type is None:
        return True
    return task_type in allowed_task_types


def _stable_offset(*, source_sample_id: str, strategy_count: int) -> int:
    if strategy_count <= 1:
        return 0
    digest = hashlib.sha256(source_sample_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % strategy_count


def _as_opt_str(value: object) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text or None
