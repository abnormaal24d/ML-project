"""Coverage settings consistency checks kept inside the config leaf."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from config.settings.root import Settings


def validate_coverage_settings_consistency(
    *, settings: Settings
) -> tuple[str, ...]:
    coverage = settings.coverage
    errors: list[str] = []

    media_kinds = set(coverage.kinds.media_kinds)

    for kind, target in coverage.targets.modality_targets.items():
        normalized_kind = normalize_kind(kind)
        if normalized_kind not in media_kinds:
            errors.append(
                f"coverage.targets.modality_targets.{kind} is not listed in "
                "coverage.kinds.media_kinds"
            )
        if nonnegative_int(target) != int(target):
            errors.append(
                f"coverage.targets.modality_targets.{kind} must be non-negative"
            )

    for task_name, target in coverage.targets.task_targets.items():
        if nonnegative_int(target) <= 0:
            continue

        normalized_task = normalize_kind(task_name)
        media_kind = coverage.kinds.task_to_media_kind.get(normalized_task)
        if media_kind is None:
            errors.append(
                f"coverage.targets.task_targets.{task_name} requires "
                f"coverage.kinds.task_to_media_kind.{task_name}"
            )
            continue

        if media_kind not in media_kinds:
            errors.append(
                f"coverage.kinds.task_to_media_kind.{task_name}={media_kind!r} "
                "is not listed in coverage.kinds.media_kinds"
            )

    return tuple(errors)


def normalize_kind(value: object) -> str:
    """Normalize a configured coverage key without importing domain code."""

    return str(value or "").strip().lower()


def nonnegative_int(value: object, *, default: int = 0) -> int:
    """Coerce a configured count while rejecting booleans and negatives."""

    if isinstance(value, bool):
        return default
    try:
        return max(0, int(str(value).strip()))
    except (TypeError, ValueError):
        return default
