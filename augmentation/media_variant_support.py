"""Shared helpers for building media augmentation variants."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from augmentation.outcomes.augmentation_result import AugmentationRejection
from augmentation.outcomes.rejection_reason import AugmentationRejectionReason
from mmcrawler_datasets.schema import MultimodalSample


def resolve_dataset_root(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return Path(value).resolve()


def resolve_source_path(
    *,
    dataset_root: Path,
    value: str | Path,
    error_message: str,
) -> Path:
    path = Path(value)
    resolved = (
        path.resolve()
        if path.is_absolute()
        else (dataset_root / path).resolve()
    )
    if not resolved.is_relative_to(dataset_root):
        raise ValueError(error_message)
    return resolved


def remove_incomplete_file(
    path: Path, *, logger: Any, event_name: str
) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError as exc:
        logger.warning(
            event_name,
            extra={"path": str(path), "error_type": type(exc).__name__},
        )


def media_rejection(
    *,
    sample: MultimodalSample,
    reason: AugmentationRejectionReason,
    message: str | None,
    operation: str,
    modality: str,
) -> AugmentationRejection:
    return AugmentationRejection(
        reason=reason,
        sample_id=sample.sample_id,
        variant_name=operation,
        modality=modality,
        message=message,
    )


def preserved_metadata(
    metadata: dict[str, object],
    policy: str,
    safe_fields: frozenset[str],
) -> dict[str, object]:
    if policy == "strip_all":
        return {}
    if policy == "preserve_all":
        return dict(metadata)
    return {
        key: value for key, value in metadata.items() if key in safe_fields
    }
