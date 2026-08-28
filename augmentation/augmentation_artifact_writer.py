"""Write augmentation report artifacts next to augmented split files."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mmcrawler_datasets.record_components.coercion import require_mapping
from mmcrawler_datasets.snapshots.output_writer import (
    write_snapshot_metadata,
    write_snapshot_rows,
)

if TYPE_CHECKING:
    from augmentation.outcomes.augmentation_result import AugmentationReport


def build_augmentation_summary(
    *,
    train_rows: tuple[dict[str, object], ...],
    val_rows: tuple[dict[str, object], ...],
    test_rows: tuple[dict[str, object], ...],
) -> dict[str, object]:
    """Build neutral post-augmentation counts for snapshot metadata."""

    split_rows = {
        "train": train_rows,
        "val": val_rows,
        "test": test_rows,
    }
    split_counts = {
        split_name: len(rows) for split_name, rows in split_rows.items()
    }
    all_rows = (*train_rows, *val_rows, *test_rows)
    modalities = _count_modalities(rows=all_rows)
    tasks = _count_tasks(rows=all_rows)
    tasks_by_split = {
        split_name: _count_tasks(rows=rows)
        for split_name, rows in split_rows.items()
    }
    return {
        "samples_total": len(all_rows),
        "splits": split_counts,
        "modalities": modalities,
        "tasks": tasks,
        "tasks_by_split": tasks_by_split,
    }


def _count_modalities(
    *,
    rows: tuple[dict[str, object], ...],
) -> dict[str, int]:
    counts = Counter(_bucket_value(row.get("modality")) for row in rows)
    return dict(sorted(counts.items()))


def _count_tasks(
    *,
    rows: tuple[dict[str, object], ...],
) -> dict[str, int]:
    counts = Counter(
        _bucket_value(require_mapping(row, "task_target").get("task_type"))
        for row in rows
    )
    return dict(sorted(counts.items()))


def _bucket_value(value: object) -> str:
    text = str(value or "unknown").strip()
    return text or "unknown"


def write_augmentation_artifacts(
    *,
    built_at: datetime,
    output_directory: Path,
    original_train_rows: tuple[dict[str, object], ...],
    augmented_train_rows: tuple[dict[str, object], ...],
    val_rows: tuple[dict[str, object], ...],
    test_rows: tuple[dict[str, object], ...],
    report: AugmentationReport,
) -> None:
    variants_added = max(
        0,
        len(augmented_train_rows) - len(original_train_rows),
    )
    lineage_rows = tuple(
        build_lineage_row(row=row)
        for row in augmented_train_rows
        if row.get("augmentation_source_sample_id") is not None
    )
    rejection_rows = tuple(
        _rejection_row(rejection)
        for rejection in getattr(report, "rejected_augmentations", ())
    )
    media_transform_applied = any(
        bool(row.get("augmentation_media_transform_applied"))
        for row in augmented_train_rows
    )
    variants_by_name = report.variants_by_name
    variants_by_modality = report.variants_by_modality
    rejected_by_reason = report.rejected_by_reason

    payload = {
        "augmentation_type": _augmentation_type(
            variants_by_modality=variants_by_modality,
            media_transform_applied=media_transform_applied,
        ),
        "built_at": built_at.isoformat(),
        "enabled": True,
        "original_train_samples": len(original_train_rows),
        "augmented_train_samples": len(augmented_train_rows),
        "val_samples": len(val_rows),
        "test_samples": len(test_rows),
        "samples_total": (
            len(augmented_train_rows) + len(val_rows) + len(test_rows)
        ),
        "variants_added": variants_added,
        "variants_by_name": dict(variants_by_name),
        "variants_by_operation": dict(
            report.variants_by_operation or variants_by_name
        ),
        "variants_by_modality": dict(variants_by_modality),
        "variants_by_task_type": dict(report.variants_by_task_type),
        "rejected_by_reason": dict(rejected_by_reason),
        "rejections_by_modality": dict(report.rejections_by_modality),
        "rejected_augmented_count": sum(
            int(value) for value in rejected_by_reason.values()
        ),
        "media_outputs": dict(report.media_outputs),
        "media_transform_applied": media_transform_applied,
        "quality_checks_passed": bool(report.quality_checks_passed),
        "quality_checks": dict(report.quality_checks),
        "quality_check_failures": list(report.quality_check_failures),
    }

    write_snapshot_metadata(
        path=output_directory / "augmentation_report.json",
        payload=payload,
    )
    write_snapshot_rows(
        path=output_directory / "augmentation_lineage.jsonl",
        rows=lineage_rows,
    )
    write_snapshot_rows(
        path=output_directory / "rejected_augmentations.jsonl",
        rows=rejection_rows,
    )


def build_lineage_row(
    *,
    row: Mapping[str, object],
) -> dict[str, object]:
    task_target = require_mapping(dict(row), "task_target")
    output_path = row.get(
        "augmentation_output_path"
    ) or _canonical_source_location(row=row)
    payload = {
        "sample_id": row.get("sample_id"),
        "source_sample_id": row.get("augmentation_source_sample_id"),
        "augmentation_name": row.get("augmentation_name"),
        "modality": row.get("modality"),
        "task_type": task_target.get("task_type"),
        "media_transform_applied": bool(
            row.get("augmentation_media_transform_applied")
        ),
        "output_path": output_path,
        "source_sha256": row.get("augmentation_source_sha256"),
        "output_sha256": row.get("augmentation_output_sha256"),
        "config_hash": row.get("augmentation_config_hash"),
        "implementation_hash": row.get("augmentation_implementation_hash"),
        "implementation_version": row.get(
            "augmentation_implementation_version"
        ),
    }
    return {key: value for key, value in payload.items() if value is not None}


def _augmentation_type(
    *,
    variants_by_modality: Mapping[str, int],
    media_transform_applied: bool,
) -> str:
    if not variants_by_modality:
        return "none"
    if media_transform_applied:
        return "mixed"
    if set(variants_by_modality) == {"text"}:
        return "text"
    return "multimodal"


def _canonical_source_location(
    *,
    row: Mapping[str, object],
) -> object | None:
    raw_objects = row.get("objects")
    if raw_objects is None:
        return None
    if not isinstance(raw_objects, list):
        raise ValueError("schema 3.0 field 'objects' must be an array")
    for item in raw_objects:
        if not isinstance(item, Mapping):
            raise ValueError("every objects[] item must be an object")
        location: object | None = item.get("object_path")
        if location is None:
            location = item.get("object_url")
        if location is not None:
            return location
    return None


def _rejection_row(rejection: Any) -> dict[str, object]:
    if hasattr(rejection, "to_row"):
        payload = rejection.to_row()
        if isinstance(payload, dict):
            return dict(payload)
    if isinstance(rejection, Mapping):
        return dict(rejection)
    return {"reason": str(rejection)}


__all__ = [
    "build_augmentation_summary",
    "build_lineage_row",
    "write_augmentation_artifacts",
    "_augmentation_type",
    "_canonical_source_location",
    "_rejection_row",
]
