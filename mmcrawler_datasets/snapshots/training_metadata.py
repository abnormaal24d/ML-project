"""Build and persist canonical training-snapshot metadata evidence."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from config.settings.datasets import (
    DatasetPathSettings,
    TrainingSnapshotAssemblerSettings,
)
from mmcrawler_datasets.assembly.build import SampleBuildResult
from mmcrawler_datasets.safe_io import load_bounded_json_object
from mmcrawler_datasets.snapshots.output_writer import (
    SnapshotOutputSettings,
    write_snapshot_metadata,
)
from mmcrawler_datasets.training_samples.models import TrainingSample

_REMOVED_SUMMARY_KEYS = frozenset(
    {
        "sample_count",
        "split_counts",
        "train_samples",
        "val_samples",
        "test_samples",
        "modality_counts",
        "task_types",
        "task_counts",
    }
)


class UnsupportedSnapshotSchemaError(ValueError):
    """Raised when augmentation receives a pre-schema-3 metadata file."""


def write_training_metadata(
    *,
    training_root: Path,
    snapshot_id: str,
    curated_snapshot_directory: Path,
    settings: TrainingSnapshotAssemblerSettings,
    dataset_paths: DatasetPathSettings,
    samples: SampleBuildResult,
    tokenizer_identity: Mapping[str, object],
    output_settings: SnapshotOutputSettings,
) -> None:
    """Persist manifest and statistics for one completed snapshot."""

    evidence = _dataset_evidence(samples)
    output_paths: dict[str, object] = {}
    if output_settings.write_jsonl:
        output_paths["splits"] = {
            "train": (
                Path(dataset_paths.training_splits_directory)
                / dataset_paths.training_train_filename
            ).as_posix(),
            "val": (
                Path(dataset_paths.training_splits_directory)
                / dataset_paths.training_val_filename
            ).as_posix(),
            "test": (
                Path(dataset_paths.training_splits_directory)
                / dataset_paths.training_test_filename
            ).as_posix(),
        }
    if output_settings.write_shards:
        output_paths["shard_index"] = output_settings.shard_index_filename

    write_snapshot_metadata(
        training_root / dataset_paths.dataset_manifest_filename,
        {
            "schema_version": settings.training_schema_version,
            "snapshot_id": snapshot_id,
            "final": True,
            "status": "completed",
            "valid": True,
            **evidence,
            "curated_snapshot_directory": str(curated_snapshot_directory),
            "tokenizer": dict(tokenizer_identity),
            "paths": output_paths,
            "outputs": {
                "jsonl": output_settings.write_jsonl,
                "shards": output_settings.write_shards,
                "shard_format": (
                    output_settings.shard_format
                    if output_settings.write_shards
                    else None
                ),
            },
        },
    )
    write_snapshot_metadata(
        training_root / dataset_paths.stats_filename,
        {
            "schema_version": settings.training_schema_version,
            "snapshot_id": snapshot_id,
            **evidence,
            "tokenizer_sha256": tokenizer_identity["sha256"],
        },
    )


def _dataset_evidence(samples: SampleBuildResult) -> dict[str, object]:
    all_samples = samples.all_samples
    return {
        "samples_total": len(all_samples),
        "splits": _sorted_counts(samples.split_counts),
        "modalities": _sorted_counts(_modality_counts(all_samples)),
        "tasks": _sorted_counts(_sample_counts(all_samples, "task_type")),
        "tasks_by_split": {
            "train": _sorted_counts(
                _sample_counts(samples.train_samples, "task_type")
            ),
            "val": _sorted_counts(
                _sample_counts(samples.val_samples, "task_type")
            ),
            "test": _sorted_counts(
                _sample_counts(samples.test_samples, "task_type")
            ),
        },
    }


def _modality_counts(
    samples: tuple[TrainingSample, ...],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for sample in samples:
        modality = str(sample.modality or "text")
        counts[modality] = counts.get(modality, 0) + 1
        for output in sample.task_target.output_modalities:
            name = str(output)
            counts[name] = counts.get(name, 0) + 1
    return counts


def _sample_counts(
    samples: tuple[Any, ...],
    field_name: str,
) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                str(getattr(sample, field_name, "") or "unknown")
                for sample in samples
            ).items()
        )
    )


def _sorted_counts(counts: Mapping[str, int]) -> dict[str, int]:
    return {
        str(name): max(0, int(count)) for name, count in sorted(counts.items())
    }


def update_augmented_training_metadata(
    *,
    training_root: Path,
    dataset_paths: DatasetPathSettings,
    augmentation_summary: Mapping[str, object],
) -> None:
    """Rewrite persisted metadata with neutral post-augmentation counts.

    ``augmentation_summary`` is a canonical mapping produced by the
    augmentation artifact writer; this module never imports augmentation.
    """
    summary = dict(augmentation_summary)
    validation_summary = _update_validation_report(
        path=training_root / dataset_paths.validation_report_filename,
        summary=summary,
    )
    _update_manifest(
        path=training_root / dataset_paths.dataset_manifest_filename,
        summary=summary,
        validation_summary=validation_summary,
    )
    _update_stats(
        path=training_root / dataset_paths.stats_filename,
        summary=summary,
    )
    _update_dataset_card(
        path=training_root / dataset_paths.dataset_card_filename,
        summary=summary,
    )


def replace_summary(
    *,
    payload: dict[str, Any],
    summary: Mapping[str, object],
) -> None:
    """Replace persisted count evidence with the canonical schema."""

    unexpected = _REMOVED_SUMMARY_KEYS.intersection(payload)
    if unexpected:
        raise UnsupportedSnapshotSchemaError(
            f"unsupported summary fields for schema 3.0: {sorted(unexpected)}"
        )

    payload.update(
        {
            "samples_total": _summary_count(summary, "samples_total"),
            "splits": _summary_counts(summary, "splits"),
            "modalities": _summary_counts(summary, "modalities"),
            "tasks": _summary_counts(summary, "tasks"),
            "tasks_by_split": _summary_counts_by_split(summary),
        }
    )


def _update_manifest(
    *,
    path: Path,
    summary: dict[str, object],
    validation_summary: dict[str, Any],
) -> None:
    payload = _read_optional_json_object(path=path)
    replace_summary(payload=payload, summary=summary)
    updated_validation_summary = dict(validation_summary)
    replace_summary(payload=updated_validation_summary, summary=summary)
    payload["validation_summary"] = updated_validation_summary
    if "valid" in updated_validation_summary:
        payload["valid"] = bool(updated_validation_summary["valid"])
    write_snapshot_metadata(path=path, payload=payload)


def _update_validation_report(
    *,
    path: Path,
    summary: dict[str, object],
) -> dict[str, Any]:
    payload = _read_optional_json_object(path=path)
    replace_summary(payload=payload, summary=summary)
    write_snapshot_metadata(path=path, payload=payload)
    return payload


def _update_stats(*, path: Path, summary: dict[str, object]) -> None:
    payload = _read_optional_json_object(path=path)
    replace_summary(payload=payload, summary=summary)
    write_snapshot_metadata(path=path, payload=payload)


def _update_dataset_card(*, path: Path, summary: dict[str, object]) -> None:
    payload = _read_optional_json_object(path=path)
    stats = _object_payload(payload.get("stats"))
    replace_summary(payload=stats, summary=summary)
    payload["stats"] = stats
    task_cards = payload.get("task_cards")
    if isinstance(task_cards, dict):
        payload["task_cards"] = _updated_task_cards(
            existing=task_cards,
            task_counts=_summary_counts(summary, "tasks"),
            modality_counts=_summary_counts(summary, "modalities"),
        )
    write_snapshot_metadata(path=path, payload=payload)


def _read_optional_json_object(*, path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return load_bounded_json_object(path=path)


def _object_payload(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _summary_count(summary: Mapping[str, object], field: str) -> int:
    value = summary.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise UnsupportedSnapshotSchemaError(
            f"summary field {field!r} must be a non-negative integer"
        )
    return value


def _summary_counts(
    summary: Mapping[str, object],
    field: str,
) -> dict[str, int]:
    value = summary.get(field)
    if not isinstance(value, Mapping):
        raise UnsupportedSnapshotSchemaError(
            f"summary field {field!r} must be an object"
        )
    counts: dict[str, int] = {}
    for raw_name, raw_count in value.items():
        name = str(raw_name).strip()
        if not name:
            raise UnsupportedSnapshotSchemaError(
                f"summary field {field!r} contains an empty key"
            )
        if (
            isinstance(raw_count, bool)
            or not isinstance(raw_count, int)
            or raw_count < 0
        ):
            raise UnsupportedSnapshotSchemaError(
                f"summary count {field}.{name} must be a non-negative integer"
            )
        counts[name] = raw_count
    return counts


def _summary_counts_by_split(
    summary: Mapping[str, object],
) -> dict[str, dict[str, int]]:
    raw_splits = summary.get("tasks_by_split")
    if not isinstance(raw_splits, Mapping):
        raise UnsupportedSnapshotSchemaError(
            "summary field 'tasks_by_split' must be an object"
        )
    return {
        str(split_name): _summary_counts(
            {"counts": counts},
            "counts",
        )
        for split_name, counts in raw_splits.items()
    }


def _updated_task_cards(
    *,
    existing: dict[str, object],
    task_counts: Mapping[str, int],
    modality_counts: Mapping[str, int],
) -> dict[str, object]:
    updated = dict(existing)
    for task_type, count in sorted(task_counts.items()):
        raw_card = updated.get(task_type)
        card = dict(raw_card) if isinstance(raw_card, dict) else {}
        card["sample_count"] = count
        card["modalities_observed"] = dict(modality_counts)
        updated[task_type] = card
    return updated


__all__ = [
    "UnsupportedSnapshotSchemaError",
    "replace_summary",
    "update_augmented_training_metadata",
    "write_training_metadata",
]
