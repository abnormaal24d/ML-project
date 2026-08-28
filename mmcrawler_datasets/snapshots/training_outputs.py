"""Materialize configured training-split, rejection and derived outputs."""

from __future__ import annotations

import shutil
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from config.settings.datasets import (
    DatasetPathSettings,
    TrainingDatasetWriterSettings,
)
from mmcrawler_datasets.assembly.build import SampleBuildResult
from mmcrawler_datasets.record_components.coercion import require_mapping
from mmcrawler_datasets.snapshots.errors import SnapshotBuildError
from mmcrawler_datasets.snapshots.output_writer import (
    SnapshotOutputSettings,
    write_shard_index,
    write_snapshot_rows,
    write_webdataset_shards,
)
from mmcrawler_datasets.snapshots.rejected_sample_reports import (
    ensure_rejected_rows_report,
    write_pair_rejections,
)

_SPLIT_NAMES = ("train", "val", "test")
_SAFE_GROUP_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
)
_ALLOWED_MODALITIES = frozenset(
    {
        "text",
        "image",
        "audio",
        "video",
        "document",
    }
)


def output_settings_from(
    settings: TrainingDatasetWriterSettings,
) -> SnapshotOutputSettings:
    """Project resolved writer settings into snapshot output dependencies."""

    return SnapshotOutputSettings(
        write_jsonl=settings.write_jsonl,
        write_shards=settings.write_shards,
        shard_format=settings.shard_format,
        max_samples_per_shard=settings.shard_max_samples,
        max_bytes_per_shard=settings.shard_max_bytes,
        shards_directory=settings.training_shards_directory,
        shard_index_filename=settings.shard_index_filename,
    )


def write_training_outputs(
    *,
    training_directory: Path,
    samples: SampleBuildResult,
    dataset_paths: DatasetPathSettings,
    output_settings: SnapshotOutputSettings,
) -> None:
    """Persist every configured split representation and rejection report."""

    split_directory = (
        training_directory / dataset_paths.training_splits_directory
    )
    rows_by_split = {
        "train": tuple(sample.to_dict() for sample in samples.train_samples),
        "val": tuple(sample.to_dict() for sample in samples.val_samples),
        "test": tuple(sample.to_dict() for sample in samples.test_samples),
    }

    if output_settings.write_jsonl:
        write_snapshot_rows(
            split_directory / dataset_paths.training_train_filename,
            rows_by_split["train"],
        )
        write_snapshot_rows(
            split_directory / dataset_paths.training_val_filename,
            rows_by_split["val"],
        )
        write_snapshot_rows(
            split_directory / dataset_paths.training_test_filename,
            rows_by_split["test"],
        )

    if output_settings.write_shards:
        entries = write_webdataset_shards(
            training_directory=training_directory,
            rows_by_split=rows_by_split,
            output_settings=output_settings,
        )
        shard_index_path = (
            training_directory / output_settings.shard_index_filename
        )
        write_shard_index(path=shard_index_path, entries_by_split=entries)
        if not shard_index_path.is_file():
            raise SnapshotBuildError(
                "shard output was requested but no shard index was produced"
            )

    write_pair_rejections(
        training_directory=training_directory,
        rows=samples.pair_rejections,
    )
    ensure_rejected_rows_report(training_directory=training_directory)


def rewrite_derived_views(
    *,
    paths: DatasetPathSettings,
    output_directory: Path,
    train_rows: tuple[dict[str, object], ...],
    val_rows: tuple[dict[str, object], ...],
    test_rows: tuple[dict[str, object], ...],
) -> None:
    split_rows = {
        "train": train_rows,
        "val": val_rows,
        "test": test_rows,
    }
    filenames = {
        "train": paths.training_train_filename,
        "val": paths.training_val_filename,
        "test": paths.training_test_filename,
    }
    _rewrite_grouped_views(
        root=output_directory / paths.training_modalities_directory,
        split_rows=split_rows,
        filenames=filenames,
        field_name="modality",
    )
    _rewrite_grouped_views(
        root=output_directory / paths.training_tasks_directory,
        split_rows=split_rows,
        filenames=filenames,
        field_name="task_type",
    )


def _rewrite_grouped_views(
    *,
    root: Path,
    split_rows: Mapping[str, Iterable[Mapping[str, Any]]],
    filenames: dict[str, str],
    field_name: str,
) -> None:
    if root.exists():
        shutil.rmtree(root)

    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: {split: [] for split in _SPLIT_NAMES}
    )
    for split_name, rows in split_rows.items():
        for row in rows:
            group_name = _resolve_group_name(row=row, field_name=field_name)
            grouped[group_name][split_name].append(row)

    for group_name, rows_by_split in sorted(grouped.items()):
        group_root = _contained_group_root(root=root, group_name=group_name)
        for split_name in _SPLIT_NAMES:
            write_snapshot_rows(
                path=group_root / filenames[split_name],
                rows=rows_by_split[split_name],
            )


def _resolve_group_name(
    *,
    row: Mapping[str, Any],
    field_name: str,
) -> str:
    if field_name == "task_type":
        from schemas.multimodal_tasks import canonical_task_name

        raw_task = require_mapping(dict(row), "task_target").get("task_type")
        if raw_task is not None and not isinstance(raw_task, str):
            raise ValueError("task_type must be a string")
        # Canonicalize first; never derive filesystem paths from raw task_type.
        return _require_safe_group_name(
            canonical_task_name(raw_task),
            field="task_type",
        )

    if field_name == "modality":
        modality = _require_safe_group_name(
            row.get("modality"),
            field="modality",
        )
        if modality not in _ALLOWED_MODALITIES:
            raise ValueError(
                f"modality is not in the allowed set: {modality!r}"
            )
        return modality

    raise ValueError(f"unsupported derived-view field: {field_name!r}")


def _require_safe_group_name(
    value: object,
    *,
    field: str,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")

    name = value.strip()
    if not name or name in {".", ".."}:
        raise ValueError(f"{field} is not a safe directory name")

    if any(character not in _SAFE_GROUP_CHARACTERS for character in name):
        raise ValueError(f"{field} contains unsupported characters")

    return name


def _contained_group_root(*, root: Path, group_name: str) -> Path:
    """Resolve a group directory and reject any path that escapes ``root``."""

    root_resolved = root.resolve()
    group_root = (root / group_name).resolve()
    if not group_root.is_relative_to(root_resolved):
        raise ValueError("derived-view group escapes its root")
    return group_root


__all__ = [
    "output_settings_from",
    "rewrite_derived_views",
    "write_training_outputs",
    "_contained_group_root",
    "_require_safe_group_name",
    "_resolve_group_name",
    "_rewrite_grouped_views",
]
