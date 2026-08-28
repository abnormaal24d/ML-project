"""Rebalance sample groups while preserving split isolation."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from config.environment.default_values import (
    DEFAULT_DATASET_SPLIT_NAMES,
    DEFAULT_TEST_SPLIT_NAME,
    DEFAULT_TRAIN_SPLIT_NAME,
    DEFAULT_VAL_SPLIT_NAME,
)

if TYPE_CHECKING:
    from mmcrawler_datasets.training_samples.models import TrainingSample


def _rebalance_modality_splits(
    *,
    split_samples: dict[str, list[TrainingSample]],
) -> dict[str, list[TrainingSample]]:
    rebalanced = {
        split: list(samples) for split, samples in split_samples.items()
    }

    by_modality: dict[str, list[TrainingSample]] = {}
    for split in DEFAULT_DATASET_SPLIT_NAMES:
        for sample in rebalanced.get(split, []):
            by_modality.setdefault(sample.modality, []).append(sample)

    for modality, samples in by_modality.items():
        if len(samples) < 2:
            continue

        present_splits = {sample.split for sample in samples}

        if DEFAULT_VAL_SPLIT_NAME not in present_splits:
            _move_group_for_modality(
                split_samples=rebalanced,
                modality=modality,
                target_split=DEFAULT_VAL_SPLIT_NAME,
            )
            present_splits = {
                sample.split
                for split in DEFAULT_DATASET_SPLIT_NAMES
                for sample in rebalanced.get(split, [])
                if sample.modality == modality
            }

        if len(samples) >= 3 and DEFAULT_TEST_SPLIT_NAME not in present_splits:
            _move_group_for_modality(
                split_samples=rebalanced,
                modality=modality,
                target_split=DEFAULT_TEST_SPLIT_NAME,
            )

    _ensure_group_safe(split_samples=rebalanced)
    return rebalanced


def _rebalance_task_splits(
    *,
    split_samples: dict[str, list[TrainingSample]],
) -> dict[str, list[TrainingSample]]:
    rebalanced = {
        split: list(samples) for split, samples in split_samples.items()
    }

    by_task: dict[str, list[TrainingSample]] = {}
    for split in DEFAULT_DATASET_SPLIT_NAMES:
        for sample in rebalanced.get(split, []):
            by_task.setdefault(sample.task_target.task_type, []).append(sample)

    for task_type, samples in by_task.items():
        if len(samples) < 2:
            continue

        present_splits = {sample.split for sample in samples}
        if DEFAULT_VAL_SPLIT_NAME not in present_splits:
            _move_group_for_task(
                split_samples=rebalanced,
                task_type=task_type,
                target_split=DEFAULT_VAL_SPLIT_NAME,
            )
            present_splits = {
                sample.split
                for split in DEFAULT_DATASET_SPLIT_NAMES
                for sample in rebalanced.get(split, [])
                if sample.task_target.task_type == task_type
            }
        if len(samples) >= 3 and DEFAULT_TEST_SPLIT_NAME not in present_splits:
            _move_group_for_task(
                split_samples=rebalanced,
                task_type=task_type,
                target_split=DEFAULT_TEST_SPLIT_NAME,
            )

    _ensure_group_safe(split_samples=rebalanced)
    return rebalanced


def _rebalance_minimum_split_sizes(
    *,
    split_samples: dict[str, list[TrainingSample]],
    min_train_samples: int,
    min_val_samples: int,
    min_test_samples: int,
) -> dict[str, list[TrainingSample]]:
    rebalanced = {
        split: list(samples) for split, samples in split_samples.items()
    }

    _fill_split_minimum(
        split_samples=rebalanced,
        target_split=DEFAULT_TRAIN_SPLIT_NAME,
        min_samples=min_train_samples,
        source_splits=(DEFAULT_TEST_SPLIT_NAME, DEFAULT_VAL_SPLIT_NAME),
    )
    _fill_split_minimum(
        split_samples=rebalanced,
        target_split=DEFAULT_VAL_SPLIT_NAME,
        min_samples=min_val_samples,
        source_splits=(DEFAULT_TRAIN_SPLIT_NAME,),
    )
    _fill_split_minimum(
        split_samples=rebalanced,
        target_split=DEFAULT_TEST_SPLIT_NAME,
        min_samples=min_test_samples,
        source_splits=(DEFAULT_TRAIN_SPLIT_NAME,),
    )

    _ensure_group_safe(split_samples=rebalanced)
    return rebalanced


def _fill_split_minimum(
    *,
    split_samples: dict[str, list[TrainingSample]],
    target_split: str,
    min_samples: int,
    source_splits: tuple[str, ...],
) -> None:
    while len(split_samples.get(target_split, [])) < min_samples:
        moved = _move_one_safe_group(
            split_samples=split_samples,
            target_split=target_split,
            source_splits=source_splits,
        )
        if not moved:
            return


def _move_one_safe_group(
    *,
    split_samples: dict[str, list[TrainingSample]],
    target_split: str,
    source_splits: tuple[str, ...],
) -> bool:
    for source_split in _viable_source_splits_by_size(
        split_samples=split_samples,
        target_split=target_split,
        source_splits=source_splits,
    ):
        if _move_one_safe_group_from_split(
            split_samples=split_samples,
            source_split=source_split,
            target_split=target_split,
        ):
            return True

    return False


def _move_one_safe_group_from_split(
    *,
    split_samples: dict[str, list[TrainingSample]],
    source_split: str,
    target_split: str,
) -> bool:
    source_samples = split_samples.get(source_split, [])
    grouped_indices: dict[str, list[int]] = {}
    grouped_samples: dict[str, list[TrainingSample]] = {}

    for index, sample in enumerate(source_samples):
        group_key = _sample_group_key(sample=sample)
        grouped_indices.setdefault(group_key, []).append(index)
        grouped_samples.setdefault(group_key, []).append(sample)

    if len(grouped_samples) <= 1:
        return False

    target_group_keys = {
        _sample_group_key(sample=sample)
        for sample in split_samples.get(target_split, [])
    }

    ordered_groups = sorted(
        grouped_samples.items(),
        key=lambda item: len(item[1]),
        reverse=(target_split == DEFAULT_TRAIN_SPLIT_NAME),
    )
    for group_key, samples in ordered_groups:
        if group_key in target_group_keys:
            continue

        moved_samples = [
            replace(
                sample,
                split=target_split,
                sample_id=rebalanced_sample_id(
                    sample=sample,
                    target_split=target_split,
                ),
            )
            for sample in samples
        ]

        for index in sorted(grouped_indices[group_key], reverse=True):
            del source_samples[index]

        split_samples.setdefault(target_split, []).extend(moved_samples)
        return True

    return False


def _viable_source_splits_by_size(
    *,
    split_samples: dict[str, list[TrainingSample]],
    target_split: str,
    source_splits: tuple[str, ...],
) -> tuple[str, ...]:
    candidates = [
        split
        for split in source_splits
        if split != target_split and split_samples.get(split)
    ]
    candidates.sort(
        key=lambda split: len(split_samples.get(split, ())),
        reverse=True,
    )
    return tuple(candidates)


def _move_group_for_modality(
    *,
    split_samples: dict[str, list[TrainingSample]],
    modality: str,
    target_split: str,
) -> None:
    source_samples = split_samples.get(DEFAULT_TRAIN_SPLIT_NAME, [])
    grouped_indices: dict[str, list[int]] = {}
    grouped_samples: dict[str, list[TrainingSample]] = {}

    for index, sample in enumerate(source_samples):
        group_key = _sample_group_key(sample=sample)
        grouped_indices.setdefault(group_key, []).append(index)
        grouped_samples.setdefault(group_key, []).append(sample)

    for group_key, samples in grouped_samples.items():
        if not any(sample.modality == modality for sample in samples):
            continue
        if not _group_safe_to_move(
            split_samples=split_samples,
            group_key=group_key,
            target_split=target_split,
        ):
            continue

        moved_samples = [
            replace(
                sample,
                split=target_split,
                sample_id=rebalanced_sample_id(
                    sample=sample,
                    target_split=target_split,
                ),
            )
            for sample in samples
        ]

        for index in sorted(grouped_indices[group_key], reverse=True):
            del source_samples[index]

        split_samples.setdefault(target_split, []).extend(moved_samples)
        return


def _move_group_for_task(
    *,
    split_samples: dict[str, list[TrainingSample]],
    task_type: str,
    target_split: str,
) -> None:
    source_samples = split_samples.get(DEFAULT_TRAIN_SPLIT_NAME, [])
    grouped_indices: dict[str, list[int]] = {}
    grouped_samples: dict[str, list[TrainingSample]] = {}

    for index, sample in enumerate(source_samples):
        group_key = _sample_group_key(sample=sample)
        grouped_indices.setdefault(group_key, []).append(index)
        grouped_samples.setdefault(group_key, []).append(sample)

    for group_key, samples in grouped_samples.items():
        if not any(
            sample.task_target.task_type == task_type for sample in samples
        ):
            continue
        if not _group_safe_to_move(
            split_samples=split_samples,
            group_key=group_key,
            target_split=target_split,
        ):
            continue

        moved_samples = [
            replace(
                sample,
                split=target_split,
                sample_id=rebalanced_sample_id(
                    sample=sample,
                    target_split=target_split,
                ),
            )
            for sample in samples
        ]

        for index in sorted(grouped_indices[group_key], reverse=True):
            del source_samples[index]

        split_samples.setdefault(target_split, []).extend(moved_samples)
        return


def _group_safe_to_move(
    *,
    split_samples: dict[str, list[TrainingSample]],
    group_key: str,
    target_split: str,
) -> bool:
    if any(
        _sample_group_key(sample=sample) == group_key
        for sample in split_samples.get(target_split, [])
    ):
        return False

    source_group_count = sum(
        1
        for sample in split_samples.get(DEFAULT_TRAIN_SPLIT_NAME, [])
        if _sample_group_key(sample=sample) == group_key
    )
    if source_group_count == 0:
        return False

    train_group_total = len(
        {
            _sample_group_key(sample=sample)
            for sample in split_samples.get(DEFAULT_TRAIN_SPLIT_NAME, [])
        }
    )
    return train_group_total > 1


def _sample_group_key(*, sample: TrainingSample) -> str:
    """Return the split group key used while rebalancing samples.

    The document id must be preferred over duplicate-cluster keys. Otherwise a
    rebalancing pass can move a text chunk, document sample, image, audio, or
    video sample from the same source document into a different split.
    """

    for name, candidate in (
        ("source_document", sample.source_document_id),
        ("content_family", sample.content_family_id),
        ("normalized_url", sample.normalized_url),
        ("document", sample.document_id),
        ("near_duplicate_cluster", sample.near_duplicate_cluster_id),
        ("exact_duplicate", sample.exact_duplicate_key),
        ("source_url", sample.source_url),
        ("object", sample.object_id),
        ("chunk", sample.chunk_id),
        ("sample", sample.sample_id),
    ):
        if candidate:
            return f"{name}:{candidate}"

    return f"sample:{sample.sample_id}"


def _ensure_group_safe(
    *,
    split_samples: dict[str, list[TrainingSample]],
) -> None:
    group_to_split: dict[str, str] = {}
    for split_name in DEFAULT_DATASET_SPLIT_NAMES:
        for sample in split_samples.get(split_name, ()):
            group_key = _sample_group_key(sample=sample)
            previous_split = group_to_split.get(group_key)
            if previous_split is None:
                group_to_split[group_key] = split_name
                continue
            if previous_split != split_name:
                raise ValueError(
                    "split rebalancing would leak source group "
                    f"{group_key}:{previous_split}->{split_name}"
                )


def rebalanced_sample_id(*, sample: Any, target_split: str) -> str:
    """Build the canonical ID for a sample moved to a different split."""
    return f"{sample.sample_id or 'sample'}:{target_split}"
