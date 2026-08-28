"""Batch and weighted samplers over MultimodalJsonlDataset.refs."""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from torch.utils.data import DistributedSampler, WeightedRandomSampler

from multimodal.tasks.registry import (
    get_task,
    task_family_for,
)
from schemas.multimodal_tasks import canonical_task_name

if TYPE_CHECKING:
    from config.multimodal.training_settings import TrainingSettings
    from mmcrawler_datasets.dataset import MultimodalJsonlDataset
    from mmcrawler_datasets.schema import DatasetSplit


class HardNegativeBatchSampler:
    """Colocate explicit negative_ids when available."""

    def __init__(
        self,
        *,
        dataset: MultimodalJsonlDataset,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.batch_size = max(1, batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self._negative_indices_by_anchor = self._index_negatives()
        self.has_explicit_negatives = any(
            bool(indices)
            for indices in self._negative_indices_by_anchor.values()
        )

    def __iter__(self) -> Iterator[list[int]]:
        # Dataset order must be reproducible; it is not a security boundary.
        rng = random.Random(self.seed + self.epoch)  # nosec B311
        order = list(range(len(self.dataset)))
        if self.shuffle:
            rng.shuffle(order)

        pending = order[:]
        used: set[int] = set()
        while pending:
            anchor_index = pending.pop(0)
            if anchor_index in used:
                continue
            batch = [anchor_index]
            used.add(anchor_index)

            for negative_index in self._negative_indices_by_anchor.get(
                anchor_index,
                (),
            ):
                if len(batch) >= self.batch_size:
                    break
                if negative_index in used:
                    continue
                batch.append(negative_index)
                used.add(negative_index)

            while pending and len(batch) < self.batch_size:
                candidate = pending.pop(0)
                if candidate in used:
                    continue
                batch.append(candidate)
                used.add(candidate)

            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _index_negatives(self) -> dict[int, tuple[int, ...]]:
        id_to_index: dict[str, int] = {}
        for index, ref in enumerate(self.dataset.refs):
            id_to_index[str(ref.sample_id)] = index
            if ref.positive_id:
                id_to_index[str(ref.positive_id)] = index

        negative_indices_by_anchor: dict[int, tuple[int, ...]] = {}
        for index, ref in enumerate(self.dataset.refs):
            definition = get_task(ref.task_type)
            if definition is None or not definition.supports_hard_negatives:
                negative_indices_by_anchor[index] = ()
                continue
            negative_indices = [
                id_to_index[negative_id]
                for negative_id in ref.negative_ids
                if negative_id in id_to_index
                and id_to_index[negative_id] != index
            ]
            negative_indices_by_anchor[index] = tuple(
                dict.fromkeys(negative_indices)
            )
        return negative_indices_by_anchor


class TaskModalityBatchSampler:
    """Keep task type and modality signature homogeneous within batches."""

    def __init__(
        self,
        *,
        dataset: MultimodalJsonlDataset,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        seed: int,
        curriculum_schedule: tuple[str, ...] = (),
    ) -> None:
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0
        self.curriculum_schedule = tuple(
            canonical_task_name(task)
            for task in curriculum_schedule
            if str(task).strip()
        )

    def __iter__(self) -> Iterator[list[int]]:
        # Dataset order must be reproducible; it is not a security boundary.
        rng = random.Random(self.seed + self.epoch)  # nosec B311
        groups = self._group_indices()
        for key in _ordered_curriculum_keys(
            keys=list(groups),
            curriculum_schedule=self.curriculum_schedule,
            shuffle=self.shuffle,
            rng=rng,
        ):
            indices = list(groups[key])
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch

    def __len__(self) -> int:
        total = 0
        for indices in self._group_indices().values():
            if self.drop_last:
                total += len(indices) // self.batch_size
            else:
                total += (
                    len(indices) + self.batch_size - 1
                ) // self.batch_size
        return total

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _group_indices(self) -> dict[tuple[str, tuple[str, ...]], list[int]]:
        groups: dict[tuple[str, tuple[str, ...]], list[int]] = {}
        for index, ref in enumerate(self.dataset.refs):
            key = (
                canonical_task_name(ref.task_type),
                ref.modality_signature,
            )
            groups.setdefault(key, []).append(index)
        return groups


def build_task_weighted_sampler(
    *,
    dataset: MultimodalJsonlDataset,
    task_sampling_weights: Mapping[str, float],
    task_family_sampling_weights: Mapping[str, float] | None = None,
    seed: int,
) -> WeightedRandomSampler | None:
    refs = dataset.refs
    if not refs:
        return None

    task_weights = {
        canonical_task_name(task_type): float(weight)
        for task_type, weight in task_sampling_weights.items()
    }
    family_weights = {
        str(family).strip().lower(): float(weight)
        for family, weight in (task_family_sampling_weights or {}).items()
        if str(family).strip()
    }
    family_counts = Counter(task_family_for(ref.task_type) for ref in refs)
    weights = [
        _task_weight(
            task_type=ref.task_type,
            task_weights=task_weights,
            family_weights=family_weights,
            family_counts=family_counts,
        )
        for ref in refs
    ]
    if not weights or all(weight <= 0.0 for weight in weights):
        return None
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def select_sampling(
    *,
    dataset: MultimodalJsonlDataset,
    split: DatasetSplit,
    settings: TrainingSettings,
    shuffle_training: bool,
    batch_size: int,
) -> tuple[Any, Any]:
    """Choose sampler / batch_sampler for one split."""

    from mmcrawler_datasets.schema import DatasetSplit as Split

    if split is not Split.TRAIN:
        return None, None

    if settings.use_hard_negative_sampler:
        hard = HardNegativeBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle_training,
            drop_last=bool(settings.drop_last),
            seed=int(settings.seed),
        )
        if hard.has_explicit_negatives:
            return None, hard

    if settings.task_aware_batching and not settings.dynamic_sampling:
        return None, TaskModalityBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle_training,
            drop_last=bool(settings.drop_last),
            seed=int(settings.seed),
            curriculum_schedule=settings.curriculum_schedule,
        )

    if settings.dynamic_sampling:
        return (
            build_task_weighted_sampler(
                dataset=dataset,
                task_sampling_weights=settings.task_sampling_weights,
                task_family_sampling_weights=(
                    settings.task_family_sampling_weights
                ),
                seed=int(settings.seed),
            ),
            None,
        )
    return None, None


class DistributedIndexSampler:
    """Partition an arbitrary deterministic index sampler across ranks."""

    def __init__(
        self,
        *,
        sampler: Iterable[int],
        num_replicas: int,
        rank: int,
        drop_last: bool,
        seed: int,
    ) -> None:
        self.sampler = sampler
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        self._seed_source()
        indices = list(self.sampler)
        total_size = self._total_size(len(indices))
        if self.drop_last:
            indices = indices[:total_size]
        elif indices and len(indices) < total_size:
            repeats = math.ceil((total_size - len(indices)) / len(indices))
            indices.extend((indices * repeats)[: total_size - len(indices)])
        return iter(indices[self.rank : total_size : self.num_replicas])

    def __len__(self) -> int:
        source_size = len(self.sampler)  # type: ignore[arg-type]
        if self.drop_last:
            return source_size // self.num_replicas
        return math.ceil(source_size / self.num_replicas)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        setter = getattr(self.sampler, "set_epoch", None)
        if callable(setter):
            setter(self.epoch)

    def _total_size(self, source_size: int) -> int:
        if self.drop_last:
            return source_size - (source_size % self.num_replicas)
        return math.ceil(source_size / self.num_replicas) * self.num_replicas

    def _seed_source(self) -> None:
        generator = getattr(self.sampler, "generator", None)
        if isinstance(generator, torch.Generator):
            generator.manual_seed(self.seed + self.epoch)


class DistributedBatchSampler:
    """Partition complete custom batches without duplicating full datasets."""

    def __init__(
        self,
        *,
        batch_sampler: Iterable[list[int]],
        num_replicas: int,
        rank: int,
        drop_last: bool,
    ) -> None:
        self.batch_sampler = batch_sampler
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.drop_last = bool(drop_last)

    def __iter__(self) -> Iterator[list[int]]:
        batches = list(self.batch_sampler)
        total_size = self._total_size(len(batches))
        if self.drop_last:
            batches = batches[:total_size]
        elif batches and len(batches) < total_size:
            repeats = math.ceil((total_size - len(batches)) / len(batches))
            batches.extend((batches * repeats)[: total_size - len(batches)])
        return iter(batches[self.rank : total_size : self.num_replicas])

    def __len__(self) -> int:
        source_size = len(self.batch_sampler)  # type: ignore[arg-type]
        if self.drop_last:
            return source_size // self.num_replicas
        return math.ceil(source_size / self.num_replicas)

    def set_epoch(self, epoch: int) -> None:
        setter = getattr(self.batch_sampler, "set_epoch", None)
        if callable(setter):
            setter(int(epoch))

    def _total_size(self, source_size: int) -> int:
        if self.drop_last:
            return source_size - (source_size % self.num_replicas)
        return math.ceil(source_size / self.num_replicas) * self.num_replicas


def apply_distributed_sampling(
    *,
    dataset: MultimodalJsonlDataset,
    sampler: Any,
    batch_sampler: Any,
    shuffle: bool,
    drop_last: bool,
    seed: int,
) -> tuple[Any, Any]:
    """Partition the chosen sampling rules across initialized ranks."""

    if not dist.is_available() or not dist.is_initialized():
        return sampler, batch_sampler
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if batch_sampler is not None:
        return None, DistributedBatchSampler(
            batch_sampler=batch_sampler,
            num_replicas=world_size,
            rank=rank,
            drop_last=drop_last,
        )
    if sampler is not None:
        return (
            DistributedIndexSampler(
                sampler=sampler,
                num_replicas=world_size,
                rank=rank,
                drop_last=drop_last,
                seed=seed,
            ),
            None,
        )
    return (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        ),
        None,
    )


def _ordered_curriculum_keys(
    *,
    keys: list[tuple[str, tuple[str, ...]]],
    curriculum_schedule: tuple[str, ...],
    shuffle: bool,
    rng: random.Random,
) -> list[tuple[str, tuple[str, ...]]]:
    curriculum_rank = {
        canonical_task_name(task): rank
        for rank, task in enumerate(curriculum_schedule)
    }
    ordered = list(keys)
    if shuffle:
        rng.shuffle(ordered)
    return sorted(
        ordered,
        key=lambda key: (
            curriculum_rank.get(key[0], len(curriculum_rank)),
            key[0],
            key[1],
        ),
    )


def _task_weight(
    *,
    task_type: str,
    task_weights: Mapping[str, float],
    family_weights: Mapping[str, float],
    family_counts: Counter[str | None],
) -> float:
    normalized = canonical_task_name(task_type)
    family = task_family_for(task_type)
    task_weight = float(task_weights.get(normalized, 1.0))
    family_weight = float(family_weights.get(family or "", 1.0))
    family_count = max(1, int(family_counts.get(family, 1)))
    return max(0.0, task_weight * family_weight / family_count)
