"""Stateless assembly of task-specific training samples."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from config.environment.default_values import (
    DEFAULT_DATASET_SPLIT_NAMES,
    DEFAULT_TEST_SPLIT_NAME,
    DEFAULT_TRAIN_SPLIT_NAME,
    DEFAULT_VAL_SPLIT_NAME,
)
from mmcrawler_datasets.schema import SplitAssigner
from mmcrawler_datasets.selection.pipeline import select_samples
from mmcrawler_datasets.selection.quality import finalize_sample
from mmcrawler_datasets.snapshots.curated import CuratedSnapshot
from mmcrawler_datasets.splitting.group_keys import collect_group_keys
from mmcrawler_datasets.splitting.rebalancing import (
    _rebalance_minimum_split_sizes,
    _rebalance_modality_splits,
    _rebalance_task_splits,
)
from mmcrawler_datasets.training_samples.models import TrainingSample
from mmcrawler_datasets.training_samples.targets import TrainingTaskTarget
from multimodal.tasks.registry import SAMPLE_BUILDER_TASKS, require_task

from .audio import build_audio_samples
from .document import build_doc_samples
from .image import build_image_samples
from .text import build_text_samples
from .video import build_video_samples

if TYPE_CHECKING:
    from config.settings.datasets import (
        DatasetValidatorSettings,
        SplitAssignerSettings,
        TrainingSnapshotAssemblerSettings,
    )
    from mmcrawler_datasets.materialization.audio_generation import (
        AudioGenerationTargetMaterializer,
    )
    from mmcrawler_datasets.materialization.video_generation import (
        VideoGenerationTargetMaterializer,
    )
    from preprocessing.privacy.text_privacy import PiiDetector


@dataclass(frozen=True, slots=True)
class SampleBuildResult:
    """Built split samples and rejected pair diagnostics."""

    train_samples: tuple[TrainingSample, ...]
    val_samples: tuple[TrainingSample, ...]
    test_samples: tuple[TrainingSample, ...]
    pair_rejections: tuple[dict[str, object], ...] = ()

    @property
    def all_samples(self) -> tuple[TrainingSample, ...]:
        return self.train_samples + self.val_samples + self.test_samples

    @property
    def pair_rejected_by_reason(self) -> dict[str, int]:
        return dict(
            sorted(
                Counter(
                    str(row.get("reason") or "unknown")
                    for row in self.pair_rejections
                ).items()
            )
        )

    @property
    def split_counts(self) -> dict[str, int]:
        return {
            DEFAULT_TRAIN_SPLIT_NAME: len(self.train_samples),
            DEFAULT_VAL_SPLIT_NAME: len(self.val_samples),
            DEFAULT_TEST_SPLIT_NAME: len(self.test_samples),
        }


def build_samples(
    snapshot: CuratedSnapshot,
    *,
    settings: TrainingSnapshotAssemblerSettings,
    split_settings: SplitAssignerSettings,
    validator_settings: DatasetValidatorSettings,
    project_root: Path,
    pii_detector: PiiDetector,
    snapshot_directory: Path,
    materialization_directory: Path,
    snapshot_id: str,
    enabled_tasks: frozenset[str] | None = None,
    audio_materializer_factory: (
        Callable[[Path], AudioGenerationTargetMaterializer] | None
    ) = None,
    video_materializer_factory: (
        Callable[[Path], VideoGenerationTargetMaterializer] | None
    ) = None,
    require_transcript_for_audio_text_pair: bool = False,
) -> SampleBuildResult:
    """Build, balance, enrich and select every enabled sample task."""

    split_assigner = SplitAssigner(
        train_ratio=split_settings.train_ratio,
        val_ratio=split_settings.val_ratio,
        test_ratio=split_settings.test_ratio,
    )
    text_cache: dict[str, str | None] = {}
    split_by_group = split_assigner.assign_many(
        keys=collect_group_keys(
            snapshot=snapshot,
            require_allow_training=validator_settings.require_allow_training,
            snapshot_directory=snapshot_directory,
            document_text_cache=text_cache,
        )
    )
    documents = snapshot.documents_by_id
    rejections: list[dict[str, object]] = []
    candidates = (
        *build_text_samples(
            snapshot.chunks,
            documents,
            split_by_group,
            split_assigner=split_assigner,
            require_allow_training=validator_settings.require_allow_training,
            snapshot_id=snapshot_id,
            schema_version=settings.training_schema_version,
            enabled_tasks=enabled_tasks,
        ),
        *build_doc_samples(
            snapshot.documents,
            split_by_group,
            split_assigner=split_assigner,
            require_allow_training=validator_settings.require_allow_training,
            snapshot_id=snapshot_id,
            snapshot_directory=snapshot_directory,
            text_cache=text_cache,
            schema_version=settings.training_schema_version,
            enabled_tasks=enabled_tasks,
        ),
        *build_image_samples(
            snapshot.images,
            documents,
            split_by_group,
            split_assigner=split_assigner,
            require_allow_training=validator_settings.require_allow_training,
            snapshot_id=snapshot_id,
            snapshot_directory=snapshot_directory,
            project_root=project_root,
            rejections=rejections,
            schema_version=settings.training_schema_version,
            enabled_tasks=enabled_tasks,
        ),
        *build_audio_samples(
            snapshot.audio,
            documents,
            split_by_group,
            split_assigner=split_assigner,
            require_allow_training=validator_settings.require_allow_training,
            snapshot_id=snapshot_id,
            snapshot_directory=snapshot_directory,
            project_root=project_root,
            rejections=rejections,
            materialization_directory=materialization_directory,
            materializer_factory=audio_materializer_factory,
            schema_version=settings.training_schema_version,
            enabled_tasks=enabled_tasks,
            require_transcript_for_audio_text_pair=(
                require_transcript_for_audio_text_pair
            ),
        ),
        *build_video_samples(
            snapshot.video,
            documents,
            split_by_group,
            split_assigner=split_assigner,
            require_allow_training=validator_settings.require_allow_training,
            snapshot_id=snapshot_id,
            snapshot_directory=snapshot_directory,
            project_root=project_root,
            rejections=rejections,
            materialization_directory=materialization_directory,
            materializer_factory=video_materializer_factory,
            schema_version=settings.training_schema_version,
            enabled_tasks=enabled_tasks,
        ),
    )
    candidates = (
        *candidates,
        *_build_cross_modal_samples(
            samples=candidates,
            enabled_tasks=enabled_tasks,
        ),
    )
    grouped = _group_by_split(candidates)
    grouped = _rebalance(grouped, validator_settings)
    finalized = tuple(
        finalize_sample(
            sample=sample,
            settings=settings,
            snapshot_id=snapshot_id,
            validator_settings=validator_settings,
            project_root=project_root,
            pii_detector=pii_detector,
        )
        for split_samples in grouped.values()
        for sample in split_samples
    )
    selected = select_samples(
        finalized,
        settings,
        validator_settings,
    )
    balanced = _rebalance(_group_by_split(selected), validator_settings)
    return SampleBuildResult(
        train_samples=tuple(balanced.get(DEFAULT_TRAIN_SPLIT_NAME, ())),
        val_samples=tuple(balanced.get(DEFAULT_VAL_SPLIT_NAME, ())),
        test_samples=tuple(balanced.get(DEFAULT_TEST_SPLIT_NAME, ())),
        pair_rejections=tuple(rejections),
    )


_PAIR_TASKS = frozenset(
    {
        "document_text_pair",
        "pdf_text_pair",
        "image_text_pair",
        "audio_text_pair",
        "video_text_pair",
    }
)


def _build_cross_modal_samples(
    *,
    samples: tuple[TrainingSample, ...],
    enabled_tasks: frozenset[str] | None,
) -> tuple[TrainingSample, ...]:
    """Derive retrieval and consistency rows from genuine aligned pairs."""

    if enabled_tasks is None or not (
        {"multimodal_retrieval", "cross_modal_consistency"} & enabled_tasks
    ):
        return ()

    pair_samples = tuple(
        sample
        for sample in samples
        if sample.task_target.task_type in _PAIR_TASKS
        and sample.text.strip()
        and _positive_identity(sample) is not None
    )
    derived: list[TrainingSample] = []
    for sample in pair_samples:
        positive_id = _positive_identity(sample)
        if positive_id is None:
            continue
        negative_candidates = sorted(
            (
                candidate
                for candidate in pair_samples
                if candidate.split == sample.split
                and candidate.content_family_id != sample.content_family_id
                and (candidate_id := _positive_identity(candidate)) is not None
                and candidate_id != positive_id
            ),
            key=lambda candidate: (
                candidate.modality != sample.modality,
                -_lexical_overlap(sample.text, candidate.text),
                str(_positive_identity(candidate)),
            ),
        )
        selected_negatives: list[TrainingSample] = []
        seen_negative_ids: set[str] = set()
        for candidate in negative_candidates:
            candidate_id = _positive_identity(candidate)
            if candidate_id is None or candidate_id in seen_negative_ids:
                continue
            selected_negatives.append(candidate)
            seen_negative_ids.add(candidate_id)
            if len(selected_negatives) == 4:
                break
        negatives = tuple(
            str(_positive_identity(candidate))
            for candidate in selected_negatives
        )
        if not negatives:
            continue
        negative_verification: tuple[dict[str, object], ...] = tuple(
            {
                "negative_id": str(_positive_identity(candidate)),
                "different_content_family": True,
                "query_content_family_id": sample.content_family_id,
                "candidate_content_family_id": candidate.content_family_id,
                "same_modality": candidate.modality == sample.modality,
                "lexical_overlap": _lexical_overlap(
                    sample.text, candidate.text
                ),
                "verification_method": "content_family_exclusion_and_ranked_overlap",
            }
            for candidate in selected_negatives
        )

        if "multimodal_retrieval" in enabled_tasks:
            definition = require_task("multimodal_retrieval")
            derived.append(
                replace(
                    sample,
                    sample_id=f"{sample.sample_id}:multimodal_retrieval",
                    task_target=TrainingTaskTarget(
                        task_type=definition.name,
                        task_family=definition.family,
                        positive_id=positive_id,
                        negative_ids=negatives,
                        negative_verification=negative_verification,
                        alignment_score=sample.task_target.alignment_score,
                    ),
                    builder_source="aligned_pair_retrieval",
                    content_hash=None,
                )
            )

        if "cross_modal_consistency" in enabled_tasks:
            definition = require_task("cross_modal_consistency")
            derived.append(
                replace(
                    sample,
                    sample_id=f"{sample.sample_id}:cross_modal_consistency",
                    task_target=TrainingTaskTarget(
                        task_type=definition.name,
                        task_family=definition.family,
                        positive_id=positive_id,
                        negative_ids=negatives,
                        negative_verification=negative_verification,
                        alignment_score=sample.task_target.alignment_score,
                    ),
                    builder_source="aligned_pair_consistency",
                    content_hash=None,
                )
            )
    return tuple(derived)


def _lexical_overlap(left: str, right: str) -> float:
    """Return a deterministic token-overlap score for hard-negative ranking."""

    left_tokens = {token for token in left.lower().split() if token}
    right_tokens = {token for token in right.lower().split() if token}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _positive_identity(sample: TrainingSample) -> str | None:
    for value in (sample.object_id, sample.document_id, sample.sample_id):
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _group_by_split(
    samples: tuple[TrainingSample, ...],
) -> dict[str, list[TrainingSample]]:
    grouped: dict[str, list[TrainingSample]] = {
        name: [] for name in DEFAULT_DATASET_SPLIT_NAMES
    }
    for sample in samples:
        grouped.setdefault(sample.split, []).append(sample)
    return grouped


def _rebalance(
    split_samples: dict[str, list[TrainingSample]],
    settings: DatasetValidatorSettings,
) -> dict[str, list[TrainingSample]]:
    rebalanced = _rebalance_modality_splits(split_samples=split_samples)
    rebalanced = _rebalance_task_splits(split_samples=rebalanced)
    return _rebalance_minimum_split_sizes(
        split_samples=rebalanced,
        min_train_samples=settings.min_train_samples,
        min_val_samples=settings.min_val_samples,
        min_test_samples=settings.min_test_samples,
    )


__all__ = [
    "SAMPLE_BUILDER_TASKS",
    "SampleBuildResult",
    "build_samples",
]
