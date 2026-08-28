"""Mutable run state for one TrainingDatasetAugmenter.augment() call."""

from __future__ import annotations

from pathlib import Path

from augmentation.augmentation_run_quality import (
    assess_augmentation_run_quality,
)
from augmentation.outcomes.augmentation_result import (
    AugmentationRejection,
    AugmentationReport,
)
from augmentation.outcomes.rejection_reason import AugmentationRejectionReason
from augmentation.text.text_identity import text_identity
from config.environment.default_values import DEFAULT_TRAIN_SPLIT_NAME
from mmcrawler_datasets.schema import MultimodalSample


class AugmentationRunState:
    """Track identities, accepted variants and rejections for one run."""

    def __init__(
        self,
        *,
        samples: tuple[MultimodalSample, ...],
    ) -> None:
        self._original_samples = len(samples)
        self._seen_keys = {
            identity
            for sample in samples
            if (identity := self._identity(sample=sample))
        }
        self._variants_by_name: dict[str, int] = {}
        self._variants_by_operation: dict[str, int] = {}
        self._variants_by_modality: dict[str, int] = {}
        self._variants_by_task_type: dict[str, int] = {}
        self._variants_by_split: dict[str, int] = {}
        self._rejected_by_reason: dict[str, int] = {}
        self._rejections_by_modality: dict[str, int] = {}
        self._rejected_augmentations: list[AugmentationRejection] = []

    def claim(self, *, sample: MultimodalSample) -> bool:
        identity = self._identity(sample=sample)
        if identity is None or identity in self._seen_keys:
            return False
        self._seen_keys.add(identity)
        return True

    def add_variant(
        self,
        *,
        augmentation_name: str,
        variant: MultimodalSample,
        fallback_split: str,
    ) -> None:
        metadata = variant.metadata or {}
        modality = str(metadata.get("modality") or variant.modality)
        task_type = str(metadata.get("task_type") or "unknown")
        split = str(metadata.get("split") or fallback_split)
        _increment_counter(
            counter=self._variants_by_name, key=augmentation_name
        )
        _increment_counter(
            counter=self._variants_by_operation,
            key=str(metadata.get("augmentation_name") or augmentation_name),
        )
        _increment_counter(counter=self._variants_by_modality, key=modality)
        _increment_counter(counter=self._variants_by_task_type, key=task_type)
        _increment_counter(counter=self._variants_by_split, key=split)

    def reject(
        self,
        *,
        reason: AugmentationRejectionReason,
        sample: MultimodalSample | None = None,
        variant_name: str | None = None,
        message: str | None = None,
    ) -> None:
        self.reject_row(
            rejection=_rejection_row(
                reason=reason,
                sample=sample,
                variant_name=variant_name,
                message=message,
            ),
            fallback_sample=sample,
        )

    def reject_row(
        self,
        *,
        rejection: AugmentationRejection,
        fallback_sample: MultimodalSample | None = None,
    ) -> None:
        completed_rejection = _complete_rejection(
            rejection=rejection,
            sample=fallback_sample,
        )
        _increment_counter(
            counter=self._rejected_by_reason,
            key=completed_rejection.reason,
        )
        _increment_counter(
            counter=self._rejections_by_modality,
            key=completed_rejection.modality or "unknown",
        )
        self._rejected_augmentations.append(completed_rejection)

    def build(
        self,
        *,
        dataset: tuple[MultimodalSample, ...],
        dataset_root: str | Path | None = None,
    ) -> AugmentationReport:
        quality = assess_augmentation_run_quality(
            dataset=dataset,
            dataset_root=dataset_root,
        )
        return AugmentationReport(
            enabled=True,
            original_samples=self._original_samples,
            augmented_samples=len(dataset),
            variants_added=len(dataset) - self._original_samples,
            variants_by_name=self._variants_by_name,
            variants_by_operation=self._variants_by_operation,
            variants_by_modality=self._variants_by_modality,
            variants_by_task_type=self._variants_by_task_type,
            variants_by_split=self._variants_by_split,
            rejected_by_reason=self._rejected_by_reason,
            rejections_by_modality=self._rejections_by_modality,
            media_outputs=quality.media_outputs,
            quality_checks_passed=quality.passed,
            quality_checks=quality.checks,
            quality_check_failures=quality.failures,
            rejected_augmentations=tuple(self._rejected_augmentations),
        )

    @staticmethod
    def _identity(
        *,
        sample: MultimodalSample,
    ) -> tuple[object, ...] | None:
        sample_text_identity = text_identity(sample.text)
        if not sample_text_identity:
            return None
        metadata = sample.metadata or {}
        modality = str(metadata.get("modality") or sample.modality)
        task_type = str(metadata.get("task_type") or "unknown")
        object_paths = _object_paths(sample=sample)
        return (
            sample_text_identity,
            modality,
            task_type,
            sample.label,
            sample.source_url,
            metadata.get("augmentation_name"),
            object_paths,
        )


def _object_paths(*, sample: MultimodalSample) -> tuple[str | None, ...]:
    return (
        str(sample.image.path) if sample.image and sample.image.path else None,
        str(sample.audio.path) if sample.audio and sample.audio.path else None,
        str(sample.video.path) if sample.video and sample.video.path else None,
    )


def _increment_counter(*, counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1


def _rejection_row(
    *,
    reason: AugmentationRejectionReason,
    sample: MultimodalSample | None,
    variant_name: str | None,
    message: str | None,
) -> AugmentationRejection:
    if sample is None:
        return AugmentationRejection(
            reason=reason,
            variant_name=variant_name,
            message=message,
        )
    metadata = sample.metadata or {}
    return AugmentationRejection(
        reason=reason,
        sample_id=sample.sample_id,
        variant_name=variant_name,
        split=str(metadata.get("split") or DEFAULT_TRAIN_SPLIT_NAME),
        modality=str(metadata.get("modality") or sample.modality),
        task_type=str(metadata.get("task_type") or "unknown"),
        label=sample.label,
        source_url=sample.source_url,
        text_length=len(sample.text or ""),
        message=message,
    )


def _complete_rejection(
    *,
    rejection: AugmentationRejection,
    sample: MultimodalSample | None,
) -> AugmentationRejection:
    if sample is None:
        return rejection
    metadata = sample.metadata or {}
    return AugmentationRejection(
        reason=rejection.reason,
        sample_id=rejection.sample_id or sample.sample_id,
        variant_name=rejection.variant_name,
        split=(
            rejection.split
            or str(metadata.get("split") or DEFAULT_TRAIN_SPLIT_NAME)
        ),
        modality=rejection.modality
        or str(metadata.get("modality") or sample.modality),
        task_type=rejection.task_type
        or str(metadata.get("task_type") or "unknown"),
        label=rejection.label if rejection.label is not None else sample.label,
        source_url=rejection.source_url or sample.source_url,
        text_length=(
            rejection.text_length
            if rejection.text_length is not None
            else len(sample.text or "")
        ),
        message=rejection.message,
    )
