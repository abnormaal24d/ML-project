"""Dataset-level text-field augmentation workflow for multimodal training."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

from augmentation.augmentation_run_accumulator import AugmentationRunState
from augmentation.outcomes.augmentation_result import (
    AugmentationRejection,
    AugmentationReport,
    AugmentationResult,
)
from augmentation.outcomes.rejection_reason import AugmentationRejectionReason
from config.environment.default_values import DEFAULT_TRAIN_SPLIT_NAME
from mmcrawler_datasets.schema import MultimodalSample

if TYPE_CHECKING:
    from pathlib import Path

    from augmentation.text.text_field_augmenter import (
        TextFieldAugmenter,
    )
    from config.augmentation.augmentation_settings import AugmentationSettings
    from logger.project_logger import ProjectLogger

MediaAugmenter = Callable[
    ...,
    tuple[
        tuple[tuple[str, "MultimodalSample"], ...],
        tuple[AugmentationRejection, ...],
    ],
]


def _sample_split(*, sample: MultimodalSample) -> str:
    raw_split = (sample.metadata or {}).get("split")
    split = str(raw_split or DEFAULT_TRAIN_SPLIT_NAME).strip().lower()
    return split or DEFAULT_TRAIN_SPLIT_NAME


def _augmentation_decision(
    *,
    sample: MultimodalSample,
) -> tuple[bool, AugmentationRejectionReason | None]:
    split = _sample_split(sample=sample)
    if split != DEFAULT_TRAIN_SPLIT_NAME:
        return False, AugmentationRejectionReason.NON_TRAIN_SPLIT
    return True, None


class TrainingDatasetAugmenter:
    """Apply deterministic train-split text-field augmentation."""

    def __init__(
        self,
        *,
        settings: AugmentationSettings,
        sample_augmenter: TextFieldAugmenter,
        logger: ProjectLogger,
        media_augmenters: tuple[MediaAugmenter, ...] = (),
    ) -> None:
        self._settings = settings
        self._sample_augmenter = sample_augmenter
        self._logger = logger
        self._media_augmenters = media_augmenters

    def augment(
        self,
        *,
        dataset: tuple[MultimodalSample, ...],
        dataset_root: str | Path | None = None,
    ) -> AugmentationResult:
        if not self._settings.enabled:
            report = AugmentationReport(
                enabled=False,
                original_samples=len(dataset),
                augmented_samples=len(dataset),
                variants_added=0,
            )
            return AugmentationResult(dataset=dataset, report=report)

        augmented_samples: list[MultimodalSample] = list(dataset)
        state = AugmentationRunState(samples=dataset)

        for sample in dataset:
            sample_split = _sample_split(sample=sample)
            allowed, rejected_reason = _augmentation_decision(sample=sample)

            if not allowed:
                state.reject(
                    reason=(
                        rejected_reason
                        or AugmentationRejectionReason.RULES_SKIP
                    ),
                    sample=sample,
                )
                continue

            for (
                duplicate_reason,
                variants,
                rejections,
            ) in self.iter_sample_augmentations(
                sample=sample,
                dataset_root=dataset_root,
            ):
                for rejection in rejections:
                    state.reject_row(
                        rejection=rejection,
                        fallback_sample=sample,
                    )
                for augmentation_name, variant in variants:
                    if not state.claim(sample=variant):
                        state.reject(
                            reason=duplicate_reason,
                            sample=sample,
                            variant_name=augmentation_name,
                        )
                        continue

                    augmented_samples.append(variant)
                    state.add_variant(
                        augmentation_name=augmentation_name,
                        variant=variant,
                        fallback_split=sample_split,
                    )

        augmented_dataset = tuple(augmented_samples)
        report = state.build(
            dataset=augmented_dataset,
            dataset_root=dataset_root,
        )

        self._logger.info(
            "multimodal_augmentation_completed",
            augmentation_scope="train",
            train_original_samples=report.original_samples,
            train_augmented_samples=report.augmented_samples,
            train_variants_added=report.variants_added,
        )
        return AugmentationResult(dataset=augmented_dataset, report=report)

    def iter_sample_augmentations(
        self,
        *,
        sample: MultimodalSample,
        dataset_root: str | Path | None,
    ) -> Iterator[
        tuple[
            AugmentationRejectionReason,
            tuple[tuple[str, MultimodalSample], ...],
            tuple[AugmentationRejection, ...],
        ]
    ]:
        """Stream per-sample augmentation steps in deterministic order.

        Emits the text-field augmentation first, then each media augmenter in
        declaration order. Expected per-attempt failures are reported as
        rejections by the individual augmenters; fatal errors propagate.
        """

        text_result = self._sample_augmenter.augment_with_rejections(
            sample=sample,
        )
        yield (
            AugmentationRejectionReason.DUPLICATE_TEXT_DATASET,
            text_result.variants,
            text_result.rejections,
        )
        for media_augmenter in self._media_augmenters:
            variants, rejections = media_augmenter(
                sample=sample,
                dataset_root=dataset_root,
            )
            yield (
                AugmentationRejectionReason.DUPLICATE_MEDIA_VARIANT_DATASET,
                variants,
                rejections,
            )
