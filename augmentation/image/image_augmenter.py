"""Coordinate deterministic image augmentation operations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from augmentation.annotations.annotation_safety import (
    non_transformable_annotations,
    rejection_message,
)
from augmentation.image.content_aware_crop import (
    CropWindow,
    select_crop_windows,
)
from augmentation.image.image_artifact_writer import load_prepared_image
from augmentation.image.image_operation_executor import ImageOperationExecutor
from augmentation.image.image_operations import resolve_image_operations
from augmentation.media_variant_support import (
    resolve_dataset_root,
    resolve_source_path,
)
from augmentation.outcomes.augmentation_result import AugmentationRejection
from augmentation.outcomes.rejection_reason import AugmentationRejectionReason
from augmentation.variant_lineage import file_sha256
from logger.project_logger import ProjectLogger
from mmcrawler_datasets.schema import MultimodalSample

if TYPE_CHECKING:
    from collections.abc import Callable

    from PIL.Image import Image

    from augmentation.outcomes.media_validation_outcome import (
        MediaValidationOutcome,
    )
    from config.augmentation.image_settings import ImageAugmentationSettings


class ImageAugmenter:
    """Validate input, plan operation variants, and delegate execution."""

    def __init__(
        self,
        *,
        settings: ImageAugmentationSettings,
        operation_executor: ImageOperationExecutor,
        validate_input: Callable[..., MediaValidationOutcome],
        logger: ProjectLogger,
    ) -> None:
        self._settings = settings
        self._operations = resolve_image_operations(settings.operations)
        self._validate_input = validate_input
        self._executor = operation_executor
        self._logger = logger
        self._logger.debug("image_augmenter_initialized")

    def enabled_operations(self) -> tuple[str, ...]:
        return self._settings.operations if self._settings.enabled else ()

    def augment(
        self, *, sample: MultimodalSample, dataset_root: str | Path | None
    ) -> tuple[
        tuple[tuple[str, MultimodalSample], ...],
        tuple[AugmentationRejection, ...],
    ]:
        if not self._settings.enabled:
            return (), ()
        if dataset_root is None:
            return (), (
                _rejection(
                    sample,
                    AugmentationRejectionReason.DATASET_ROOT_MISSING,
                    "image",
                ),
            )
        if sample.image is None or sample.image.path is None:
            return (), ()
        unsafe = non_transformable_annotations(
            sample=sample, media_kind="image"
        )
        if unsafe:
            return (), (
                _rejection(
                    sample,
                    AugmentationRejectionReason.MEDIA_ANNOTATIONS_NOT_TRANSFORMABLE,
                    "image",
                    rejection_message(fields=unsafe),
                ),
            )
        root = resolve_dataset_root(dataset_root)
        if root is None:
            return (), (
                _rejection(
                    sample,
                    AugmentationRejectionReason.DATASET_ROOT_MISSING,
                    "image",
                ),
            )
        try:
            source_path = resolve_source_path(
                dataset_root=root,
                value=sample.image.path,
                error_message="image_source_path_escapes_dataset_root",
            )
        except ValueError as exc:
            return (), (
                _rejection(
                    sample,
                    AugmentationRejectionReason.INVALID_IMAGE,
                    "image",
                    str(exc),
                ),
            )
        validation = self._validate_input(
            path=source_path,
            declared_mime_type=sample.image.mime_type,
            declared_byte_size=sample.image.byte_size,
        )
        if not validation.accepted:
            return (), (
                _rejection(
                    sample, AugmentationRejectionReason.INVALID_IMAGE, "image"
                ),
            )
        try:
            source_sha256 = file_sha256(path=source_path)
            prepared, source_info = load_prepared_image(source_path)
        except ImportError:
            return (), (
                _rejection(
                    sample,
                    AugmentationRejectionReason.IMAGE_BACKEND_MISSING,
                    "image",
                ),
            )
        except (OSError, ValueError) as exc:
            return (), (
                _rejection(
                    sample,
                    AugmentationRejectionReason.IMAGE_DECODE_FAILED,
                    "image",
                    str(exc),
                ),
            )

        variants: list[tuple[str, MultimodalSample]] = []
        rejections: list[AugmentationRejection] = []
        for operation in self._operations:
            windows = self._crop_windows(
                operation=operation, prepared=prepared, sample=sample
            )
            if windows == ():
                rejections.append(
                    _rejection(
                        sample,
                        AugmentationRejectionReason.CONTENT_AWARE_CROP_NO_SAFE_WINDOW,
                        operation,
                    )
                )
                continue
            for index, crop_window in enumerate(windows):
                variant_operation = (
                    operation
                    if crop_window is None
                    else f"{operation}_{index}"
                )
                outcome = self._executor.execute(
                    sample=sample,
                    root=root,
                    source_path=source_path,
                    source_sha256=source_sha256,
                    prepared=prepared,
                    source_info=source_info,
                    operation=operation,
                    variant_operation=variant_operation,
                    crop_window=crop_window,
                )
                if outcome.variant is not None:
                    variants.append((variant_operation, outcome.variant))
                if outcome.rejection is not None:
                    rejections.append(outcome.rejection)
        return tuple(variants), tuple(rejections)

    def _crop_windows(
        self, *, operation: str, prepared: Image, sample: MultimodalSample
    ) -> tuple[CropWindow | None, ...]:
        if operation != "content_aware_crop":
            return (None,)
        return select_crop_windows(
            image=prepared,
            sample=sample,
            width=self._settings.crop_width,
            height=self._settings.crop_height,
            candidate_count=self._settings.crop_candidate_count,
            variant_count=self._settings.crop_variant_count,
            minimum_annotation_coverage=self._settings.minimum_annotation_coverage,
            strategy=self._settings.crop_strategy,
            seed_key=sample.sample_id,
        )


def _rejection(
    sample: MultimodalSample,
    reason: AugmentationRejectionReason,
    variant: str,
    message: str | None = None,
) -> AugmentationRejection:
    return AugmentationRejection(
        reason=reason,
        sample_id=sample.sample_id,
        variant_name=variant,
        modality="image",
        message=message,
    )
