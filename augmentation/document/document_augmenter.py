"""Coordinate text-only and page-only document augmentation workflows."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from augmentation.annotations.annotation_safety import (
    non_transformable_annotations,
    rejection_message,
)
from augmentation.document.document_page_augmenter import DocumentPageAugmenter
from augmentation.document.document_text_augmenter import DocumentTextAugmenter
from augmentation.media_variant_support import (
    resolve_dataset_root,
    resolve_source_path,
)
from augmentation.outcomes.augmentation_result import AugmentationRejection
from augmentation.outcomes.rejection_reason import AugmentationRejectionReason
from logger.project_logger import ProjectLogger
from mmcrawler_datasets.schema import ModalityObject, MultimodalSample

if TYPE_CHECKING:
    from config.augmentation.document_settings import (
        DocumentAugmentationSettings,
    )

_PAGE_OPERATIONS = frozenset({"page_image", "layout_preserving"})
_TEXT_OPERATIONS = frozenset({"text_span", "ocr_normalization"})
_DOCUMENT_OPERATIONS = _PAGE_OPERATIONS | _TEXT_OPERATIONS


class DocumentAugmenter:
    """Validate and route each document operation to its SRP component."""

    def __init__(
        self,
        *,
        settings: DocumentAugmentationSettings,
        text_augmenter: DocumentTextAugmenter,
        page_augmenter: DocumentPageAugmenter,
        logger: ProjectLogger,
    ) -> None:
        unknown = set(settings.operations) - _DOCUMENT_OPERATIONS
        if unknown:
            raise ValueError(
                f"unknown document augmentation operations: {sorted(unknown)}"
            )
        self._settings = settings
        self._operations = tuple(settings.operations)
        self._text_augmenter = text_augmenter
        self._page_augmenter = page_augmenter
        self._logger = logger
        self._logger.debug("document_augmenter_initialized")

    @property
    def mode(self) -> str:
        return self._settings.mode

    def media_transforms_enabled(self) -> bool:
        return (
            self._settings.enabled and self._settings.mode != "text_field_only"
        )

    def enabled_operations(self) -> tuple[str, ...]:
        return self._operations if self._settings.enabled else ()

    def augment(
        self, *, sample: MultimodalSample, dataset_root: str | Path | None
    ) -> tuple[
        tuple[tuple[str, MultimodalSample], ...],
        tuple[AugmentationRejection, ...],
    ]:
        if (
            not self._settings.enabled
            or _sample_modality(sample) != "document"
        ):
            return (), ()
        if dataset_root is None:
            return (), (
                _rejection(
                    sample,
                    AugmentationRejectionReason.DATASET_ROOT_MISSING,
                    "document",
                ),
            )
        unsafe = non_transformable_annotations(
            sample=sample, media_kind="document"
        )
        if unsafe:
            return (), (
                _rejection(
                    sample,
                    AugmentationRejectionReason.MEDIA_ANNOTATIONS_NOT_TRANSFORMABLE,
                    "document",
                    rejection_message(fields=unsafe),
                ),
            )

        root = resolve_dataset_root(dataset_root)
        if root is None:
            return (), (
                _rejection(
                    sample,
                    AugmentationRejectionReason.DATASET_ROOT_MISSING,
                    "document",
                ),
            )
        source_page = _modality_path(sample.image, root)
        variants: list[tuple[str, MultimodalSample]] = []
        rejections: list[AugmentationRejection] = []
        for operation in self._operations:
            if operation in _PAGE_OPERATIONS:
                if not self.media_transforms_enabled():
                    continue
                if source_page is None or not source_page.is_file():
                    rejections.append(
                        _rejection(
                            sample,
                            AugmentationRejectionReason.DOCUMENT_PAGE_IMAGE_MISSING,
                            operation,
                        )
                    )
                    continue
                variant, rejection = self._page_augmenter.augment(
                    sample=sample,
                    root=root,
                    source_page=source_page,
                    operation=operation,
                )
            else:
                variant, rejection = self._text_augmenter.augment(
                    sample=sample,
                    root=root,
                    source_page=source_page,
                    operation=operation,
                )
            if variant is not None:
                variants.append((f"document_{operation}", variant))
            if rejection is not None:
                rejections.append(rejection)
        return tuple(variants), tuple(rejections)


def _sample_modality(sample: MultimodalSample) -> str:
    return (
        str((sample.metadata or {}).get("modality") or sample.modality)
        .strip()
        .lower()
    )


def _modality_path(source: object, root: Path) -> Path | None:
    if not isinstance(source, ModalityObject) or source.path is None:
        return None
    try:
        return resolve_source_path(
            dataset_root=root,
            value=source.path,
            error_message="document_source_path_escapes_dataset_root",
        )
    except ValueError:
        return None


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
        modality="document",
        message=message,
    )
