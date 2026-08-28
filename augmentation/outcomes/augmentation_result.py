"""Typed outcomes emitted by the augmentation workflow."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

from augmentation.outcomes.rejection_reason import AugmentationRejectionReason

if TYPE_CHECKING:
    from mmcrawler_datasets.schema import MultimodalSample


@dataclass(frozen=True, slots=True)
class AugmentationRejection:
    """One rejected augmentation attempt with sample-level context."""

    reason: AugmentationRejectionReason
    sample_id: str | None = None
    variant_name: str | None = None
    split: str | None = None
    modality: str | None = None
    task_type: str | None = None
    label: int | None = None
    source_url: str | None = None
    text_length: int | None = None
    message: str | None = None

    def to_row(self) -> dict[str, object]:
        """Return a compact JSONL row without empty values."""

        row = {
            "reason": str(self.reason),
            "sample_id": self.sample_id,
            "variant_name": self.variant_name,
            "split": self.split,
            "modality": self.modality,
            "task_type": self.task_type,
            "label": self.label,
            "source_url": self.source_url,
            "text_length": self.text_length,
            "message": self.message,
        }
        return {key: value for key, value in row.items() if value is not None}


@dataclass(frozen=True, slots=True)
class AugmentationReport:
    """Immutable summary emitted by the augmentation workflow."""

    enabled: bool
    original_samples: int
    augmented_samples: int
    variants_added: int
    variants_by_name: Mapping[str, int] = field(
        default_factory=lambda: MappingProxyType({})
    )
    variants_by_operation: Mapping[str, int] = field(
        default_factory=lambda: MappingProxyType({})
    )
    variants_by_modality: Mapping[str, int] = field(
        default_factory=lambda: MappingProxyType({})
    )
    variants_by_task_type: Mapping[str, int] = field(
        default_factory=lambda: MappingProxyType({})
    )
    variants_by_split: Mapping[str, int] = field(
        default_factory=lambda: MappingProxyType({})
    )
    rejected_by_reason: Mapping[str, int] = field(
        default_factory=lambda: MappingProxyType({})
    )
    rejections_by_modality: Mapping[str, int] = field(
        default_factory=lambda: MappingProxyType({})
    )
    media_outputs: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )
    quality_checks_passed: bool = True
    quality_checks: Mapping[str, int] = field(
        default_factory=lambda: MappingProxyType({})
    )
    quality_check_failures: tuple[str, ...] = ()
    rejected_augmentations: tuple[AugmentationRejection, ...] = ()


@dataclass(frozen=True, slots=True)
class AugmentationResult:
    """Augmented dataset plus immutable augmentation report."""

    dataset: tuple[MultimodalSample, ...]
    report: AugmentationReport
