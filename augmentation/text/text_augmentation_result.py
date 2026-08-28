"""Sample-level augmentation output with detailed rejection context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from augmentation.outcomes.augmentation_result import AugmentationRejection
    from mmcrawler_datasets.schema import MultimodalSample


@dataclass(frozen=True, slots=True)
class SampleAugmentationResult:
    """Variants and rejected attempts produced for one source sample."""

    variants: tuple[tuple[str, MultimodalSample], ...] = ()
    rejections: tuple[AugmentationRejection, ...] = ()

    @classmethod
    def rejected(
        cls,
        *rejections: AugmentationRejection,
    ) -> SampleAugmentationResult:
        """Return a result that only records rejected augmentation attempts."""

        return cls(variants=(), rejections=tuple(rejections))
