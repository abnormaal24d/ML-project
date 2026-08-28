"""Basic cross-section validation rules."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from config.validation.coverage_settings import (
    validate_coverage_settings_consistency,
)

if TYPE_CHECKING:
    from config.settings.root import Settings

_SPLIT_TOTAL_TOLERANCE = 1e-6


def _validate_dataset_splits(settings: Settings) -> None:
    """Validate the relational invariant for dataset split ratios."""

    split_settings = (
        ("curation", settings.datasets.splits.curation),
        ("training", settings.datasets.splits.training),
    )

    for split_name, splits in split_settings:
        ratios = (
            float(splits.train_ratio),
            float(splits.val_ratio),
            float(splits.test_ratio),
        )

        if not all(math.isfinite(value) for value in ratios):
            raise ValueError(f"{split_name} split ratios must be finite numbers")

        if any(value < 0.0 or value > 1.0 for value in ratios):
            raise ValueError(
                f"{split_name} split ratios must be between 0.0 and 1.0"
            )

        if not math.isclose(
            math.fsum(ratios),
            1.0,
            rel_tol=0.0,
            abs_tol=_SPLIT_TOTAL_TOLERANCE,
        ):
            raise ValueError(f"{split_name} split ratios must sum to 1.0")


def _validate_coverage(settings: Settings) -> None:
    """Delegate coverage consistency validation to its owning module."""

    messages = tuple(validate_coverage_settings_consistency(settings=settings))
    if messages:
        raise ValueError(
            "coverage settings are inconsistent: "
            + "; ".join(str(message) for message in messages)
        )
