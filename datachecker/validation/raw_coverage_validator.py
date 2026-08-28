"""Raw crawl coverage validation (modality counts vs minimums)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from config.validation.coverage_settings import nonnegative_int, normalize_kind

_RAW_COVERAGE_LOW = "raw_modality_coverage_below_min"
_RAW_MODALITY_MISSING = "raw_modality_missing"


@dataclass(frozen=True, slots=True)
class CoverageValidationResult:
    """Outcome of coverage validation."""

    is_valid: bool
    errors: tuple[str, ...] = ()
    counts: dict[str, int] | None = None
    minimums: dict[str, int] | None = None


class RawCoverageValidator:
    """Validate raw crawl modality counts against configured minimums."""

    def __init__(
        self,
        *,
        minimum_modality_counts: dict[str, int],
    ) -> None:
        self._minimum_modality_counts = {
            normalize_kind(kind): nonnegative_int(minimum)
            for kind, minimum in minimum_modality_counts.items()
            if normalize_kind(kind)
        }

    def validate_counts(
        self,
        *,
        counts: Mapping[str, int],
    ) -> CoverageValidationResult:
        normalized_counts = {
            normalize_kind(kind): nonnegative_int(count)
            for kind, count in counts.items()
            if normalize_kind(kind)
        }
        errors: list[str] = []
        for kind, minimum in sorted(self._minimum_modality_counts.items()):
            if minimum <= 0:
                continue
            current = normalized_counts.get(kind, 0)
            if current < minimum:
                errors.append(
                    f"{_RAW_COVERAGE_LOW}:{kind}:{current}/{minimum}"
                )
            if current == 0:
                errors.append(
                    f"{_RAW_MODALITY_MISSING}:{kind}:target>{minimum}"
                )

        return CoverageValidationResult(
            is_valid=not errors,
            errors=tuple(errors),
            counts=normalized_counts,
            minimums=dict(self._minimum_modality_counts),
        )
