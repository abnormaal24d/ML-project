"""Configured coverage gap analysis."""

from __future__ import annotations

from config.coverage.settings import CoverageSettings
from config.validation.coverage_settings import nonnegative_int, normalize_kind


class CoverageGapAnalyzer:
    """Parse validation errors and map them to configured media gaps."""

    def __init__(self, *, settings: CoverageSettings) -> None:
        self._settings = settings

    def parse_current_minimum(
        self,
        value: str,
    ) -> tuple[int | None, int | None]:
        current_text, separator, minimum_text = str(value).partition("/")
        if not separator:
            return None, None
        try:
            return int(current_text), int(minimum_text)
        except ValueError:
            return None, None

    def parse_count_validation_error(
        self,
        raw_error: str,
    ) -> tuple[str, int, int]:
        parts = str(raw_error).split(":")
        if len(parts) < 2:
            return "", 0, 0

        current, minimum = self.parse_current_minimum(parts[-1])
        if current is None or minimum is None:
            return "", 0, 0

        name = parts[0]
        errors = self._settings.error_names

        text_sample_errors = {
            errors.total_samples_below_min,
            errors.train_samples_below_min,
            errors.val_samples_below_min,
            errors.test_samples_below_min,
        }
        if name in text_sample_errors:
            return "task:text_pretrain", current, minimum

        modality_errors = {
            errors.modality_coverage_below_min,
            errors.split_modality_coverage_below_min,
            errors.raw_modality_coverage_below_min,
        }
        if name in modality_errors and len(parts) >= 3:
            return f"modality:{normalize_kind(parts[-2])}", current, minimum

        if name == errors.task_coverage_below_min and len(parts) >= 3:
            return f"task:{normalize_kind(parts[-2])}", current, minimum

        return "", 0, 0

    def gaps_from_validation_errors(
        self,
        validation_errors: tuple[str, ...] | list[str],
    ) -> dict[str, int]:
        gaps: dict[str, int] = {}
        for raw_error in validation_errors:
            gap_name, current, minimum = self.parse_count_validation_error(
                raw_error
            )
            missing = max(0, minimum - current)
            if not gap_name or missing <= 0:
                continue
            gaps[gap_name] = max(gaps.get(gap_name, 0), missing)
        return dict(sorted(gaps.items()))

    def missing_by_media_kind(
        self,
        coverage_gaps: dict[str, int],
    ) -> dict[str, int]:
        missing_by_kind: dict[str, int] = {}
        kind_settings = self._settings.kinds
        focus_settings = self._settings.focus

        for gap_name, raw_missing in coverage_gaps.items():
            missing = nonnegative_int(raw_missing)
            if missing <= 0:
                continue

            prefix, separator, value = str(gap_name).partition(":")
            if not separator:
                continue

            normalized_value = normalize_kind(value)
            if prefix == "task":
                media_kind = kind_settings.task_to_media_kind.get(
                    normalized_value
                )
                weight = focus_settings.task_gap_weight
            elif prefix == "modality":
                media_kind = kind_settings.modality_to_media_kind.get(
                    normalized_value
                )
                weight = focus_settings.modality_gap_weight
            else:
                continue

            if media_kind is None:
                continue

            normalized_media_kind = normalize_kind(media_kind)
            if normalized_media_kind not in set(kind_settings.media_kinds):
                continue

            missing_by_kind[normalized_media_kind] = (
                missing_by_kind.get(normalized_media_kind, 0)
                + missing * weight
            )

        return dict(sorted(missing_by_kind.items()))

    def merge_gaps(self, *gap_dicts: dict[str, int]) -> dict[str, int]:
        merged: dict[str, int] = {}
        for gap_dict in gap_dicts:
            for name, value in (gap_dict or {}).items():
                merged[name] = max(merged.get(name, 0), nonnegative_int(value))
        return dict(sorted(merged.items()))

    def has_missing_media(self, coverage_gaps: dict[str, int]) -> bool:
        focusable = set(self._settings.kinds.focusable_media_kinds)
        missing = self.missing_by_media_kind(coverage_gaps)
        return any(
            value > 0 for kind, value in missing.items() if kind in focusable
        )
