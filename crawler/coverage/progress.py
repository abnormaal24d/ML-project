"""Bounded progress tracking for coverage-focused workflow recrawls."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config.coverage.settings import CoverageSettings


@dataclass(frozen=True, slots=True)
class CoverageProgressDecision:
    """Decision after observing one invalid training snapshot."""

    should_recrawl: bool
    attempt_count: int
    blocked_reason: str | None = None
    details: tuple[str, ...] = ()


class CoverageProgressTracker:
    """Detect repeated no-progress validation failures and stop recrawling."""

    def __init__(
        self,
        *,
        settings: CoverageSettings,
        max_no_progress_attempts: int | None = None,
    ) -> None:
        self._settings = settings
        configured_attempts = (
            max_no_progress_attempts
            if max_no_progress_attempts is not None
            else self._settings.progress.max_no_progress_attempts
        )
        self._max_no_progress_attempts = max(1, int(configured_attempts))
        self._last_fingerprint: tuple[tuple[str, int], ...] | None = None
        self._same_fingerprint_count = 0

    def observe_validation_failure(
        self,
        *,
        validation_payload: dict[str, Any],
        validation_errors: tuple[str, ...],
        validation_report_path: Path,
        training_directory: Path,
        coverage_gaps: dict[str, int],
        missing_by_kind: dict[str, int],
    ) -> CoverageProgressDecision:
        """Return whether another coverage-focused crawl is still justified."""

        fingerprint = _coverage_fingerprint(
            settings=self._settings,
            validation_payload=validation_payload,
            coverage_gaps=coverage_gaps,
            validation_errors=validation_errors,
        )
        if fingerprint == self._last_fingerprint:
            self._same_fingerprint_count += 1
        else:
            self._last_fingerprint = fingerprint
            self._same_fingerprint_count = 1

        media_blocked = self._media_feasibility_decision(
            validation_payload=validation_payload,
            validation_report_path=validation_report_path,
            training_directory=training_directory,
            coverage_gaps=coverage_gaps,
            missing_by_kind=missing_by_kind,
        )
        if media_blocked is not None:
            return media_blocked

        if self._same_fingerprint_count < self._max_no_progress_attempts:
            return CoverageProgressDecision(
                should_recrawl=True,
                attempt_count=self._same_fingerprint_count,
            )

        details = _blocked_details(
            settings=self._settings,
            validation_payload=validation_payload,
            validation_errors=validation_errors,
            validation_report_path=validation_report_path,
            training_directory=training_directory,
            coverage_gaps=coverage_gaps,
        )
        return CoverageProgressDecision(
            should_recrawl=False,
            attempt_count=self._same_fingerprint_count,
            blocked_reason=self._settings.progress.blocked_reason,
            details=details,
        )

    def _media_feasibility_decision(
        self,
        *,
        validation_payload: dict[str, Any],
        validation_report_path: Path,
        training_directory: Path,
        coverage_gaps: dict[str, int],
        missing_by_kind: dict[str, int],
    ) -> CoverageProgressDecision | None:
        """Block a recrawl that cannot make media coverage progress.

        Gaps that require media assets are not worth a recrawl when the raw
        crawl already found zero media assets: asset discovery/admission is
        the defect, not crawl volume. Returns a blocked decision or None.
        """
        raw_media_total = _raw_media_total(validation_payload)
        if raw_media_total is None or raw_media_total > 0:
            return None

        if not any(
            missing_by_kind.get(kind, 0) > 0
            for kind in ("image", "audio", "video")
        ):
            return None

        return CoverageProgressDecision(
            should_recrawl=False,
            attempt_count=self._same_fingerprint_count,
            blocked_reason="no_raw_media_found",
            details=(
                "coverage gaps require media assets, but the raw crawl "
                "found 0 media assets",
                "fix asset discovery/admission before recrawling",
                f"validation_report={validation_report_path}",
                f"training_directory={training_directory}",
                f"coverage_gaps={dict(sorted(coverage_gaps.items()))}",
                f"raw_media_found={raw_media_total}",
            ),
        )


def _coverage_fingerprint(
    *,
    settings: CoverageSettings,
    validation_payload: dict[str, Any],
    coverage_gaps: dict[str, int],
    validation_errors: tuple[str, ...],
) -> tuple[tuple[str, int], ...]:
    counts: dict[str, int] = {}
    for group_name in ("modalities", "tasks"):
        raw_counts = validation_payload.get(group_name)
        if isinstance(raw_counts, dict):
            for key, value in raw_counts.items():
                counts[f"{group_name}:{key}"] = _as_int(value)
    for key, value in coverage_gaps.items():
        counts[f"gap:{key}"] = _as_int(value)
    for error in validation_errors:
        if str(error).startswith(
            settings.progress.duplicate_sample_error_prefix
        ):
            counts["error:duplicate_sample"] = (
                counts.get("error:duplicate_sample", 0) + 1
            )
    return tuple(sorted(counts.items()))


def _blocked_details(
    *,
    settings: CoverageSettings,
    validation_payload: dict[str, Any],
    validation_errors: tuple[str, ...],
    validation_report_path: Path,
    training_directory: Path,
    coverage_gaps: dict[str, int],
) -> tuple[str, ...]:
    modalities = validation_payload.get("modalities")
    tasks = validation_payload.get("tasks")
    details = [
        "training snapshot validation made no coverage progress",
        f"validation_report={validation_report_path}",
        f"training_directory={training_directory}",
        f"coverage_gaps={dict(sorted(coverage_gaps.items()))}",
    ]
    if isinstance(modalities, dict):
        details.append(f"modalities={dict(sorted(modalities.items()))}")
    if isinstance(tasks, dict):
        details.append(f"tasks={dict(sorted(tasks.items()))}")
    duplicate_errors = [
        error
        for error in validation_errors
        if error.startswith(settings.progress.duplicate_sample_error_prefix)
    ]
    if duplicate_errors:
        details.append(f"duplicate_sample_ids={duplicate_errors[:5]}")
    return tuple(details)


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _raw_media_total(validation_payload: dict[str, Any]) -> int | None:
    """Return total raw media assets reported by the validation flow.

    Returns None when the payload carries no coverage-flow diagnosis.
    """
    diagnosis = validation_payload.get("coverage_flow_diagnosis")
    if not isinstance(diagnosis, dict):
        return None

    raw_counts = diagnosis.get("raw")
    if not isinstance(raw_counts, dict):
        return None

    total = 0
    for kind in ("image", "audio", "video"):
        try:
            total += max(0, int(raw_counts.get(kind) or 0))
        except (TypeError, ValueError):
            continue
    return total
