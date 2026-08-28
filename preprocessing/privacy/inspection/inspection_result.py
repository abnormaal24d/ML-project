"""Immutable aggregate returned by every privacy inspection.

This module owns the raw ``InspectionResult`` produced by local inspectors and
the release-safe projection ``MediaAnalysisEvidence`` used by the media
privacy release policy.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from preprocessing.privacy.inspection.detector import DetectorRun
from preprocessing.privacy.inspection.finding import PrivacyFinding
from preprocessing.privacy.inspection.inspection_coverage import (
    InspectionCoverage,
)


@dataclass(frozen=True, slots=True)
class InspectionResult:
    subject_digest: str
    findings: tuple[PrivacyFinding, ...]
    coverage: InspectionCoverage
    detector_runs: tuple[DetectorRun, ...]
    completed: bool
    errors: tuple[str, ...] = ()
    detector_versions: tuple[tuple[str, str], ...] = ()

    @property
    def has_findings(self) -> bool:
        return bool(self.findings)

    @property
    def finding_counts(self) -> dict[str, int]:
        return dict(Counter(item.finding_type.value for item in self.findings))

    @property
    def safe_to_assess(self) -> bool:
        return self.completed and self.coverage.complete and not self.errors


@dataclass(frozen=True, slots=True)
class MediaFindingEvidence:
    """One normalized local detector finding without raw private values."""

    finding_type: str
    detector_name: str
    detector_version: str
    confidence: float
    field_name: str
    bounding_box: tuple[int, int, int, int] | None
    time_range_ms: tuple[int, int] | None
    frame_index: int | None
    attributes: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "type": self.finding_type,
            "detector": self.detector_name,
            "version": self.detector_version,
            "confidence": self.confidence,
            "field_name": self.field_name,
            "bounding_box": list(self.bounding_box)
            if self.bounding_box
            else None,
            "time_range_ms": list(self.time_range_ms)
            if self.time_range_ms
            else None,
            "frame_index": self.frame_index,
            "attributes": dict(self.attributes),
        }


@dataclass(frozen=True, slots=True)
class MediaAnalysisEvidence:
    """Serializable evidence generated from a local ``InspectionResult``."""

    subject_sha256: str
    completed_checks: frozenset[str]
    findings: tuple[MediaFindingEvidence, ...]
    checked_ranges_ms: tuple[tuple[int, int], ...]
    detector_versions: tuple[tuple[str, str], ...]
    coverage_complete: bool
    unchecked_fields: frozenset[str]
    errors: tuple[str, ...]
    valid: bool
    reasons: tuple[str, ...]

    def finding_types(self) -> frozenset[str]:
        return frozenset(item.finding_type for item in self.findings)

    @property
    def primary_failure_reason(self) -> str | None:
        """Return the most actionable local inspection failure."""

        if self.errors:
            return self.errors[0]
        if self.unchecked_fields:
            return f"local_inspection_unchecked:{sorted(self.unchecked_fields)[0]}"
        if self.reasons:
            return self.reasons[0]
        return None

    @property
    def clean(self) -> bool:
        """Return whether the inspection was valid and found no private data."""

        return self.valid and not self.findings

    def to_dict(self) -> dict[str, object]:
        return {
            "source": "local_inspection",
            "subject_sha256": self.subject_sha256,
            "completed_checks": sorted(self.completed_checks),
            "findings": [item.to_dict() for item in self.findings],
            "checked_ranges_ms": [
                list(item) for item in self.checked_ranges_ms
            ],
            "detector_versions": [
                list(item) for item in self.detector_versions
            ],
            "coverage_complete": self.coverage_complete,
            "unchecked_fields": sorted(self.unchecked_fields),
            "errors": list(self.errors),
            "valid": self.valid,
            "reasons": list(self.reasons),
        }


def media_analysis_evidence(
    inspection: InspectionResult,
    *,
    expected_digest: str,
) -> MediaAnalysisEvidence:
    """Normalize a local inspection and independently verify byte binding."""

    reasons: list[str] = []
    if not expected_digest:
        reasons.append("local_inspection_subject_unavailable")
    if inspection.subject_digest != expected_digest:
        reasons.append("local_inspection_subject_mismatch")
    if not inspection.completed:
        reasons.append("local_inspection_incomplete")
    if not inspection.coverage.complete:
        reasons.append("local_inspection_coverage_incomplete")
    reasons.extend(
        f"local_inspection_unchecked:{name}"
        for name in sorted(inspection.coverage.unchecked_fields)
    )
    errors = tuple(
        dict.fromkeys(
            (*inspection.errors, *inspection.coverage.detector_failures)
        )
    )
    reasons.extend(f"local_detector_failure:{error}" for error in errors)

    findings = tuple(
        MediaFindingEvidence(
            finding_type=finding.finding_type.value,
            detector_name=finding.detector_name,
            detector_version=finding.detector_version,
            confidence=finding.confidence,
            field_name=finding.location.field_name,
            bounding_box=(
                (
                    finding.location.bounding_box.x,
                    finding.location.bounding_box.y,
                    finding.location.bounding_box.width,
                    finding.location.bounding_box.height,
                )
                if finding.location.bounding_box is not None
                else None
            ),
            time_range_ms=(
                (
                    finding.location.time_range.start_ms,
                    finding.location.time_range.end_ms,
                )
                if finding.location.time_range is not None
                else None
            ),
            frame_index=finding.location.frame_index,
            attributes=tuple(
                sorted(
                    (str(key), str(value))
                    for key, value in finding.attributes.items()
                )
            ),
        )
        for finding in inspection.findings
    )
    checked_ranges = (
        inspection.coverage.checked_audio_ranges_ms
        or inspection.coverage.checked_video_ranges_ms
    )
    versions = tuple(
        sorted(
            {
                *inspection.detector_versions,
                *(
                    (run.detector_name, run.detector_version)
                    for run in inspection.detector_runs
                ),
            }
        )
    )
    unique_reasons = tuple(dict.fromkeys(reasons))
    return MediaAnalysisEvidence(
        subject_sha256=inspection.subject_digest,
        completed_checks=inspection.coverage.checked_fields,
        findings=findings,
        checked_ranges_ms=checked_ranges,
        detector_versions=versions,
        coverage_complete=inspection.coverage.complete,
        unchecked_fields=inspection.coverage.unchecked_fields,
        errors=errors,
        valid=not unique_reasons,
        reasons=unique_reasons,
    )


__all__ = [
    "InspectionResult",
    "MediaAnalysisEvidence",
    "MediaFindingEvidence",
    "media_analysis_evidence",
]
