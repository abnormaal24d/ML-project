"""Execute all text detectors with isolated failure reporting."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterable

from preprocessing.privacy.inspection.content_readers.text_content import (
    TextContent,
)
from preprocessing.privacy.inspection.detector import (
    DetectorRun,
    TextDetectorInput,
)
from preprocessing.privacy.inspection.detector_registry import DetectorRegistry
from preprocessing.privacy.inspection.finding import PrivacyFinding
from preprocessing.privacy.inspection.finding_type import FindingType
from preprocessing.privacy.inspection.inspection_coverage import (
    InspectionCoverage,
)
from preprocessing.privacy.inspection.inspection_result import InspectionResult


def inspect_text(
    content: TextContent,
    registry: DetectorRegistry,
) -> InspectionResult:
    subject_digest = hashlib.sha256(content.text.encode("utf-8")).hexdigest()
    findings: list[PrivacyFinding] = []
    runs: list[DetectorRun] = []
    failures: list[str] = []
    if not registry.text_detectors:
        failures.append("text_detector_registry_empty")
    for detector in registry.text_detectors:
        started = time.perf_counter_ns()
        try:
            detected = tuple(
                detector.detect(
                    TextDetectorInput(
                        text=content.text,
                        field_name=content.field_name,
                        language=content.language,
                        country=content.country,
                    )
                )
            )
            findings.extend(detected)
            runs.append(
                DetectorRun(
                    detector_name=detector.name,
                    detector_version=detector.version,
                    completed=True,
                    finding_count=len(detected),
                    elapsed_ms=_elapsed_ms(started),
                )
            )
        except Exception as exc:
            failure = f"{detector.name}:{type(exc).__name__}"
            failures.append(failure)
            runs.append(
                DetectorRun(
                    detector_name=detector.name,
                    detector_version=detector.version,
                    completed=False,
                    finding_count=0,
                    elapsed_ms=_elapsed_ms(started),
                    failure=failure,
                )
            )
    coverage = InspectionCoverage(
        checked_fields=(
            frozenset({content.field_name})
            if registry.text_detectors
            else frozenset()
        ),
        required_fields=frozenset({content.field_name}),
        detector_failures=tuple(failures),
    )
    return InspectionResult(
        subject_digest=subject_digest,
        findings=_deduplicate_findings(findings),
        coverage=coverage,
        detector_runs=tuple(runs),
        completed=not failures and coverage.complete,
        errors=tuple(failures),
    )


def inspect_text_fields(
    *,
    fields: dict[str, str],
    registry: DetectorRegistry,
    language: str | None = None,
    country: str | None = None,
    required_fields: frozenset[str] | None = None,
    subject_bytes: bytes | None = None,
) -> InspectionResult:
    all_findings: list[PrivacyFinding] = []
    all_runs: list[DetectorRun] = []
    failures: list[str] = []
    checked: set[str] = set()
    for field_name, value in fields.items():
        result = inspect_text(
            TextContent(value, field_name, language, country), registry
        )
        checked.update(result.coverage.checked_fields)
        all_findings.extend(result.findings)
        all_runs.extend(result.detector_runs)
        failures.extend(result.errors)
    raw = subject_bytes if subject_bytes is not None else repr(fields).encode()
    required = (
        frozenset(fields) if required_fields is None else required_fields
    )
    unique_failures = tuple(dict.fromkeys(failures))
    coverage = InspectionCoverage(
        checked_fields=frozenset(checked),
        required_fields=required,
        detector_failures=unique_failures,
    )
    return InspectionResult(
        subject_digest=hashlib.sha256(raw).hexdigest(),
        findings=_deduplicate_findings(all_findings),
        coverage=coverage,
        detector_runs=tuple(all_runs),
        completed=not failures and coverage.complete,
        errors=unique_failures,
    )


def _deduplicate_findings(
    findings: Iterable[PrivacyFinding],
) -> tuple[PrivacyFinding, ...]:
    unique: dict[
        tuple[FindingType, str, int | None, int | None],
        PrivacyFinding,
    ] = {}
    for finding in findings:
        span = finding.location.text_span
        key = (
            finding.finding_type,
            finding.location.field_name,
            None if span is None else span.start,
            None if span is None else span.end,
        )
        existing = unique.get(key)
        if existing is None or finding.confidence > existing.confidence:
            unique[key] = finding

    ordered = sorted(
        unique.values(),
        key=lambda item: (
            _STRUCTURED_PRIORITY.get(item.finding_type.value, 100),
            item.location.field_name,
            item.location.text_span.start
            if item.location.text_span is not None
            else -1,
            -(
                item.location.text_span.end - item.location.text_span.start
                if item.location.text_span is not None
                else 0
            ),
        ),
    )
    selected: list[PrivacyFinding] = []
    for candidate in ordered:
        if candidate.finding_type.value in _EXCLUSIVE_STRUCTURED_TYPES:
            if any(_exclusive_overlap(candidate, item) for item in selected):
                continue
        selected.append(candidate)
    return tuple(
        sorted(
            selected,
            key=lambda item: (
                item.location.field_name,
                item.location.text_span.start
                if item.location.text_span is not None
                else -1,
                item.finding_type.value,
            ),
        )
    )


def _exclusive_overlap(
    left: PrivacyFinding,
    right: PrivacyFinding,
) -> bool:
    if right.finding_type.value not in _EXCLUSIVE_STRUCTURED_TYPES:
        return False
    if left.location.field_name != right.location.field_name:
        return False
    left_span = left.location.text_span
    right_span = right.location.text_span
    if left_span is None or right_span is None:
        return False
    return (
        left_span.start < right_span.end and right_span.start < left_span.end
    )


_EXCLUSIVE_STRUCTURED_TYPES = frozenset(
    {
        "belgian_national_number",
        "payment_card",
        "iban",
        "date_of_birth",
        "passport_number",
        "ip_address",
        "phone_number",
    }
)
_STRUCTURED_PRIORITY = {
    "belgian_national_number": 0,
    "payment_card": 1,
    "iban": 2,
    "date_of_birth": 3,
    "passport_number": 4,
    "ip_address": 5,
    "phone_number": 6,
}


def _elapsed_ms(started_ns: int) -> int:
    return max(0, (time.perf_counter_ns() - started_ns) // 1_000_000)
