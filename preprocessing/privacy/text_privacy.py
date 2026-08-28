"""PII detection and privacy rules for preprocessed text.

Text preprocessing uses the central privacy package at
``preprocessing.privacy`` for detection, assessment, and remediation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

from preprocessing.privacy.inspection.content_readers.text_content import (
    TextContent,
)
from preprocessing.privacy.inspection.detector_registry import (
    DetectorRegistry,
)
from preprocessing.privacy.inspection.inspect_text import inspect_text

if TYPE_CHECKING:
    from config.preprocessing.text_settings import PrivacyDetectionSettings


@dataclass(frozen=True, slots=True)
class PiiDetectionResult:
    """Sanitized result produced by the central privacy inspection engine."""

    finding_counts: dict[str, int]
    spans: tuple[dict[str, object], ...] = ()
    completed: bool = True
    coverage_complete: bool = True
    assessment_outcome: str = "accept"
    risk_level: str = "none"
    errors: tuple[str, ...] = ()

    @property
    def has_findings(self) -> bool:
        return any(count > 0 for count in self.finding_counts.values())

    @property
    def has_secret_findings(self) -> bool:
        return any(
            self.finding_counts.get(name, 0) > 0
            for name in _SECRET_FINDING_TYPES
        )

    @property
    def has_personal_findings(self) -> bool:
        return any(
            self.finding_counts.get(name, 0) > 0
            for name in _REDACTABLE_FINDING_TYPES
        )

    @property
    def total_findings(self) -> int:
        return sum(self.finding_counts.values())

    @property
    def inspection_complete(self) -> bool:
        return self.completed and self.coverage_complete and not self.errors


@dataclass(frozen=True, slots=True)
class _NormalizedSpan:
    finding_type: str
    start: int
    end: int


class PiiDetector:
    """Detect and classify privacy findings for preprocessing only."""

    def __init__(
        self,
        *,
        settings: PrivacyDetectionSettings,
        registry: DetectorRegistry,
    ) -> None:
        self._settings = settings
        self._registry = registry

    def detect(self, *, text: str) -> PiiDetectionResult:
        if not self._settings.enabled:
            return PiiDetectionResult(
                finding_counts={},
                spans=(),
                completed=False,
                coverage_complete=False,
                assessment_outcome="quarantine",
                risk_level="unknown",
                errors=("pii_detection_disabled",),
            )

        inspection = inspect_text(
            TextContent(text=text, field_name="text"),
            self._registry,
        )
        counts = inspection.finding_counts
        outcome, risk_level = _classify_findings(
            finding_counts=counts,
            inspection_complete=(
                inspection.completed and inspection.coverage.complete
            ),
        )
        spans: tuple[dict[str, object], ...] = tuple(
            {
                "type": finding.finding_type.value,
                "start": finding.location.text_span.start,
                "end": finding.location.text_span.end,
                "confidence": finding.confidence,
                "detector": finding.detector_name,
            }
            for finding in inspection.findings
            if finding.location.text_span is not None
        )
        return PiiDetectionResult(
            finding_counts=counts,
            spans=spans,
            completed=inspection.completed,
            coverage_complete=inspection.coverage.complete,
            assessment_outcome=outcome,
            risk_level=risk_level,
            errors=inspection.errors,
        )

    @property
    def registry(self) -> DetectorRegistry:
        """Return the immutable registry shared by all local inspectors."""

        return self._registry

    @property
    def quarantine_on_detection(self) -> bool:
        return self._settings.quarantine_on_detection

    def has_secret_findings(self, *, result: PiiDetectionResult) -> bool:
        return result.has_secret_findings

    def redact(self, *, text: str, result: PiiDetectionResult) -> str:
        spans = _merge_redaction_spans(text=text, spans=result.spans)
        redacted = text
        for span in reversed(spans):
            replacement = f"[REDACTED_{span.finding_type.upper()}]"
            redacted = (
                redacted[: span.start] + replacement + redacted[span.end :]
            )
        return redacted


def _classify_findings(
    *,
    finding_counts: dict[str, int],
    inspection_complete: bool,
) -> tuple[str, str]:
    """Map detector output directly to the preprocessing privacy action."""

    if not inspection_complete:
        return "quarantine", "critical"
    present = {name for name, count in finding_counts.items() if count > 0}
    if present & _SECRET_FINDING_TYPES:
        return "escalate", "critical"
    if present & _RESTRICTED_FINDING_TYPES:
        return "reject", "critical"
    if present & _REVIEW_FINDING_TYPES:
        return "review", "critical"
    if len(present & _REIDENTIFICATION_FINDING_TYPES) >= 3:
        return "review", "high"
    if present & _REDACTABLE_FINDING_TYPES:
        return "remediate", "high"
    return "accept", "none"


def redact_text(
    *,
    text: str,
    spans: tuple[dict[str, object], ...],
    replacement: str = "[REDACTED]",
) -> str:
    """Return text with remediable personal-data spans replaced."""

    safe_spans = tuple(
        span
        for span in spans
        if str(span.get("type") or "") in _REDACTABLE_FINDING_TYPES
    )
    normalized = _merge_redaction_spans(text=text, spans=safe_spans)
    redacted = text
    for span in reversed(normalized):
        redacted = redacted[: span.start] + replacement + redacted[span.end :]
    return redacted


def _merge_redaction_spans(
    *,
    text: str,
    spans: Iterable[dict[str, object]],
) -> tuple[_NormalizedSpan, ...]:
    normalized: list[_NormalizedSpan] = []
    for raw_span in spans:
        finding_type = str(raw_span.get("type") or "unknown").strip().lower()
        if finding_type not in _REDACTABLE_FINDING_TYPES:
            continue
        start = _span_int(raw_span.get("start"))
        end = _span_int(raw_span.get("end"))
        if start >= end or end > len(text):
            continue
        normalized.append(
            _NormalizedSpan(
                finding_type=finding_type,
                start=start,
                end=end,
            )
        )

    if not normalized:
        return ()

    normalized.sort(key=lambda span: (span.start, span.end))
    merged: list[_NormalizedSpan] = []
    current_group: list[_NormalizedSpan] = [normalized[0]]
    current_end = normalized[0].end

    for span in normalized[1:]:
        if span.start < current_end:
            current_group.append(span)
            current_end = max(current_end, span.end)
            continue
        merged.append(_collapse_span_group(current_group))
        current_group = [span]
        current_end = span.end

    merged.append(_collapse_span_group(current_group))
    return tuple(merged)


def _collapse_span_group(spans: list[_NormalizedSpan]) -> _NormalizedSpan:
    representative = min(
        spans,
        key=lambda span: (
            _FINDING_PRIORITY.get(span.finding_type, 100),
            span.start,
            -(span.end - span.start),
        ),
    )
    return _NormalizedSpan(
        finding_type=representative.finding_type,
        start=min(span.start for span in spans),
        end=max(span.end for span in spans),
    )


def _span_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if not isinstance(value, (int, float, str, bytes, bytearray)):
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


_SECRET_FINDING_TYPES: frozenset[str] = frozenset(
    {
        "api_credential",
        "cloud_credential",
        "oauth_token",
        "jwt_token",
        "session_credential",
        "private_key",
        "basic_auth_credential",
        "database_credential",
    }
)
_RESTRICTED_FINDING_TYPES: frozenset[str] = frozenset(
    {
        "belgian_national_number",
        "passport_number",
        "identity_document",
        "payment_card",
    }
)
_REVIEW_FINDING_TYPES: frozenset[str] = frozenset(
    {
        "health_information",
        "financial_information",
        "political_information",
        "religious_information",
        "criminal_information",
        "minor_information",
        "face",
        "voice_identity",
        "signature",
    }
)
_REIDENTIFICATION_FINDING_TYPES: frozenset[str] = frozenset(
    {
        "person_name",
        "postal_address",
        "date_of_birth",
        "geographic_location",
        "organization",
        "ip_address",
    }
)
_REDACTABLE_FINDING_TYPES: frozenset[str] = frozenset(
    {
        "person_name",
        "email_address",
        "phone_number",
        "postal_address",
        "geographic_location",
        "date_of_birth",
        "iban",
        "ip_address",
        "license_plate",
    }
)
_FINDING_PRIORITY: dict[str, int] = {
    "belgian_national_number": 0,
    "payment_card": 1,
    "iban": 2,
    "date_of_birth": 3,
    "passport_number": 4,
    "email_address": 5,
    "ip_address": 6,
    "phone_number": 7,
    "postal_address": 8,
    "person_name": 9,
    "geographic_location": 10,
    "license_plate": 11,
}

