"""Reusable regex detector with safe evidence output."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Pattern, Sequence

from preprocessing.privacy.inspection.detector import TextDetectorInput
from preprocessing.privacy.inspection.evidence_location import (
    EvidenceLocation,
    TextSpan,
)
from preprocessing.privacy.inspection.finding import (
    PrivacyFinding,
    stable_finding_id,
)
from preprocessing.privacy.inspection.finding_type import FindingType
from preprocessing.privacy.inspection.value_digest import (
    digest_sensitive_value,
)

Validator = Callable[[str], bool]
Normalizer = Callable[[str], str]


@dataclass(frozen=True, slots=True)
class PatternSpec:
    finding_type: FindingType
    pattern: Pattern[str]
    confidence: float
    validator: Validator | None = None
    normalizer: Normalizer | None = None
    country: str | None = None


class PatternDetector:
    def __init__(
        self,
        *,
        name: str,
        version: str,
        specifications: Sequence[PatternSpec],
    ) -> None:
        self.name = name
        self.version = version
        self._specifications = tuple(specifications)

    def detect(self, item: TextDetectorInput) -> tuple[PrivacyFinding, ...]:
        findings: list[PrivacyFinding] = []
        for specification in self._specifications:
            for match in specification.pattern.finditer(item.text):
                value = match.group(0)
                if specification.validator is not None:
                    if not specification.validator(value):
                        continue
                normalized = (
                    specification.normalizer(value)
                    if specification.normalizer is not None
                    else value
                )
                location = EvidenceLocation(
                    field_name=item.field_name,
                    text_span=TextSpan(match.start(), match.end()),
                    page_number=item.page_number,
                )
                value_digest = digest_sensitive_value(normalized)
                findings.append(
                    PrivacyFinding(
                        finding_id=stable_finding_id(
                            finding_type=specification.finding_type,
                            detector_name=self.name,
                            detector_version=self.version,
                            location=location,
                            normalized_value_digest=value_digest,
                        ),
                        finding_type=specification.finding_type,
                        confidence=specification.confidence,
                        location=location,
                        detector_name=self.name,
                        detector_version=self.version,
                        normalized_value_digest=value_digest,
                        country=specification.country or item.country,
                        language=item.resolved_language(),
                    )
                )
        return tuple(findings)


def compile_pattern(
    pattern: str, *, ignore_case: bool = False
) -> Pattern[str]:
    flags = re.IGNORECASE if ignore_case else 0
    return re.compile(pattern, flags)
