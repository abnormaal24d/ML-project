"""Sentence-window detector for sensitive contextual phrases."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

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


@dataclass(frozen=True, slots=True)
class ContextPhrase:
    expression: re.Pattern[str]
    confidence: float


class ContextPhraseDetector:
    def __init__(
        self,
        *,
        name: str,
        version: str,
        finding_type: FindingType,
        phrases: Sequence[ContextPhrase],
    ) -> None:
        self.name = name
        self.version = version
        self._finding_type = finding_type
        self._phrases = tuple(phrases)

    def detect(self, item: TextDetectorInput) -> tuple[PrivacyFinding, ...]:
        findings: list[PrivacyFinding] = []
        occupied: set[tuple[int, int]] = set()
        for phrase_item in self._phrases:
            for match in phrase_item.expression.finditer(item.text):
                start, end = _sentence_window(
                    item.text, match.start(), match.end()
                )
                if (start, end) in occupied:
                    continue
                occupied.add((start, end))
                location = EvidenceLocation(
                    field_name=item.field_name,
                    text_span=TextSpan(start, end),
                    page_number=item.page_number,
                )
                findings.append(
                    PrivacyFinding(
                        finding_id=stable_finding_id(
                            finding_type=self._finding_type,
                            detector_name=self.name,
                            detector_version=self.version,
                            location=location,
                        ),
                        finding_type=self._finding_type,
                        confidence=phrase_item.confidence,
                        location=location,
                        detector_name=self.name,
                        detector_version=self.version,
                        country=item.country,
                        language=item.resolved_language(),
                    )
                )
        return tuple(findings)


def phrase(expression: str, confidence: float = 0.82) -> ContextPhrase:
    return ContextPhrase(re.compile(expression, re.IGNORECASE), confidence)


def _sentence_window(text: str, start: int, end: int) -> tuple[int, int]:
    left = max(text.rfind(".", 0, start), text.rfind("\n", 0, start)) + 1
    right_candidates = [
        position
        for position in (text.find(".", end), text.find("\n", end))
        if position >= 0
    ]
    right = min(right_candidates) + 1 if right_candidates else len(text)
    return min(max(0, left), start), max(end, right)
