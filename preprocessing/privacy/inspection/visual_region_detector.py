"""Transforms trusted visual-region detections into privacy findings."""

from __future__ import annotations

from typing import Sequence

from preprocessing.privacy.inspection.detector import VisualRegion
from preprocessing.privacy.inspection.evidence_location import (
    BoundingBox,
    EvidenceLocation,
    TimeRange,
)
from preprocessing.privacy.inspection.finding import (
    PrivacyFinding,
    stable_finding_id,
)
from preprocessing.privacy.inspection.finding_type import FindingType


class VisualRegionDetector:
    def __init__(
        self,
        *,
        name: str,
        version: str,
        accepted_categories: frozenset[str],
        finding_type: FindingType,
        minimum_confidence: float = 0.7,
    ) -> None:
        self.name = name
        self.version = version
        self._categories = accepted_categories
        self._finding_type = finding_type
        self._minimum_confidence = minimum_confidence

    def detect_regions(
        self,
        *,
        field_name: str,
        regions: Sequence[VisualRegion],
    ) -> tuple[PrivacyFinding, ...]:
        findings: list[PrivacyFinding] = []
        for region in regions:
            if region.category.casefold() not in self._categories:
                continue
            if region.confidence < self._minimum_confidence:
                continue
            time_range = None
            if region.timestamp_ms is not None:
                time_range = TimeRange(
                    region.timestamp_ms,
                    region.timestamp_ms + 1,
                )
            location = EvidenceLocation(
                field_name=field_name,
                bounding_box=BoundingBox(
                    region.x,
                    region.y,
                    region.width,
                    region.height,
                ),
                frame_index=region.frame_index,
                time_range=time_range,
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
                    confidence=region.confidence,
                    location=location,
                    detector_name=self.name,
                    detector_version=self.version,
                )
            )
        return tuple(findings)
