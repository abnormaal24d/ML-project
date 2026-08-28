"""Normalized evidence produced by any privacy detector."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from preprocessing.privacy.inspection.evidence_location import EvidenceLocation
from preprocessing.privacy.inspection.finding_type import FindingType


@dataclass(frozen=True, slots=True)
class PrivacyFinding:
    finding_id: str
    finding_type: FindingType
    confidence: float
    location: EvidenceLocation
    detector_name: str
    detector_version: str
    normalized_value_digest: str | None = None
    country: str | None = None
    language: str | None = None
    attributes: dict[str, str | int | float | bool] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.finding_id.strip():
            raise ValueError("finding_id must not be blank")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")
        if not self.detector_name.strip():
            raise ValueError("detector_name must not be blank")
        if not self.detector_version.strip():
            raise ValueError("detector_version must not be blank")


def stable_finding_id(
    *,
    finding_type: FindingType,
    detector_name: str,
    detector_version: str,
    location: EvidenceLocation,
    normalized_value_digest: str | None = None,
) -> str:
    """Create a deterministic identifier from non-sensitive evidence."""

    payload = {
        "finding_type": finding_type.value,
        "detector_name": detector_name,
        "detector_version": detector_version,
        "field_name": location.field_name,
        "text_span": (
            [location.text_span.start, location.text_span.end]
            if location.text_span
            else None
        ),
        "page_number": location.page_number,
        "bounding_box": (
            [
                location.bounding_box.x,
                location.bounding_box.y,
                location.bounding_box.width,
                location.bounding_box.height,
            ]
            if location.bounding_box
            else None
        ),
        "time_range": (
            [location.time_range.start_ms, location.time_range.end_ms]
            if location.time_range
            else None
        ),
        "frame_index": location.frame_index,
        "normalized_value_digest": normalized_value_digest,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = ["PrivacyFinding", "stable_finding_id"]
