"""Immutable detector selection by content type."""

from __future__ import annotations

from dataclasses import dataclass

from preprocessing.privacy.inspection.detector import (
    TextDetector,
    VisualDetector,
)


@dataclass(frozen=True, slots=True)
class DetectorRegistry:
    text_detectors: tuple[TextDetector, ...]
    visual_detectors: tuple[VisualDetector, ...] = ()

    def detector_versions(self) -> dict[str, str]:
        versions = {
            detector.name: detector.version for detector in self.text_detectors
        }
        versions.update(
            {
                detector.name: detector.version
                for detector in self.visual_detectors
            }
        )
        return versions
