"""Compose the detector registry used by every local privacy inspector."""

from dataclasses import dataclass

from config.preprocessing.text_settings import PrivacyDetectionSettings
from preprocessing.privacy.inspection.credentials import (
    build_credential_detectors,
)
from preprocessing.privacy.inspection.detector_registry import DetectorRegistry
from preprocessing.privacy.inspection.named_entities import (
    build_named_entity_detectors,
)
from preprocessing.privacy.inspection.sensitive_information import (
    build_sensitive_information_detectors,
)
from preprocessing.privacy.inspection.structured_identifiers import (
    build_structured_identifier_detectors,
)
from preprocessing.privacy.inspection.visual_identifiers import (
    build_visual_identifier_detectors,
)
from preprocessing.privacy.text_privacy import PiiDetector


@dataclass(frozen=True, slots=True)
class PrivacyInspectionServices:
    """Stateless privacy dependencies shared by all preprocessors."""

    registry: DetectorRegistry
    pii_detector: PiiDetector


def build_default_detector_registry() -> DetectorRegistry:
    """Build fresh detector instances while preserving canonical ordering."""

    return DetectorRegistry(
        text_detectors=(
            *build_structured_identifier_detectors(),
            *build_credential_detectors(),
            *build_named_entity_detectors(),
            *build_sensitive_information_detectors(),
        ),
        visual_detectors=build_visual_identifier_detectors(),
    )


def build_privacy_inspection_services(
    *,
    settings: PrivacyDetectionSettings,
) -> PrivacyInspectionServices:
    """Build one registry shared by text and all local media inspectors."""

    registry = build_default_detector_registry()
    return PrivacyInspectionServices(
        registry=registry,
        pii_detector=PiiDetector(settings=settings, registry=registry),
    )


__all__ = [
    "PrivacyInspectionServices",
    "build_default_detector_registry",
    "build_privacy_inspection_services",
]
