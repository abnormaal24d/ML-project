"""Explicit privacy test composition helpers."""

from config.preprocessing.text_settings import PrivacyDetectionSettings
from orchestration.composition.privacy.privacy_inspection_services import (
    build_default_detector_registry,
)
from preprocessing.privacy.text_privacy import PiiDetector


def build_test_pii_detector(
    settings: PrivacyDetectionSettings | None = None,
) -> PiiDetector:
    return PiiDetector(
        settings=settings or PrivacyDetectionSettings(),
        registry=build_default_detector_registry(),
    )
