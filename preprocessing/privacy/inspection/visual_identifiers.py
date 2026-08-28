"""Consolidated visual-identifiers privacy detectors."""

from preprocessing.privacy.inspection.detector import VisualDetector
from preprocessing.privacy.inspection.finding_type import FindingType
from preprocessing.privacy.inspection.visual_region_detector import (
    VisualRegionDetector,
)


def build_face_detector() -> VisualRegionDetector:
    return VisualRegionDetector(
        name="face",
        version="1.0.0",
        accepted_categories=frozenset({"face"}),
        finding_type=FindingType.FACE,
    )


def build_identity_document_detector() -> VisualRegionDetector:
    return VisualRegionDetector(
        name="identity_document",
        version="1.0.0",
        accepted_categories=frozenset(
            {"identity_document", "passport", "id_card"}
        ),
        finding_type=FindingType.IDENTITY_DOCUMENT,
    )


def build_license_plate_detector() -> VisualRegionDetector:
    return VisualRegionDetector(
        name="license_plate",
        version="1.0.0",
        accepted_categories=frozenset({"license_plate", "number_plate"}),
        finding_type=FindingType.LICENSE_PLATE,
    )


def build_machine_readable_code_detector() -> VisualRegionDetector:
    return VisualRegionDetector(
        name="machine_readable_code",
        version="1.0.0",
        accepted_categories=frozenset(
            {"qr_code", "barcode", "machine_readable_code"}
        ),
        finding_type=FindingType.MACHINE_READABLE_CODE,
    )


def build_signature_detector() -> VisualRegionDetector:
    return VisualRegionDetector(
        name="signature",
        version="1.0.0",
        accepted_categories=frozenset({"signature"}),
        finding_type=FindingType.SIGNATURE,
    )


def build_visual_identifier_detectors() -> tuple[VisualDetector, ...]:
    """Construct visual-identifier detectors in stable order."""

    return (
        build_face_detector(),
        build_license_plate_detector(),
        build_identity_document_detector(),
        build_signature_detector(),
        build_machine_readable_code_detector(),
    )
