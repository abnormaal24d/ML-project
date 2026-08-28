"""Tests for mask padding in image remediation."""

from __future__ import annotations

from io import BytesIO

from PIL import Image

from preprocessing.privacy.inspection.evidence_location import (
    BoundingBox,
    EvidenceLocation,
)
from preprocessing.privacy.inspection.finding import PrivacyFinding
from preprocessing.privacy.inspection.finding_type import FindingType
from preprocessing.privacy.remediation.images.mask_sensitive_regions import (
    mask_sensitive_regions,
)


def _create_test_image(width: int = 400, height: int = 300) -> bytes:
    """Create a simple test PNG image."""
    img = Image.new("RGB", (width, height), color="white")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _finding(
    *,
    box: BoundingBox,
    finding_type: FindingType = FindingType.SIGNATURE,
    field_name: str = "visual_content",
) -> PrivacyFinding:
    return PrivacyFinding(
        finding_id=f"{finding_type.value}-{field_name}",
        finding_type=finding_type,
        confidence=0.99,
        location=EvidenceLocation(
            field_name=field_name,
            bounding_box=box,
        ),
        detector_name="test-detector",
        detector_version="1",
    )


def test_mask_sensitive_regions_basic() -> None:
    """Basic masking without padding."""
    payload = _create_test_image()
    finding = _finding(
        box=BoundingBox(x=100, y=100, width=50, height=50),
    )

    result = mask_sensitive_regions(
        payload=payload,
        findings=(finding,),
    )

    assert len(result) > 0
    # Verify it's a valid PNG
    img = Image.open(BytesIO(result))
    assert img.size == (400, 300)


def test_mask_sensitive_regions_with_default_padding() -> None:
    """Default padding is applied to boxes."""
    payload = _create_test_image()
    finding = _finding(
        box=BoundingBox(x=100, y=100, width=100, height=100),
    )

    result = mask_sensitive_regions(
        payload=payload,
        findings=(finding,),
        default_padding=(0.30, 0.35),
    )

    img = Image.open(BytesIO(result))
    # The masked region should be larger than original due to padding
    # Original box: x=100..200, y=100..200
    # With 30% x padding: 30px each side -> 70..230
    # With 35% y padding: 35px each side -> 65..235
    # Check that pixels in padded region are black
    px = img.load()
    # Center of original box should be black
    assert px[150, 150] == (0, 0, 0)
    # Padded area should also be black
    assert px[85, 85] == (0, 0, 0)
    assert px[215, 215] == (0, 0, 0)
    # Outside padded area should be white
    assert px[50, 50] == (255, 255, 255)


def test_mask_sensitive_regions_padding_clamped_to_image_bounds() -> None:
    """Padding is clamped to image boundaries."""
    payload = _create_test_image(400, 300)
    finding = _finding(
        box=BoundingBox(x=0, y=0, width=50, height=50),  # At top-left corner
    )

    result = mask_sensitive_regions(
        payload=payload,
        findings=(finding,),
        default_padding=(1.0, 1.0),  # Large padding
    )

    img = Image.open(BytesIO(result))
    px = img.load()
    # Should not crash and should be clamped to 0,0
    assert px[0, 0] == (0, 0, 0)
    # Area up to 100px should be masked (50 + 100% padding)
    assert px[100, 100] == (0, 0, 0)
    # Beyond 150px should be white
    assert px[200, 200] == (255, 255, 255)


def test_mask_multiple_boxes() -> None:
    """Multiple boxes are all masked."""
    payload = _create_test_image()
    findings = (
        _finding(
            box=BoundingBox(x=50, y=50, width=50, height=50),
        ),
        _finding(
            box=BoundingBox(x=200, y=150, width=100, height=50),
        ),
    )

    result = mask_sensitive_regions(
        payload=payload,
        findings=findings,
        default_padding=(0.0, 0.0),
    )

    img = Image.open(BytesIO(result))
    px = img.load()
    assert px[75, 75] == (0, 0, 0)  # First box center
    assert px[250, 175] == (0, 0, 0)  # Second box center
    assert px[0, 0] == (255, 255, 255)  # Outside boxes


def test_mask_sensitive_regions_preserves_image_format() -> None:
    """Output is valid PNG."""
    payload = _create_test_image()
    finding = _finding(
        box=BoundingBox(x=100, y=100, width=50, height=50),
    )

    result = mask_sensitive_regions(
        payload=payload,
        findings=(finding,),
    )

    img = Image.open(BytesIO(result))
    assert img.format == "PNG"
    assert img.mode == "RGB"


def test_face_uses_face_padding() -> None:
    finding = _finding(
        box=BoundingBox(
            x=100,
            y=100,
            width=100,
            height=100,
        ),
        finding_type=FindingType.FACE,
    )

    result = mask_sensitive_regions(
        payload=_create_test_image(),
        findings=(finding,),
        face_padding=(0.30, 0.30),
        default_padding=(0.0, 0.0),
    )

    px = Image.open(BytesIO(result)).load()

    assert px[75, 75] == (0, 0, 0)
    assert px[60, 60] == (255, 255, 255)


def test_license_plate_uses_plate_padding() -> None:
    finding = _finding(
        box=BoundingBox(
            x=100,
            y=100,
            width=100,
            height=100,
        ),
        finding_type=FindingType.LICENSE_PLATE,
    )

    result = mask_sensitive_regions(
        payload=_create_test_image(),
        findings=(finding,),
        plate_padding=(0.20, 0.20),
        default_padding=(0.0, 0.0),
    )

    px = Image.open(BytesIO(result)).load()

    assert px[85, 85] == (0, 0, 0)
    assert px[70, 70] == (255, 255, 255)


def test_ocr_finding_uses_ocr_padding() -> None:
    finding = _finding(
        box=BoundingBox(
            x=100,
            y=100,
            width=100,
            height=100,
        ),
        finding_type=FindingType.EMAIL_ADDRESS,
        field_name="ocr_text",
    )

    result = mask_sensitive_regions(
        payload=_create_test_image(),
        findings=(finding,),
        ocr_padding=(0.10, 0.15),
        default_padding=(0.0, 0.0),
    )

    px = Image.open(BytesIO(result)).load()

    assert px[95, 90] == (0, 0, 0)
    assert px[80, 80] == (255, 255, 255)


def test_license_plate_padding_wins_over_ocr_padding() -> None:
    finding = _finding(
        box=BoundingBox(
            x=100,
            y=100,
            width=100,
            height=100,
        ),
        finding_type=FindingType.LICENSE_PLATE,
        field_name="ocr_text",
    )

    result = mask_sensitive_regions(
        payload=_create_test_image(),
        findings=(finding,),
        plate_padding=(0.20, 0.20),
        ocr_padding=(0.0, 0.0),
    )

    px = Image.open(BytesIO(result)).load()

    assert px[85, 85] == (0, 0, 0)
    assert px[70, 70] == (255, 255, 255)
