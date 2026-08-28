"""Mask image regions and emit metadata-free PNG bytes."""

from __future__ import annotations

from io import BytesIO

from preprocessing.privacy.inspection.evidence_location import BoundingBox
from preprocessing.privacy.inspection.finding import PrivacyFinding
from preprocessing.privacy.inspection.finding_type import FindingType


def _padded_box(
    box: BoundingBox,
    *,
    image_width: int,
    image_height: int,
    x_ratio: float = 0.30,
    y_ratio: float = 0.35,
) -> BoundingBox:
    pad_x = round(box.width * x_ratio)
    pad_y = round(box.height * y_ratio)

    left = max(0, box.x - pad_x)
    top = max(0, box.y - pad_y)
    right = min(image_width, box.x + box.width + pad_x)
    bottom = min(image_height, box.y + box.height + pad_y)

    return BoundingBox(
        x=left,
        y=top,
        width=max(1, right - left),
        height=max(1, bottom - top),
    )


def mask_sensitive_regions(
    *,
    payload: bytes,
    findings: tuple[PrivacyFinding, ...],
    image_width: int | None = None,
    image_height: int | None = None,
    face_padding: tuple[float, float] = (0.30, 0.35),
    plate_padding: tuple[float, float] = (0.20, 0.20),
    ocr_padding: tuple[float, float] = (0.10, 0.15),
    default_padding: tuple[float, float] = (0.10, 0.15),
) -> bytes:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required for image remediation") from exc

    with Image.open(BytesIO(payload)) as source:
        image = source.convert("RGB")
        img_width, img_height = image.size

        if image_width is None:
            image_width = img_width
        if image_height is None:
            image_height = img_height

        draw = ImageDraw.Draw(image)
        for finding in findings:
            box = finding.location.bounding_box
            if box is None:
                raise ValueError(
                    "image privacy finding requires a bounding box"
                )

            if finding.finding_type is FindingType.FACE:
                padding = face_padding
            elif finding.finding_type is FindingType.LICENSE_PLATE:
                padding = plate_padding
            elif finding.location.field_name == "ocr_text":
                padding = ocr_padding
            else:
                padding = default_padding

            padded = _padded_box(
                box,
                image_width=image_width,
                image_height=image_height,
                x_ratio=padding[0],
                y_ratio=padding[1],
            )
            draw.rectangle(
                (
                    padded.x,
                    padded.y,
                    padded.x + padded.width,
                    padded.y + padded.height,
                ),
                fill=(0, 0, 0),
            )
        target = BytesIO()
        image.save(target, format="PNG", optimize=True)
        return target.getvalue()
