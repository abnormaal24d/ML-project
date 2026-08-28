"""Contract tests for the canonical persisted curated image record."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from mmcrawler_datasets.curated.evidence import (
    AssetContextRecord,
    PrivacyClearanceRecord,
)
from mmcrawler_datasets.curated.image import CuratedImageRecord


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _privacy_clearance_dict() -> dict[str, object]:
    text = "hello"
    text_digest = _digest(text.encode())
    media_digest = _digest(b"media")
    return {
        "status": "approved",
        "input_digest": media_digest,
        "output_digest": media_digest,
        "checked_fields": ["media_decode"],
        "required_fields": ["media_decode"],
        "approved_text_fields": [
            {
                "name": "caption_text",
                "value": text,
                "input_digest": text_digest,
                "output_digest": text_digest,
            }
        ],
        "approved_objects": [],
        "inspection_digest": _digest(b"inspection"),
        "assessment_digest": _digest(b"assessment"),
        "remediation_verified": False,
        "derivation_digest": None,
        "reasons": [],
    }


def _image() -> CuratedImageRecord:
    return CuratedImageRecord(
        schema_version="3.0",
        snapshot_id="snapshot-1",
        image_id="image-1",
        object_id="object-1",
        source_run_id="run-1",
        media_path="media/image-1.png",
        image_mime_type="image/png",
        source_url="https://example.test/image-1.png",
        parent_document_id=None,
        page_title=None,
        alt_text=None,
        figcaption=None,
        surrounding_text=None,
        caption_text="An image",
        caption_source="caption",
        caption_quality_score=0.9,
        context_score=0.8,
        ocr_preview=None,
        image_width=100,
        image_height=100,
        image_format="PNG",
        image_average_hash=None,
        split=None,
        allow_training=True,
        license="CC0",
        trainable=True,
        curated_media_status="trainable",
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("allow_training", "definitely-not"),
        ("image_is_animated", 1),
        ("trainable", "false"),
    ),
)
def test_image_rejects_non_boolean_values(
    field: str,
    value: object,
) -> None:
    row = _image().to_dict()
    row[field] = value

    with pytest.raises(ValueError, match="must be a boolean"):
        CuratedImageRecord.from_dict(row)


@pytest.mark.parametrize("value", (100.5, "100", False))
def test_image_rejects_non_integer_values(value: object) -> None:
    row = _image().to_dict()
    row["image_width"] = value

    with pytest.raises(ValueError, match="must be an integer"):
        CuratedImageRecord.from_dict(row)


@pytest.mark.parametrize(
    "value",
    ("NaN", float("nan"), float("inf"), False),
)
def test_image_rejects_invalid_float_values(value: object) -> None:
    row = _image().to_dict()
    row["ocr_confidence"] = value

    with pytest.raises(ValueError):
        CuratedImageRecord.from_dict(row)


def test_image_rejects_non_string_fingerprint() -> None:
    row = _image().to_dict()
    row["image_phash"] = {"a": 1}

    with pytest.raises(ValueError, match="image_phash must be a string"):
        CuratedImageRecord.from_dict(row)


def test_image_rejects_invalid_nested_ocr_item() -> None:
    row = _image().to_dict()
    row["ocr_boxes"] = [{"text": "valid"}, "invalid"]

    with pytest.raises(ValueError, match=r"ocr_boxes\[1\]"):
        CuratedImageRecord.from_dict(row)


def test_image_from_dict_normalizes_ocr_lists_to_tuples() -> None:
    row = _image().to_dict()
    row["ocr_boxes"] = [{"text": "valid"}]
    row["ocr_lines"] = [{"text": "line"}]

    decoded = CuratedImageRecord.from_dict(row)

    assert decoded.ocr_boxes == ({"text": "valid"},)
    assert decoded.ocr_lines == ({"text": "line"},)
    assert type(decoded.ocr_boxes) is tuple
    assert type(decoded.ocr_lines) is tuple


@pytest.mark.parametrize("mutation", ("unknown", "missing"))
def test_image_requires_exact_wire_fields(mutation: str) -> None:
    row = _image().to_dict()
    if mutation == "unknown":
        row["vendor_metadata"] = "not allowed"
    else:
        del row["allow_training"]

    with pytest.raises(ValueError):
        CuratedImageRecord.from_dict(row)


@pytest.mark.parametrize(
    "path",
    (
        "../escape.png",
        "/absolute/image.png",
        r"C:\media\image.png",
        "media\\image.png",
    ),
)
def test_image_rejects_non_relative_media_paths(path: str) -> None:
    row = _image().to_dict()
    row["media_path"] = path

    with pytest.raises(ValueError, match="project-relative"):
        CuratedImageRecord.from_dict(row)


def test_image_rejects_non_relative_normalized_media_path() -> None:
    row = _image().to_dict()
    row["normalized_media_path"] = r"C:\media\image.png"

    with pytest.raises(ValueError, match="project-relative"):
        CuratedImageRecord.from_dict(row)


def test_image_canonicalizes_media_path() -> None:
    decoded = CuratedImageRecord.from_dict(_image().to_dict())

    assert decoded.media_path == "media/image-1.png"


def test_image_privacy_clearance_round_trip() -> None:
    record = replace(
        _image(),
        privacy_clearance=PrivacyClearanceRecord.model_validate(
            _privacy_clearance_dict()
        ),
    )

    wire = record.to_dict()
    assert wire["privacy_clearance"]["status"] == "approved"
    decoded = CuratedImageRecord.from_dict(wire)

    assert decoded == record
    assert decoded.privacy_clearance.permits_training
    assert decoded.privacy_clearance.approved_text("caption_text") == "hello"


def test_image_rejects_non_object_privacy_clearance() -> None:
    row = _image().to_dict()
    row["privacy_clearance"] = "not-an-object"

    with pytest.raises(
        ValueError, match="privacy_clearance must be an object"
    ):
        CuratedImageRecord.from_dict(row)


def test_image_sparse_asset_context_round_trip() -> None:
    record = replace(
        _image(),
        asset_context=AssetContextRecord.from_mapping(
            {"safety_status": "approved"}
        ),
    )

    wire = record.to_dict()
    assert wire["asset_context"] == {"safety_status": "approved"}
    decoded = CuratedImageRecord.from_dict(wire)

    assert decoded == record
    assert decoded.asset_context.safety_status == "approved"


def test_image_sparse_asset_context_filters_null_fields() -> None:
    record = replace(
        _image(),
        asset_context=AssetContextRecord.from_mapping(
            {
                "safety_status": "approved",
                "fetch_record_id": "fetch-1",
                "media_identity": "identity-1",
            }
        ),
    )

    wire = record.to_dict()

    assert wire["asset_context"] == {
        "safety_status": "approved",
        "fetch_record_id": "fetch-1",
        "media_identity": "identity-1",
    }


def test_image_rejects_unknown_asset_context_wire_fields() -> None:
    row = _image().to_dict()
    row["asset_context"] = {"safety_status": "approved", "raw_html": "x"}

    with pytest.raises(ValueError, match="unknown fields"):
        CuratedImageRecord.from_dict(row)


def test_image_rejects_non_object_asset_context() -> None:
    row = _image().to_dict()
    row["asset_context"] = "not-an-object"

    with pytest.raises(ValueError, match="asset_context must be an object"):
        CuratedImageRecord.from_dict(row)


def test_image_round_trip_uses_producer_contract() -> None:
    record = _image()

    decoded = CuratedImageRecord.from_dict(record.to_dict())

    assert decoded == record
    assert type(decoded) is CuratedImageRecord


def test_image_producer_rejects_non_finite_number() -> None:
    with pytest.raises(ValueError, match="image_quality_score"):
        replace(_image(), image_quality_score=float("nan"))


def test_image_producer_rejects_non_string_fingerprint() -> None:
    with pytest.raises(ValueError, match="image_phash"):
        replace(_image(), image_phash={"a": 1})


def test_image_producer_rejects_untyped_asset_context() -> None:
    with pytest.raises(TypeError, match="asset_context must be typed"):
        replace(_image(), asset_context={"safety_status": "approved"})
