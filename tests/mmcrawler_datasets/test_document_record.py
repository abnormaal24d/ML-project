from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from mmcrawler_datasets.curated.document import (
    ChunkRecord,
    CuratedDocumentRecord,
)
from mmcrawler_datasets.curated.evidence import PrivacyClearanceRecord


def _document() -> CuratedDocumentRecord:
    return CuratedDocumentRecord(
        schema_version="3.0",
        snapshot_id="snapshot-1",
        document_id="document-1",
        source_run_id="run-1",
        source_fetch_record_id="fetch-1",
        object_id="object-1",
        requested_url="https://example.test/document",
        final_url="https://example.test/document",
        normalized_url="https://example.test/document",
        domain="example.test",
        path="/document",
        modality="html",
        language="en",
        title="Document",
        text_path="text/document-1.txt",
        markdown_path=None,
        raw_storage_path="restricted://object-1",
        raw_byte_size=10,
        extracted_char_count=10,
        extracted_token_count_estimate=3,
        boilerplate_ratio=0.1,
        code_block_count=0,
        quality_score=0.9,
        quality_bucket="high",
        rejection_reason=None,
        content_role="article",
        discovery_useful=True,
        exact_duplicate_key="exact-1",
        near_duplicate_cluster_id=None,
        is_near_duplicate=False,
        license="CC0",
        license_url=None,
        allow_training=True,
        created_at="2026-07-27T00:00:00+00:00",
    )


def _chunk() -> ChunkRecord:
    return ChunkRecord(
        schema_version="3.0",
        snapshot_id="snapshot-1",
        chunk_id="chunk-1",
        document_id="document-1",
        chunk_index=0,
        start_char=0,
        end_char=13,
        token_count_estimate=3,
        text="Approved text",
        language="en",
        title="Document",
        section_path=("Introduction",),
        quality_score=0.9,
        exact_duplicate_key="exact-1",
        near_duplicate_cluster_id=None,
        split="train",
    )


def _clearance_record() -> PrivacyClearanceRecord:
    text = "Approved body text"
    text_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    media_digest = hashlib.sha256(b"media").hexdigest()
    return PrivacyClearanceRecord.model_validate(
        {
            "status": "approved",
            "input_digest": media_digest,
            "output_digest": media_digest,
            "checked_fields": ["body"],
            "required_fields": ["body"],
            "approved_text_fields": [
                {
                    "name": "body",
                    "value": text,
                    "input_digest": text_digest,
                    "output_digest": text_digest,
                }
            ],
            "approved_objects": [],
            "inspection_digest": media_digest,
            "assessment_digest": media_digest,
            "remediation_verified": False,
            "derivation_digest": None,
            "reasons": [],
        }
    )


def _document_with_clearance() -> CuratedDocumentRecord:
    return replace(_document(), privacy_clearance=_clearance_record())


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("allow_training", "definitely-not"),
        ("allow_training", 1),
        ("discovery_useful", "true"),
        ("is_near_duplicate", 0),
    ),
)
def test_document_rejects_non_boolean_values(
    field: str,
    value: object,
) -> None:
    row = _document().to_dict()
    row[field] = value

    with pytest.raises(ValueError, match="must be a boolean"):
        CuratedDocumentRecord.from_dict(row)


@pytest.mark.parametrize("value", (44100.9, "44100", True))
def test_document_rejects_non_integer_values(value: object) -> None:
    row = _document().to_dict()
    row["raw_byte_size"] = value

    with pytest.raises(ValueError, match="must be an integer"):
        CuratedDocumentRecord.from_dict(row)


@pytest.mark.parametrize(
    "value",
    ("NaN", float("nan"), float("inf"), True),
)
def test_document_rejects_invalid_float_values(value: object) -> None:
    row = _document().to_dict()
    row["quality_score"] = value

    with pytest.raises(ValueError):
        CuratedDocumentRecord.from_dict(row)


def test_document_rejects_non_string_text() -> None:
    row = _document().to_dict()
    row["license"] = {"a": 1}

    with pytest.raises(ValueError, match="license must be a string"):
        CuratedDocumentRecord.from_dict(row)


@pytest.mark.parametrize("mutation", ("unknown", "missing"))
def test_document_requires_exact_wire_fields(mutation: str) -> None:
    row = _document().to_dict()
    if mutation == "unknown":
        row["vendor_metadata"] = "not allowed"
    else:
        del row["allow_training"]

    with pytest.raises(ValueError, match="invalid document record"):
        CuratedDocumentRecord.from_dict(row)


def test_document_round_trip_uses_producer_contract() -> None:
    record = _document()

    decoded = CuratedDocumentRecord.from_dict(record.to_dict())

    assert decoded == record
    assert type(decoded) is CuratedDocumentRecord


def test_document_producer_rejects_invalid_governance_boolean() -> None:
    with pytest.raises(ValueError, match="allow_training"):
        replace(_document(), allow_training="definitely-not")


def test_document_text_path_must_be_contained_relative_path() -> None:
    for text_path in ("C:\\etc\\passwd", "text/../escape.txt"):
        with pytest.raises(
            ValueError,
            match="curated path must be project-relative",
        ):
            replace(_document(), text_path=text_path)


def test_document_rejects_backslash_path() -> None:
    with pytest.raises(
        ValueError,
        match="curated path must be project-relative",
    ):
        replace(_document(), text_path=r"text\sub\document-1.txt")


def test_document_from_dict_accepts_missing_privacy_clearance() -> None:
    row = {
        key: value
        for key, value in _document().to_dict().items()
        if key != "privacy_clearance"
    }

    decoded = CuratedDocumentRecord.from_dict(row)

    assert decoded.privacy_clearance is None


def test_document_from_dict_rejects_non_object_privacy_clearance() -> None:
    row = _document().to_dict()
    row["privacy_clearance"] = "not-an-object"

    with pytest.raises(
        ValueError,
        match="privacy_clearance must be an object",
    ):
        CuratedDocumentRecord.from_dict(row)


def test_document_from_dict_decodes_privacy_clearance_record() -> None:
    record = _document_with_clearance()

    decoded = CuratedDocumentRecord.from_dict(record.to_dict())

    assert decoded.privacy_clearance == record.privacy_clearance
    assert type(decoded.privacy_clearance) is PrivacyClearanceRecord


def test_document_from_dict_rejects_invalid_privacy_clearance() -> None:
    row = _document_with_clearance().to_dict()
    row["privacy_clearance"]["status"] = "not-a-status"

    with pytest.raises(ValueError):
        CuratedDocumentRecord.from_dict(row)


def test_document_to_dict_serializes_privacy_clearance_wire_shape() -> None:
    record = _document_with_clearance()

    payload = record.to_dict()

    assert payload["privacy_clearance"] == record.privacy_clearance.to_dict()


def test_document_producer_rejects_untyped_privacy_clearance() -> None:
    with pytest.raises(TypeError, match="privacy_clearance must be typed"):
        replace(_document(), privacy_clearance={})


def test_chunk_round_trip_uses_producer_contract() -> None:
    record = _chunk()

    decoded = ChunkRecord.from_dict(record.to_dict())

    assert decoded == record
    assert type(decoded) is ChunkRecord


@pytest.mark.parametrize("mutation", ("unknown", "missing"))
def test_chunk_requires_exact_wire_fields(mutation: str) -> None:
    row = _chunk().to_dict()
    if mutation == "unknown":
        row["vendor_metadata"] = "not allowed"
    else:
        del row["quality_score"]

    with pytest.raises(ValueError, match="invalid chunk record"):
        ChunkRecord.from_dict(row)


@pytest.mark.parametrize("value", (1.5, "1", False))
def test_chunk_rejects_non_integer_values(value: object) -> None:
    row = _chunk().to_dict()
    row["chunk_index"] = value

    with pytest.raises(ValueError, match="chunk_index must be an integer"):
        ChunkRecord.from_dict(row)


@pytest.mark.parametrize("value", ("NaN", float("nan"), float("inf"), True))
def test_chunk_rejects_invalid_float_values(value: object) -> None:
    row = _chunk().to_dict()
    row["quality_score"] = value

    with pytest.raises(ValueError):
        ChunkRecord.from_dict(row)


def test_chunk_rejects_non_string_text() -> None:
    row = _chunk().to_dict()
    row["text"] = {"a": 1}

    with pytest.raises(ValueError, match="text must be a string"):
        ChunkRecord.from_dict(row)


def test_chunk_rejects_non_string_section_path_item() -> None:
    row = _chunk().to_dict()
    row["section_path"] = ["Introduction", 5]

    with pytest.raises(ValueError, match=r"section_path\[1\]"):
        ChunkRecord.from_dict(row)


def test_chunk_section_path_list_normalized_to_tuple() -> None:
    row = _chunk().to_dict()
    row["section_path"] = ["Introduction", "Body"]

    decoded = ChunkRecord.from_dict(row)

    assert decoded.section_path == ("Introduction", "Body")
    assert type(decoded.section_path) is tuple


def test_chunk_rejects_missing_section_path() -> None:
    row = {
        key: value
        for key, value in _chunk().to_dict().items()
        if key != "section_path"
    }

    with pytest.raises(ValueError, match="invalid chunk record"):
        ChunkRecord.from_dict(row)


def test_chunk_rejects_end_before_start() -> None:
    row = _chunk().to_dict()
    row["end_char"] = 2
    row["start_char"] = 5

    with pytest.raises(
        ValueError, match="end_char must not precede start_char"
    ):
        ChunkRecord.from_dict(row)


def test_chunk_producer_rejects_list_section_path() -> None:
    with pytest.raises(TypeError, match="section_path must be a tuple"):
        replace(_chunk(), section_path=["Introduction"])
