"""Contract tests for the neutral timed-media wire contract.

The strict decode behavior of the audio/video records lives next to the
contract itself, at the owner:

1. ``timed_media.py`` stays dependency-neutral: it must not import any
   higher domain package (crawler, preprocessing, training, evaluator,
   orchestration, ...).
2. The canonical audio/video JSON schemas remain byte-identical to the
   fixed golden SHA-256 digests below.
3. Rows are decoded strictly through ``CuratedAudioRecord.model_validate``
   and ``CuratedVideoRecord.model_validate`` without a wrapper layer.

Any refactor of the timed-media contract must keep all invariants intact.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from mmcrawler_datasets.curated.evidence import AssetContextRecord
from mmcrawler_datasets.curated.timed_media import (
    CuratedAudioRecord,
    CuratedVideoRecord,
    timed_media_contract_sha256,
)

CONTRACT_PATH = (
    Path(__file__).resolve().parents[2]
    / "mmcrawler_datasets"
    / "curated"
    / "timed_media.py"
)

_FORBIDDEN_DOMAIN_PACKAGES = (
    "augmentation",
    "crawler",
    "datachecker",
    "evaluator",
    "multimodal",
    "orchestration",
    "preprocessing",
    "training",
)

GOLDEN_AUDIO_SCHEMA_SHA256 = (
    "ee33313b6b5a0fa074dec10ce4af34ff7c5700495e03a71e887c998c948739bf"
)
GOLDEN_VIDEO_SCHEMA_SHA256 = (
    "bac6b3f89e44ace3f693e2bc9a2dc2518cb7b006c44101a5c9d2570ec1e0c7fd"
)


def test_timed_media_contract_has_no_domain_imports() -> None:
    tree = ast.parse(CONTRACT_PATH.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    assert not imported.intersection(_FORBIDDEN_DOMAIN_PACKAGES)


@pytest.mark.parametrize(
    ("record_type", "expected_sha256"),
    (
        (CuratedAudioRecord, GOLDEN_AUDIO_SCHEMA_SHA256),
        (CuratedVideoRecord, GOLDEN_VIDEO_SCHEMA_SHA256),
    ),
)
def test_schema_hash(
    record_type: type[CuratedAudioRecord] | type[CuratedVideoRecord],
    expected_sha256: str,
) -> None:
    assert timed_media_contract_sha256(record_type) == expected_sha256


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _clearance() -> dict[str, object]:
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
                "name": "transcript_text",
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


def _common_row(*, media_id: str, media_path: str) -> dict[str, object]:
    return {
        "schema_version": "3.0",
        "snapshot_id": "snapshot-1",
        "media_id": media_id,
        "object_id": f"object-{media_id}",
        "source_run_id": "run-1",
        "source_url": f"https://example.test/{media_id}",
        "media_path": media_path,
        "media_mime_type": "application/octet-stream",
        "domain": "example.test",
        "language": "en",
        "parent_document_id": None,
        "page_title": None,
        "surrounding_text": None,
        "html_context": None,
        "transcript_text": "hello",
        "transcript_preview": None,
        "transcript_language": "en",
        "transcript_segments": [],
        "context_score": 0.8,
        "quality_score": 0.9,
        "fetch_mode": None,
        "asset_fetch_mode": None,
        "is_complete_payload": True,
        "observed_bytes": 5,
        "source_content_length": None,
        "source_content_type": None,
        "fetch_duration_seconds": None,
        "payload_sha256": _digest(b"media"),
        "media_fingerprint": None,
        "near_duplicate_cluster_id": None,
        "allow_training": True,
        "license": "CC0",
        "license_url": None,
        "governance_note": None,
        "robots_status": None,
        "terms_source": None,
        "usage_rules": None,
        "privacy_clearance": _clearance(),
        "safety_status": "passed",
        "asset_context": AssetContextRecord.from_mapping(
            {"safety_status": "passed"}
        ).to_dict(),
        "trainable": True,
        "curated_media_status": "trainable",
        "curated_rejection_reason": None,
    }


def _audio_row() -> dict[str, object]:
    return {
        **_common_row(
            media_id="audio-1",
            media_path="media/audio-1.wav",
        ),
        "modality": "audio",
        "normalized_audio_path": "media/audio-1.wav",
        "target_audio_path": "media/audio-1.wav",
        "audio_duration_seconds": None,
        "audio_sample_rate": None,
        "audio_channels": None,
        "audio_loudness_lufs": None,
        "audio_chromaprint": None,
    }


def _video_row() -> dict[str, object]:
    return {
        **_common_row(
            media_id="video-1",
            media_path="media/video-1.mp4",
        ),
        "modality": "video",
        "normalized_video_path": "media/video-1.mp4",
        "target_video_path": "media/video-1.mp4",
        "video_duration_seconds": None,
        "video_width": None,
        "video_height": None,
        "frame_ocr_text": None,
        "frame_ocr_preview": None,
        "keyframes": [],
        "video_keyframe_phashes": None,
    }


@pytest.mark.parametrize(
    ("decoder", "row_factory"),
    (
        (CuratedAudioRecord.model_validate, _audio_row),
        (CuratedVideoRecord.model_validate, _video_row),
    ),
)
def test_timed_media_rejects_unknown_fields(decoder, row_factory) -> None:
    row = row_factory()
    row["vendor_metadata"] = "must not survive"

    with pytest.raises(
        ValidationError, match="Extra inputs are not permitted"
    ):
        decoder(row)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("allow_training", "definitely-not"),
        ("is_complete_payload", 1),
        ("observed_bytes", 5.0),
        ("fetch_duration_seconds", "0.1"),
    ),
)
def test_audio_rejects_implicit_coercion(field: str, value: object) -> None:
    row = _audio_row()
    row[field] = value

    with pytest.raises(ValidationError):
        CuratedAudioRecord.model_validate(row)


def test_audio_rejects_fractional_sample_rate() -> None:
    row = _audio_row()
    row["audio_sample_rate"] = 44100.9

    with pytest.raises(ValidationError):
        CuratedAudioRecord.model_validate(row)


@pytest.mark.parametrize("value", (float("nan"), float("inf")))
def test_audio_rejects_non_finite_duration(value: float) -> None:
    row = _audio_row()
    row["audio_duration_seconds"] = value

    with pytest.raises(ValidationError):
        CuratedAudioRecord.model_validate(row)


def test_audio_rejects_invalid_nested_segment() -> None:
    row = _audio_row()
    row["transcript_segments"] = [{"text": 123}]

    with pytest.raises(ValidationError):
        CuratedAudioRecord.model_validate(row)


@pytest.mark.parametrize(
    "path",
    ("../escape.wav", "/absolute/audio.wav", r"C:\media\audio.wav"),
)
def test_audio_rejects_non_relative_media_paths(path: str) -> None:
    row = _audio_row()
    row["media_path"] = path
    row["normalized_audio_path"] = path
    row["target_audio_path"] = path

    with pytest.raises(ValidationError, match="project-relative"):
        CuratedAudioRecord.model_validate(row)


def test_audio_rejects_missing_wire_field() -> None:
    row = _audio_row()
    del row["source_run_id"]

    with pytest.raises(ValidationError, match="Field required"):
        CuratedAudioRecord.model_validate(row)


def test_audio_rejects_coercive_nested_privacy_boolean() -> None:
    row = _audio_row()
    clearance = dict(row["privacy_clearance"])
    clearance["remediation_verified"] = "false"
    row["privacy_clearance"] = clearance

    with pytest.raises(ValidationError):
        CuratedAudioRecord.model_validate(row)


def test_audio_rejects_unknown_nested_asset_context_field() -> None:
    row = _audio_row()
    context = dict(row["asset_context"])
    context["raw_html"] = "must not survive"
    row["asset_context"] = context

    with pytest.raises(ValidationError):
        CuratedAudioRecord.model_validate(row)


def test_video_rejects_uncleared_keyframes() -> None:
    row = _video_row()
    row["keyframes"] = [
        {
            "frame_path": "media/frame.jpg",
            "timestamp_seconds": 1.0,
        }
    ]

    with pytest.raises(ValidationError, match="privacy clearance"):
        CuratedVideoRecord.model_validate(row)


def test_curated_audio_round_trip_uses_canonical_fields() -> None:
    record = CuratedAudioRecord.model_validate(_audio_row())

    assert record.transcript_text == "hello"
    assert record.to_dict()["transcript_text"] == "hello"
    assert "transcript" not in record.to_dict()
    assert CuratedAudioRecord.model_validate(record.to_dict()) == record


def test_curated_video_round_trip_uses_canonical_fields() -> None:
    record = CuratedVideoRecord.model_validate(_video_row())

    assert record.transcript_text == "hello"
    assert record.to_dict()["transcript_text"] == "hello"
    assert "transcript" not in record.to_dict()
    assert CuratedVideoRecord.model_validate(record.to_dict()) == record
