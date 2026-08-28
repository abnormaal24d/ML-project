"""Contract tests for the dataset-owned curated snapshot reader."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from config.settings.datasets import (
    CuratedDatasetWriterSettings,
    DatasetPathSettings,
)
from crawler.curation.publishing.dataset_export.curated_dataset_writer import (
    CuratedDatasetWriter,
)
from crawler.curation.publishing.dataset_export.jsonl_writer import JsonlWriter
from crawler.curation.snapshots.manifest import CuratedSnapshotManifest
from mmcrawler_datasets.curated.document import (
    ChunkRecord,
    CuratedDocumentRecord,
)
from mmcrawler_datasets.curated.evidence import PrivacyClearanceRecord
from mmcrawler_datasets.curated.image import CuratedImageRecord
from mmcrawler_datasets.curated.timed_media import (
    CURATED_AUDIO_CONTRACT_SHA256,
    CURATED_VIDEO_CONTRACT_SHA256,
    CuratedAudioRecord,
    CuratedVideoRecord,
)
from mmcrawler_datasets.snapshots.curated import (
    CuratedSnapshot,
    SnapshotContractError,
    read_snapshot,
)
from schemas.versions import CURATED_DATASET_SCHEMA_VERSION


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
                "name": "body",
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


def _document() -> CuratedDocumentRecord:
    return CuratedDocumentRecord(
        schema_version=CURATED_DATASET_SCHEMA_VERSION,
        snapshot_id="snapshot-1",
        document_id="document-1",
        source_run_id="run-1",
        source_fetch_record_id="fetch-1",
        object_id="object-1",
        requested_url="https://example.test/doc",
        final_url="https://example.test/doc",
        normalized_url="https://example.test/doc",
        domain="example.test",
        path="media/doc.txt",
        modality="text",
        language="en",
        title="Example",
        text_path="media/doc.txt",
        markdown_path=None,
        raw_storage_path="raw/doc.txt",
        raw_byte_size=5,
        extracted_char_count=5,
        extracted_token_count_estimate=1,
        boilerplate_ratio=0.1,
        code_block_count=0,
        quality_score=0.9,
        quality_bucket="gold",
        rejection_reason=None,
        content_role="primary",
        discovery_useful=True,
        exact_duplicate_key="document-1",
        near_duplicate_cluster_id=None,
        is_near_duplicate=False,
        license="CC0",
        license_url=None,
        allow_training=True,
        created_at="2026-01-01T00:00:00Z",
        privacy_clearance=PrivacyClearanceRecord.model_validate(_clearance()),
    )


def _chunk() -> ChunkRecord:
    return ChunkRecord(
        schema_version=CURATED_DATASET_SCHEMA_VERSION,
        snapshot_id="snapshot-1",
        chunk_id="chunk-1",
        document_id="document-1",
        chunk_index=0,
        start_char=0,
        end_char=5,
        token_count_estimate=1,
        text="hello",
        language="en",
        title="Example",
        section_path=("section-1",),
        quality_score=0.9,
        exact_duplicate_key="chunk-1",
        near_duplicate_cluster_id=None,
        split="train",
    )


def _image() -> CuratedImageRecord:
    return CuratedImageRecord(
        schema_version=CURATED_DATASET_SCHEMA_VERSION,
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


def _audio() -> CuratedAudioRecord:
    return CuratedAudioRecord.model_validate(_audio_row())


def _video() -> CuratedVideoRecord:
    return CuratedVideoRecord.model_validate(_video_row())


def _common_row(*, media_id: str, media_path: str) -> dict[str, object]:
    return {
        "schema_version": CURATED_DATASET_SCHEMA_VERSION,
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
        "asset_context": {
            "safety_status": "passed",
            "fetch_record_id": "fetch-1",
            "parent_fetch_record_id": "fetch-1",
            "parent_stable_url_id": "url-1",
            "media_identity": "identity-1",
            "fetch_mode": "direct",
            "asset_fetch_mode": "direct",
            "source_page_url": "https://example.test/page",
            "embed_host": "example.test",
        },
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


def _write_snapshot(
    *,
    snapshot_directory: Path,
    dataset_paths: DatasetPathSettings,
    documents: tuple[CuratedDocumentRecord, ...] = (),
    chunks: tuple[ChunkRecord, ...] = (),
    images: tuple[CuratedImageRecord, ...] = (),
    audio: tuple[CuratedAudioRecord, ...] = (),
    video: tuple[CuratedVideoRecord, ...] = (),
) -> CuratedDatasetWriter:
    writer = CuratedDatasetWriter(
        settings=CuratedDatasetWriterSettings(),
        dataset_paths=dataset_paths,
        snapshot_directory=snapshot_directory,
        jsonl_writer=JsonlWriter(),
    )
    writer.write_documents(documents=documents)
    writer.write_chunks(chunks=chunks)
    writer.write_images(images=images)
    writer.write_audio(records=audio)
    writer.write_video(records=video)
    CuratedSnapshotManifest.write(
        snapshot_directory / dataset_paths.snapshot_manifest_filename,
        schema_version=CURATED_DATASET_SCHEMA_VERSION,
        curated_audio_contract_sha256=CURATED_AUDIO_CONTRACT_SHA256,
        curated_video_contract_sha256=CURATED_VIDEO_CONTRACT_SHA256,
    )
    return writer


def test_read_snapshot_round_trips_all_modalities(tmp_path: Path) -> None:
    dataset_paths = DatasetPathSettings()
    snapshot_directory = tmp_path / "snapshot"
    expected_documents = (_document(),)
    expected_chunks = (_chunk(),)
    expected_images = (_image(),)
    expected_audio = (_audio(),)
    expected_video = (_video(),)
    _write_snapshot(
        snapshot_directory=snapshot_directory,
        dataset_paths=dataset_paths,
        documents=expected_documents,
        chunks=expected_chunks,
        images=expected_images,
        audio=expected_audio,
        video=expected_video,
    )

    snapshot = read_snapshot(
        dataset_paths=dataset_paths,
        snapshot_directory=snapshot_directory,
    )

    assert isinstance(snapshot, CuratedSnapshot)
    assert snapshot.documents == expected_documents
    assert snapshot.chunks == expected_chunks
    assert snapshot.images == expected_images
    assert snapshot.audio == expected_audio
    assert snapshot.video == expected_video
    assert snapshot.documents_by_id == {"document-1": expected_documents[0]}


def test_read_snapshot_accepts_existing_empty_entity_files(
    tmp_path: Path,
) -> None:
    dataset_paths = DatasetPathSettings()
    snapshot_directory = tmp_path / "snapshot"
    _write_snapshot(
        snapshot_directory=snapshot_directory,
        dataset_paths=dataset_paths,
    )

    snapshot = read_snapshot(
        dataset_paths=dataset_paths,
        snapshot_directory=snapshot_directory,
    )

    assert snapshot.documents == ()
    assert snapshot.chunks == ()
    assert snapshot.images == ()
    assert snapshot.audio == ()
    assert snapshot.video == ()


def test_read_snapshot_rejects_missing_entity_file(tmp_path: Path) -> None:
    dataset_paths = DatasetPathSettings()
    snapshot_directory = tmp_path / "snapshot"
    snapshot_directory.mkdir(parents=True)
    CuratedSnapshotManifest.write(
        snapshot_directory / dataset_paths.snapshot_manifest_filename,
        schema_version=CURATED_DATASET_SCHEMA_VERSION,
        curated_audio_contract_sha256=CURATED_AUDIO_CONTRACT_SHA256,
        curated_video_contract_sha256=CURATED_VIDEO_CONTRACT_SHA256,
    )

    with pytest.raises(SnapshotContractError, match="missing"):
        read_snapshot(
            dataset_paths=dataset_paths,
            snapshot_directory=snapshot_directory,
        )


def test_read_snapshot_rejects_corrupt_entity_row(tmp_path: Path) -> None:
    dataset_paths = DatasetPathSettings()
    snapshot_directory = tmp_path / "snapshot"
    _write_snapshot(
        snapshot_directory=snapshot_directory,
        dataset_paths=dataset_paths,
    )
    entities = snapshot_directory / dataset_paths.curated_entities_directory
    (entities / dataset_paths.curated_audio_filename).write_text(
        '{"schema_version":"3.0"\n', encoding="utf-8"
    )

    with pytest.raises(SnapshotContractError, match="invalid JSONL"):
        read_snapshot(
            dataset_paths=dataset_paths,
            snapshot_directory=snapshot_directory,
        )


def test_read_snapshot_rejects_contract_digest_mismatch(
    tmp_path: Path,
) -> None:
    dataset_paths = DatasetPathSettings()
    snapshot_directory = tmp_path / "snapshot"
    _write_snapshot(
        snapshot_directory=snapshot_directory,
        dataset_paths=dataset_paths,
    )
    CuratedSnapshotManifest.write(
        snapshot_directory / dataset_paths.snapshot_manifest_filename,
        schema_version=CURATED_DATASET_SCHEMA_VERSION,
        curated_audio_contract_sha256="0" * 64,
        curated_video_contract_sha256=CURATED_VIDEO_CONTRACT_SHA256,
    )

    with pytest.raises(
        SnapshotContractError, match="contract digest mismatch"
    ):
        read_snapshot(
            dataset_paths=dataset_paths,
            snapshot_directory=snapshot_directory,
        )


def test_read_snapshot_rejects_wrong_manifest_schema(tmp_path: Path) -> None:
    dataset_paths = DatasetPathSettings()
    snapshot_directory = tmp_path / "snapshot"
    _write_snapshot(
        snapshot_directory=snapshot_directory,
        dataset_paths=dataset_paths,
    )
    CuratedSnapshotManifest.write(
        snapshot_directory / dataset_paths.snapshot_manifest_filename,
        schema_version="9.9",
        curated_audio_contract_sha256=CURATED_AUDIO_CONTRACT_SHA256,
        curated_video_contract_sha256=CURATED_VIDEO_CONTRACT_SHA256,
    )

    with pytest.raises(SnapshotContractError, match="schema mismatch"):
        read_snapshot(
            dataset_paths=dataset_paths,
            snapshot_directory=snapshot_directory,
        )


def test_read_snapshot_rejects_wrong_entity_schema(tmp_path: Path) -> None:
    dataset_paths = DatasetPathSettings()
    snapshot_directory = tmp_path / "snapshot"
    _write_snapshot(
        snapshot_directory=snapshot_directory,
        dataset_paths=dataset_paths,
    )
    entities = snapshot_directory / dataset_paths.curated_entities_directory
    (entities / dataset_paths.curated_images_filename).write_text(
        '{"schema_version":"9.9","image_id":"image-1"}\n',
        encoding="utf-8",
    )

    with pytest.raises(SnapshotContractError, match="schema mismatch"):
        read_snapshot(
            dataset_paths=dataset_paths,
            snapshot_directory=snapshot_directory,
        )


def test_read_snapshot_rejects_missing_manifest(tmp_path: Path) -> None:
    dataset_paths = DatasetPathSettings()
    snapshot_directory = tmp_path / "snapshot"
    _write_snapshot(
        snapshot_directory=snapshot_directory,
        dataset_paths=dataset_paths,
    )
    (snapshot_directory / dataset_paths.snapshot_manifest_filename).unlink()

    with pytest.raises(SnapshotContractError, match="cannot stat"):
        read_snapshot(
            dataset_paths=dataset_paths,
            snapshot_directory=snapshot_directory,
        )
