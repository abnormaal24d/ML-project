from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from config.settings.datasets import (
    CuratedDatasetWriterSettings,
    DatasetPathSettings,
)
from crawler.curation.media.cleared_timed_media_records import (
    build_privacy_cleared_timed_media,
)
from crawler.curation.publishing.dataset_export.curated_dataset_writer import (
    CuratedDatasetWriter,
)
from crawler.curation.publishing.dataset_export.jsonl_writer import JsonlWriter
from crawler.curation.snapshots.manifest import CuratedSnapshotManifest
from mmcrawler_datasets.assembly.audio import build_audio_samples
from mmcrawler_datasets.assembly.video import build_video_samples
from mmcrawler_datasets.curated.timed_media import (
    CURATED_AUDIO_CONTRACT_SHA256,
    CURATED_VIDEO_CONTRACT_SHA256,
    CuratedAudioRecord,
    CuratedVideoRecord,
)
from mmcrawler_datasets.schema import SplitAssigner
from mmcrawler_datasets.snapshots.curated import read_snapshot
from preprocessing.preprocessed_media import (
    PreprocessedAudio,
    PreprocessedVideo,
)
from preprocessing.preprocessing_quality import PreprocessingQualityResult
from preprocessing.privacy.clearance import (
    ApprovedTextField,
    PrivacyClearance,
    PrivacyClearanceStatus,
)


class _PassThroughAudioMaterializer:
    def __init__(self, output_root: Path) -> None:
        self._output_root = output_root

    def materialize(self, sample, *, project_root: Path):
        target = self._output_root / "target_audio_tokens.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"audio tokens")
        return replace(
            sample,
            task_target=replace(
                sample.task_target,
                target_audio_tokens_path=(
                    target.relative_to(project_root).as_posix()
                ),
            ),
        )


class _PassThroughVideoMaterializer:
    def materialize(self, sample, *, project_root: Path):
        del project_root
        return sample


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _clearance(*, media: bytes, transcript: str) -> PrivacyClearance:
    text_digest = _digest(transcript.encode())
    media_digest = _digest(media)
    return PrivacyClearance(
        status=PrivacyClearanceStatus.APPROVED,
        input_digest=media_digest,
        output_digest=media_digest,
        checked_fields=frozenset({"media_decode"}),
        required_fields=frozenset({"media_decode"}),
        approved_text_fields=(
            ApprovedTextField(
                name="transcript_text",
                value=transcript,
                input_digest=text_digest,
                output_digest=text_digest,
            ),
        ),
        inspection_digest=_digest(b"inspection" + media),
        assessment_digest=_digest(b"assessment" + media),
    )


def _raw_entry(*, source_id: str, media: bytes, content_type: str):
    record = SimpleNamespace(
        fetch_record_id=source_id,
        parent_fetch_record_id=None,
        parent_stable_url_id=None,
        parent_url=None,
        media_identity=f"identity-{source_id}",
        fetch_mode="full",
        asset_fetch_mode="full",
        is_complete_payload=True,
        source_page_url=None,
        embed_host=None,
        governance={
            "training": {"allowed": True},
            "license": {"expression": "CC0"},
        },
        object_id=f"object-{source_id}",
        run_id="run-1",
        observed_bytes=len(media),
        source_content_length=len(media),
        source_content_type=content_type,
        fetch_duration_seconds=0.1,
        payload_sha256=_digest(media),
    )
    return SimpleNamespace(record=record)


def _quality(modality: str) -> PreprocessingQualityResult:
    return PreprocessingQualityResult(
        score=0.9,
        bucket="gold",
        rejection_reason=None,
        token_count_estimate=1,
        modality=modality,
    )


def test_producer_writer_reader_and_training_assembly_round_trip(
    tmp_path: Path,
) -> None:
    transcript = "Public transcript"
    audio_bytes = b"canonical audio payload"
    video_bytes = b"canonical video payload"
    audio_path = tmp_path / "audio.wav"
    video_path = tmp_path / "video.mp4"
    audio_path.write_bytes(audio_bytes)
    video_path.write_bytes(video_bytes)

    audio_item = PreprocessedAudio(
        media_id="audio-1",
        source_id="source-audio",
        source_url="https://example.test/audio.wav",
        normalized_url="https://example.test/audio.wav",
        domain="example.test",
        media_path=str(audio_path),
        mime_type="audio/wav",
        duration_seconds=2.0,
        transcript_text=transcript,
        transcript_language="en",
        transcript_segments=(
            {"start_ms": 0, "end_ms": 2000, "text": transcript},
        ),
        quality=_quality("audio"),
        normalized_audio_path=str(audio_path),
        sample_rate=16_000,
        channels=1,
        loudness_lufs=-18.0,
        safety_status="passed",
        privacy_clearance=_clearance(
            media=audio_bytes,
            transcript=transcript,
        ),
    )
    video_item = PreprocessedVideo(
        media_id="video-1",
        source_id="source-video",
        source_url="https://example.test/video.mp4",
        normalized_url="https://example.test/video.mp4",
        domain="example.test",
        media_path=str(video_path),
        mime_type="video/mp4",
        duration_seconds=3.0,
        width=640,
        height=480,
        transcript_text=transcript,
        transcript_language="en",
        transcript_segments=(
            {
                "start_seconds": 0.0,
                "end_seconds": 3.0,
                "text": transcript,
            },
        ),
        frame_ocr_text=None,
        keyframe_paths=(),
        quality=_quality("video"),
        normalized_video_path=str(video_path),
        safety_status="passed",
        privacy_clearance=_clearance(
            media=video_bytes,
            transcript=transcript,
        ),
    )

    with pytest.raises(TypeError, match="context_score must be numeric"):
        build_privacy_cleared_timed_media(
            snapshot_id="snapshot-1",
            schema_version="3.0",
            modality="audio",
            raw_entries=(
                _raw_entry(
                    source_id="source-audio",
                    media=audio_bytes,
                    content_type="audio/wav",
                ),
            ),
            documents=(),
            preprocessed=(
                replace(
                    audio_item,
                    alignment_signals={"context_score": "invalid"},
                ),
            ),
            project_root=tmp_path,
        )

    audio_records = build_privacy_cleared_timed_media(
        snapshot_id="snapshot-1",
        schema_version="3.0",
        modality="audio",
        raw_entries=(
            _raw_entry(
                source_id="source-audio",
                media=audio_bytes,
                content_type="audio/wav",
            ),
        ),
        documents=(),
        preprocessed=(audio_item,),
        project_root=tmp_path,
    )
    video_records = build_privacy_cleared_timed_media(
        snapshot_id="snapshot-1",
        schema_version="3.0",
        modality="video",
        raw_entries=(
            _raw_entry(
                source_id="source-video",
                media=video_bytes,
                content_type="video/mp4",
            ),
        ),
        documents=(),
        preprocessed=(video_item,),
        project_root=tmp_path,
    )

    assert set(audio_records[0].to_dict()) == set(
        CuratedAudioRecord.model_fields
    )
    assert set(video_records[0].to_dict()) == set(
        CuratedVideoRecord.model_fields
    )

    snapshot_directory = tmp_path / "snapshot"
    dataset_paths = DatasetPathSettings()
    writer = CuratedDatasetWriter(
        settings=CuratedDatasetWriterSettings(),
        dataset_paths=dataset_paths,
        snapshot_directory=snapshot_directory,
        jsonl_writer=JsonlWriter(),
    )
    writer.write_documents(documents=())
    writer.write_chunks(chunks=())
    writer.write_images(images=())
    writer.write_audio(records=audio_records)
    writer.write_video(records=video_records)
    CuratedSnapshotManifest.write(
        snapshot_directory / dataset_paths.snapshot_manifest_filename,
        schema_version="3.0",
        curated_audio_contract_sha256=CURATED_AUDIO_CONTRACT_SHA256,
        curated_video_contract_sha256=CURATED_VIDEO_CONTRACT_SHA256,
    )

    snapshot = read_snapshot(
        dataset_paths=dataset_paths,
        snapshot_directory=snapshot_directory,
    )

    assert snapshot.audio == audio_records
    assert snapshot.video == video_records
    assert snapshot.audio[0].context_score is None
    assert snapshot.video[0].context_score is None
    assert snapshot.audio[0].transcript_segments[0].start_seconds == 0.0
    assert snapshot.audio[0].transcript_segments[0].end_seconds == 2.0

    split_assigner = SplitAssigner(
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0,
    )
    audio_samples = build_audio_samples(
        snapshot.audio,
        {},
        {},
        split_assigner=split_assigner,
        require_allow_training=True,
        snapshot_id="snapshot-1",
        snapshot_directory=snapshot_directory,
        materialization_directory=tmp_path / "materialized",
        project_root=tmp_path,
        rejections=[],
        materializer_factory=_PassThroughAudioMaterializer,
        require_transcript_for_audio_text_pair=True,
    )
    video_samples = build_video_samples(
        snapshot.video,
        {},
        {},
        split_assigner=split_assigner,
        require_allow_training=True,
        snapshot_id="snapshot-1",
        snapshot_directory=snapshot_directory,
        materialization_directory=tmp_path / "materialized",
        project_root=tmp_path,
        rejections=[],
        materializer_factory=lambda _output: _PassThroughVideoMaterializer(),
    )

    assert isinstance(audio_samples[0].privacy_clearance, PrivacyClearance)
    assert isinstance(video_samples[0].privacy_clearance, PrivacyClearance)
    assert any(sample.modality == "audio" for sample in audio_samples)
    assert any(sample.modality == "video" for sample in video_samples)


def test_typed_writer_rejects_unvalidated_mappings(tmp_path: Path) -> None:
    writer = CuratedDatasetWriter(
        settings=CuratedDatasetWriterSettings(),
        dataset_paths=DatasetPathSettings(),
        snapshot_directory=tmp_path / "snapshot",
        jsonl_writer=JsonlWriter(),
    )

    with pytest.raises(TypeError, match="CuratedAudioRecord"):
        writer.write_audio(records=({"media_id": "untyped"},))
