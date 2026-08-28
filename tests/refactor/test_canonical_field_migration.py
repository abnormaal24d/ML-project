from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from crawler.analysis.enrichment.audio.audio_analyzer import (
    AudioAnalysisResult,
)
from crawler.analysis.enrichment.video.video_analysis_result import (
    VideoAnalysisResult,
)
from crawler.analysis.enrichment.video.video_enrichment_payload import (
    _metadata_payload,
)
from crawler.processing.handlers.audio_handler import AudioHandler
from evaluator.leakage.indexing import build_index
from logger.formatters import PlainFormatter
from mmcrawler_datasets.assembly.text_pairing import pairability_score
from mmcrawler_datasets.curated.timed_media import (
    CuratedAudioRecord,
    CuratedVideoRecord,
)
from mmcrawler_datasets.curated.training_projection import (
    TrainingTimedMediaInput,
)
from preprocessing.media.speech.speaker_diarizer import (
    _coerce_backend_segments,
    _segments_from_transcript_hints,
)
from preprocessing.media.video.video_scene_analysis import (
    segments_overlap,
)


def _leakage_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "dataset_id": "dataset",
        "sample_id": "sample",
        "partition": "train",
        "lineage_key": "lineage",
        "modality": "text",
        "content_hash": "a" * 64,
    }
    row.update(overrides)
    return row


def test_leakage_requires_canonical_modality_field() -> None:
    row = _leakage_row(content_type="text")
    row.pop("modality")

    with pytest.raises(ValueError, match="canonical 'modality'"):
        build_index((row,), max_records=10)


def test_epoch_formatter_ignores_removed_loss_alias() -> None:
    record = logging.LogRecord(
        name="training.runtime",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="multimodal_training_epoch_completed",
        args=(),
        exc_info=None,
    )
    fields = {
        "epoch": 1,
        "epochs": 2,
        "loss": 9.5,
        "val_loss": 2.0,
        "test_loss": 3.0,
        "batches": 4,
        "learning_rate": 0.001,
        "trend": "stable",
    }
    record.__dict__.update(fields)
    record.__dict__["_project_event"] = record.msg
    record.__dict__["_project_field_keys"] = tuple(fields)

    rendered = PlainFormatter(compact_context=True).format(record)

    assert "train_loss=None" in rendered
    assert "train_loss=9.5" not in rendered


def test_audio_enrichment_producer_emits_only_prefixed_fields() -> None:
    handler = object.__new__(AudioHandler)
    handler._settings = SimpleNamespace(  # type: ignore[attr-defined]
        extract_metadata=True
    )
    analysis = AudioAnalysisResult(
        payload_path="audio.wav",
        metadata={
            "duration_seconds": 2.5,
            "sample_rate": 16_000,
            "channels": 2,
            "bitrate": 256_000,
            "container": "wav",
        },
        duration_seconds=2.5,
        sample_rate=16_000,
        channels=2,
        bitrate=256_000,
    )

    payload = handler._build_audio_enrichment_fields(
        analysis=analysis,
        run_transcription=False,
    )

    assert payload["audio_duration_seconds"] == 2.5
    assert payload["audio_sample_rate"] == 16_000
    assert payload["audio_channels"] == 2
    assert payload["audio_bitrate"] == 256_000
    assert payload["container"] == "wav"
    for removed in ("duration_seconds", "sample_rate", "channels", "bitrate"):
        assert removed not in payload


def test_audio_contract_rejects_removed_generic_fields() -> None:
    """The curated audio contract forbids generic duration_seconds aliases."""
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        CuratedAudioRecord.model_validate(
            {"duration_seconds": 1.0, "audio_duration_seconds": 1.0}
        )


def test_video_contract_rejects_removed_generic_fields() -> None:
    """The curated video contract forbids generic width aliases."""
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        CuratedVideoRecord.model_validate({"width": 640, "video_width": 640})


def test_video_enrichment_producer_emits_only_prefixed_fields() -> None:
    analysis = VideoAnalysisResult(
        metadata={
            "duration_seconds": 3.0,
            "width": 1920,
            "height": 1080,
            "fps": 25.0,
            "frame_count": 75,
            "container": "mp4",
        },
        metadata_status="ok",
    )

    payload = _metadata_payload(analysis=analysis)

    assert payload["video_duration_seconds"] == 3.0
    assert payload["video_width"] == 1920
    assert payload["video_height"] == 1080
    assert payload["video_fps"] == 25.0
    assert payload["video_frame_count"] == 75
    assert payload["container"] == "mp4"
    for removed in (
        "duration_seconds",
        "width",
        "height",
        "fps",
        "frame_count",
    ):
        assert removed not in payload


def test_pairability_uses_only_asset_fetch_mode() -> None:
    old_only = pairability_score(
        media_record={"fetch_mode": "full_payload"},
        parent_text=None,
    )
    canonical = pairability_score(
        media_record={"asset_fetch_mode": "full_payload"},
        parent_text=None,
    )

    assert old_only == 0.0
    assert canonical == 0.1


def test_pairability_asset_context_asset_fetch_mode_counts() -> None:
    context_only = pairability_score(
        media_record={
            "asset_context": {
                "asset_fetch_mode": "full_payload",
            },
        },
        parent_text=None,
    )

    assert context_only == 0.1


def test_pairability_top_level_transcript_yields_score() -> None:
    transcript_only = pairability_score(
        media_record={"transcript_text": "some transcript text"},
        parent_text=None,
    )

    assert transcript_only == 0.35


def test_pairability_top_level_frame_ocr_yields_score() -> None:
    ocr_only = pairability_score(
        media_record={"frame_ocr_text": "some ocr text"},
        parent_text=None,
    )

    assert ocr_only == 0.35


def test_pairability_legacy_fetch_mode_does_not_count() -> None:
    legacy_only = pairability_score(
        media_record={"fetch_mode": "full_payload"},
        parent_text=None,
    )

    assert legacy_only == 0.0


def test_transcript_hint_diarization_requires_canonical_segment_fields() -> (
    None
):
    old_segment = {
        "speaker": "speaker-1",
        "start": 1.0,
        "end": 2.0,
    }
    canonical_segment = {
        "speaker_id": "speaker-1",
        "start_seconds": 1.0,
        "end_seconds": 2.0,
    }

    assert _segments_from_transcript_hints([old_segment]) == []
    assert _segments_from_transcript_hints([canonical_segment]) == [
        {
            "speaker_id": "speaker-1",
            "start_seconds": 1.0,
            "end_seconds": 2.0,
            "confidence": None,
            "overlapping_speech": False,
        }
    ]


def test_mapping_diarization_backend_requires_canonical_segment_fields() -> (
    None
):
    old_result = {
        "speaker_segments": [
            {"speaker": "speaker-1", "start": 1.0, "end": 2.0}
        ]
    }
    canonical_result = {
        "segments": [
            {
                "speaker_id": "speaker-1",
                "start_seconds": 1.0,
                "end_seconds": 2.0,
            }
        ]
    }

    assert _coerce_backend_segments(old_result) == []
    assert _coerce_backend_segments(canonical_result) == [
        {
            "speaker_id": "speaker-1",
            "start_seconds": 1.0,
            "end_seconds": 2.0,
            "confidence": None,
            "overlapping_speech": False,
            "embedding": None,
        }
    ]


def test_video_scene_overlap_uses_canonical_timing_fields_only() -> None:
    assert not segments_overlap(
        start_seconds=10.2,
        end_seconds=10.3,
        segment={"start": 10.0, "end": 11.0},
    )
    assert segments_overlap(
        start_seconds=10.2,
        end_seconds=10.3,
        segment={"start_seconds": 10.0, "end_seconds": 11.0},
    )


def test_training_timed_media_input_to_dict_preserves_fields() -> None:
    record = TrainingTimedMediaInput(
        media_id="media-1",
        media_path="/tmp/media.mp3",
        transcript_text=None,
        source_url="https://example.test",
        allow_training=True,
        parent_document_id=None,
        modality="audio",
        domain="example.test",
        page_title=None,
        transcript_preview=None,
        surrounding_text=None,
        html_context=None,
        media_mime_type="audio/mpeg",
        transcript_segments=(),
        license=None,
        license_url=None,
        governance_note=None,
        robots_status="unknown",
        terms_source=None,
        usage_rules="unknown",
        asset_fetch_mode="full_payload",
        is_complete_payload=True,
        near_duplicate_cluster_id=None,
        media_fingerprint=None,
        privacy_clearance=SimpleNamespace(  # type: ignore[attr-defined]
            permits_training=True,
            approved_text=lambda _: _,
            to_dict=lambda: {},
        ),
        asset_context=SimpleNamespace(  # type: ignore[attr-defined]
            to_dict=lambda: {},
        ),
        language=None,
        context_score=None,
        quality_score=0.5,
        trainable=True,
        curated_rejection_reason=None,
    )
    result = record.to_dict()
    assert result["asset_fetch_mode"] == "full_payload"
    assert result["is_complete_payload"] is True
    assert result["language"] is None
    assert result["context_score"] is None
    assert result["quality_score"] == 0.5
    assert result["trainable"] is True
    assert result["curated_rejection_reason"] is None


def test_modality_quality_threshold_is_enforced() -> None:
    from mmcrawler_datasets.selection.quality import quality_reject
    from mmcrawler_datasets.training_samples.models import TrainingSample
    from mmcrawler_datasets.training_samples.targets import TrainingTaskTarget

    sample = TrainingSample(
        schema_version="1.0",
        sample_id="sample-1",
        snapshot_id="snap-1",
        split="train",
        modality="image",
        task_target=TrainingTaskTarget(
            task_type="image_text_pair",
            task_family="pair",
            target_text="text",
            positive_id="obj-1",
        ),
        text="a cat",
        quality_score=0.3,
        context_score=None,
        pairability_score=0.9,
        pair_source="caption",
    )
    from config.settings.datasets import (
        DatasetValidatorSettings,
        TrainingSnapshotAssemblerSettings,
    )

    settings = TrainingSnapshotAssemblerSettings(
        min_pairability_score=0.0,
        min_alignment_score=0.0,
        min_caption_quality_score=0.0,
        language_rules="multilingual",
        accepted_languages=("en",),
        min_language_confidence=0.0,
        dataset_version_prefix="v",
        processing_version="1",
    )
    validator_settings = DatasetValidatorSettings(
        min_quality_score_by_modality={"image": 0.5},
        min_context_score_by_modality={},
        min_alignment_score_by_modality={},
        min_alignment_score_by_task={},
        require_allow_training=False,
        min_train_samples=0,
        min_val_samples=0,
        min_test_samples=0,
        model_min_total_samples=0,
        model_min_train_samples=0,
        model_min_val_samples=0,
        model_min_test_samples=0,
        model_min_training_batches=0,
        model_max_test_train_loss_ratio=None,
        max_test_train_loss_ratio=None,
        require_dataset_card=False,
        require_model_card=False,
        require_pii_passed=False,
        require_safety_passed=False,
        require_known_license=False,
        require_license_evidence=False,
        require_license_url_or_terms=False,
        require_evaluation_metrics=False,
        min_evaluation_metrics={},
    )
    reason = quality_reject(sample, settings, validator_settings)
    assert reason is not None


def test_context_threshold_is_enforced() -> None:
    from mmcrawler_datasets.selection.quality import quality_reject
    from mmcrawler_datasets.training_samples.models import TrainingSample
    from mmcrawler_datasets.training_samples.targets import TrainingTaskTarget

    sample = TrainingSample(
        schema_version="1.0",
        sample_id="sample-1",
        snapshot_id="snap-1",
        split="train",
        modality="image",
        task_target=TrainingTaskTarget(
            task_type="image_text_pair",
            task_family="pair",
            target_text="text",
            positive_id="obj-1",
        ),
        text="a cat",
        quality_score=0.9,
        context_score=None,
        pairability_score=0.9,
        pair_source="caption",
    )
    from config.settings.datasets import (
        DatasetValidatorSettings,
        TrainingSnapshotAssemblerSettings,
    )

    settings = TrainingSnapshotAssemblerSettings(
        min_pairability_score=0.0,
        min_alignment_score=0.0,
        min_caption_quality_score=0.0,
        language_rules="multilingual",
        accepted_languages=("en",),
        min_language_confidence=0.0,
        dataset_version_prefix="v",
        processing_version="1",
    )
    validator_settings = DatasetValidatorSettings(
        min_quality_score_by_modality={},
        min_context_score_by_modality={"image": 0.35},
        min_alignment_score_by_modality={},
        min_alignment_score_by_task={},
        require_allow_training=False,
        min_train_samples=0,
        min_val_samples=0,
        min_test_samples=0,
        model_min_total_samples=0,
        model_min_train_samples=0,
        model_min_val_samples=0,
        model_min_test_samples=0,
        model_min_training_batches=0,
        model_max_test_train_loss_ratio=None,
        max_test_train_loss_ratio=None,
        require_dataset_card=False,
        require_model_card=False,
        require_pii_passed=False,
        require_safety_passed=False,
        require_known_license=False,
        require_license_evidence=False,
        require_license_url_or_terms=False,
        require_evaluation_metrics=False,
        min_evaluation_metrics={},
    )
    reason = quality_reject(sample, settings, validator_settings)
    assert reason is not None


def test_task_alignment_threshold_is_enforced() -> None:
    from mmcrawler_datasets.selection.quality import quality_reject
    from mmcrawler_datasets.training_samples.models import TrainingSample
    from mmcrawler_datasets.training_samples.targets import TrainingTaskTarget

    sample = TrainingSample(
        schema_version="1.0",
        sample_id="sample-1",
        snapshot_id="snap-1",
        split="train",
        modality="image",
        task_target=TrainingTaskTarget(
            task_type="image_text_pair",
            task_family="pair",
            target_text="text",
            positive_id="obj-1",
            alignment_score=0.2,
        ),
        text="a cat",
        quality_score=0.9,
        context_score=0.9,
        pairability_score=0.9,
        pair_source="caption",
    )
    from config.settings.datasets import (
        DatasetValidatorSettings,
        TrainingSnapshotAssemblerSettings,
    )

    settings = TrainingSnapshotAssemblerSettings(
        min_pairability_score=0.0,
        min_alignment_score=0.0,
        min_caption_quality_score=0.0,
        language_rules="multilingual",
        accepted_languages=("en",),
        min_language_confidence=0.0,
        dataset_version_prefix="v",
        processing_version="1",
    )
    validator_settings = DatasetValidatorSettings(
        min_quality_score_by_modality={},
        min_context_score_by_modality={},
        min_alignment_score_by_modality={},
        min_alignment_score_by_task={"image_text_pair": 0.5},
        require_allow_training=False,
        min_train_samples=0,
        min_val_samples=0,
        min_test_samples=0,
        model_min_total_samples=0,
        model_min_train_samples=0,
        model_min_val_samples=0,
        model_min_test_samples=0,
        model_min_training_batches=0,
        model_max_test_train_loss_ratio=None,
        max_test_train_loss_ratio=None,
        require_dataset_card=False,
        require_model_card=False,
        require_pii_passed=False,
        require_safety_passed=False,
        require_known_license=False,
        require_license_evidence=False,
        require_license_url_or_terms=False,
        require_evaluation_metrics=False,
        min_evaluation_metrics={},
    )
    reason = quality_reject(sample, settings, validator_settings)
    assert reason is not None


def test_strengest_of_general_modality_and_task_alignment_threshold() -> None:
    from mmcrawler_datasets.selection.quality import quality_reject
    from mmcrawler_datasets.training_samples.models import TrainingSample
    from mmcrawler_datasets.training_samples.targets import TrainingTaskTarget

    sample = TrainingSample(
        schema_version="1.0",
        sample_id="sample-1",
        snapshot_id="snap-1",
        split="train",
        modality="image",
        task_target=TrainingTaskTarget(
            task_type="image_text_pair",
            task_family="pair",
            target_text="text",
            positive_id="obj-1",
            alignment_score=0.4,
        ),
        text="a cat",
        quality_score=0.9,
        context_score=0.9,
        pairability_score=0.9,
        pair_source="caption",
    )
    from config.settings.datasets import (
        DatasetValidatorSettings,
        TrainingSnapshotAssemblerSettings,
    )

    settings = TrainingSnapshotAssemblerSettings(
        min_pairability_score=0.0,
        min_alignment_score=0.3,
        min_caption_quality_score=0.0,
        language_rules="multilingual",
        accepted_languages=("en",),
        min_language_confidence=0.0,
        dataset_version_prefix="v",
        processing_version="1",
    )
    validator_settings = DatasetValidatorSettings(
        min_quality_score_by_modality={},
        min_context_score_by_modality={},
        min_alignment_score_by_modality={"image": 0.5},
        min_alignment_score_by_task={},
        require_allow_training=False,
        min_train_samples=0,
        min_val_samples=0,
        min_test_samples=0,
        model_min_total_samples=0,
        model_min_train_samples=0,
        model_min_val_samples=0,
        model_min_test_samples=0,
        model_min_training_batches=0,
        model_max_test_train_loss_ratio=None,
        max_test_train_loss_ratio=None,
        require_dataset_card=False,
        require_model_card=False,
        require_pii_passed=False,
        require_safety_passed=False,
        require_known_license=False,
        require_license_evidence=False,
        require_license_url_or_terms=False,
        require_evaluation_metrics=False,
        min_evaluation_metrics={},
    )
    reason = quality_reject(sample, settings, validator_settings)
    assert reason is not None


def test_document_without_context_score_is_not_affected_by_candidate_document_context_threshold() -> (
    None
):
    from mmcrawler_datasets.selection.quality import quality_reject
    from mmcrawler_datasets.training_samples.fingerprints import (
        ContentFingerprints,
    )
    from mmcrawler_datasets.training_samples.models import TrainingSample
    from mmcrawler_datasets.training_samples.targets import TrainingTaskTarget

    sample = TrainingSample(
        schema_version="1.0",
        sample_id="sample-1",
        snapshot_id="snap-1",
        split="train",
        modality="document",
        task_target=TrainingTaskTarget(
            task_type="document_text_pair",
            task_family="pair",
            target_text="text",
            positive_id="obj-1",
        ),
        text="a document",
        quality_score=0.9,
        context_score=None,
        pairability_score=0.9,
        pair_source="text",
        content_fingerprints=ContentFingerprints(
            emitted_text_sha256="a" * 64,
            normalized_text_sha256="b" * 64,
            text_shingle_profile=("token1", "token2"),
            image_ahash=None,
            image_dhash=None,
            image_phash=None,
            audio_chromaprint=None,
            video_keyframe_phashes=None,
            document_layout_sha256="c" * 64,
            document_page_phashes=None,
        ),
    )
    from config.settings.datasets import (
        DatasetValidatorSettings,
        TrainingSnapshotAssemblerSettings,
    )

    settings = TrainingSnapshotAssemblerSettings(
        min_pairability_score=0.0,
        min_alignment_score=0.0,
        min_caption_quality_score=0.0,
        language_rules="multilingual",
        accepted_languages=("en",),
        min_language_confidence=0.0,
        dataset_version_prefix="v",
        processing_version="1",
    )
    validator_settings = DatasetValidatorSettings(
        min_quality_score_by_modality={},
        min_context_score_by_modality={},
        min_alignment_score_by_modality={},
        min_alignment_score_by_task={},
        require_allow_training=False,
        min_train_samples=0,
        min_val_samples=0,
        min_test_samples=0,
        model_min_total_samples=0,
        model_min_train_samples=0,
        model_min_val_samples=0,
        model_min_test_samples=0,
        model_min_training_batches=0,
        model_max_test_train_loss_ratio=None,
        max_test_train_loss_ratio=None,
        require_dataset_card=False,
        require_model_card=False,
        require_pii_passed=False,
        require_safety_passed=False,
        require_known_license=False,
        require_license_evidence=False,
        require_license_url_or_terms=False,
        require_evaluation_metrics=False,
        min_evaluation_metrics={},
    )
    reason = quality_reject(sample, settings, validator_settings)
    assert reason is None
