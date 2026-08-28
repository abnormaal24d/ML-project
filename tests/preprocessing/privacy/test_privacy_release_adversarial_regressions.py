"""Fail-closed regressions discovered during the privacy release recheck."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from config.collection.modality_acceptance import ImageAcceptanceSettings
from config.media_toolchain import MediaToolchainSettings
from config.preprocessing.media_settings import (
    ImageValidationSettings,
    MediaPrivacySettings,
)
from config.preprocessing.text_settings import PrivacyDetectionSettings
from crawler.curation.media.cleared_image_records import (
    build_privacy_cleared_images,
)
from mmcrawler_datasets.selection.privacy_clearance import (
    verify_sample_clearance,
)
from mmcrawler_datasets.training_samples.models import (
    TrainingSample,
    TrainingTaskTarget,
    TrainingTextSpan,
)
from preprocessing.media.adapters.embedded_metadata import (
    FfmpegEmbeddedMetadataAdapter,
)
from preprocessing.media.image.image_preprocessor import ImagePreprocessor
from preprocessing.media.privacy_inspection import inspect_media_privacy
from preprocessing.media.transcript_segment_normalizer import (
    normalize_segments,
)
from preprocessing.preprocessed_media import PreprocessedImage
from preprocessing.preprocessing_input import (
    LanguageEvidence,
    PreprocessingInput,
)
from preprocessing.preprocessing_quality import PreprocessingQualityResult
from preprocessing.privacy.clearance import (
    ApprovedTextField,
    PrivacyClearance,
    PrivacyClearanceStatus,
)
from preprocessing.privacy.field_inspection import (
    inspect_text_fields_for_release,
)
from preprocessing.privacy.inspection.content_readers.text_content import (
    TextContent,
)
from preprocessing.privacy.inspection.detector_registry import DetectorRegistry
from preprocessing.privacy.inspection.inspect_text import inspect_text
from preprocessing.privacy.inspection.inspection_coverage import (
    InspectionCoverage,
    ranges_cover_duration,
)
from preprocessing.privacy.inspection.inspection_result import InspectionResult
from preprocessing.privacy.inspection.local_content_factories import (
    LocalImagePrivacyContentFactory,
    LocalVideoPrivacyContentFactory,
)
from preprocessing.privacy.inspection.local_visual_analysis import (
    LocalVisualAnalysis,
    OpenCvVisualPrivacyAnalyzer,
)
from tests.support.privacy import build_test_pii_detector

TEST_MAX_DECODE_PIXELS = 100_000


class _Logger:
    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None


class _NoTextOcr:
    def extract(self, *, image_bytes: bytes):
        assert image_bytes
        return None


class _CleanVisual:
    def analyze_bytes(self, **kwargs) -> LocalVisualAnalysis:
        assert kwargs["payload"]
        return LocalVisualAnalysis(
            regions=(),
            completed=True,
            detector_versions={"test-visual": "1"},
        )


def _safe_inspection(path: Path) -> InspectionResult:
    return InspectionResult(
        subject_digest=_file_sha(path),
        findings=(),
        coverage=InspectionCoverage(
            checked_fields=frozenset(),
            required_fields=frozenset(),
        ),
        detector_runs=(),
        completed=True,
    )


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str:
    return _sha(path.read_bytes())


def _analysis(path: Path, checks: tuple[str, ...], findings=()):
    return {
        "subject_sha256": _file_sha(path),
        "valid": True,
        "checked_ranges_ms": [],
        "checks": {
            name: {
                "completed": True,
                "detector": f"{name}-detector",
                "version": "1",
            }
            for name in checks
        },
        "findings": list(findings),
    }


def _jpeg(path: Path, *, artist: str | None = None) -> None:
    image = Image.new("RGB", (128, 128), "white")
    exif = Image.Exif()
    if artist:
        exif[315] = artist
    image.save(path, format="JPEG", exif=exif)


def _image_preprocessor() -> ImagePreprocessor:
    return ImagePreprocessor(
        logger=_Logger(),
        settings=ImageValidationSettings(),
        modality_acceptance=ImageAcceptanceSettings(
            fetch_max_bytes=10_000_000,
            preprocessing_max_bytes=10_000_000,
            max_decode_pixels=TEST_MAX_DECODE_PIXELS,
        ),
        pii_detector=build_test_pii_detector(),
        privacy_content_factory=LocalImagePrivacyContentFactory(
            ocr_engine=_NoTextOcr(),
            visual_analyzer=_CleanVisual(),
            max_decode_pixels=TEST_MAX_DECODE_PIXELS,
        ),
        embedded_metadata_adapter=FfmpegEmbeddedMetadataAdapter(
            toolchain=MediaToolchainSettings(),
            settings=MediaPrivacySettings(),
            required=False,
        ),
        now=lambda: datetime(2026, 8, 26, 9, 0, tzinfo=UTC),
        generate_id=lambda: "privacy-adversarial-run",
    )


def _approved_clearance(
    media: Path, fields: dict[str, str]
) -> PrivacyClearance:
    approved = tuple(
        ApprovedTextField(
            name=name,
            value=value,
            input_digest=_sha(value.encode()),
            output_digest=_sha(value.encode()),
        )
        for name, value in fields.items()
    )
    return PrivacyClearance(
        status=PrivacyClearanceStatus.APPROVED,
        input_digest=_file_sha(media),
        output_digest=_file_sha(media),
        checked_fields=frozenset(fields),
        required_fields=frozenset(fields),
        approved_text_fields=approved,
        inspection_digest=_sha(b"inspection"),
        assessment_digest=_sha(b"assessment"),
    )


def test_disabled_privacy_detection_is_fail_closed() -> None:
    result = inspect_text_fields_for_release(
        fields={"title": "Jan Peeters jan.peeters@example.com"},
        detector=build_test_pii_detector(
            PrivacyDetectionSettings(enabled=False)
        ),
    )

    assert not result.clearance.permits_training
    assert result.clearance.status is PrivacyClearanceStatus.INCOMPLETE


def test_metadata_key_pii_does_not_enter_clearance(tmp_path: Path) -> None:
    media = tmp_path / "media.bin"
    media.write_bytes(b"media")
    item = PreprocessingInput(
        source_id="media",
        source_url="https://example.test/media",
        normalized_url="https://example.test/media",
        domain="example.test",
        path="/media",
        language_evidence=LanguageEvidence(language="en"),
        modality="image",
        media_path=str(media),
        payload={
            "jan.peeters@example.com": "safe value",
            "privacy_analysis": _analysis(media, ()),
        },
    )

    result = inspect_media_privacy(
        item=item,
        detector=build_test_pii_detector(),
        fields={"caption_text": "Public scientific image"},
        inspection=_safe_inspection(media),
        media_path=str(media),
        content_field_prefixes=("ocr_text",),
        object_id=item.source_id,
    )

    serialized = json.dumps(result.clearance.to_dict(), sort_keys=True)
    assert "jan.peeters@example.com" not in serialized


def test_embedded_exif_is_locally_analyzed_and_stripped(
    tmp_path: Path,
) -> None:
    media = tmp_path / "exif.jpg"
    _jpeg(media, artist="jan.peeters@example.com")
    item = PreprocessingInput(
        source_id="image",
        source_url="https://example.test/exif.jpg",
        normalized_url="https://example.test/exif.jpg",
        domain="example.test",
        path="/exif.jpg",
        language_evidence=LanguageEvidence(language="en"),
        title="Public image",
        modality="image",
        mime_type="image/jpeg",
        media_path=str(media),
        byte_size=media.stat().st_size,
        payload={
            "caption_text": "Public scientific instrument",
            # metadata_inspection is intentionally absent.
            "privacy_analysis": _analysis(
                media, ("ocr_analysis", "visual_analysis")
            ),
        },
    )

    result = _image_preprocessor().process(inputs=(item,))

    assert len(result.items) == 1
    output = Path(result.items[0].media_path)
    assert output.name == "exif.metadata-clean.jpg"
    with Image.open(output) as cleaned:
        assert not cleaned.getexif()


def test_remediation_does_not_preserve_raw_normalized_path(
    tmp_path: Path,
) -> None:
    original = tmp_path / "Jan_Peeters_original.jpg"
    sanitized = tmp_path / "sanitized.jpg"
    _jpeg(original)
    _jpeg(sanitized)
    all_checks = (
        "media_decode",
        "ocr_analysis",
        "visual_analysis",
        "metadata_inspection",
    )
    item = PreprocessingInput(
        source_id="image",
        source_url="https://example.test/image.jpg",
        normalized_url="https://example.test/image.jpg",
        domain="example.test",
        path="/image.jpg",
        language_evidence=LanguageEvidence(language="en"),
        title="Public image",
        modality="image",
        mime_type="image/jpeg",
        media_path=str(original),
        byte_size=original.stat().st_size,
        payload={
            "caption_text": "Public scientific instrument",
            "ocr_text": "jan.peeters@example.com",
            "normalized_media_path": str(original),
            "privacy_sanitized_media_path": str(sanitized),
            "privacy_analysis": _analysis(
                original, ("ocr_analysis", "visual_analysis")
            ),
            "privacy_residual_analysis": _analysis(sanitized, all_checks),
        },
    )

    result = _image_preprocessor().process(inputs=(item,))
    assert len(result.items) == 1
    image = result.items[0]

    # Caller-selected normalized/remediation paths and self-declared evidence
    # are ignored. The exact typed source remains authoritative.
    assert image.media_path == str(original)
    assert image.normalized_media_path == str(original)
    assert image.media_path != str(sanitized)


def test_raw_governance_and_parent_url_cannot_reenter_curation(
    tmp_path: Path,
) -> None:
    media = tmp_path / "safe.jpg"
    _jpeg(media)
    clearance = _approved_clearance(
        media,
        {
            "caption_text": "Approved caption",
            "source_url": "https://example.test/safe.jpg",
        },
    )
    item = PreprocessedImage(
        media_id="safe",
        source_id="source",
        source_url="https://example.test/safe.jpg",
        normalized_url="https://example.test/safe.jpg",
        domain="example.test",
        media_path=str(media),
        mime_type="image/jpeg",
        width=128,
        height=128,
        normalized_media_path=str(media),
        ocr_text=None,
        ocr_confidence=None,
        ocr_language="en",
        ocr_quality_score=None,
        quality=PreprocessingQualityResult(
            score=0.9,
            bucket="gold",
            rejection_reason=None,
            token_count_estimate=2,
            modality="image",
        ),
        alignment_signals={
            "caption_source": "caption",
            "caption_quality_score": 0.9,
        },
        safety_status="passed",
        privacy_clearance=clearance,
    )
    record = SimpleNamespace(
        fetch_record_id="source",
        parent_fetch_record_id=None,
        parent_stable_url_id=None,
        media_identity="identity",
        fetch_mode="full",
        asset_fetch_mode="full",
        is_complete_payload=True,
        source_page_url=(
            "https://example.test/users/jan.peeters@example.com?x=1"
        ),
        embed_host="jan.peeters@example.com",
        governance={
            "training": {
                "allowed": True,
                "reason": "Approved by jan.peeters@example.com",
            },
            "license": {
                "expression": "CC0 - owner jan.peeters@example.com",
                "evidence_url": (
                    "https://example.test/license?email="
                    "jan.peeters@example.com"
                ),
            },
            "usage_policy": "Contact jan.peeters@example.com",
        },
        object_id="object",
        run_id="run",
        observed_bytes=media.stat().st_size,
        byte_size=media.stat().st_size,
        source_content_length=media.stat().st_size,
        source_content_type="image/jpeg",
        fetch_duration_seconds=0.1,
        payload_sha256=_file_sha(media),
        kind="image",
        storage_relative_path=media.name,
        asset_metadata_only=False,
        requested_url="https://example.test/safe.jpg",
        final_url="https://example.test/safe.jpg",
        normalized_url="https://example.test/safe.jpg",
        domain="example.test",
        path="/safe.jpg",
        language="en",
        encoding=None,
        mime_type="image/jpeg",
        keyframe_paths=(),
        thumbnail_path=None,
    )

    row = build_privacy_cleared_images(
        snapshot_id="snapshot",
        schema_version="3",
        raw_entries=(SimpleNamespace(record=record),),
        documents=(),
        preprocessed=(item,),
        project_root=tmp_path,
    )[0]

    assert "jan.peeters@example.com" not in json.dumps(
        row.to_dict(), sort_keys=True, default=str
    )


def test_curated_images_only_from_privacy_cleared_outputs(
    tmp_path: Path,
) -> None:
    media = tmp_path / "safe.jpg"
    _jpeg(media)

    def preprocessed_item(
        *,
        clearance: PrivacyClearance | None,
    ) -> PreprocessedImage:
        return PreprocessedImage(
            media_id="safe",
            source_id="source",
            source_url="https://example.test/safe.jpg",
            normalized_url="https://example.test/safe.jpg",
            domain="example.test",
            media_path=str(media),
            mime_type="image/jpeg",
            width=128,
            height=128,
            normalized_media_path=str(media),
            ocr_text=None,
            ocr_confidence=None,
            ocr_language="en",
            ocr_quality_score=None,
            quality=PreprocessingQualityResult(
                score=0.9,
                bucket="gold",
                rejection_reason=None,
                token_count_estimate=2,
                modality="image",
            ),
            alignment_signals={"caption_source": "caption"},
            safety_status="passed",
            privacy_clearance=clearance,
        )

    record = SimpleNamespace(
        fetch_record_id="source",
        parent_fetch_record_id=None,
        parent_stable_url_id=None,
        media_identity="identity",
        fetch_mode="full",
        asset_fetch_mode="full",
        source_page_url="https://example.test/safe.jpg",
        embed_host=None,
        governance={},
        object_id="object",
        run_id="run",
        observed_bytes=media.stat().st_size,
        byte_size=media.stat().st_size,
        kind="image",
        storage_relative_path="safe.jpg",
        mime_type="image/jpeg",
    )
    raw_entries = (SimpleNamespace(record=record),)

    blocked_clearance = _approved_clearance(
        media,
        {"caption_text": "Approved caption"},
    )
    object.__setattr__(
        blocked_clearance, "status", PrivacyClearanceStatus.INCOMPLETE
    )

    rows = build_privacy_cleared_images(
        snapshot_id="snapshot",
        schema_version="3",
        raw_entries=raw_entries,
        documents=(),
        preprocessed=(
            preprocessed_item(clearance=None),
            preprocessed_item(clearance=blocked_clearance),
            preprocessed_item(
                clearance=_approved_clearance(
                    media, {"caption_text": "Approved caption"}
                )
            ),
        ),
        project_root=tmp_path,
    )

    # Only the approved item may produce a curated row; clearance-less and
    # denied items must be skipped (fail closed).
    assert len(rows) == 1
    assert rows[0].caption_text == "Approved caption"


def test_training_verifier_checks_all_serialized_text_fields(
    tmp_path: Path,
) -> None:
    text = "A safe scientific sentence with enough useful content."
    clearance = inspect_text_fields_for_release(
        fields={"body": text},
        detector=build_test_pii_detector(),
    ).clearance
    sample = TrainingSample(
        sample_id="sample",
        snapshot_id="snapshot",
        modality="text",
        text=text,
        title="Patient Jan Peeters jan.peeters@example.com",
        source_url=("https://example.test/?email=jan.peeters@example.com"),
        text_spans=(
            TrainingTextSpan(
                text="Call +32 470 12 34 56",
                source="raw",
            ),
        ),
        task_target=TrainingTaskTarget(task_type="text_pretrain"),
        privacy_clearance=clearance,
    )

    result = verify_sample_clearance(
        sample,
        project_root=tmp_path,
        residual_pii_detector=build_test_pii_detector(),
    )

    assert not result.valid


def test_normalized_transcript_ranges_are_recognized() -> None:
    segments = normalize_segments(
        [
            {
                "start_seconds": 0,
                "end_seconds": 2,
                "text": "Safe transcript",
            }
        ]
    )

    ranges = tuple(
        (
            int(float(segment["start_seconds"]) * 1000),
            int(float(segment["end_seconds"]) * 1000),
        )
        for segment in segments
    )
    assert ranges_cover_duration(ranges=ranges, duration_ms=2000)


def test_unreleased_raw_metadata_values_do_not_enter_clearance(
    tmp_path: Path,
) -> None:
    media = tmp_path / "safe.bin"
    media.write_bytes(b"safe")
    raw_path = "/patients/Jan_Peeters/original.jpg"
    item = PreprocessingInput(
        source_id="safe",
        source_url="https://example.test/safe",
        normalized_url="https://example.test/safe",
        domain="example.test",
        path="/safe",
        language_evidence=LanguageEvidence(language="en"),
        modality="image",
        media_path=str(media),
        payload={
            "normalized_media_path": raw_path,
            "privacy_analysis": _analysis(media, ()),
        },
    )

    result = inspect_media_privacy(
        item=item,
        detector=build_test_pii_detector(),
        fields={"caption_text": "Public scientific image"},
        inspection=_safe_inspection(media),
        media_path=str(media),
        content_field_prefixes=("ocr_text",),
        object_id=item.source_id,
    )

    assert raw_path not in json.dumps(result.clearance.to_dict())


def test_non_private_exif_is_stripped_to_authoritative_derivative(
    tmp_path: Path,
) -> None:
    original = tmp_path / "original.jpg"
    sanitized = tmp_path / "clean.jpg"
    _jpeg(original, artist="scanner-v1")
    item = PreprocessingInput(
        source_id="image",
        source_url="https://example.test/image.jpg",
        normalized_url="https://example.test/image.jpg",
        domain="example.test",
        path="/image.jpg",
        language_evidence=LanguageEvidence(language="en"),
        title="Public image",
        modality="image",
        mime_type="image/jpeg",
        media_path=str(original),
        byte_size=original.stat().st_size,
        payload={
            "caption_text": "Public scientific instrument",
            "privacy_sanitized_media_path": str(sanitized),
            "privacy_analysis": _analysis(
                original, ("ocr_analysis", "visual_analysis")
            ),
        },
    )

    result = _image_preprocessor().process(inputs=(item,))

    assert len(result.items) == 1
    image = result.items[0]
    output = Path(image.media_path)
    assert output.name == "original.metadata-clean.jpg"
    assert output != sanitized
    assert image.normalized_media_path == str(output)
    with Image.open(output) as cleaned:
        assert not cleaned.getexif()


def test_empty_text_detector_registry_is_incomplete() -> None:
    result = inspect_text(
        TextContent(
            text="public text",
            field_name="body",
            language="en",
            country=None,
        ),
        DetectorRegistry(text_detectors=()),
    )

    assert result.completed is False
    assert result.coverage.complete is False
    assert result.coverage.unchecked_fields == frozenset({"body"})
    assert result.errors == ("text_detector_registry_empty",)


class _ExplodingVisual:
    def analyze_bytes(self, **kwargs):
        del kwargs
        raise RuntimeError("visual backend failed")


class _ExplodingProbeReader:
    def probe(self, path):
        del path
        raise RuntimeError("probe failed")


class _QrDetectorWithoutFindings:
    def detectMulti(self, frame):
        del frame
        return False, None


class _Cv2WithoutBarcode:
    def QRCodeDetector(self):
        return _QrDetectorWithoutFindings()


def test_image_visual_backend_exception_is_captured(tmp_path: Path) -> None:
    media = tmp_path / "image.jpg"
    _jpeg(media)
    item = PreprocessingInput(
        source_id="image-exception",
        source_url="https://example.test/image.jpg",
        normalized_url="https://example.test/image.jpg",
        domain="example.test",
        path="/image.jpg",
        language_evidence=LanguageEvidence(language="en"),
        modality="image",
        media_path=str(media),
    )
    factory = LocalImagePrivacyContentFactory(
        ocr_engine=_NoTextOcr(),
        visual_analyzer=_ExplodingVisual(),
        max_decode_pixels=TEST_MAX_DECODE_PIXELS,
    )

    content = factory.build(
        item=item,
        media_path=media,
        metadata={},
        residual=False,
    )

    assert content.visual_analysis_completed is False
    assert (
        "local_visual_analysis_failure:RuntimeError" in content.analysis_errors
    )


def test_video_probe_exception_is_captured(tmp_path: Path) -> None:
    media = tmp_path / "video.bin"
    media.write_bytes(b"not-a-real-video")
    item = PreprocessingInput(
        source_id="video-exception",
        source_url="https://example.test/video.bin",
        normalized_url="https://example.test/video.bin",
        domain="example.test",
        path="/video.bin",
        language_evidence=LanguageEvidence(language="en"),
        modality="video",
        media_path=str(media),
    )
    factory = LocalVideoPrivacyContentFactory(
        ocr_engine=_NoTextOcr(),
        visual_analyzer=_CleanVisual(),
        reader=_ExplodingProbeReader(),
        frame_processor=object(),
        audio_stream_probe=lambda _path: False,
    )

    content = factory.build(
        item=item,
        media_path=media,
        metadata={},
        duration_ms=1_000,
        transcript_segments=(),
        residual=False,
    )

    assert "video_probe_failure:RuntimeError" in content.analysis_errors
    assert content.visual_analysis_completed is False


def test_missing_barcode_backend_is_fail_closed() -> None:
    analyzer = OpenCvVisualPrivacyAnalyzer.__new__(OpenCvVisualPrivacyAnalyzer)
    analyzer._cv2 = _Cv2WithoutBarcode()
    errors: list[str] = []

    regions = analyzer._machine_readable_regions(
        frame=object(),
        frame_index=None,
        timestamp_ms=None,
        errors=errors,
    )

    assert regions == ()
    assert errors == ["barcode_detector_unavailable"]
