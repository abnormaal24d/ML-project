from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from config.collection.modality_acceptance import ImageAcceptanceSettings
from config.media_toolchain import MediaToolchainSettings
from config.preprocessing.media_settings import (
    ImageValidationSettings,
    MediaPrivacySettings,
)
from preprocessing.media.adapters.embedded_metadata import (
    FfmpegEmbeddedMetadataAdapter,
)
from preprocessing.media.image.image_preprocessor import ImagePreprocessor
from preprocessing.media.privacy_inspection import inspect_media_privacy
from preprocessing.preprocessing_input import (
    LanguageEvidence,
    PreprocessingInput,
)
from preprocessing.privacy.artifacts import DerivationReceipt
from preprocessing.privacy.clearance import PrivacyClearanceStatus
from preprocessing.privacy.inspection.content_readers.audio_content import (
    AudioContent,
    TranscriptSegment,
)
from preprocessing.privacy.inspection.detector import VisualRegion
from preprocessing.privacy.inspection.inspect_audio import inspect_audio
from preprocessing.privacy.inspection.inspect_image import inspect_image
from preprocessing.privacy.inspection.inspection_result import (
    media_analysis_evidence,
)
from preprocessing.privacy.inspection.local_content_factories import (
    LocalImagePrivacyContentFactory,
)
from preprocessing.privacy.inspection.local_visual_analysis import (
    LocalVisualAnalysis,
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


class _VisualSequence:
    def __init__(self, *results: LocalVisualAnalysis) -> None:
        self._results = list(results)

    def analyze_bytes(self, **kwargs) -> LocalVisualAnalysis:
        assert kwargs["payload"]
        if len(self._results) > 1:
            return self._results.pop(0)
        return self._results[0]


class _CleanVisual:
    def analyze_bytes(self, **kwargs) -> LocalVisualAnalysis:
        assert kwargs["payload"]
        return LocalVisualAnalysis(
            regions=(),
            completed=True,
            detector_versions={"test-visual": "1"},
        )


def _item(
    path: Path,
    *,
    payload: dict[str, object] | None = None,
) -> PreprocessingInput:
    return PreprocessingInput(
        source_id="image",
        source_url="https://example.test/image.png",
        normalized_url="https://example.test/image.png",
        domain="example.test",
        path="/image.png",
        language_evidence=LanguageEvidence(language="en"),
        title="Public image",
        modality="image",
        mime_type="image/png",
        media_path=str(path),
        byte_size=path.stat().st_size,
        payload={"caption_text": "Public scene", **(payload or {})},
    )


def _preprocessor(factory=None) -> ImagePreprocessor:
    factory = factory or LocalImagePrivacyContentFactory(
        ocr_engine=None,
        visual_analyzer=_CleanVisual(),
        max_decode_pixels=TEST_MAX_DECODE_PIXELS,
    )
    return ImagePreprocessor(
        logger=_Logger(),
        settings=ImageValidationSettings(),
        modality_acceptance=ImageAcceptanceSettings(
            fetch_max_bytes=1_000_000,
            preprocessing_max_bytes=1_000_000,
            max_decode_pixels=TEST_MAX_DECODE_PIXELS,
        ),
        pii_detector=build_test_pii_detector(),
        privacy_content_factory=factory,
        embedded_metadata_adapter=FfmpegEmbeddedMetadataAdapter(
            toolchain=MediaToolchainSettings(),
            settings=MediaPrivacySettings(),
            required=False,
        ),
        now=lambda: datetime(2026, 8, 26, 9, 0, tzinfo=UTC),
        generate_id=lambda: "privacy-evidence-run",
    )


def test_local_inspection_is_bound_to_exact_media_bytes(
    tmp_path: Path,
) -> None:
    image = tmp_path / "safe.png"
    Image.new("RGB", (128, 128), "white").save(image)
    detector = build_test_pii_detector()
    factory = LocalImagePrivacyContentFactory(
        ocr_engine=_NoTextOcr(),
        visual_analyzer=_CleanVisual(),
        max_decode_pixels=TEST_MAX_DECODE_PIXELS,
    )
    inspection = inspect_image(
        factory.build(
            item=_item(image),
            media_path=image,
            metadata={},
            residual=False,
        ),
        detector.registry,
    )

    evidence = media_analysis_evidence(
        inspection,
        expected_digest=hashlib.sha256(b"different").hexdigest(),
    )

    assert evidence.valid is False
    assert "local_inspection_subject_mismatch" in evidence.reasons


def test_local_image_privacy_inspection_respects_decode_pixel_limit(
    tmp_path: Path,
) -> None:
    image = tmp_path / "over-limit.png"
    Image.new("RGB", (11, 10), "white").save(image)
    factory = LocalImagePrivacyContentFactory(
        ocr_engine=None,
        visual_analyzer=_CleanVisual(),
        max_decode_pixels=100,
    )

    content = factory.build(
        item=_item(image),
        media_path=image,
        metadata={},
        residual=False,
    )

    assert content.media_decode_completed is False
    assert "image_decode_failed" in content.analysis_errors


def test_self_declared_privacy_payload_cannot_override_missing_local_backend(
    tmp_path: Path,
) -> None:
    image = tmp_path / "self-declared.png"
    Image.new("RGB", (128, 128), "white").save(image)
    malicious = {
        "privacy_analysis": {
            "subject_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
            "valid": True,
            "checks": {
                "ocr_analysis": {
                    "completed": True,
                    "detector": "totally-untrusted",
                    "version": "banana",
                },
                "visual_analysis": {
                    "completed": True,
                    "detector": "totally-untrusted",
                    "version": "banana",
                },
            },
            "findings": [],
        }
    }

    result = _preprocessor().process(inputs=(_item(image, payload=malicious),))

    assert result.items == ()
    assert result.quarantine_records
    reasons = result.quarantine_records[0].quality_signals.get(
        "privacy_reasons", []
    )
    assert any(
        "local_ocr_backend_unavailable" in str(reason) for reason in reasons
    )


def test_identity_document_from_local_visual_inspection_is_rejected(
    tmp_path: Path,
) -> None:
    image = tmp_path / "identity.png"
    Image.new("RGB", (128, 128), "white").save(image)
    visual = _VisualSequence(
        LocalVisualAnalysis(
            regions=(
                VisualRegion(
                    category="identity_document",
                    confidence=0.99,
                    x=0,
                    y=0,
                    width=128,
                    height=128,
                ),
            ),
            completed=True,
            detector_versions={"local-id": "1"},
        )
    )
    factory = LocalImagePrivacyContentFactory(
        ocr_engine=_NoTextOcr(),
        visual_analyzer=visual,
        max_decode_pixels=TEST_MAX_DECODE_PIXELS,
    )

    result = _preprocessor(factory).process(inputs=(_item(image),))

    assert result.items == ()
    assert result.quarantine_records[0].reason == "identity_document_detected"


def test_local_face_detection_creates_fresh_derivative_and_rescans_it(
    tmp_path: Path,
) -> None:
    original = tmp_path / "original.png"
    attacker_path = tmp_path / "attacker-selected.png"
    Image.new("RGB", (128, 128), "white").save(original)
    Image.new("RGB", (128, 128), "blue").save(attacker_path)
    visual = _VisualSequence(
        LocalVisualAnalysis(
            regions=(
                VisualRegion(
                    category="face",
                    confidence=0.98,
                    x=10,
                    y=10,
                    width=20,
                    height=20,
                ),
            ),
            completed=True,
            detector_versions={"local-face": "1"},
        ),
        LocalVisualAnalysis(
            regions=(),
            completed=True,
            detector_versions={"local-face": "1"},
        ),
    )
    factory = LocalImagePrivacyContentFactory(
        ocr_engine=_NoTextOcr(),
        visual_analyzer=visual,
        max_decode_pixels=TEST_MAX_DECODE_PIXELS,
    )
    item = _item(
        original,
        payload={"privacy_sanitized_media_path": str(attacker_path)},
    )

    result = _preprocessor(factory).process(inputs=(item,))

    assert len(result.items) == 1
    output = Path(result.items[0].media_path)
    assert output != attacker_path
    assert output.name == "original.privacy-sanitized.png"
    with Image.open(output) as image:
        assert image.getpixel((15, 15)) == (0, 0, 0)
    with Image.open(attacker_path) as image:
        assert image.getpixel((15, 15)) == (0, 0, 255)
    assert (
        result.items[0].privacy_clearance.status
        is PrivacyClearanceStatus.REMEDIATED
    )


def test_identifiable_voice_without_authorization_is_rejected_locally(
    tmp_path: Path,
) -> None:
    media = tmp_path / "voice.wav"
    media.write_bytes(b"locally decoded voice")
    item = PreprocessingInput(
        source_id="voice",
        source_url="https://example.test/voice.wav",
        normalized_url="https://example.test/voice.wav",
        domain="example.test",
        path="/voice.wav",
        language_evidence=LanguageEvidence(language="en"),
        modality="audio",
        media_path=str(media),
    )
    content = AudioContent(
        subject_bytes=media.read_bytes(),
        duration_ms=2_000,
        transcript_segments=(
            TranscriptSegment(0, 2_000, "Public sentence", None),
        ),
        transcript_checked_ranges_ms=((0, 2_000),),
        transcript_analysis_completed=True,
        metadata={},
        metadata_analysis_completed=True,
        language="en",
        country=None,
        full_decode_completed=True,
        speaker_analysis_completed=True,
        background_speech_analysis_completed=True,
        voice_analysis_completed=True,
        voice_identity_detected=True,
        voice_identity_authorized=False,
        audio_fingerprint="fingerprint",
        detector_versions={"local-audio": "1"},
    )
    detector = build_test_pii_detector()
    inspection = inspect_audio(content, detector.registry)

    result = inspect_media_privacy(
        item=item,
        object_id=item.source_id,
        detector=detector,
        fields={"transcript_text": "Public sentence"},
        inspection=inspection,
        media_path=str(media),
        content_field_prefixes=("transcript",),
    )

    assert result.clearance.status is PrivacyClearanceStatus.REJECTED
    assert (
        result.rejection_reason == "identifiable_voice_without_authorization"
    )


def test_incomplete_local_audio_coverage_fails_closed(tmp_path: Path) -> None:
    media = tmp_path / "partial.wav"
    media.write_bytes(b"audio")
    item = PreprocessingInput(
        source_id="partial",
        source_url="https://example.test/partial.wav",
        normalized_url="https://example.test/partial.wav",
        domain="example.test",
        path="/partial.wav",
        language_evidence=LanguageEvidence(language="en"),
        modality="audio",
        media_path=str(media),
    )
    content = AudioContent(
        subject_bytes=media.read_bytes(),
        duration_ms=2_000,
        transcript_segments=(TranscriptSegment(0, 1_000, "Partial", None),),
        transcript_checked_ranges_ms=((0, 1_000),),
        transcript_analysis_completed=False,
        metadata={},
        metadata_analysis_completed=True,
        language="en",
        country=None,
        full_decode_completed=True,
        speaker_analysis_completed=True,
        background_speech_analysis_completed=True,
        voice_analysis_completed=True,
        voice_identity_detected=False,
        voice_identity_authorized=False,
        audio_fingerprint="fingerprint",
    )
    detector = build_test_pii_detector()
    inspection = inspect_audio(content, detector.registry)

    result = inspect_media_privacy(
        item=item,
        object_id=item.source_id,
        detector=detector,
        fields={"transcript_text": "Partial"},
        inspection=inspection,
        media_path=str(media),
        content_field_prefixes=("transcript",),
    )

    assert not result.clearance.permits_training
    assert result.clearance.status is PrivacyClearanceStatus.INCOMPLETE
    assert any(
        "transcript_coverage" in reason for reason in result.clearance.reasons
    )


def test_local_video_factory_inspects_every_decoded_frame(
    tmp_path: Path,
) -> None:
    cv2 = __import__("cv2")
    import numpy as np

    from preprocessing.media.adapters.opencv_video import (
        OpenCvFrameProcessor,
        OpenCvVideoReader,
    )
    from preprocessing.privacy.inspection.inspect_video import inspect_video
    from preprocessing.privacy.inspection.local_content_factories import (
        LocalVideoPrivacyContentFactory,
    )

    class _FrameVisual:
        def analyze_frame(self, **kwargs) -> LocalVisualAnalysis:
            assert kwargs["frame"] is not None
            return LocalVisualAnalysis(
                regions=(),
                completed=True,
                detector_versions={"test-video-visual": "1"},
            )

    video = tmp_path / "safe.avi"
    writer = cv2.VideoWriter(
        str(video),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (64, 64),
    )
    assert writer.isOpened()
    for value in (0, 64, 128, 192, 255):
        writer.write(np.full((64, 64, 3), value, dtype=np.uint8))
    writer.release()

    item = PreprocessingInput(
        source_id="video",
        source_url="https://example.test/safe.avi",
        normalized_url="https://example.test/safe.avi",
        domain="example.test",
        path="/safe.avi",
        language_evidence=LanguageEvidence(language="en"),
        modality="video",
        media_path=str(video),
    )
    factory = LocalVideoPrivacyContentFactory(
        ocr_engine=_NoTextOcr(),
        visual_analyzer=_FrameVisual(),
        reader=OpenCvVideoReader(),
        frame_processor=OpenCvFrameProcessor(),
        audio_stream_probe=lambda _path: False,
        max_frames=100,
    )
    content = factory.build(
        item=item,
        media_path=video,
        metadata={},
        duration_ms=1_000,
        transcript_segments=(),
        residual=False,
    )
    detector = build_test_pii_detector()
    inspection = inspect_video(content, detector.registry)

    assert content.inspected_frame_count == content.decoded_frame_count == 5
    assert content.uninspected_intervals_ms == ()
    assert inspection.safe_to_assess


def test_local_document_factory_enforces_page_coverage() -> None:
    from preprocessing.privacy.inspection.inspect_document import (
        inspect_document,
    )
    from preprocessing.privacy.inspection.local_content_factories import (
        LocalDocumentPrivacyContentFactory,
    )

    item = PreprocessingInput(
        source_id="document",
        source_url="https://example.test/document.txt",
        normalized_url="https://example.test/document.txt",
        domain="example.test",
        path="/document.txt",
        language_evidence=LanguageEvidence(language="en"),
        modality="document",
        ocr_text="Public document body",
    )
    content = LocalDocumentPrivacyContentFactory().build(
        item=item,
        normalized_text="Public document body",
        title="Public document",
        metadata={},
    )
    inspection = inspect_document(content, build_test_pii_detector().registry)

    assert inspection.safe_to_assess
    assert inspection.coverage.checked_pages == frozenset({1})
    assert "document_page_coverage" in inspection.coverage.checked_fields


def test_video_inspection_uses_local_complete_coverage(tmp_path: Path) -> None:
    from preprocessing.privacy.inspection.content_readers.video_content import (
        FrameText,
        VideoContent,
    )
    from preprocessing.privacy.inspection.inspect_video import inspect_video

    media = tmp_path / "video.bin"
    media.write_bytes(b"exact-local-video-bytes")
    digest = hashlib.sha256(media.read_bytes()).hexdigest()
    content = VideoContent(
        subject_bytes=media.read_bytes(),
        subject_sha256=digest,
        duration_ms=1_000,
        decoded_frame_count=2,
        inspected_frame_count=2,
        scene_count=1,
        transcript_segments=(
            TranscriptSegment(0, 1_000, "Public narration", None),
        ),
        transcript_checked_ranges_ms=((0, 1_000),),
        frame_text=(
            FrameText(0, 0, "", "frame-0"),
            FrameText(1, 500, "", "frame-1"),
        ),
        frame_ocr_checked_ranges_ms=((0, 1_000),),
        visual_regions=(),
        checked_video_ranges_ms=((0, 1_000),),
        uninspected_intervals_ms=(),
        tracking_completed=True,
        audio_inspection_completed=True,
        metadata_inspection_completed=True,
        residual_scan_completed=True,
        detector_versions={"local-video": "1"},
        visual_analysis_completed=True,
        audio_fingerprint="video-audio-fingerprint",
        metadata={},
        language="en",
        country=None,
    )
    detector = build_test_pii_detector()
    inspection = inspect_video(content, detector.registry)
    item = PreprocessingInput(
        source_id="video",
        source_url="https://example.test/video",
        normalized_url="https://example.test/video",
        domain="example.test",
        path="/video",
        language_evidence=LanguageEvidence(language="en"),
        modality="video",
        media_path=str(media),
    )

    result = inspect_media_privacy(
        item=item,
        object_id=item.source_id,
        detector=detector,
        fields={"transcript_text": "Public narration"},
        inspection=inspection,
        media_path=str(media),
        content_field_prefixes=("transcript", "frame_ocr"),
    )

    assert inspection.completed
    assert result.clearance.permits_training
    assert result.analysis_evidence.to_dict()["source"] == "local_inspection"


def test_document_inspection_requires_local_page_coverage() -> None:
    from preprocessing.privacy.inspection.content_readers.document_content import (
        DocumentContent,
        DocumentPage,
    )
    from preprocessing.privacy.inspection.inspect_document import (
        inspect_document,
    )

    detector = build_test_pii_detector()
    complete = inspect_document(
        DocumentContent(
            subject_bytes=b"page one",
            title="Public report",
            pages=(DocumentPage(page_number=1, text="Public page"),),
            metadata={},
            language="en",
            country=None,
            expected_page_count=1,
        ),
        detector.registry,
    )
    incomplete = inspect_document(
        DocumentContent(
            subject_bytes=b"page one only",
            title="Public report",
            pages=(DocumentPage(page_number=1, text="Public page"),),
            metadata={},
            language="en",
            country=None,
            expected_page_count=2,
        ),
        detector.registry,
    )

    assert complete.completed
    assert "document_page_coverage" in complete.coverage.checked_fields
    assert not incomplete.completed
    assert "document_page_coverage" in incomplete.coverage.unchecked_fields


def test_active_flows_call_local_inspectors_and_never_read_payload_evidence() -> (
    None
):
    import ast

    project_root = Path(__file__).resolve().parents[3]
    call_sites = {
        "preprocessing/media/image/image_preprocessor.py": "inspect_image",
        "preprocessing/media/audio/audio_preprocessor.py": "inspect_audio",
        "preprocessing/media/video/video_preprocessor.py": "inspect_video",
        "preprocessing/text/text_preparation.py": "inspect_document",
    }
    for relative, expected_call in call_sites.items():
        source = (project_root / relative).read_text(encoding="utf-8")
        tree = ast.parse(source)
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert expected_call in called, (
            f"{relative} does not call {expected_call}"
        )

    for path in (project_root / "preprocessing").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(
                node.func, ast.Attribute
            ):
                continue
            if node.func.attr != "get" or not node.args:
                continue
            key = node.args[0]
            assert not (
                isinstance(key, ast.Constant)
                and key.value
                in {"privacy_analysis", "privacy_residual_analysis"}
            ), f"untrusted privacy evidence consumed by {path}"


def test_preexisting_canonical_destination_is_never_reused(
    tmp_path: Path,
) -> None:
    original = tmp_path / "source.png"
    preexisting = tmp_path / "source.privacy-sanitized.png"
    Image.new("RGB", (128, 128), "red").save(original)
    Image.new("RGB", (128, 128), "blue").save(preexisting)
    visual = _VisualSequence(
        LocalVisualAnalysis(
            regions=(
                VisualRegion(
                    category="face",
                    confidence=0.99,
                    x=10,
                    y=10,
                    width=20,
                    height=20,
                ),
            ),
            completed=True,
            detector_versions={"local-face": "1"},
        ),
        LocalVisualAnalysis(
            regions=(),
            completed=True,
            detector_versions={"local-face": "1"},
        ),
    )
    factory = LocalImagePrivacyContentFactory(
        ocr_engine=_NoTextOcr(),
        visual_analyzer=visual,
        max_decode_pixels=TEST_MAX_DECODE_PIXELS,
    )

    result = _preprocessor(factory).process(inputs=(_item(original),))

    assert len(result.items) == 1
    output = Path(result.items[0].media_path)
    assert output.name == preexisting.name
    assert output != preexisting
    assert output.parent != tmp_path
    with Image.open(output) as image:
        assert image.getpixel((15, 15)) == (0, 0, 0)
        assert image.getpixel((100, 100)) == (255, 0, 0)
    with Image.open(preexisting) as image:
        assert image.getpixel((100, 100)) == (0, 0, 255)
    clearance = result.items[0].privacy_clearance
    assert clearance is not None
    assert clearance.status is PrivacyClearanceStatus.REMEDIATED
    assert clearance.derivation_digest is not None
    receipt_path = output.with_name(f"{output.name}.receipt.json")
    assert receipt_path.is_file()
    receipt = DerivationReceipt(
        **json.loads(receipt_path.read_text(encoding="utf-8"))
    )
    assert receipt.derivation_digest == clearance.derivation_digest
    assert (
        hashlib.sha256(receipt_path.read_bytes()).hexdigest()
        == receipt.receipt_digest
    )
