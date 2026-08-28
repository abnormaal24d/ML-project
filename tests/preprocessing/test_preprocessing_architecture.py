"""Architecture guards for the slim preprocessing package."""

from __future__ import annotations

import ast
from pathlib import Path

PREPROCESSING_ROOT = Path(__file__).resolve().parents[2] / "preprocessing"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MEDIA_ADAPTERS_ROOT = PREPROCESSING_ROOT / "media" / "adapters"
PRIVACY_ROOT = PREPROCESSING_ROOT / "privacy"

FORBIDDEN_PREPROCESSING_IMPORTS = {
    "crawler",
    "training",
    "orchestration",
    "augmentation",
    "media_io",
}

FORBIDDEN_EXTERNAL_LIBS = {
    "cv2",
    "av",
    "mutagen",
    "pyannote",
    "pytesseract",
    "rapidocr",
    "pypdf",
    "PIL",
    "pillow",
    "soundfile",
}

BANNED_GLOBAL_GETTERS = (
    "get_video_reader",
    "get_frame_processor",
    "get_container_probe",
    "install_media_backends",
    "reset_media_backends",
    "get_video_clip_writer",
    "get_audio_extractor",
    "get_audio_decoder",
    "get_video_normalizer",
    "MediaBackends",
    "ensure_media_backends_installed",
    "build_media_backends",
)

_MIGRATED_ANALYSIS_PACKAGES = (
    "crawler.analysis.enrichment.ocr",
    "crawler.analysis.enrichment.speech",
    "crawler.analysis.enrichment.documents.document_text",
    "crawler.analysis.enrichment.documents.document_text_reader",
    "crawler.analysis.enrichment.documents.document_ocr_extractor",
    "crawler.analysis.enrichment.video.mp4_tail_metadata",
    "crawler.analysis.enrichment.video.video_action_recognition",
    "crawler.analysis.enrichment.video.video_audio_extractor",
    "crawler.analysis.enrichment.video.video_frame_ocr",
    "crawler.analysis.enrichment.video.video_keyframe_selector",
    "crawler.analysis.enrichment.video.video_ocr",
    "crawler.analysis.enrichment.video.video_scene_analysis",
    "crawler.analysis.enrichment.video.video_semantic_outputs",
)
_MIGRATION_SCAN_ROOTS = (
    "augmentation",
    "config",
    "crawler",
    "datachecker",
    "evaluator",
    "logger",
    "mmcrawler_datasets",
    "multimodal",
    "orchestration",
    "preprocessing",
    "schemas",
    "shared",
    "training",
    "tests",
)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    return modules


def _imported_module_paths(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module
        ):
            modules.add(node.module)
    return modules


def test_preprocessing_core_layout_is_stable() -> None:
    content = {
        path.relative_to(PREPROCESSING_ROOT).as_posix()
        for path in PREPROCESSING_ROOT.rglob("*.py")
        if path.name != "__init__.py"
    }
    required = {
        "media/audio/audio_emotion_analyzer.py",
        "media/audio/audio_event_analyzer.py",
        "media/audio/audio_preprocessor.py",
        "media/base_media_preprocessor.py",
        "media/image/image_blur_score.py",
        "media/image/image_hashes.py",
        "media/image/image_metadata_reader.py",
        "media/image/image_preprocessor.py",
        "media/media_fingerprint.py",
        "media/media_input_validation.py",
        "media/ports.py",
        "media/transcript_segment_normalizer.py",
        "media/video/mp4_tail_metadata.py",
        "media/video/video_action_recognition.py",
        "media/video/video_frame_ocr.py",
        "media/video/video_keyframe_selector.py",
        "media/video/video_ocr.py",
        "media/video/video_preprocessor.py",
        "media/video/video_safety_scanner.py",
        "media/video/video_scene_analysis.py",
        "media/video/video_semantic_outputs.py",
        "media/adapters/pillow_image.py",
        "media/adapters/opencv_video.py",
        "media/adapters/pyav_media.py",
        "media/adapters/audio_decode.py",
        "media/adapters/document_ocr_extractor.py",
        "media/adapters/document_text_reader.py",
        "media/adapters/pyannote_adapter.py",
        "media/adapters/rapidocr_engine.py",
        "media/adapters/tesseract_engine.py",
        "media/adapters/whisper_model_loader.py",
        "media/document/document_text.py",
        "multimodal_preprocessor.py",
        "preprocessed_document.py",
        "preprocessed_media.py",
        "preprocessing_input.py",
        "preprocessing_quality.py",
        "preprocessing_result.py",
        "provenance.py",
        "text/text_language.py",
        "text/text_metadata.py",
        "text/text_preparation.py",
        "text/text_preprocessor.py",
        "privacy/text_privacy.py",
        "text/text_quality.py",
    }
    missing = sorted(required - content)
    assert not missing, f"missing preprocessing modules: {missing}"


def test_pure_media_helpers_have_preprocessing_owners() -> None:
    expected = (
        "media/audio/audio_emotion_analyzer.py",
        "media/audio/audio_event_analyzer.py",
        "media/image/image_blur_score.py",
        "media/image/image_hashes.py",
        "media/image/image_metadata_reader.py",
        "media/video/mp4_tail_metadata.py",
        "media/video/video_action_recognition.py",
        "media/video/video_frame_ocr.py",
        "media/video/video_keyframe_selector.py",
        "media/video/video_ocr.py",
        "media/video/video_scene_analysis.py",
        "media/video/video_semantic_outputs.py",
    )
    removed_crawler_modules = (
        "crawler/analysis/enrichment/audio/audio_emotion_analyzer.py",
        "crawler/analysis/enrichment/audio/audio_event_analyzer.py",
        "crawler/analysis/enrichment/audio/audio_sample_decoding.py",
        "crawler/analysis/enrichment/image/image_blur_score.py",
        "crawler/analysis/enrichment/image/image_hashes.py",
        "crawler/analysis/enrichment/image/image_metadata_reader.py",
        "crawler/analysis/enrichment/video/mp4_tail_metadata.py",
        "crawler/analysis/enrichment/video/video_action_recognition.py",
        "crawler/analysis/enrichment/video/video_audio_extractor.py",
        "crawler/analysis/enrichment/video/video_frame_ocr.py",
        "crawler/analysis/enrichment/video/video_keyframe_selector.py",
        "crawler/analysis/enrichment/video/video_ocr.py",
        "crawler/analysis/enrichment/video/video_scene_analysis.py",
        "crawler/analysis/enrichment/video/video_semantic_outputs.py",
    )

    assert all((PREPROCESSING_ROOT / path).is_file() for path in expected)
    assert not any(
        (PROJECT_ROOT / path).exists() for path in removed_crawler_modules
    )


def test_preprocessing_has_no_forbidden_package_imports() -> None:
    violations: list[str] = []
    for path in PREPROCESSING_ROOT.rglob("*.py"):
        imported = _imported_modules(path)
        forbidden = imported & FORBIDDEN_PREPROCESSING_IMPORTS
        if forbidden:
            rel = path.relative_to(PREPROCESSING_ROOT)
            violations.append(f"{rel}: {sorted(forbidden)}")
    assert not violations, "Forbidden imports in preprocessing:\n" + "\n".join(
        violations
    )


def test_codec_libraries_only_live_in_media_adapters() -> None:
    violations: list[str] = []
    for path in PREPROCESSING_ROOT.rglob("*.py"):
        if (
            MEDIA_ADAPTERS_ROOT in path.parents
            or path.parent == MEDIA_ADAPTERS_ROOT
            or PRIVACY_ROOT in path.parents
            or path.parent == PRIVACY_ROOT
        ):
            continue
        imported = _imported_modules(path)
        bad = imported & FORBIDDEN_EXTERNAL_LIBS
        if bad:
            rel = path.relative_to(PREPROCESSING_ROOT)
            violations.append(f"{rel}: {sorted(bad)}")
    assert not violations, (
        "External codec/OCR libs outside adapters:\n" + "\n".join(violations)
    )


def test_no_global_media_backend_registry() -> None:
    violations: list[str] = []
    for path in PREPROCESSING_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        hits = [name for name in BANNED_GLOBAL_GETTERS if name in text]
        if hits:
            violations.append(f"{path.as_posix()}: {hits}")
    for path in (PROJECT_ROOT / "orchestration").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        hits = [
            name
            for name in (
                "MediaBackends",
                "ensure_media_backends_installed",
                "build_media_backends",
            )
            if name in text
        ]
        if hits:
            violations.append(f"{path.as_posix()}: {hits}")
    assert not violations, (
        "Global media backend registry remnants:\n" + "\n".join(violations)
    )


def test_domain_services_do_not_construct_concrete_dependencies() -> None:
    """Concrete privacy and codec adapters are composed outside the domain."""

    forbidden_construction = {
        "media/image/image_preprocessor.py": (
            "LocalImagePrivacyContentFactory(",
        ),
        "media/audio/audio_preprocessor.py": (
            "LocalAudioPrivacyContentFactory(",
        ),
        "media/video/video_preprocessor.py": (
            "LocalVideoPrivacyContentFactory(",
            "OpenCvVideoReader(",
        ),
        "text/text_preparation.py": ("LocalDocumentPrivacyContentFactory(",),
        "media/ocr/ocr_engine.py": (
            "RapidOcrEngine(",
            "TesseractOcrEngine(",
        ),
    }
    violations = [
        f"{relative}: {token}"
        for relative, tokens in forbidden_construction.items()
        for token in tokens
        if token in (PREPROCESSING_ROOT / relative).read_text(encoding="utf-8")
    ]
    assert not violations, "Domain construction leaks:\n" + "\n".join(
        violations
    )


def test_base_composition_defers_optional_video_adapter_import() -> None:
    """The base distribution must import without media-extra dependencies."""

    path = (
        PROJECT_ROOT
        / "orchestration"
        / "composition"
        / "preprocessing_dependencies.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    top_level_imports = {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "preprocessing.media.adapters.opencv_video" not in top_level_imports


def test_media_io_package_is_not_referenced() -> None:
    hits: list[str] = []
    self_path = Path(__file__).resolve()
    for path in PROJECT_ROOT.rglob("*.py"):
        if path.resolve() == self_path:
            continue
        if any(
            part in path.parts
            for part in ("__pycache__", ".git", ".venv", "venv")
        ):
            continue
        text = path.read_text(encoding="utf-8")
        if "media_io" in text or "MEDIA_IO_ROOT" in text:
            hits.append(path.as_posix())
    assert not hits, "media_io references remain:\n" + "\n".join(hits)


def test_crawler_independent_media_support_is_preprocessing_owned() -> None:
    expected = (
        PREPROCESSING_ROOT / "media" / "adapters" / "pyannote_adapter.py",
        PREPROCESSING_ROOT / "media" / "adapters" / "rapidocr_engine.py",
        PREPROCESSING_ROOT / "media" / "adapters" / "tesseract_engine.py",
        PREPROCESSING_ROOT / "media" / "adapters" / "whisper_model_loader.py",
        PREPROCESSING_ROOT / "media" / "ocr" / "ocr_engine.py",
        PREPROCESSING_ROOT / "media" / "speech" / "speech_transcriber.py",
        PREPROCESSING_ROOT / "media" / "document" / "document_text.py",
        PREPROCESSING_ROOT / "media" / "video" / "mp4_tail_metadata.py",
        PREPROCESSING_ROOT / "media" / "video" / "video_action_recognition.py",
        PREPROCESSING_ROOT / "media" / "video" / "video_frame_ocr.py",
        PREPROCESSING_ROOT / "media" / "video" / "video_keyframe_selector.py",
        PREPROCESSING_ROOT / "media" / "video" / "video_ocr.py",
        PREPROCESSING_ROOT / "media" / "video" / "video_scene_analysis.py",
        PREPROCESSING_ROOT / "media" / "video" / "video_semantic_outputs.py",
        MEDIA_ADAPTERS_ROOT / "document_text_reader.py",
        MEDIA_ADAPTERS_ROOT / "document_ocr_extractor.py",
    )
    assert all(path.is_file() for path in expected)

    legacy_roots = (
        PROJECT_ROOT / "crawler" / "analysis" / "enrichment" / "ocr",
        PROJECT_ROOT / "crawler" / "analysis" / "enrichment" / "speech",
    )
    legacy_sources = [
        path
        for root in legacy_roots
        if root.exists()
        for path in root.rglob("*.py")
    ]
    legacy_sources.extend(
        path
        for path in (
            PROJECT_ROOT
            / "crawler"
            / "analysis"
            / "enrichment"
            / "documents"
            / "document_text.py",
            PROJECT_ROOT
            / "crawler"
            / "analysis"
            / "enrichment"
            / "documents"
            / "document_text_reader.py",
            PROJECT_ROOT
            / "crawler"
            / "analysis"
            / "enrichment"
            / "documents"
            / "document_ocr_extractor.py",
            PROJECT_ROOT
            / "crawler"
            / "analysis"
            / "enrichment"
            / "video"
            / "mp4_tail_metadata.py",
            PROJECT_ROOT
            / "crawler"
            / "analysis"
            / "enrichment"
            / "video"
            / "video_action_recognition.py",
            PROJECT_ROOT
            / "crawler"
            / "analysis"
            / "enrichment"
            / "video"
            / "video_audio_extractor.py",
            PROJECT_ROOT
            / "crawler"
            / "analysis"
            / "enrichment"
            / "video"
            / "video_frame_ocr.py",
            PROJECT_ROOT
            / "crawler"
            / "analysis"
            / "enrichment"
            / "video"
            / "video_keyframe_selector.py",
            PROJECT_ROOT
            / "crawler"
            / "analysis"
            / "enrichment"
            / "video"
            / "video_ocr.py",
            PROJECT_ROOT
            / "crawler"
            / "analysis"
            / "enrichment"
            / "video"
            / "video_scene_analysis.py",
            PROJECT_ROOT
            / "crawler"
            / "analysis"
            / "enrichment"
            / "video"
            / "video_semantic_outputs.py",
        )
        if path.exists()
    )
    assert not legacy_sources, legacy_sources

    violations: list[str] = []
    for root_name in _MIGRATION_SCAN_ROOTS:
        for path in (PROJECT_ROOT / root_name).rglob("*.py"):
            for imported in _imported_module_paths(path):
                if any(
                    imported == legacy or imported.startswith(f"{legacy}.")
                    for legacy in _MIGRATED_ANALYSIS_PACKAGES
                ):
                    violations.append(
                        f"{path.relative_to(PROJECT_ROOT)}: {imported}"
                    )
    assert not violations, (
        "Legacy crawler media-support imports:\n"
        + "\n".join(sorted(violations))
    )


def test_preprocessing_package_initializer_is_empty() -> None:
    initializer = PROJECT_ROOT / "preprocessing" / "__init__.py"
    assert initializer.stat().st_size == 0


_PRIVACY_INSPECTION_PATH = (
    PREPROCESSING_ROOT / "media" / "privacy_inspection.py"
)

_ALLOWED_PRIVACY_INSPECTION_IMPORTS = {
    "__future__",
    "dataclasses",
    "preprocessing.preprocessing_input",
    "preprocessing.privacy.artifacts",
    "preprocessing.privacy.clearance",
    "preprocessing.privacy.field_inspection",
    "preprocessing.privacy.inspection.inspection_result",
    "preprocessing.privacy.text_privacy",
}

_FORBIDDEN_PRIVACY_INSPECTION_TOKENS = (
    "ApprovedObject(",
    "ApprovedTextField(",
    "PrivacyArtifactWorkspace",
    "build_receipt",
    "canonical_sha256",
    "derive_privacy_media_path",
    "hashlib",
    "inspect.getsourcefile",
    "is_file",
    "json",
    "metadata_text_fields",
    "privacy_artifact_name",
    "read_bytes",
    "replace(",
    "spans",
    "stat(",
    "verify_published_artifact",
)


def test_privacy_inspection_is_a_pure_release_policy_coordinator() -> None:
    """privacy_inspection.py only coordinates the media release policy."""

    path = _PRIVACY_INSPECTION_PATH
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))

    imported = _imported_module_paths(path)
    unexpected = imported - _ALLOWED_PRIVACY_INSPECTION_IMPORTS
    assert not unexpected, (
        "unexpected privacy_inspection imports: "
        + ", ".join(sorted(unexpected))
    )

    tokens = [
        token
        for token in _FORBIDDEN_PRIVACY_INSPECTION_TOKENS
        if token in text
    ]
    assert not tokens, "privacy_inspection ownership leaks: " + ", ".join(
        tokens
    )

    defined = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    assert "inspect_media_privacy" in defined
    assert "MediaPrivacyResult" in defined


def test_embedded_metadata_strip_lives_on_the_base_preprocessor() -> None:
    """The metadata-clean transform is owned by the shared base class."""

    base = PREPROCESSING_ROOT / "media" / "base_media_preprocessor.py"
    text = base.read_text(encoding="utf-8")
    assert "_prepare_embedded_metadata" in text
    assert "privacy_artifact_name" in text
    text = _PRIVACY_INSPECTION_PATH.read_text(encoding="utf-8")
    assert "prepare_embedded_metadata" not in text


def test_behavioral_class_surface_is_small() -> None:
    """Only substantive algorithm/orchestrator classes remain public."""

    allowed = {
        "MultimodalPreprocessor",
        "TextPreprocessor",
        "TextInputPreparer",
        "TextQualityScorer",
        "LanguageDetector",
        "BaseMediaPreprocessor",
        "ImagePreprocessor",
        "AudioPreprocessor",
        "VideoPreprocessor",
        "AudioFingerprintError",
        "DocumentTextUnavailableError",
        "AudioEmotionAnalyzer",
        "AudioEventAnalyzer",
        "ImageBlurEstimator",
        "ImageMetadataReader",
        "OcrBackendFailure",
        "OcrBackendUnavailable",
        "PillowGrayscaleArrayConverter",
        "PillowImageHashCalculator",
        "PillowImageMetadataAssembler",
        "ProsodyExtractor",
        "ProsodyValidationError",
        "SpeakerDiarizer",
        "SpeechTranscriber",
        "Mp4TailMetadataReader",
        "VideoFrameTextExtractionService",
        "VideoSafetyScanner",
    }
    found: set[str] = set()
    for path in PREPROCESSING_ROOT.rglob("*.py"):
        if (
            MEDIA_ADAPTERS_ROOT in path.parents
            or path.parent == MEDIA_ADAPTERS_ROOT
            or PRIVACY_ROOT in path.parents
            or path.parent == PRIVACY_ROOT
        ):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and not node.name.startswith(
                "_"
            ):
                decorators = {
                    getattr(d, "id", None)
                    or getattr(getattr(d, "func", None), "id", None)
                    for d in node.decorator_list
                }
                if "dataclass" in decorators:
                    continue
                bases = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        bases.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        bases.append(base.attr)
                if (
                    "StrEnum" in bases
                    or "Enum" in bases
                    or "Protocol" in bases
                ):
                    continue
                if node.name.endswith("Result") or node.name.endswith(
                    "Record"
                ):
                    continue
                found.add(node.name)
    unexpected = found - allowed
    missing = allowed - found
    assert not unexpected, (
        f"unexpected behavioral classes: {sorted(unexpected)}"
    )
    assert not missing, f"missing behavioral classes: {sorted(missing)}"
