"""Architecture boundary guards for the extraction cutover.

These tests lock the production architecture invariants from the deep
extraction plan so dual routes and layer leaks do not return silently.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EXTRACTION_ROOT = _REPO_ROOT / "crawler" / "extraction"
_MODALITIES_ROOT = _EXTRACTION_ROOT / "modalities"
_PAYLOADS_ROOT = _EXTRACTION_ROOT / "payloads"


def _python_files(root: Path) -> tuple[Path, ...]:
    return tuple(sorted(root.rglob("*.py")))


def _module_source_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".", maxsplit=1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.module is None:
                continue
            module = node.module or ""
            if module:
                imports.add(module.split(".", maxsplit=1)[0])
    return imports


def test_extraction_package_does_not_import_preprocessing() -> None:
    offenders: list[str] = []
    for path in _python_files(_EXTRACTION_ROOT):
        imports = _module_source_imports(path)
        if "preprocessing" in imports:
            offenders.append(str(path.relative_to(_REPO_ROOT)))
    assert offenders == []


def test_payload_result_dataclasses_exclude_enrichment_fields() -> None:
    from crawler.extraction.payloads.audio_payload_extractor import (
        AudioPayloadExtractionResult,
    )
    from crawler.extraction.payloads.document_payload_extractor import (
        DocumentPayloadExtractionResult,
    )
    from crawler.extraction.payloads.image_payload_extractor import (
        ImagePayloadExtractionResult,
    )
    from crawler.extraction.payloads.video_payload_extractor import (
        VideoPayloadExtractionResult,
    )

    forbidden = {
        "ocr_text",
        "transcript_text",
        "blur_variance",
        "quality_score",
        "keyframes",
        "scene_graph",
        "text_preview",
        "extracted_text",
    }
    for cls in (
        ImagePayloadExtractionResult,
        AudioPayloadExtractionResult,
        VideoPayloadExtractionResult,
        DocumentPayloadExtractionResult,
    ):
        fields = set(cls.__dataclass_fields__)
        assert fields.isdisjoint(forbidden), cls.__name__


def test_removed_extraction_modules_stay_absent() -> None:
    for name in (
        "crawler.analysis.enrichment.pages.page_analyzer",
        "crawler.extraction.assets.asset_extractor",
        "crawler.extraction.urls.extractor",
        "preprocessing.cleaning.html_processing",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(name)


def test_production_page_composition_uses_page_content_extractor() -> None:
    from orchestration.composition.runtime.handler_composition.page import (
        build_page_handler,
    )

    source = inspect.getsource(build_page_handler)
    assert "PageContentExtractor" in source
    assert "PageAnalyzer" not in source
    assert "AssetExtractor" not in source


def test_production_feed_composition_uses_feed_content_extractor() -> None:
    from orchestration.composition.runtime.handler_composition.feed import (
        build_feed_handler,
    )

    source = inspect.getsource(build_feed_handler)
    assert "FeedContentExtractor" in source
    assert "FeedAnalyzer" not in source
    assert "analyzer=" not in source
    assert "feed_content_extractor=" in source


def test_production_media_composition_uses_payload_extractors() -> None:
    from orchestration.composition.runtime.handler_composition import (
        audio as audio_module,
    )
    from orchestration.composition.runtime.handler_composition import (
        document as document_module,
    )
    from orchestration.composition.runtime.handler_composition import (
        image as image_module,
    )
    from orchestration.composition.runtime.handler_composition import (
        video as video_module,
    )

    image_handler_source = inspect.getsource(image_module.build_image_handler)
    image_source = inspect.getsource(image_module._build_image_analyzer)
    assert "ImagePayloadExtractor" in image_source
    assert "max_decode_pixels=image_acceptance.max_decode_pixels" in (
        image_handler_source
    )
    assert (
        image_source.count(
            "max_decode_pixels=image_acceptance.max_decode_pixels"
        )
        == 2
    )

    audio_source = inspect.getsource(audio_module._build_audio_analyzer)
    assert "AudioPayloadExtractor" in audio_source

    document_source = inspect.getsource(
        document_module._build_document_analyzer
    )
    assert "DocumentPayloadExtractor" in document_source

    video_source = inspect.getsource(video_module._build_video_analyzer)
    assert "VideoPayloadExtractor" in video_source
    assert "clip_writer=OpenCvVideoClipWriter()" in video_source


def test_document_composition_uses_one_payload_classification_path() -> None:
    from orchestration.composition.runtime.handler_composition.document import (
        _build_document_analyzer,
    )

    source = inspect.getsource(_build_document_analyzer)
    assert "DocumentPayloadExtractor" in source
    assert "DocumentTextReader" in source
    assert "PdfTextReader" in source
    assert "settings=document_settings" not in source
    for removed_wrapper in (
        "DocumentPreviewReader",
        "PdfPageTextExtractor",
        "PyPdf2PdfReaderLoader",
        "document_metadata_runtime",
    ):
        assert removed_wrapper not in source


def test_reference_extractors_do_not_traverse_element_subtrees() -> None:
    """Reference extractors must consume structural ownership from the index."""

    reference_files = (
        "image_extractor.py",
        "audio_extractor.py",
        "video_extractor.py",
        "document_extractor.py",
    )
    for name in reference_files:
        source = (_MODALITIES_ROOT / name).read_text(encoding="utf-8")
        assert ".find_all(" not in source
        assert "_find_children" not in source
        assert "_is_nested_under" not in source


def test_page_extraction_result_has_no_parsed_document_field() -> None:
    from crawler.extraction.modalities.page_content_extractor import (
        PageExtractionResult,
    )

    fields = set(PageExtractionResult.__dataclass_fields__)
    assert "parsed_document" not in fields
    assert "document" not in fields
    assert "soup" not in fields


def test_preprocessing_input_has_no_html_body() -> None:
    from preprocessing.preprocessing_input import PreprocessingInput

    fields = set(getattr(PreprocessingInput, "model_fields", {}) or {})
    if not fields and hasattr(PreprocessingInput, "__annotations__"):
        fields = set(PreprocessingInput.__annotations__)
    assert "html_body" not in fields


def test_removed_feed_module_is_absent() -> None:
    removed_module = (
        _REPO_ROOT
        / "crawler"
        / "analysis"
        / "enrichment"
        / "feeds"
        / "feed_analyzer.py"
    )
    assert not removed_module.exists()
