"""Page preprocessing uses ExtractedTextContent without HTML re-parse."""

from __future__ import annotations

import importlib

import pytest

from preprocessing.preprocessing_input import (
    ExtractedTextContent,
    LanguageEvidence,
    PreprocessingInput,
)
from preprocessing.privacy.inspection.local_content_factories import (
    LocalDocumentPrivacyContentFactory,
)
from preprocessing.text.text_preparation import TextInputPreparer
from tests.support.privacy import build_test_pii_detector


def test_normalizer_uses_extracted_text_content() -> None:
    content = ExtractedTextContent(
        text="Main body text about science.",
        markdown="# Title\n\nMain body text about science.",
        headings=("Title",),
        code_block_count=0,
        boilerplate_ratio=0.05,
        warnings=(),
    )
    item = PreprocessingInput(
        source_id="s1",
        source_url="https://example.test/a",
        normalized_url="https://example.test/a",
        domain="example.test",
        path="/a",
        language_evidence=LanguageEvidence(language="en"),
        title="Title",
        modality="text",
        extracted_text_content=content,
    )
    prepared = TextInputPreparer(
        pii_detector=build_test_pii_detector(),
        document_content_factory=LocalDocumentPrivacyContentFactory(),
    ).prepare_valid_input(batch_index=0, item=item)
    assert prepared.rejection_reason is None
    assert prepared.extracted_document is not None
    assert prepared.extracted_document.headings == ("Title",)
    assert prepared.extracted_document.markdown.startswith("# Title")
    assert "science" in prepared.normalized_text


def test_active_preprocessing_does_not_import_html_extractors() -> None:
    for module_name in (
        "preprocessing.cleaning.preprocessing_html_text_extractor",
        "preprocessing.cleaning.preprocessing_markdown_renderer",
        "preprocessing.cleaning.html_processing",
        "preprocessing.cleaning.preprocessed_html_document",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_name)

    from orchestration.composition import (
        dataset_dependencies as dataset_deps,
    )
    from orchestration.composition import (
        preprocessing_dependencies as pre_deps,
    )

    assert hasattr(pre_deps, "build_multimodal_preprocessor")
    assert not hasattr(pre_deps, "build_text_chunk_splitter")
    assert hasattr(dataset_deps, "build_text_chunk_splitter")
    assert not hasattr(pre_deps, "PreprocessingComponents")


def test_preprocessing_input_has_no_html_body_field() -> None:
    fields = PreprocessingInput.__dataclass_fields__
    assert "html_body" not in fields
    assert "extracted_text_content" in fields
