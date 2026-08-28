"""Text preprocessing emits release records; curation owns selection."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from config.preprocessing.text_settings import (
    PreprocessingInputValidationSettings,
    TextQualityScorerSettings,
)
from preprocessing.preprocessing_input import (
    ExtractedTextContent,
    LanguageEvidence,
    PreprocessingInput,
)
from preprocessing.privacy.clearance import PrivacyClearanceStatus
from preprocessing.privacy.inspection.local_content_factories import (
    LocalDocumentPrivacyContentFactory,
)
from preprocessing.text.text_language import LanguageDetector
from preprocessing.text.text_metadata import build_text_metadata
from preprocessing.text.text_preparation import (
    PreparedTextDocument,
    TextInputPreparer,
)
from preprocessing.text.text_preprocessor import TextPreprocessor
from preprocessing.text.text_quality import TextQualityScorer
from tests.support.logging import TEST_LOGGER
from tests.support.privacy import build_test_pii_detector


def _input(*, source_id: str) -> PreprocessingInput:
    words = [f"term{index}" for index in range(100)]
    text = "This is a technical document " + " ".join(words) * 4
    return PreprocessingInput(
        source_id=source_id,
        source_url="https://example.test/shared",
        normalized_url="https://example.test/shared",
        domain="example.test",
        path="/shared",
        language_evidence=LanguageEvidence(
            language="en",
            confidence=0.99,
            source="crawler_test",
        ),
        modality="text",
        extracted_text_content=ExtractedTextContent(
            text=text,
            markdown=text,
            headings=("Technical document",),
            code_block_count=0,
            boilerplate_ratio=0.0,
            warnings=(),
        ),
    )


def test_text_preprocessor_does_not_select_url_or_exact_duplicates() -> None:
    preprocessor = TextPreprocessor(
        quality_scorer=TextQualityScorer(
            settings=TextQualityScorerSettings(),
            language_detector=LanguageDetector(),
        ),
        input_validation=PreprocessingInputValidationSettings(),
        input_preparer=TextInputPreparer(
            pii_detector=build_test_pii_detector(),
            document_content_factory=LocalDocumentPrivacyContentFactory(),
        ),
        logger=TEST_LOGGER,
        max_workers=1,
    )

    result = preprocessor.process(
        inputs=(_input(source_id="source-a"), _input(source_id="source-b"))
    )

    assert len(result.documents) == 2
    assert (
        result.documents[0].exact_duplicate_key
        == result.documents[1].exact_duplicate_key
    )
    assert all(
        document.near_duplicate_cluster_id is None
        for document in result.documents
    )
    assert result.diagnostics["dataset_selection_owner"] == "curation"


def _structured_input(*, source_id: str) -> PreprocessingInput:
    text = (
        "Structured document\n"
        "Page one text with measurements.\n"
        "Page two text with an overview figure."
    )
    return PreprocessingInput(
        source_id=source_id,
        source_url="https://example.test/structured",
        normalized_url="https://example.test/structured",
        domain="example.test",
        path="/structured",
        language_evidence=LanguageEvidence(
            language="en",
            confidence=0.99,
            source="crawler_test",
        ),
        modality="text",
        extracted_text_content=ExtractedTextContent(
            text=text,
            markdown=text,
            headings=("Structured document",),
            code_block_count=0,
            boilerplate_ratio=0.0,
            warnings=(),
        ),
        payload={
            "pages": [
                {
                    "page_number": 1,
                    "text": "Structured document\nPage one text with measurements.",
                    "text_start": 0,
                    "text_end": 56,
                },
                {
                    "page_number": 2,
                    "text": "Page two text with an overview figure.",
                    "text_start": 57,
                    "text_end": 96,
                },
            ],
            "tables": [
                {
                    "table_id": "table-1",
                    "page_number": 1,
                    "caption": "Measurements",
                    "cells": [["a", "b"]],
                    "confidence": 0.91,
                    "bounding_box": {"x": 1, "y": 2, "w": 3, "h": 4},
                }
            ],
            "figures": [
                {
                    "figure_id": "figure-1",
                    "page_number": 2,
                    "caption": "Overview",
                    "image_path": "figures/overview.png",
                    "confidence": 0.88,
                    "bounding_box": {"x": 5, "y": 6, "w": 7, "h": 8},
                }
            ],
        },
    )


def test_text_preprocessor_preserves_structural_identity_end_to_end() -> None:
    preprocessor = TextPreprocessor(
        quality_scorer=TextQualityScorer(
            settings=TextQualityScorerSettings(),
            language_detector=LanguageDetector(),
        ),
        input_validation=PreprocessingInputValidationSettings(),
        input_preparer=TextInputPreparer(
            pii_detector=build_test_pii_detector(),
            document_content_factory=LocalDocumentPrivacyContentFactory(),
        ),
        logger=TEST_LOGGER,
        max_workers=1,
    )

    result = preprocessor.process(
        inputs=(_structured_input(source_id="source-a"),)
    )

    document = result.documents[0]
    assert [page.page_number for page in document.pages] == [1, 2]
    assert document.tables[0].table_id == "table-1"
    assert document.tables[0].page_number == 1
    assert document.tables[0].caption == "Measurements"
    assert document.tables[0].cells == (("a", "b"),)
    assert document.tables[0].bounding_box == (1.0, 2.0, 4.0, 6.0)
    assert document.tables[0].confidence == 0.91
    assert document.figures[0].figure_id == "figure-1"
    assert document.figures[0].page_number == 2
    assert document.figures[0].caption == "Overview"
    assert document.figures[0].image_path is None
    assert document.figures[0].bounding_box == (5.0, 6.0, 12.0, 14.0)
    assert document.figures[0].confidence == 0.88
    assert document.headings[0].text == "Structured document"
    assert document.reading_order == tuple(
        block.block_id for block in document.text_blocks
    )
    assert document.source_text_mapping


def test_privacy_cleared_document_rejects_post_build_text_tampering() -> None:
    document = (
        _privacy_test_preprocessor()
        .process(inputs=(_structured_input(source_id="tamper-proof"),))
        .documents[0]
    )

    with pytest.raises(ValueError, match="exact privacy approval"):
        replace(
            document,
            tables=(replace(document.tables[0], caption="owner@example.com"),),
        )
    with pytest.raises(ValueError, match="exact privacy approval"):
        replace(
            document,
            tables=(replace(document.tables[0], cells=(("rogue", "b"),)),),
        )
    with pytest.raises(ValueError, match="exact privacy approval"):
        replace(
            document,
            figures=(
                replace(document.figures[0], caption="owner@example.com"),
            ),
        )
    with pytest.raises(ValueError, match="exact privacy approval"):
        replace(
            document,
            headings=(
                replace(
                    document.headings[0],
                    text="Structured",
                    text_end=len("Structured"),
                ),
            ),
        )
    with pytest.raises(ValueError, match="object-level privacy approval"):
        replace(
            document,
            pages=(
                replace(
                    document.pages[0],
                    rendered_image_path="pages/1.png",
                ),
                document.pages[1],
            ),
        )


def test_text_metadata_carries_no_document_structure_roundtrip() -> None:
    source_text = "Heading\nPage one text.\nPage two text."
    prepared = PreparedTextDocument(
        title="Structured document",
        text=source_text,
        markdown=source_text,
        headings=("Heading",),
        code_block_count=0,
        boilerplate_ratio=0.0,
        warnings=(),
    )
    metadata = build_text_metadata(
        extracted_document=prepared,
        normalized_text=source_text,
        language="en",
        path="/document.pdf",
        title="Structured document",
        rejection_reason=None,
    )

    assert metadata.headings == ("Heading",)
    assert metadata.heading_count == 1
    assert "document_structure" not in metadata.extra
    assert "removed_text_ratio" in metadata.extra


def _privacy_test_preprocessor() -> TextPreprocessor:
    return TextPreprocessor(
        quality_scorer=TextQualityScorer(
            settings=TextQualityScorerSettings(),
            language_detector=LanguageDetector(),
        ),
        input_validation=PreprocessingInputValidationSettings(),
        input_preparer=TextInputPreparer(
            pii_detector=build_test_pii_detector(),
            document_content_factory=LocalDocumentPrivacyContentFactory(),
        ),
        logger=TEST_LOGGER,
        max_workers=1,
    )


def test_text_preprocessor_redacts_every_released_structure_field() -> None:
    page_text = (
        "Owner body@example.com\n"
        "This technical page contains enough public explanatory text to be "
        "accepted while all private contact values are removed from every "
        "released structural representation."
    )
    item = PreprocessingInput(
        source_id="privacy-structure",
        source_url="https://example.test/privacy-structure",
        normalized_url="https://example.test/privacy-structure",
        domain="example.test",
        path="/privacy-structure",
        modality="text",
        extracted_text_content=ExtractedTextContent(
            text=page_text,
            markdown=page_text,
            headings=("Owner body@example.com",),
            code_block_count=0,
            boilerplate_ratio=0.0,
            warnings=(),
        ),
        payload={
            "pages": [
                {
                    "page_number": 1,
                    "text": page_text,
                    "rendered_image_path": "pages/page-owner@example.com.png",
                }
            ],
            "blocks": [
                {
                    "block_id": "contact-block",
                    "page_number": 1,
                    "text": page_text,
                    "text_start": 0,
                    "text_end": len(page_text),
                    "reading_order": 0,
                }
            ],
            "tables": [
                {
                    "table_id": "table-owner@example.com",
                    "page_number": 1,
                    "caption": "Maintainer table-owner@example.com",
                    "cells": [["Team", "cell-owner@example.com"]],
                }
            ],
            "figures": [
                {
                    "figure_id": "figure-owner@example.com",
                    "page_number": 1,
                    "caption": "Reviewer figure-owner@example.com",
                    "image_path": "figures/path-owner@example.com/image.png",
                }
            ],
        },
    )

    result = _privacy_test_preprocessor().process(inputs=(item,))

    assert not result.skipped_sources
    document = result.documents[0]
    assert document.privacy_clearance is not None
    assert (
        document.privacy_clearance.status is PrivacyClearanceStatus.REMEDIATED
    )
    assert (
        document.privacy_clearance.output_digest
        == hashlib.sha256(document.text.encode("utf-8")).hexdigest()
    )
    assert "@" not in document.text
    assert "@" not in document.markdown
    assert "@" not in document.text_blocks[0].text
    assert "@" not in (document.tables[0].caption or "")
    assert "@" not in document.tables[0].cells[0][1]
    assert "@" not in (document.figures[0].caption or "")
    assert document.figures[0].image_path is None
    assert document.pages[0].rendered_image_path is None
    assert "@" not in document.headings[0].text
    table_identity_digest = hashlib.sha256(
        b"table\n0\n[REDACTED_EMAIL_ADDRESS]"
    ).hexdigest()[:16]
    figure_identity_digest = hashlib.sha256(
        b"figure\n0\n[REDACTED_EMAIL_ADDRESS]"
    ).hexdigest()[:16]
    assert document.tables[0].table_id == (
        f"table:privacy:{table_identity_digest}"
    )
    assert document.figures[0].figure_id == (
        f"figure:privacy:{figure_identity_digest}"
    )
    approved_names = {
        field.name for field in document.privacy_clearance.approved_text_fields
    }
    assert {
        "structure:table:0:caption",
        "structure:table:0:table_id",
        "structure:table:0:cell:0:1",
        "structure:figure:0:caption",
        "structure:figure:0:figure_id",
        "structure:figure:0:image_path",
        "structure:page:0:rendered_image_path",
    }.issubset(approved_names)


def test_page_redaction_rebuilds_spans_blocks_and_headings() -> None:
    first_page = (
        "Contact a@b.co\n"
        "The first page explains a stable and public technical process in "
        "enough detail for preprocessing."
    )
    second_page = (
        "Second page\n"
        "The second page contains additional public implementation notes and "
        "remains aligned after the replacement changes the first page length."
    )
    source_text = f"{first_page}\n{second_page}"
    second_start = len(first_page) + 1
    item = PreprocessingInput(
        source_id="page-redaction",
        source_url="https://example.test/page-redaction",
        normalized_url="https://example.test/page-redaction",
        domain="example.test",
        path="/page-redaction",
        modality="text",
        extracted_text_content=ExtractedTextContent(
            text=source_text,
            markdown=source_text,
            headings=("Contact a@b.co", "Second page"),
            code_block_count=0,
            boilerplate_ratio=0.0,
            warnings=(),
        ),
        payload={
            "pages": [
                {"page_number": 1, "text": first_page},
                {"page_number": 2, "text": second_page},
            ],
            "blocks": [
                {
                    "block_id": "page-one",
                    "page_number": 1,
                    "text": first_page,
                    "text_start": 0,
                    "text_end": len(first_page),
                    "reading_order": 0,
                },
                {
                    "block_id": "page-two",
                    "page_number": 2,
                    "text": second_page,
                    "text_start": second_start,
                    "text_end": second_start + len(second_page),
                    "reading_order": 1,
                },
            ],
        },
    )

    result = _privacy_test_preprocessor().process(inputs=(item,))

    assert not result.skipped_sources
    document = result.documents[0]
    assert "[REDACTED_EMAIL_ADDRESS]" in document.text
    assert len(document.text) > len(source_text)
    assert document.pages[0].text_start == 0
    assert document.pages[0].text_end == document.pages[1].text_start - 1
    assert document.pages[1].text_end == len(document.text)
    for page, block in zip(document.pages, document.text_blocks, strict=True):
        canonical_page = document.text[page.text_start : page.text_end]
        assert block.text == canonical_page
        assert (block.text_start, block.text_end) == (
            page.text_start,
            page.text_end,
        )
    assert document.headings[0].text == ("Contact [REDACTED_EMAIL_ADDRESS]")
    assert all(
        mapping.normalized_end <= len(document.text)
        for mapping in document.source_text_mapping
    )
    for mapping in document.source_text_mapping:
        assert (
            source_text[mapping.source_start : mapping.source_end]
            == (
                document.text[
                    mapping.normalized_start : mapping.normalized_end
                ]
            )
        )
    redacted_start = source_text.index("a@b.co")
    redacted_end = redacted_start + len("a@b.co")
    assert all(
        mapping.source_end <= redacted_start
        or mapping.source_start >= redacted_end
        for mapping in document.source_text_mapping
    )


def test_structure_paths_require_separate_visual_object_approval() -> None:
    item = _structured_input(source_id="unsafe-paths")
    pages = item.payload["pages"]
    figures = item.payload["figures"]
    tables = item.payload["tables"]
    assert isinstance(pages, list)
    assert isinstance(pages[0], dict)
    assert isinstance(figures, list)
    assert isinstance(figures[0], dict)
    assert isinstance(tables, list)
    assert isinstance(tables[0], dict)
    pages[0]["rendered_image_path"] = "pages/1.png"
    figures[0]["image_path"] = "figures/overview.png"
    tables[0]["table_id"] = "../../arbitrary id"
    item.payload["blocks"] = [
        {
            "block_id": "../../arbitrary block",
            "page_number": 1,
            "text": "Structured document",
            "text_start": 0,
            "text_end": len("Structured document"),
            "reading_order": 0,
            "source": "attacker-string",
            "block_type": "../../private-label",
        }
    ]

    result = _privacy_test_preprocessor().process(inputs=(item,))

    assert not result.skipped_sources
    document = result.documents[0]
    assert document.pages[0].rendered_image_path is None
    assert document.figures[0].image_path is None
    assert document.tables[0].table_id.startswith("table:privacy:")
    assert "/" not in document.tables[0].table_id
    assert all(block.source == "native" for block in document.text_blocks)
    assert all(
        block.block_type == "paragraph" for block in document.text_blocks
    )


def test_leading_blank_page_retains_valid_zero_width_span() -> None:
    visible_page = (
        "Visible page\n"
        "This public technical explanation remains canonical even when the "
        "source document begins with a page that contains no extractable text."
    )
    item = PreprocessingInput(
        source_id="blank-first-page",
        source_url="https://example.test/blank-first-page",
        normalized_url="https://example.test/blank-first-page",
        domain="example.test",
        path="/blank-first-page",
        modality="text",
        extracted_text_content=ExtractedTextContent(
            text=visible_page,
            markdown=visible_page,
            headings=("Visible page",),
            code_block_count=0,
            boilerplate_ratio=0.0,
            warnings=(),
        ),
        payload={
            "pages": [
                {"page_number": 1, "text": ""},
                {"page_number": 2, "text": visible_page},
            ]
        },
    )

    result = _privacy_test_preprocessor().process(inputs=(item,))

    assert not result.skipped_sources
    document = result.documents[0]
    assert document.text == visible_page
    assert (document.pages[0].text_start, document.pages[0].text_end) == (
        0,
        0,
    )
    assert (document.pages[1].text_start, document.pages[1].text_end) == (
        0,
        len(visible_page),
    )


def test_malformed_structure_is_quarantined_without_aborting_batch() -> None:
    malformed = _structured_input(source_id="malformed")
    tables = malformed.payload["tables"]
    assert isinstance(tables, list)
    assert isinstance(tables[0], dict)
    tables[0]["confidence"] = 2.0
    invalid_cells = _structured_input(source_id="invalid-cells")
    invalid_cell_tables = invalid_cells.payload["tables"]
    assert isinstance(invalid_cell_tables, list)
    assert isinstance(invalid_cell_tables[0], dict)
    invalid_cell_tables[0]["cells"] = [["approved", 42]]
    invalid_page = _structured_input(source_id="invalid-page")
    invalid_pages = invalid_page.payload["pages"]
    assert isinstance(invalid_pages, list)
    assert isinstance(invalid_pages[0], dict)
    invalid_pages[0]["page_number"] = 0
    valid = _structured_input(source_id="valid-after-malformed")

    result = _privacy_test_preprocessor().process(
        inputs=(malformed, invalid_cells, invalid_page, valid)
    )

    assert [document.source_id for document in result.documents] == [
        "valid-after-malformed"
    ]
    assert result.skipped_sources["malformed"] == (
        "preprocessing_exception:ValueError"
    )
    assert result.skipped_sources["invalid-cells"] == (
        "preprocessing_exception:ValueError"
    )
    assert result.skipped_sources["invalid-page"] == (
        "privacy_structure_remediation_inconsistent"
    )
