"""Curation selects URL and content candidates after preprocessing."""

from __future__ import annotations

from types import SimpleNamespace

from crawler.curation.documents.assembler import (
    _ProvisionalDocument,
    _select_best_by_key,
)
from preprocessing.preprocessed_document import (
    PreprocessedDocument,
    PreprocessingMetadata,
)
from preprocessing.preprocessing_quality import PreprocessingQualityResult


def _candidate(
    *,
    document_id: str,
    source_url: str,
    exact_key: str,
    quality: float,
    normalized_url: str | None = None,
    text: str,
) -> _ProvisionalDocument:
    document = PreprocessedDocument(
        document_id=document_id,
        source_id=document_id,
        source_url=source_url,
        title=None,
        text=text,
        markdown=text,
        language="en",
        metadata=PreprocessingMetadata(
            char_count=len(text),
            token_count_estimate=len(text.split()),
            line_count=1,
            paragraph_count=1,
            heading_count=0,
        ),
        quality=PreprocessingQualityResult(
            score=quality,
            bucket="silver",
            rejection_reason=None,
            token_count_estimate=len(text.split()),
            modality="text",
            language="en",
            alignment_score=quality,
            signals={},
        ),
        exact_duplicate_key=exact_key,
        domain="example.test",
    )
    entry = SimpleNamespace(
        record=SimpleNamespace(
            fetched_at="2026-07-31T00:00:00+00:00",
            domain="example.test",
            normalized_url=normalized_url or source_url,
        )
    )
    return _ProvisionalDocument(
        entry=entry,
        preprocessed_document=document,
        domain_governance=None,
    )


def test_best_valid_candidate_is_selected_per_url_then_exact_content() -> None:
    lower = _candidate(
        document_id="lower",
        source_url="https://EXAMPLE.test/article?utm_source=crawler",
        normalized_url="https://example.test/article",
        exact_key="content-a",
        quality=0.6,
        text="lower quality version",
    )
    better = _candidate(
        document_id="better",
        source_url="https://example.test/article",
        exact_key="content-b",
        quality=0.9,
        text="better and more complete quality version",
    )
    copied = _candidate(
        document_id="copy",
        source_url="https://mirror.test/article",
        exact_key="content-b",
        quality=0.8,
        text="same released content",
    )

    url_selected = _select_best_by_key(
        provisional=[lower, better, copied],
        key=lambda item: item.entry.record.normalized_url,
    )
    exact_selected = _select_best_by_key(
        provisional=url_selected,
        key=lambda item: item.preprocessed_document.exact_duplicate_key,
    )

    assert [
        item.preprocessed_document.document_id for item in url_selected
    ] == ["better", "copy"]
    assert [
        item.preprocessed_document.document_id for item in exact_selected
    ] == ["better"]
