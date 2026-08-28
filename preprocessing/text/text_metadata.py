"""Build preprocessing metadata for text documents."""

from __future__ import annotations

from typing import TYPE_CHECKING

from preprocessing.preprocessed_document import PreprocessingMetadata

if TYPE_CHECKING:
    from preprocessing.text.text_preparation import PreparedTextDocument


def build_text_metadata(
    *,
    extracted_document: PreparedTextDocument,
    normalized_text: str,
    language: str | None,
    path: str,
    title: str | None,
    rejection_reason: str | None,
    diagnostics: dict[str, object] | None = None,
) -> PreprocessingMetadata:
    """Build structured preprocessing metadata from extraction outputs."""

    statistics = _calculate_text_statistics(text=normalized_text)
    extra: dict[str, object] = {
        "removed_text_ratio": round(extracted_document.removed_text_ratio, 4),
    }
    extra.update(diagnostics or {})
    return PreprocessingMetadata(
        char_count=int(statistics["char_count"]),
        token_count_estimate=int(statistics["token_count_estimate"]),
        line_count=int(statistics["line_count"]),
        paragraph_count=int(statistics["paragraph_count"]),
        heading_count=len(extracted_document.headings),
        headings=extracted_document.headings,
        title=title,
        language=language,
        ascii_ratio=statistics["ascii_ratio"],
        unicode_ratio=statistics["unicode_ratio"],
        code_block_count=extracted_document.code_block_count,
        boilerplate_ratio=extracted_document.boilerplate_ratio,
        content_role=classify_content_role(
            path=path,
            title=title,
            rejection_reason=rejection_reason,
        ),
        warnings=extracted_document.warnings,
        extra=extra,
    )


def classify_content_role(
    *,
    path: str,
    title: str | None,
    rejection_reason: str | None,
) -> str:
    normalized = " ".join(
        value.lower() for value in (path, title or "") if value
    )
    if any(
        token in normalized for token in ("rss", "feed", "podcast.xml", "atom")
    ):
        return "feed_page"
    if any(
        token in normalized
        for token in (
            "multimedia",
            "gallery",
            "images",
            "videos",
            "video",
            "audio",
            "podcast",
            "media",
        )
    ):
        return "media_index_page"
    if rejection_reason == "text_too_short":
        return "thin_but_useful_page"
    if rejection_reason == "boilerplate_heavy":
        return "hub_page"
    return "content_document"


def _calculate_text_statistics(*, text: str) -> dict[str, int | float]:
    counters = _count_text_statistics(text=text)
    char_count = counters["char_count"]
    ascii_ratio = counters["ascii_count"] / char_count if char_count else 0.0
    unicode_ratio = (
        (char_count - counters["ascii_count"]) / char_count
        if char_count
        else 0.0
    )
    return {
        "char_count": char_count,
        "token_count_estimate": counters["token_count_estimate"],
        "line_count": counters["line_count"],
        "paragraph_count": counters["paragraph_count"],
        "ascii_ratio": ascii_ratio,
        "unicode_ratio": unicode_ratio,
    }


def _count_text_statistics(*, text: str) -> dict[str, int]:
    token_count = 0
    line_count = 0
    paragraph_count = 0
    ascii_count = 0
    in_space_delimited_token = False
    line_has_text = False
    paragraph_has_text = False
    newline_run = 0

    for char in text:
        if ord(char) < 128:
            ascii_count += 1
        if char.isspace():
            in_space_delimited_token = False
        elif not in_space_delimited_token:
            token_count += 1
            in_space_delimited_token = True
        if char == "\n":
            if line_has_text:
                line_count += 1
            line_has_text = False
            newline_run += 1
            if newline_run == 2:
                if paragraph_has_text:
                    paragraph_count += 1
                paragraph_has_text = False
            continue
        newline_run = 0
        if not char.isspace():
            line_has_text = True
            paragraph_has_text = True

    if line_has_text:
        line_count += 1
    if paragraph_has_text:
        paragraph_count += 1

    return {
        "char_count": len(text),
        "token_count_estimate": token_count,
        "line_count": line_count,
        "paragraph_count": paragraph_count,
        "ascii_count": ascii_count,
    }
