"""Chunk splitting for preprocessed documents."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from logger.project_logger import ProjectLogger
from preprocessing.preprocessed_document import PreprocessedDocument
from preprocessing.provenance import stable_identifier

if TYPE_CHECKING:
    from collections.abc import Callable

    from config.settings.datasets import DocumentChunkerSettings


@dataclass(frozen=True, slots=True)
class TextChunk:
    """One immutable chunk derived from a curated preprocessed document."""

    chunk_id: str
    document_id: str
    chunk_index: int
    text: str
    start_char: int
    end_char: int
    token_count_estimate: int
    section_path: tuple[str, ...]
    quality_score: float
    split: str | None
    language: str | None = None
    title: str | None = None
    exact_duplicate_key: str | None = None
    near_duplicate_cluster_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class TextChunkSplitter:
    """Split preprocessed documents into overlapping training chunks."""

    def __init__(
        self,
        *,
        settings: DocumentChunkerSettings,
        assign_split: Callable[[str], str | None],
        logger: ProjectLogger,
    ) -> None:
        self._settings = settings
        self._assign_split = assign_split
        self._logger = logger

    def build(
        self,
        *,
        documents: tuple[PreprocessedDocument, ...],
    ) -> tuple[TextChunk, ...]:
        """Build chunks for accepted preprocessed documents."""

        chunks: list[TextChunk] = []
        for document in documents:
            if document.quality.bucket == "reject":
                continue
            text = document.text.strip()
            if not text:
                continue
            tokens = _tokenize_with_offsets(text=text)
            if not tokens:
                continue

            target = max(
                self._settings.chunk_min_target_tokens,
                self._settings.chunk_target_tokens,
            )
            overlap = min(target - 1, self._settings.chunk_overlap_tokens)
            step = max(1, target - overlap)
            split_key = (
                document.near_duplicate_cluster_id
                or document.exact_duplicate_key
                or document.document_id
            )
            split = self._assign_split(split_key)
            section_path = _resolve_section_path(document=document)

            chunk_index = 0
            for paragraph_start, paragraph_end in _paragraph_token_ranges(
                text=text,
                tokens=tokens,
            ):
                for token_start in range(paragraph_start, paragraph_end, step):
                    token_end = min(token_start + target, paragraph_end)
                    window_tokens = tokens[token_start:token_end]
                    if not window_tokens:
                        continue
                    start_char = window_tokens[0].start
                    end_char = window_tokens[-1].end
                    chunk_text, start_char, end_char = _trim_chunk_bounds(
                        text=text,
                        start_char=start_char,
                        end_char=end_char,
                    )
                    if not chunk_text:
                        continue
                    chunks.append(
                        TextChunk(
                            chunk_id=stable_identifier(
                                prefix="chunk",
                                parts=(document.document_id, str(chunk_index)),
                            ),
                            document_id=document.document_id,
                            chunk_index=chunk_index,
                            text=chunk_text,
                            start_char=start_char,
                            end_char=end_char,
                            token_count_estimate=len(window_tokens),
                            section_path=section_path,
                            quality_score=document.quality.score,
                            split=split,
                            language=document.language,
                            title=document.title,
                            exact_duplicate_key=document.exact_duplicate_key,
                            near_duplicate_cluster_id=(
                                document.near_duplicate_cluster_id
                            ),
                        )
                    )
                    chunk_index += 1

                    if token_end >= paragraph_end:
                        break

        self._logger.info(
            "dataset_text_chunks_built",
            total=len(chunks),
        )
        return tuple(chunks)


@dataclass(frozen=True, slots=True)
class _Token:
    text: str
    start: int
    end: int


def _tokenize_with_offsets(*, text: str) -> list[_Token]:
    return [
        _Token(text=match.group(0), start=match.start(), end=match.end())
        for match in re.finditer(r"\S+", text)
    ]


def _paragraph_token_ranges(
    *,
    text: str,
    tokens: list[_Token],
) -> tuple[tuple[int, int], ...]:
    if not tokens:
        return ()

    paragraph_char_ranges = _paragraph_char_ranges(text=text)
    if not paragraph_char_ranges:
        return ((0, len(tokens)),)

    ranges: list[tuple[int, int]] = []
    token_index = 0
    for start_char, end_char in paragraph_char_ranges:
        while (
            token_index < len(tokens) and tokens[token_index].end <= start_char
        ):
            token_index += 1
        start_index = token_index
        while (
            token_index < len(tokens) and tokens[token_index].start < end_char
        ):
            token_index += 1
        if start_index < token_index:
            ranges.append((start_index, token_index))

    return tuple(ranges) or ((0, len(tokens)),)


def _paragraph_char_ranges(*, text: str) -> tuple[tuple[int, int], ...]:
    ranges: list[tuple[int, int]] = []
    start = 0
    for separator in re.finditer(r"\n\s*\n+", text):
        _append_trimmed_range(
            ranges=ranges,
            text=text,
            start=start,
            end=separator.start(),
        )
        start = separator.end()
    _append_trimmed_range(
        ranges=ranges,
        text=text,
        start=start,
        end=len(text),
    )
    return tuple(ranges)


def _append_trimmed_range(
    *,
    ranges: list[tuple[int, int]],
    text: str,
    start: int,
    end: int,
) -> None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start < end:
        ranges.append((start, end))


def _trim_chunk_bounds(
    *,
    text: str,
    start_char: int,
    end_char: int,
) -> tuple[str, int, int]:
    while start_char < end_char and text[start_char].isspace():
        start_char += 1
    while end_char > start_char and text[end_char - 1].isspace():
        end_char -= 1
    return text[start_char:end_char], start_char, end_char


def _resolve_section_path(
    *,
    document: PreprocessedDocument,
) -> tuple[str, ...]:
    if document.metadata.headings:
        return tuple(
            heading for heading in document.metadata.headings[:3] if heading
        )
    if document.title:
        return (document.title,)
    return ()
