"""Typed, canonical document outputs from preprocessing."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, TypedDict

from preprocessing.text.document_structure_privacy import (
    approved_structure_identity,
    is_safe_structure_block_type,
    is_safe_structure_identity,
)

if TYPE_CHECKING:
    from preprocessing.preprocessing_quality import PreprocessingQualityResult
    from preprocessing.privacy.clearance import PrivacyClearance

BoundingBox = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class DocumentPage:
    page_number: int
    text_start: int
    text_end: int
    rendered_image_path: str | None = None
    width: int | None = None
    height: int | None = None

    def __post_init__(self) -> None:
        if self.page_number < 1:
            raise ValueError("document page_number must be positive")
        if self.text_start < 0 or self.text_end < self.text_start:
            raise ValueError("document page text span is invalid")
        if self.width is not None and self.width <= 0:
            raise ValueError("document page width must be positive")
        if self.height is not None and self.height <= 0:
            raise ValueError("document page height must be positive")


@dataclass(frozen=True, slots=True)
class DocumentTextBlock:
    block_id: str
    page_number: int
    text: str
    text_start: int
    text_end: int
    reading_order: int
    block_type: str = "paragraph"
    source: str = "native"
    bounding_box: BoundingBox | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not self.block_id.strip() or not self.text.strip():
            raise ValueError("document text blocks require id and text")
        if self.page_number < 1 or self.reading_order < 0:
            raise ValueError("document text block position is invalid")
        if self.text_start < 0 or self.text_end < self.text_start:
            raise ValueError("document text block span is invalid")
        if self.source not in {"native", "ocr"}:
            raise ValueError(
                "document text block source must be native or ocr"
            )
        if not is_safe_structure_block_type(self.block_type):
            raise ValueError("document text block type is not supported")
        _validate_box(self.bounding_box)
        _validate_confidence(self.confidence)


@dataclass(frozen=True, slots=True)
class DocumentTable:
    table_id: str
    page_number: int
    cells: tuple[tuple[str, ...], ...]
    bounding_box: BoundingBox | None = None
    caption: str | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not self.table_id.strip() or self.page_number < 1:
            raise ValueError("document table identity is invalid")
        _validate_box(self.bounding_box)
        _validate_confidence(self.confidence)


@dataclass(frozen=True, slots=True)
class DocumentFigure:
    figure_id: str
    page_number: int
    bounding_box: BoundingBox | None = None
    caption: str | None = None
    image_path: str | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not self.figure_id.strip() or self.page_number < 1:
            raise ValueError("document figure identity is invalid")
        _validate_box(self.bounding_box)
        _validate_confidence(self.confidence)


@dataclass(frozen=True, slots=True)
class DocumentHeading:
    heading_id: str
    text: str
    text_start: int
    text_end: int
    level: int = 1
    page_number: int | None = None

    def __post_init__(self) -> None:
        if not self.heading_id.strip() or not self.text.strip():
            raise ValueError("document headings require id and text")
        if self.text_start < 0 or self.text_end < self.text_start:
            raise ValueError("document heading span is invalid")
        if self.level < 1:
            raise ValueError("document heading level must be positive")
        if self.page_number is not None and self.page_number < 1:
            raise ValueError("document heading page_number must be positive")


@dataclass(frozen=True, slots=True)
class SourceTextMapping:
    source_start: int
    source_end: int
    normalized_start: int
    normalized_end: int

    def __post_init__(self) -> None:
        values = (
            self.source_start,
            self.source_end,
            self.normalized_start,
            self.normalized_end,
        )
        if min(values) < 0:
            raise ValueError(
                "source text mapping offsets must be non-negative"
            )
        if self.source_end < self.source_start:
            raise ValueError("source text mapping source span is invalid")
        if self.normalized_end < self.normalized_start:
            raise ValueError("source text mapping normalized span is invalid")


class _CanonicalDocumentStructure(TypedDict):
    """Typed result of canonical document-structure normalization."""

    pages: tuple[DocumentPage, ...]
    text_blocks: tuple[DocumentTextBlock, ...]
    tables: tuple[DocumentTable, ...]
    figures: tuple[DocumentFigure, ...]
    headings: tuple[DocumentHeading, ...]
    reading_order: tuple[str, ...]
    source_text_mapping: tuple[SourceTextMapping, ...]


@dataclass(frozen=True, slots=True)
class PreprocessingMetadata:
    char_count: int
    token_count_estimate: int
    line_count: int
    paragraph_count: int
    heading_count: int
    headings: tuple[str, ...] = ()
    title: str | None = None
    language: str | None = None
    ascii_ratio: float = 0.0
    unicode_ratio: float = 0.0
    code_block_count: int = 0
    boilerplate_ratio: float = 0.0
    content_role: str = "content_document"
    warnings: tuple[str, ...] = ()
    extra: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PreprocessedDocument:
    document_id: str
    source_id: str | None
    source_url: str
    title: str | None
    text: str
    markdown: str
    language: str | None
    metadata: PreprocessingMetadata
    quality: PreprocessingQualityResult
    exact_duplicate_key: str
    near_duplicate_cluster_id: str | None = None
    is_near_duplicate: bool = False
    warnings: tuple[str, ...] = ()
    domain: str | None = None
    path: str | None = None
    allow_training: bool | None = None
    privacy_clearance: PrivacyClearance | None = None
    pages: tuple[DocumentPage, ...] = ()
    text_blocks: tuple[DocumentTextBlock, ...] = ()
    tables: tuple[DocumentTable, ...] = ()
    figures: tuple[DocumentFigure, ...] = ()
    headings: tuple[DocumentHeading, ...] = ()
    reading_order: tuple[str, ...] = ()
    source_text_mapping: tuple[SourceTextMapping, ...] = ()

    def __post_init__(self) -> None:
        page_numbers = {page.page_number for page in self.pages}
        if len(page_numbers) != len(self.pages):
            raise ValueError("document page numbers must be unique")
        text_length = len(self.text)
        for page in self.pages:
            if page.text_end > text_length:
                raise ValueError(
                    "document page text span exceeds document text"
                )
        sorted_pages = sorted(self.pages, key=lambda p: p.page_number)
        for i in range(len(sorted_pages) - 1):
            current = sorted_pages[i]
            next_page = sorted_pages[i + 1]
            if current.text_end > next_page.text_start:
                raise ValueError(
                    f"document page spans overlap: page {current.page_number} "
                    f"ends at {current.text_end}, page {next_page.page_number} "
                    f"starts at {next_page.text_start}"
                )
        block_ids = [block.block_id for block in self.text_blocks]
        if len(set(block_ids)) != len(block_ids):
            raise ValueError("document text block ids must be unique")
        for block in self.text_blocks:
            if page_numbers and block.page_number not in page_numbers:
                raise ValueError("text block references an unknown page")
            if block.text_end > text_length:
                raise ValueError(
                    "document text block span exceeds document text"
                )
        for table in self.tables:
            if page_numbers and table.page_number not in page_numbers:
                raise ValueError("table references an unknown page")
        table_ids = [table.table_id for table in self.tables]
        if len(set(table_ids)) != len(table_ids):
            raise ValueError("document table ids must be unique")
        for figure in self.figures:
            if page_numbers and figure.page_number not in page_numbers:
                raise ValueError("figure references an unknown page")
        figure_ids = [figure.figure_id for figure in self.figures]
        if len(set(figure_ids)) != len(figure_ids):
            raise ValueError("document figure ids must be unique")
        for heading in self.headings:
            if (
                page_numbers
                and heading.page_number is not None
                and heading.page_number not in page_numbers
            ):
                raise ValueError("heading references an unknown page")
            if heading.text_end > text_length:
                raise ValueError("document heading span exceeds document text")
        heading_ids = [heading.heading_id for heading in self.headings]
        if len(set(heading_ids)) != len(heading_ids):
            raise ValueError("document heading ids must be unique")
        if self.text_blocks and not self.reading_order:
            raise ValueError("reading order must cover all text blocks")
        if self.reading_order:
            if len(set(self.reading_order)) != len(self.reading_order):
                raise ValueError("reading order contains duplicate blocks")
            if set(self.reading_order) != set(block_ids):
                raise ValueError(
                    "reading order must contain every text block exactly once"
                )
        for block in self.text_blocks:
            if self.text[block.text_start : block.text_end] != block.text:
                raise ValueError(
                    "document text block must match its canonical text span"
                )
        for heading in self.headings:
            if (
                self.text[heading.text_start : heading.text_end]
                != heading.text
            ):
                raise ValueError(
                    "document heading must match its canonical text span"
                )
        self._validate_privacy_release_binding()

    def _validate_privacy_release_binding(self) -> None:
        clearance = self.privacy_clearance
        if clearance is None or not clearance.permits_training:
            return
        body_digest = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if clearance.output_digest != body_digest:
            raise ValueError(
                "privacy clearance output digest does not match document text"
            )
        _require_approved_text(clearance, name="body", value=self.text)
        _require_approved_text(
            clearance,
            name="markdown",
            value=self.markdown,
        )
        _require_approved_text(
            clearance,
            name="source_url",
            value=self.source_url,
        )
        if self.title is not None:
            _require_approved_text(
                clearance,
                name="title",
                value=self.title,
            )
        if self.path:
            _require_approved_text(
                clearance,
                name="path",
                value=self.path,
            )

        for heading_index, heading in enumerate(self.headings):
            _require_approved_text(
                clearance,
                name=f"heading:{heading_index}",
                value=heading.text,
            )
        for heading_index, metadata_heading in enumerate(
            self.metadata.headings
        ):
            _require_approved_text(
                clearance,
                name=f"extracted_heading:{heading_index}",
                value=metadata_heading,
            )
        for page in self.pages:
            if page.rendered_image_path is not None:
                raise ValueError(
                    "document page image requires object-level privacy approval"
                )
        for block_index, block in enumerate(self.text_blocks):
            if not is_safe_structure_identity(block.block_id):
                raise ValueError("document block_id is not a safe opaque ID")
            generated_page_id = f"page:{block.page_number}:native"
            if block.block_id == generated_page_id:
                continue
            approved_id = clearance.approved_text(
                f"structure:block:{block_index}:block_id"
            )
            expected_id = approved_structure_identity(
                kind="block",
                index=block_index,
                original=approved_id,
                approved=approved_id,
            )
            if block.block_id != expected_id:
                raise ValueError(
                    "document block_id lacks exact privacy binding"
                )
        for table_index, table in enumerate(self.tables):
            approved_id = clearance.approved_text(
                f"structure:table:{table_index}:table_id"
            )
            expected_id = approved_structure_identity(
                kind="table",
                index=table_index,
                original=approved_id,
                approved=approved_id,
            )
            if table.table_id != expected_id:
                raise ValueError(
                    "document table_id lacks exact privacy binding"
                )
            if table.caption is not None:
                _require_approved_text(
                    clearance,
                    name=f"structure:table:{table_index}:caption",
                    value=table.caption,
                )
            for row_index, row in enumerate(table.cells):
                for column_index, cell in enumerate(row):
                    if not cell:
                        continue
                    _require_approved_text(
                        clearance,
                        name=(
                            f"structure:table:{table_index}:cell:"
                            f"{row_index}:{column_index}"
                        ),
                        value=cell,
                    )
        for figure_index, figure in enumerate(self.figures):
            approved_id = clearance.approved_text(
                f"structure:figure:{figure_index}:figure_id"
            )
            expected_id = approved_structure_identity(
                kind="figure",
                index=figure_index,
                original=approved_id,
                approved=approved_id,
            )
            if figure.figure_id != expected_id:
                raise ValueError(
                    "document figure_id lacks exact privacy binding"
                )
            if figure.caption is not None:
                _require_approved_text(
                    clearance,
                    name=f"structure:figure:{figure_index}:caption",
                    value=figure.caption,
                )
            if figure.image_path is not None:
                raise ValueError(
                    "document figure image requires object-level privacy approval"
                )


def _require_approved_text(
    clearance: PrivacyClearance,
    *,
    name: str,
    value: str,
) -> None:
    if clearance.approved_text(name) != value:
        raise ValueError(f"{name} lacks exact privacy approval")


def build_document_structure(
    *,
    source_payload: Mapping[str, object],
    source_text: str,
    normalized_text: str,
    headings: tuple[str, ...],
) -> _CanonicalDocumentStructure:
    """Normalize raw source structure directly into typed canonical objects.

    Source identities (table_id, figure_id, object_id, block_id) are preserved
    when present. Missing identities receive deterministic content-based ids
    derived from page, text and geometry, so generated ids are stable across
    runs and independent of input order.
    """

    pages = _pages_from_payload(
        value=source_payload.get("pages"),
        normalized_text=normalized_text,
    )
    _validate_page_spans(pages, normalized_text)
    blocks = _blocks_from_payload(
        value=source_payload.get("blocks"),
        pages=pages,
        normalized_text=normalized_text,
    )
    if not blocks:
        blocks = _page_blocks(pages=pages, normalized_text=normalized_text)
    tables = _tables_from_payload(source_payload.get("tables"))
    figures = _figures_from_payload(source_payload.get("figures"))
    _validate_unique_ids(tables, "table_id", "table")
    _validate_unique_ids(figures, "figure_id", "figure")
    return {
        "pages": pages,
        "text_blocks": blocks,
        "tables": tables,
        "figures": figures,
        "headings": _headings_from_payload(
            headings=headings, normalized_text=normalized_text
        ),
        "reading_order": tuple(
            block.block_id
            for block in sorted(blocks, key=lambda item: item.reading_order)
        ),
        "source_text_mapping": _text_mapping(
            source_text=source_text,
            normalized_text=normalized_text,
        ),
    }


def _validate_page_spans(
    pages: tuple[DocumentPage, ...], normalized_text: str
) -> None:
    text_length = len(normalized_text)
    sorted_pages = sorted(pages, key=lambda p: p.page_number)
    seen_page_numbers: set[int] = set()
    for i, page in enumerate(sorted_pages):
        if page.page_number in seen_page_numbers:
            raise ValueError("document page numbers must be unique")
        seen_page_numbers.add(page.page_number)
        if page.text_end > text_length:
            raise ValueError("document page text span exceeds document text")
        if i > 0:
            prev = sorted_pages[i - 1]
            if prev.text_end > page.text_start:
                raise ValueError(
                    f"document page spans overlap: page {prev.page_number} "
                    f"ends at {prev.text_end}, page {page.page_number} "
                    f"starts at {page.text_start}"
                )


def _validate_unique_ids(
    items: tuple[object, ...], id_attr: str, kind: str
) -> None:
    seen: set[str] = set()
    for item in items:
        item_id = getattr(item, id_attr)
        if item_id in seen:
            raise ValueError(f"{kind} ids must be unique")
        seen.add(item_id)


def _pages_from_payload(
    *, value: object, normalized_text: str
) -> tuple[DocumentPage, ...]:
    rows: list[DocumentPage] = []
    for raw in _mapping_sequence(value):
        page_number = max(1, _int(raw.get("page_number"), len(rows) + 1))
        start = _optional_non_negative_int(raw.get("text_start"))
        end = _optional_non_negative_int(raw.get("text_end"))
        if start is None or end is None:
            raise ValueError(
                f"page {page_number} requires text_start and text_end"
            )
        if end < start:
            raise ValueError(
                f"page {page_number} has invalid span: {start}..{end}"
            )
        if end > len(normalized_text):
            raise ValueError(
                f"page {page_number} span {start}..{end} exceeds document text "
                f"length {len(normalized_text)}"
            )
        page_text = _text(raw.get("text"))
        if page_text is not None:
            actual = normalized_text[start:end]
            if actual != page_text:
                raise ValueError(
                    f"page {page_number} text does not match canonical text at span "
                    f"{start}..{end}"
                )
        rows.append(
            DocumentPage(
                page_number=page_number,
                text_start=start,
                text_end=end,
                rendered_image_path=_text(raw.get("rendered_image_path")),
                width=_optional_non_negative_int(raw.get("width")),
                height=_optional_non_negative_int(raw.get("height")),
            )
        )
    return tuple(sorted(rows, key=lambda item: item.page_number))


def _blocks_from_payload(
    *,
    value: object,
    pages: tuple[DocumentPage, ...],
    normalized_text: str,
) -> tuple[DocumentTextBlock, ...]:
    rows: list[DocumentTextBlock] = []
    page_numbers = {page.page_number for page in pages}
    page_number_default = pages[0].page_number if pages else 1
    for index, raw in enumerate(_mapping_sequence(value)):
        start = _optional_non_negative_int(raw.get("text_start"))
        end = _optional_non_negative_int(raw.get("text_end"))
        if start is None or end is None or end < start:
            raise ValueError(
                f"block at index {index} requires valid text_start/text_end"
            )
        if end > len(normalized_text):
            raise ValueError(
                f"block span {start}..{end} exceeds document text length "
                f"{len(normalized_text)}"
            )
        text = _text(raw.get("text")) or normalized_text[start:end]
        if not text.strip():
            continue
        page_number = (
            max(1, _int(raw.get("page_number"), 1))
            if raw.get("page_number") is not None
            else page_number_default
        )
        if page_numbers and page_number not in page_numbers:
            raise ValueError(
                f"block at index {index} references unknown page {page_number}"
            )
        source = _text(raw.get("source")) or "native"
        rows.append(
            DocumentTextBlock(
                block_id=_text(raw.get("block_id"))
                or (
                    "block:"
                    + _identity_fingerprint(
                        page_number, text, start, end, source
                    )
                ),
                page_number=page_number,
                text=text,
                text_start=start,
                text_end=end,
                reading_order=max(0, _int(raw.get("reading_order"), index)),
                block_type=_text(raw.get("block_type")) or "paragraph",
                source=source,
                bounding_box=_box(raw.get("bounding_box")),
                confidence=_optional_float(raw.get("confidence")),
            )
        )
    return tuple(rows)


def _page_blocks(
    *, pages: tuple[DocumentPage, ...], normalized_text: str
) -> tuple[DocumentTextBlock, ...]:
    rows: list[DocumentTextBlock] = []
    for index, page in enumerate(pages):
        if page.text_start == page.text_end:
            continue
        text = normalized_text[page.text_start : page.text_end]
        rows.append(
            DocumentTextBlock(
                block_id=f"page:{page.page_number}:native",
                page_number=page.page_number,
                text=text,
                text_start=page.text_start,
                text_end=page.text_end,
                reading_order=index,
            )
        )
    return tuple(rows)


def _headings_from_payload(
    *, headings: tuple[str, ...], normalized_text: str
) -> tuple[DocumentHeading, ...]:
    rows: list[DocumentHeading] = []
    search_start = 0
    for heading in headings:
        text = heading.strip()
        if not text:
            continue
        start = normalized_text.find(text, search_start)
        if start < 0:
            start = normalized_text.find(text)
        if start < 0:
            raise ValueError("document heading does not match canonical text")
        end = start + len(text)
        search_start = end
        rows.append(
            DocumentHeading(
                heading_id=(
                    "heading:" + _identity_fingerprint(text, start, end)
                ),
                text=text,
                text_start=start,
                text_end=end,
            )
        )
    return tuple(rows)


def _resolve_table_id(raw: Mapping[str, object]) -> str:
    table_id = _text(raw.get("table_id"))
    if table_id is None:
        raise ValueError("table requires table_id")
    return table_id


def _tables_from_payload(value: object) -> tuple[DocumentTable, ...]:
    rows: list[DocumentTable] = []
    for raw in _mapping_sequence(value):
        cells_value = raw.get("cells")
        cells = (
            tuple(tuple(str(cell) for cell in row) for row in cells_value)
            if isinstance(cells_value, Sequence)
            and not isinstance(cells_value, (str, bytes))
            else ()
        )
        page_number = max(1, _int(raw.get("page_number"), 1))
        caption = _text(raw.get("caption"))
        bounding_box = _box(raw.get("bounding_box"))
        table_id = _resolve_table_id(raw)
        rows.append(
            DocumentTable(
                table_id=table_id,
                page_number=page_number,
                cells=cells,
                bounding_box=bounding_box,
                caption=caption,
                confidence=_optional_float(raw.get("confidence")),
            )
        )
    return tuple(rows)


def _resolve_figure_id(raw: Mapping[str, object]) -> str:
    figure_id = _text(raw.get("figure_id"))
    if figure_id is None:
        raise ValueError("figure requires figure_id")
    return figure_id


def _figures_from_payload(value: object) -> tuple[DocumentFigure, ...]:
    rows: list[DocumentFigure] = []
    for raw in _mapping_sequence(value):
        page_number = max(1, _int(raw.get("page_number"), 1))
        caption = _text(raw.get("caption"))
        image_path = _text(raw.get("image_path"))
        bounding_box = _box(raw.get("bounding_box"))
        figure_id = _resolve_figure_id(raw)
        rows.append(
            DocumentFigure(
                figure_id=figure_id,
                page_number=page_number,
                bounding_box=bounding_box,
                caption=caption,
                image_path=image_path,
                confidence=_optional_float(raw.get("confidence")),
            )
        )
    return tuple(rows)


def _text_mapping(
    *, source_text: str, normalized_text: str
) -> tuple[SourceTextMapping, ...]:
    if not source_text or not normalized_text:
        return ()
    matcher = SequenceMatcher(
        a=source_text,
        b=normalized_text,
        autojunk=False,
    )
    rows: list[SourceTextMapping] = []
    for (
        source_start,
        normalized_start,
        length,
    ) in matcher.get_matching_blocks():
        if length <= 0:
            continue
        rows.append(
            SourceTextMapping(
                source_start=source_start,
                source_end=source_start + length,
                normalized_start=normalized_start,
                normalized_end=normalized_start + length,
            )
        )
        if len(rows) >= 2048:
            break
    return tuple(rows)


def _identity_fingerprint(*parts: object) -> str:
    """Return a stable content-based identity fragment for one object."""

    payload = "\x1f".join(_identity_part(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _identity_part(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, tuple):
        return "\x1e".join(_identity_part(item) for item in value)
    return str(value)


def _mapping_sequence(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _box(value: object) -> BoundingBox | None:
    if isinstance(value, Mapping):
        x = _float(value.get("x"))
        y = _float(value.get("y"))
        width = _float(value.get("w"))
        height = _float(value.get("h"))
        if x is None or y is None or width is None or height is None:
            return None
        return (x, y, x + width, y + height)
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 4
    ):
        return None
    try:
        x0, y0, x1, y1 = (float(item) for item in value)
        return (x0, y0, x1, y1)
    except (TypeError, ValueError):
        return None


def _validate_box(value: BoundingBox | None) -> None:
    if value is None:
        return
    if not all(math.isfinite(coordinate) for coordinate in value):
        raise ValueError("document bounding box must be finite")
    x0, y0, x1, y1 = value
    if min(value) < 0.0 or x1 < x0 or y1 < y0:
        raise ValueError("document bounding box is invalid")


def _validate_confidence(value: float | None) -> None:
    if value is not None and not 0.0 <= value <= 1.0:
        raise ValueError("document confidence must be in [0, 1]")


def _text(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _int(value: object, default: int) -> int:
    if isinstance(value, bool) or not isinstance(
        value, (str, bytes, bytearray, int, float)
    ):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: object) -> float | None:
    if (
        value is None
        or isinstance(value, bool)
        or not isinstance(value, (str, bytes, bytearray, int, float))
    ):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_non_negative_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        if value < 0:
            raise ValueError("value must be non-negative")
        return value
    if isinstance(value, float):
        if value < 0 or not value.is_integer():
            raise ValueError("value must be non-negative integer")
        return int(value)
    return None


def _optional_float(value: object) -> float | None:
    if (
        value is None
        or isinstance(value, bool)
        or not isinstance(value, (str, bytes, bytearray, int, float))
    ):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "BoundingBox",
    "DocumentFigure",
    "DocumentHeading",
    "DocumentPage",
    "DocumentTable",
    "DocumentTextBlock",
    "PreprocessedDocument",
    "PreprocessingMetadata",
    "SourceTextMapping",
    "build_document_structure",
]
