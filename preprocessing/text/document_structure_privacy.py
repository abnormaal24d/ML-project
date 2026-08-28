"""Privacy-field projection for releasable document structure."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence

_SAFE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_ALLOWED_BLOCK_TYPES = frozenset(
    {
        "caption",
        "code",
        "figure",
        "footer",
        "footnote",
        "formula",
        "header",
        "heading",
        "list",
        "list_item",
        "page_number",
        "paragraph",
        "quote",
        "table",
        "title",
    }
)


class _DocumentStructurePrivacyError(ValueError):
    """Raised when approved structure cannot match approved canonical text."""


def collect_structure_text_fields(
    *,
    source_payload: Mapping[str, object],
    normalized_pages: tuple[str, ...],
) -> dict[str, str | None]:
    """Project every text value that canonical document output may release."""

    fields: dict[str, str | None] = {}
    pages = _mapping_sequence(source_payload.get("pages"))
    for index, raw in enumerate(pages):
        page_text = (
            normalized_pages[index]
            if index < len(normalized_pages)
            else _text(raw.get("text"))
        )
        fields[_page_field(index, "text")] = page_text
        fields[_page_field(index, "rendered_image_path")] = _text(
            raw.get("rendered_image_path")
        )

    for index, raw in enumerate(
        _mapping_sequence(source_payload.get("blocks"))
    ):
        for name in ("block_id", "block_type", "source"):
            fields[_block_field(index, name)] = _text(raw.get(name))

    for table_index, raw in enumerate(
        _mapping_sequence(source_payload.get("tables"))
    ):
        fields[_table_field(table_index, "table_id")] = _text(
            raw.get("table_id")
        )
        fields[_table_field(table_index, "caption")] = _text(
            raw.get("caption")
        )
        for row_index, row in enumerate(_cell_rows(raw.get("cells"))):
            for column_index, cell in enumerate(row):
                if not isinstance(cell, str):
                    raise _DocumentStructurePrivacyError(
                        "document table cells must contain strings"
                    )
                fields[
                    _table_cell_field(
                        table_index,
                        row_index,
                        column_index,
                    )
                ] = cell

    for index, raw in enumerate(
        _mapping_sequence(source_payload.get("figures"))
    ):
        for name in ("figure_id", "caption", "image_path"):
            fields[_figure_field(index, name)] = _text(raw.get(name))
    return fields


def build_approved_structure_payload(
    *,
    source_payload: Mapping[str, object],
    approved_values: Mapping[str, str],
    original_text: str,
    approved_text: str,
    normalized_pages: tuple[str, ...],
) -> dict[str, object]:
    """Rebuild canonical structure using approved text values only."""

    pages = _approved_pages(
        source_payload=source_payload,
        approved_values=approved_values,
        normalized_pages=normalized_pages,
        approved_text=approved_text,
    )
    blocks = _approved_blocks(
        source_payload=source_payload,
        approved_values=approved_values,
        original_text=original_text,
        approved_text=approved_text,
    )
    return {
        "pages": pages,
        "blocks": blocks,
        "tables": _approved_tables(
            source_payload=source_payload,
            approved_values=approved_values,
        ),
        "figures": _approved_figures(
            source_payload=source_payload,
            approved_values=approved_values,
        ),
    }


def _approved_pages(
    *,
    source_payload: Mapping[str, object],
    approved_values: Mapping[str, str],
    normalized_pages: tuple[str, ...],
    approved_text: str,
) -> list[dict[str, object]]:
    raw_pages = _mapping_sequence(source_payload.get("pages"))
    if not raw_pages:
        return []

    approved_pages: list[str] = []
    for index, raw in enumerate(raw_pages):
        original_page = (
            normalized_pages[index]
            if index < len(normalized_pages)
            else (_text(raw.get("text")) or "")
        )
        if not original_page:
            approved_pages.append("")
            continue
        approved_page = approved_values.get(_page_field(index, "text"))
        if approved_page is None:
            raise _DocumentStructurePrivacyError(
                "non-empty document page lacks privacy approval"
            )
        approved_pages.append(approved_page)
    if "\n".join(approved_pages).strip() != approved_text:
        raise _DocumentStructurePrivacyError(
            "approved page text does not reconstruct approved body"
        )

    approved_offsets = _page_spans(
        pages=tuple(approved_pages),
        canonical_text=approved_text,
    )
    seen_page_numbers: set[int] = set()
    payload: list[dict[str, object]] = []
    for index, raw in enumerate(raw_pages):
        page_number = _positive_int(raw.get("page_number"), index + 1)
        if page_number in seen_page_numbers:
            raise _DocumentStructurePrivacyError(
                "document page numbers must be unique"
            )
        seen_page_numbers.add(page_number)
        approved_span = approved_offsets[index]
        page: dict[str, object] = {
            "page_number": page_number,
            "text": approved_pages[index],
            "text_start": approved_span[0],
            "text_end": approved_span[1],
        }
        _copy_non_text_fields(
            source=raw,
            target=page,
            names=("width", "height"),
        )
        payload.append(page)
    return payload


def _approved_blocks(
    *,
    source_payload: Mapping[str, object],
    approved_values: Mapping[str, str],
    original_text: str,
    approved_text: str,
) -> list[dict[str, object]]:
    raw_blocks = _mapping_sequence(source_payload.get("blocks"))
    if not raw_blocks or approved_text != original_text:
        return []

    approved_blocks: list[dict[str, object]] = []
    for index, raw in enumerate(raw_blocks):
        start = _non_negative_int(raw.get("text_start"))
        end = _non_negative_int(raw.get("text_end"))
        if (
            start is None
            or end is None
            or not start <= end <= len(original_text)
        ):
            return []
        canonical_segment = approved_text[start:end]
        approved_fragment = canonical_segment.strip()
        if not approved_fragment:
            return []
        relative_start = canonical_segment.find(approved_fragment)
        approved_start = start + relative_start
        approved_end = approved_start + len(approved_fragment)
        raw_fragment = _text(raw.get("text"))
        if raw_fragment is not None and raw_fragment != approved_fragment:
            return []
        page_number = _positive_int(raw.get("page_number"), 1)
        block: dict[str, object] = {
            "page_number": page_number,
            "text": approved_fragment,
            "text_start": approved_start,
            "text_end": approved_end,
        }
        for name in ("block_type", "source"):
            original = _text(raw.get(name))
            value = _unchanged_approved_value(
                original=original,
                approved=approved_values.get(_block_field(index, name)),
            )
            if original is not None and value is None:
                return []
            if value:
                block[name] = value
        if block.get("source", "native") not in {"native", "ocr"}:
            return []
        if not is_safe_structure_block_type(
            str(block.get("block_type", "paragraph"))
        ):
            return []
        block_id = approved_structure_identity(
            kind="block",
            index=index,
            original=_text(raw.get("block_id")),
            approved=approved_values.get(_block_field(index, "block_id")),
        )
        if block_id:
            block["block_id"] = block_id
        _copy_non_text_fields(
            source=raw,
            target=block,
            names=(
                "reading_order",
                "bounding_box",
                "confidence",
            ),
        )
        approved_blocks.append(block)
    return approved_blocks


def _approved_tables(
    *,
    source_payload: Mapping[str, object],
    approved_values: Mapping[str, str],
) -> list[dict[str, object]]:
    tables: list[dict[str, object]] = []
    for table_index, raw in enumerate(
        _mapping_sequence(source_payload.get("tables"))
    ):
        table: dict[str, object] = {}
        table_id = approved_structure_identity(
            kind="table",
            index=table_index,
            original=_text(raw.get("table_id")),
            approved=approved_values.get(
                _table_field(table_index, "table_id")
            ),
        )
        if table_id:
            table["table_id"] = table_id
        caption = approved_values.get(_table_field(table_index, "caption"))
        if caption:
            table["caption"] = caption
        _copy_non_text_fields(
            source=raw,
            target=table,
            names=(
                "page_number",
                "bounding_box",
                "confidence",
            ),
        )
        cells: list[list[str]] = []
        for row_index, row in enumerate(_cell_rows(raw.get("cells"))):
            approved_row: list[str] = []
            for column_index, cell in enumerate(row):
                if not isinstance(cell, str):
                    raise _DocumentStructurePrivacyError(
                        "document table cells must contain strings"
                    )
                original = cell.strip()
                field_name = _table_cell_field(
                    table_index,
                    row_index,
                    column_index,
                )
                if original:
                    approved = approved_values.get(field_name)
                    if approved is None:
                        raise _DocumentStructurePrivacyError(
                            "document table cell lacks privacy approval"
                        )
                    approved_row.append(approved)
                else:
                    approved_row.append("")
            cells.append(approved_row)
        table["cells"] = cells
        tables.append(table)
    return tables


def _approved_figures(
    *,
    source_payload: Mapping[str, object],
    approved_values: Mapping[str, str],
) -> list[dict[str, object]]:
    figures: list[dict[str, object]] = []
    for index, raw in enumerate(
        _mapping_sequence(source_payload.get("figures"))
    ):
        figure: dict[str, object] = {}
        figure_id = approved_structure_identity(
            kind="figure",
            index=index,
            original=_text(raw.get("figure_id")),
            approved=approved_values.get(_figure_field(index, "figure_id")),
        )
        if figure_id:
            figure["figure_id"] = figure_id
        caption = approved_values.get(_figure_field(index, "caption"))
        if caption:
            figure["caption"] = caption
        _copy_non_text_fields(
            source=raw,
            target=figure,
            names=(
                "page_number",
                "bounding_box",
                "confidence",
            ),
        )
        figures.append(figure)
    return figures


def _copy_non_text_fields(
    *,
    source: Mapping[str, object],
    target: dict[str, object],
    names: tuple[str, ...],
) -> None:
    for name in names:
        if name in source:
            target[name] = source[name]


def _page_spans(
    *,
    pages: tuple[str, ...],
    canonical_text: str,
) -> tuple[tuple[int, int], ...]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for page in pages:
        if not page:
            spans.append((cursor, cursor))
            continue
        start = canonical_text.find(page, cursor)
        if start < 0:
            raise _DocumentStructurePrivacyError(
                "approved page is not an ordered canonical body slice"
            )
        end = start + len(page)
        spans.append((start, end))
        cursor = end
    return tuple(spans)


def approved_structure_identity(
    *,
    kind: str,
    index: int,
    original: str | None,
    approved: str | None,
) -> str | None:
    if approved is None:
        approved = f"missing:{kind}:{index}"
    if original == approved and is_safe_structure_identity(approved):
        return approved
    digest = hashlib.sha256(
        f"{kind}\n{index}\n{approved}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{kind}:privacy:{digest}"


def is_safe_structure_identity(value: str) -> bool:
    """Return whether a released structural identity is a bounded opaque ID."""

    return _SAFE_ID_PATTERN.fullmatch(value) is not None and ".." not in value


def is_safe_structure_block_type(value: str) -> bool:
    """Return whether a released text block has a known semantic type."""

    return value in _ALLOWED_BLOCK_TYPES


def _unchanged_approved_value(
    *,
    original: str | None,
    approved: str | None,
) -> str | None:
    if original is None or approved != original:
        return None
    return approved


def _mapping_sequence(value: object) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _cell_rows(value: object) -> tuple[Sequence[object], ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise _DocumentStructurePrivacyError(
            "document table cells must be a sequence of rows"
        )
    rows: list[Sequence[object]] = []
    for row in value:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
            raise _DocumentStructurePrivacyError(
                "document table rows must be sequences"
            )
        rows.append(row)
    return tuple(rows)


def _text(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _non_negative_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, float) and value >= 0 and value.is_integer():
        return int(value)
    return None


def _positive_int(value: object, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(
        value, (str, bytes, bytearray, int, float)
    ):
        raise _DocumentStructurePrivacyError(
            "document page_number must be a positive integer"
        )
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise _DocumentStructurePrivacyError(
            "document page_number must be a positive integer"
        ) from error
    if parsed < 1 or (isinstance(value, float) and not value.is_integer()):
        raise _DocumentStructurePrivacyError(
            "document page_number must be a positive integer"
        )
    return parsed


def _page_field(index: int, name: str) -> str:
    return f"structure:page:{index}:{name}"


def _block_field(index: int, name: str) -> str:
    return f"structure:block:{index}:{name}"


def _table_field(index: int, name: str) -> str:
    return f"structure:table:{index}:{name}"


def _table_cell_field(
    table_index: int,
    row_index: int,
    column_index: int,
) -> str:
    return f"structure:table:{table_index}:cell:{row_index}:{column_index}"


def _figure_field(index: int, name: str) -> str:
    return f"structure:figure:{index}:{name}"


__all__ = [
    "approved_structure_identity",
    "build_approved_structure_payload",
    "collect_structure_text_fields",
    "is_safe_structure_block_type",
    "is_safe_structure_identity",
]
