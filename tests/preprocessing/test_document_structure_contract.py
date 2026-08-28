"""Canonical document-structure contract: raw payload to typed model."""

from __future__ import annotations

import pytest

from preprocessing.preprocessed_document import (
    DocumentFigure,
    DocumentHeading,
    DocumentPage,
    DocumentTable,
    DocumentTextBlock,
    PreprocessedDocument,
    PreprocessingMetadata,
    build_document_structure,
)
from preprocessing.preprocessing_quality import PreprocessingQualityResult

TEXT = "Heading\nPage one text.\nPage two text."


def _document(
    *,
    text: str = TEXT,
    pages: tuple[DocumentPage, ...] = (),
    text_blocks: tuple[DocumentTextBlock, ...] = (),
    tables: tuple[DocumentTable, ...] = (),
    figures: tuple[DocumentFigure, ...] = (),
    headings: tuple[DocumentHeading, ...] = (),
    reading_order: tuple[str, ...] = (),
) -> PreprocessedDocument:
    return PreprocessedDocument(
        document_id="doc-1",
        source_id="doc-1",
        source_url="https://example.test/doc",
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
            score=0.9,
            bucket="silver",
            rejection_reason=None,
            token_count_estimate=len(text.split()),
            modality="text",
            language="en",
            alignment_score=0.9,
            signals={},
        ),
        exact_duplicate_key="key",
        pages=pages,
        text_blocks=text_blocks,
        tables=tables,
        figures=figures,
        headings=headings,
        reading_order=reading_order,
    )


def _page(
    *, page_number: int = 1, text_end: int | None = None
) -> DocumentPage:
    return DocumentPage(
        page_number=page_number,
        text_start=0,
        text_end=len(TEXT) if text_end is None else text_end,
    )


def _block(
    *,
    block_id: str = "b1",
    page_number: int = 1,
    text: str = "Heading",
    text_start: int = 0,
    text_end: int = 7,
    reading_order: int = 0,
) -> DocumentTextBlock:
    return DocumentTextBlock(
        block_id=block_id,
        page_number=page_number,
        text=text,
        text_start=text_start,
        text_end=text_end,
        reading_order=reading_order,
    )


def test_build_document_structure_preserves_rich_source_identity() -> None:
    structure = build_document_structure(
        source_payload={
            "pages": [
                {
                    "page_number": 1,
                    "text": "Heading\nPage one text.",
                    "text_start": 0,
                    "text_end": 22,
                    "rendered_image_path": "pages/1.png",
                },
                {
                    "page_number": 2,
                    "text": "Page two text.",
                    "text_start": 23,
                    "text_end": 37,
                },
            ],
            "blocks": [
                {
                    "block_id": "b1",
                    "page_number": 1,
                    "text": "Heading",
                    "text_start": 0,
                    "text_end": 7,
                    "reading_order": 0,
                    "source": "ocr",
                    "block_type": "heading",
                    "confidence": 0.95,
                    "bounding_box": [1, 2, 3, 4],
                }
            ],
            "tables": [
                {
                    "table_id": "table-1",
                    "page_number": 1,
                    "caption": "Measurements",
                    "text": "a b",
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
        source_text=TEXT,
        normalized_text=TEXT,
        headings=("Heading",),
    )

    assert [page.page_number for page in structure["pages"]] == [1, 2]
    assert structure["pages"][0].rendered_image_path == "pages/1.png"
    assert structure["text_blocks"][0].block_id == "b1"
    assert structure["text_blocks"][0].source == "ocr"
    assert structure["text_blocks"][0].block_type == "heading"
    assert structure["text_blocks"][0].confidence == 0.95
    assert structure["text_blocks"][0].bounding_box == (1.0, 2.0, 3.0, 4.0)
    assert structure["tables"][0].table_id == "table-1"
    assert structure["tables"][0].cells == (("a", "b"),)
    assert structure["tables"][0].bounding_box == (1.0, 2.0, 4.0, 6.0)
    assert structure["tables"][0].confidence == 0.91
    assert structure["figures"][0].figure_id == "figure-1"
    assert structure["figures"][0].image_path == "figures/overview.png"
    assert structure["figures"][0].bounding_box == (5.0, 6.0, 12.0, 14.0)
    assert structure["figures"][0].confidence == 0.88
    assert structure["headings"][0].heading_id.startswith("heading:")
    assert structure["headings"][0].text == "Heading"
    assert structure["reading_order"] == ("b1",)
    assert structure["source_text_mapping"]


def test_raw_blocks_are_not_replaced_by_synthetic_page_blocks() -> None:
    structure = build_document_structure(
        source_payload={
            "blocks": [
                {
                    "block_id": "b1",
                    "page_number": 1,
                    "text": "Heading",
                    "text_start": 0,
                    "text_end": 7,
                    "reading_order": 0,
                    "source": "ocr",
                    "block_type": "heading",
                    "confidence": 0.95,
                    "bounding_box": [1, 2, 3, 4],
                }
            ]
        },
        source_text=TEXT,
        normalized_text=TEXT,
        headings=(),
    )

    assert structure["text_blocks"] == (
        DocumentTextBlock(
            block_id="b1",
            page_number=1,
            text="Heading",
            text_start=0,
            text_end=7,
            reading_order=0,
            block_type="heading",
            source="ocr",
            bounding_box=(1.0, 2.0, 3.0, 4.0),
            confidence=0.95,
        ),
    )
    assert structure["reading_order"] == ("b1",)


def test_generated_block_id_is_content_based_not_positional() -> None:
    structure = build_document_structure(
        source_payload={
            "blocks": [
                {
                    "page_number": 1,
                    "text": "Heading",
                    "text_start": 0,
                    "text_end": 7,
                    "reading_order": 0,
                }
            ],
            "tables": [
                {
                    "table_id": "table-1",
                    "page_number": 1,
                    "caption": "Measurements",
                    "cells": [["a"]],
                }
            ],
            "figures": [
                {
                    "figure_id": "figure-1",
                    "page_number": 2,
                    "caption": "Overview",
                    "image_path": "f.png",
                }
            ],
        },
        source_text=TEXT,
        normalized_text=TEXT,
        headings=(),
    )

    block_id = structure["text_blocks"][0].block_id
    assert block_id.startswith("block:")
    assert block_id != "block:0"
    assert structure["tables"][0].table_id == "table-1"
    assert structure["figures"][0].figure_id == "figure-1"


def test_explicit_identity_stable_across_runs_and_order() -> None:
    payload = {
        "tables": [
            {
                "table_id": "table-A",
                "page_number": 1,
                "caption": "A",
                "cells": [["x"]],
            },
            {
                "table_id": "table-B",
                "page_number": 1,
                "caption": "B",
                "cells": [["y"]],
            },
        ],
        "figures": [
            {
                "figure_id": "figure-F",
                "page_number": 2,
                "caption": "F",
                "image_path": "f.png",
            }
        ],
        "blocks": [
            {
                "page_number": 1,
                "text": "Heading",
                "text_start": 0,
                "text_end": 7,
                "reading_order": 0,
            }
        ],
    }
    first = build_document_structure(
        source_payload=payload,
        source_text=TEXT,
        normalized_text=TEXT,
        headings=(),
    )
    second = build_document_structure(
        source_payload=payload,
        source_text=TEXT,
        normalized_text=TEXT,
        headings=(),
    )
    assert [t.table_id for t in first["tables"]] == [
        t.table_id for t in second["tables"]
    ]
    assert [f.figure_id for f in first["figures"]] == [
        f.figure_id for f in second["figures"]
    ]
    assert [b.block_id for b in first["text_blocks"]] == [
        b.block_id for b in second["text_blocks"]
    ]

    reordered = build_document_structure(
        source_payload={
            "tables": [
                {
                    "table_id": "table-B",
                    "page_number": 1,
                    "caption": "B",
                    "cells": [["y"]],
                },
                {
                    "table_id": "table-A",
                    "page_number": 1,
                    "caption": "A",
                    "cells": [["x"]],
                },
            ],
            "figures": [
                {
                    "figure_id": "figure-F",
                    "page_number": 2,
                    "caption": "F",
                    "image_path": "f.png",
                }
            ],
            "blocks": [
                {
                    "page_number": 1,
                    "text": "Heading",
                    "text_start": 0,
                    "text_end": 7,
                    "reading_order": 0,
                }
            ],
        },
        source_text=TEXT,
        normalized_text=TEXT,
        headings=(),
    )
    assert {t.table_id for t in reordered["tables"]} == {
        t.table_id for t in first["tables"]
    }
    assert [t.table_id for t in reordered["tables"]] == [
        t.table_id for t in first["tables"]
    ][::-1]
    assert [f.figure_id for f in reordered["figures"]] == [
        f.figure_id for f in first["figures"]
    ]
    assert [b.block_id for b in reordered["text_blocks"]] == [
        b.block_id for b in first["text_blocks"]
    ]


def test_explicit_identity_preserved() -> None:
    structure = build_document_structure(
        source_payload={
            "tables": [
                {
                    "table_id": "table-1",
                    "page_number": 1,
                    "caption": "Measurements",
                }
            ],
            "figures": [
                {
                    "figure_id": "figure-1",
                    "page_number": 2,
                    "caption": "Overview",
                }
            ],
        },
        source_text=TEXT,
        normalized_text=TEXT,
        headings=(),
    )

    assert structure["tables"][0].table_id == "table-1"
    assert structure["figures"][0].figure_id == "figure-1"


def test_duplicate_explicit_table_ids_fail_closed() -> None:
    with pytest.raises(ValueError, match="table ids must be unique"):
        build_document_structure(
            source_payload={
                "tables": [
                    {
                        "table_id": "table-1",
                        "page_number": 1,
                        "caption": "A",
                        "cells": [["x"]],
                    },
                    {
                        "table_id": "table-1",
                        "page_number": 1,
                        "caption": "A",
                        "cells": [["x"]],
                    },
                ],
            },
            source_text=TEXT,
            normalized_text=TEXT,
            headings=(),
        )


def test_generated_page_blocks_fallback_is_fully_ordered() -> None:
    structure = build_document_structure(
        source_payload={
            "pages": [
                {
                    "page_number": 1,
                    "text": "Heading\nPage one text.",
                    "text_start": 0,
                    "text_end": 22,
                },
                {
                    "page_number": 2,
                    "text": "Page two text.",
                    "text_start": 23,
                    "text_end": 37,
                },
            ]
        },
        source_text=TEXT,
        normalized_text=TEXT,
        headings=(),
    )

    block_ids = [block.block_id for block in structure["text_blocks"]]
    assert block_ids == ["page:1:native", "page:2:native"]
    assert structure["reading_order"] == tuple(block_ids)
    assert len(set(block_ids)) == len(block_ids)


@pytest.mark.parametrize(
    ("kind", "invalid"),
    [
        ("table", DocumentTable(table_id="t", page_number=99, cells=())),
        (
            "figure",
            DocumentFigure(figure_id="f", page_number=99),
        ),
        (
            "text_block",
            DocumentTextBlock(
                block_id="b",
                page_number=99,
                text="Heading",
                text_start=0,
                text_end=7,
                reading_order=0,
            ),
        ),
        (
            "heading",
            DocumentHeading(
                heading_id="h",
                text="Heading",
                text_start=0,
                text_end=7,
                page_number=99,
            ),
        ),
    ],
)
def test_referential_integrity_rejects_unknown_page_references(
    kind: str, invalid: object
) -> None:
    with pytest.raises(ValueError, match="unknown page"):
        _document(pages=(_page(),), **{f"{kind}s": (invalid,)})


def test_duplicate_page_numbers_are_rejected() -> None:
    with pytest.raises(ValueError, match="page numbers must be unique"):
        _document(pages=(_page(), _page()))


def test_duplicate_block_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="block ids must be unique"):
        _document(
            pages=(_page(),),
            text_blocks=(
                _block(block_id="x"),
                _block(
                    block_id="x",
                    text="Page one text",
                    text_start=8,
                    text_end=22,
                    reading_order=1,
                ),
            ),
            reading_order=("x",),
        )


def test_duplicate_table_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="table ids must be unique"):
        _document(
            pages=(_page(),),
            tables=(
                DocumentTable(table_id="t", page_number=1, cells=()),
                DocumentTable(table_id="t", page_number=1, cells=()),
            ),
        )


def test_duplicate_figure_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="figure ids must be unique"):
        _document(
            pages=(_page(),),
            figures=(
                DocumentFigure(figure_id="f", page_number=1),
                DocumentFigure(figure_id="f", page_number=1),
            ),
        )


def test_duplicate_heading_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="heading ids must be unique"):
        _document(
            pages=(_page(),),
            headings=(
                DocumentHeading(
                    heading_id="h", text="Heading", text_start=0, text_end=7
                ),
                DocumentHeading(
                    heading_id="h",
                    text="Page one text",
                    text_start=8,
                    text_end=22,
                ),
            ),
        )


@pytest.mark.parametrize(
    "box",
    [
        {"x": 1, "y": 2, "w": 3, "h": 4},
        [0, 0, 1, 1],
        (1, 1, 1, 1),
    ],
)
def test_bounding_box_accepts_xywh_dicts_and_xyxy_sequences(
    box: object,
) -> None:
    structure = build_document_structure(
        source_payload={
            "tables": [
                {"table_id": "t", "page_number": 1, "bounding_box": box}
            ]
        },
        source_text=TEXT,
        normalized_text=TEXT,
        headings=(),
    )

    table = structure["tables"][0]
    assert table.bounding_box is not None
    assert all(isinstance(value, float) for value in table.bounding_box)


def test_xywh_dict_is_converted_to_xyxy_canonical() -> None:
    structure = build_document_structure(
        source_payload={
            "tables": [
                {
                    "table_id": "t",
                    "page_number": 1,
                    "bounding_box": {"x": 1, "y": 2, "w": 3, "h": 4},
                }
            ]
        },
        source_text=TEXT,
        normalized_text=TEXT,
        headings=(),
    )

    assert structure["tables"][0].bounding_box == (1.0, 2.0, 4.0, 6.0)


@pytest.mark.parametrize(
    "box",
    [
        [-1, 0, 1, 1],
        [0, -1, 1, 1],
        [1, 1, 0, 2],
        [1, 1, 2, 0],
        [0, 0, "nan", 1],
        [0, 0, 1, "nan"],
        [0, 0, "inf", 1],
        [0, 0, 1, "inf"],
        [0, 0, float("nan"), float("nan")],
    ],
)
def test_bounding_box_rejects_invalid_coordinates(box: object) -> None:
    with pytest.raises(ValueError, match="bounding box"):
        build_document_structure(
            source_payload={
                "tables": [
                    {"table_id": "t", "page_number": 1, "bounding_box": box}
                ]
            },
            source_text=TEXT,
            normalized_text=TEXT,
            headings=(),
        )


def test_block_span_beyond_document_text_is_rejected() -> None:
    with pytest.raises(ValueError, match="span exceeds document text"):
        _document(
            pages=(_page(),),
            text_blocks=(
                DocumentTextBlock(
                    block_id="b",
                    page_number=1,
                    text="Heading",
                    text_start=100,
                    text_end=200,
                    reading_order=0,
                ),
            ),
            reading_order=("b",),
        )


def test_heading_span_beyond_document_text_is_rejected() -> None:
    with pytest.raises(ValueError, match="span exceeds document text"):
        _document(
            pages=(_page(),),
            headings=(
                DocumentHeading(
                    heading_id="h",
                    text="Heading",
                    text_start=0,
                    text_end=999,
                ),
            ),
        )


def test_page_span_beyond_document_text_is_rejected() -> None:
    with pytest.raises(ValueError, match="span exceeds document text"):
        _document(pages=(_page(text_end=999),))


def test_reading_order_duplicates_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate blocks"):
        _document(
            pages=(_page(),),
            text_blocks=(_block(block_id="x"),),
            reading_order=("x", "x"),
        )


def test_reading_order_must_cover_every_block_exactly_once() -> None:
    with pytest.raises(ValueError, match="exactly once"):
        _document(
            pages=(_page(),),
            text_blocks=(
                _block(block_id="x"),
                _block(
                    block_id="y",
                    text="Page one text",
                    text_start=8,
                    text_end=22,
                    reading_order=1,
                ),
            ),
            reading_order=("x",),
        )


def test_text_blocks_require_reading_order() -> None:
    with pytest.raises(ValueError, match="must cover all text blocks"):
        _document(
            pages=(_page(),),
            text_blocks=(_block(block_id="x"),),
        )


def test_heading_model_enforces_local_invariants() -> None:
    with pytest.raises(ValueError, match="heading"):
        DocumentHeading(
            heading_id="", text="Heading", text_start=0, text_end=7
        )
    with pytest.raises(ValueError, match="heading"):
        DocumentHeading(
            heading_id="h", text="Heading", text_start=7, text_end=0
        )
    with pytest.raises(ValueError, match="heading"):
        DocumentHeading(
            heading_id="h", text="Heading", text_start=0, text_end=7, level=0
        )
    with pytest.raises(ValueError, match="heading"):
        DocumentHeading(
            heading_id="h",
            text="Heading",
            text_start=0,
            text_end=7,
            page_number=0,
        )


# --- Identity preservation regression tests ---


def test_explicit_table_id_preserved() -> None:
    structure = build_document_structure(
        source_payload={
            "tables": [
                {
                    "table_id": "explicit-table",
                    "page_number": 1,
                    "caption": "Test",
                    "cells": [["a"]],
                }
            ]
        },
        source_text=TEXT,
        normalized_text=TEXT,
        headings=(),
    )
    assert structure["tables"][0].table_id == "explicit-table"


def test_explicit_figure_id_preserved() -> None:
    structure = build_document_structure(
        source_payload={
            "figures": [
                {
                    "figure_id": "explicit-figure",
                    "page_number": 1,
                    "caption": "Test",
                    "image_path": "f.png",
                }
            ]
        },
        source_text=TEXT,
        normalized_text=TEXT,
        headings=(),
    )
    assert structure["figures"][0].figure_id == "explicit-figure"


def test_duplicate_explicit_table_ids_rejected_in_document() -> None:
    with pytest.raises(ValueError, match="table ids must be unique"):
        build_document_structure(
            source_payload={
                "tables": [
                    {
                        "table_id": "same-id",
                        "page_number": 1,
                        "cells": [["a"]],
                    },
                    {
                        "table_id": "same-id",
                        "page_number": 1,
                        "cells": [["b"]],
                    },
                ]
            },
            source_text=TEXT,
            normalized_text=TEXT,
            headings=(),
        )


def test_duplicate_explicit_figure_ids_rejected_in_document() -> None:
    with pytest.raises(ValueError, match="figure ids must be unique"):
        build_document_structure(
            source_payload={
                "figures": [
                    {
                        "figure_id": "same-id",
                        "page_number": 1,
                        "image_path": "a.png",
                    },
                    {
                        "figure_id": "same-id",
                        "page_number": 1,
                        "image_path": "b.png",
                    },
                ]
            },
            source_text=TEXT,
            normalized_text=TEXT,
            headings=(),
        )


def test_table_identity_stable_across_reordering() -> None:
    payload_a = {
        "tables": [
            {
                "table_id": "table-A",
                "page_number": 1,
                "caption": "A",
                "cells": [["a"]],
            },
            {
                "table_id": "table-B",
                "page_number": 1,
                "caption": "B",
                "cells": [["b"]],
            },
        ]
    }
    payload_b = {
        "tables": [
            {
                "table_id": "table-B",
                "page_number": 1,
                "caption": "B",
                "cells": [["b"]],
            },
            {
                "table_id": "table-A",
                "page_number": 1,
                "caption": "A",
                "cells": [["a"]],
            },
        ]
    }
    struct_a = build_document_structure(
        source_payload=payload_a,
        source_text=TEXT,
        normalized_text=TEXT,
        headings=(),
    )
    struct_b = build_document_structure(
        source_payload=payload_b,
        source_text=TEXT,
        normalized_text=TEXT,
        headings=(),
    )
    assert [t.table_id for t in struct_a["tables"]] == ["table-A", "table-B"]
    assert [t.table_id for t in struct_b["tables"]] == ["table-B", "table-A"]
    assert {t.table_id for t in struct_a["tables"]} == {
        t.table_id for t in struct_b["tables"]
    }


def test_figure_identity_stable_across_reordering() -> None:
    payload_a = {
        "figures": [
            {
                "figure_id": "fig-A",
                "page_number": 1,
                "caption": "A",
                "image_path": "a.png",
            },
            {
                "figure_id": "fig-B",
                "page_number": 1,
                "caption": "B",
                "image_path": "b.png",
            },
        ]
    }
    payload_b = {
        "figures": [
            {
                "figure_id": "fig-B",
                "page_number": 1,
                "caption": "B",
                "image_path": "b.png",
            },
            {
                "figure_id": "fig-A",
                "page_number": 1,
                "caption": "A",
                "image_path": "a.png",
            },
        ]
    }
    struct_a = build_document_structure(
        source_payload=payload_a,
        source_text=TEXT,
        normalized_text=TEXT,
        headings=(),
    )
    struct_b = build_document_structure(
        source_payload=payload_b,
        source_text=TEXT,
        normalized_text=TEXT,
        headings=(),
    )
    assert [f.figure_id for f in struct_a["figures"]] == ["fig-A", "fig-B"]
    assert [f.figure_id for f in struct_b["figures"]] == ["fig-B", "fig-A"]
    assert {f.figure_id for f in struct_a["figures"]} == {
        f.figure_id for f in struct_b["figures"]
    }


def test_duplicate_page_numbers_rejected() -> None:
    with pytest.raises(ValueError, match="page numbers must be unique"):
        build_document_structure(
            source_payload={
                "pages": [
                    {
                        "page_number": 1,
                        "text": "A",
                        "text_start": 0,
                        "text_end": 1,
                    },
                    {
                        "page_number": 1,
                        "text": "B",
                        "text_start": 1,
                        "text_end": 2,
                    },
                ]
            },
            source_text="AB",
            normalized_text="AB",
            headings=(),
        )


def test_textless_page_preserved_as_zero_length_span() -> None:
    structure = build_document_structure(
        source_payload={
            "pages": [
                {
                    "page_number": 1,
                    "text": "A",
                    "text_start": 0,
                    "text_end": 1,
                },
                {"page_number": 2, "text": "", "text_start": 1, "text_end": 1},
                {
                    "page_number": 3,
                    "text": "B",
                    "text_start": 1,
                    "text_end": 2,
                },
            ]
        },
        source_text="AB",
        normalized_text="AB",
        headings=(),
    )
    page_numbers = [p.page_number for p in structure["pages"]]
    assert page_numbers == [1, 2, 3]
    page_2 = structure["pages"][1]
    assert page_2.text_start == 1
    assert page_2.text_end == 1


def test_page_span_overlap_rejected() -> None:
    with pytest.raises(ValueError, match="page spans overlap"):
        build_document_structure(
            source_payload={
                "pages": [
                    {
                        "page_number": 1,
                        "text": "AB",
                        "text_start": 0,
                        "text_end": 2,
                    },
                    {
                        "page_number": 2,
                        "text": "BC",
                        "text_start": 1,
                        "text_end": 3,
                    },
                ]
            },
            source_text="ABC",
            normalized_text="ABC",
            headings=(),
        )


def test_page_width_height_must_be_positive() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        build_document_structure(
            source_payload={
                "pages": [
                    {
                        "page_number": 1,
                        "text": "A",
                        "text_start": 0,
                        "text_end": 1,
                        "width": -100,
                        "height": 200,
                    }
                ]
            },
            source_text="A",
            normalized_text="A",
            headings=(),
        )

    with pytest.raises(ValueError, match="height must be positive"):
        build_document_structure(
            source_payload={
                "pages": [
                    {
                        "page_number": 1,
                        "text": "A",
                        "text_start": 0,
                        "text_end": 1,
                        "width": 100,
                        "height": 0,
                    }
                ]
            },
            source_text="A",
            normalized_text="A",
            headings=(),
        )


def test_page_span_exceeds_document_text_rejected() -> None:
    with pytest.raises(ValueError, match="exceeds document text"):
        build_document_structure(
            source_payload={
                "pages": [
                    {
                        "page_number": 1,
                        "text": "A",
                        "text_start": 0,
                        "text_end": 10,
                    }
                ]
            },
            source_text="A",
            normalized_text="A",
            headings=(),
        )


def test_mismatched_page_text_and_offsets_rejected() -> None:
    with pytest.raises(ValueError, match="text does not match canonical text"):
        build_document_structure(
            source_payload={
                "pages": [
                    {
                        "page_number": 1,
                        "text": "WRONG",
                        "text_start": 0,
                        "text_end": 1,
                    }
                ]
            },
            source_text="A",
            normalized_text="A",
            headings=(),
        )


def test_preprocessed_document_validates_page_uniqueness() -> None:
    from preprocessing.preprocessed_document import (
        DocumentPage,
        PreprocessedDocument,
        PreprocessingMetadata,
    )
    from preprocessing.preprocessing_quality import PreprocessingQualityResult

    page_1 = DocumentPage(page_number=1, text_start=0, text_end=1)
    page_2 = DocumentPage(page_number=1, text_start=1, text_end=2)

    with pytest.raises(ValueError, match="page numbers must be unique"):
        PreprocessedDocument(
            document_id="doc-1",
            source_id="src-1",
            source_url="https://example.test",
            title=None,
            text="AB",
            markdown="AB",
            language="en",
            metadata=PreprocessingMetadata(
                char_count=2,
                token_count_estimate=1,
                line_count=1,
                paragraph_count=1,
                heading_count=0,
            ),
            quality=PreprocessingQualityResult(
                score=0.9,
                bucket="silver",
                rejection_reason=None,
                token_count_estimate=1,
                modality="text",
                language="en",
                alignment_score=0.9,
                signals={},
            ),
            exact_duplicate_key="key",
            pages=(page_1, page_2),
        )


def test_preprocessed_document_validates_page_span_bounds() -> None:
    from preprocessing.preprocessed_document import (
        DocumentPage,
        PreprocessedDocument,
        PreprocessingMetadata,
    )
    from preprocessing.preprocessing_quality import PreprocessingQualityResult

    page = DocumentPage(page_number=1, text_start=0, text_end=10)

    with pytest.raises(ValueError, match="exceeds document text"):
        PreprocessedDocument(
            document_id="doc-1",
            source_id="src-1",
            source_url="https://example.test",
            title=None,
            text="A",
            markdown="A",
            language="en",
            metadata=PreprocessingMetadata(
                char_count=1,
                token_count_estimate=1,
                line_count=1,
                paragraph_count=1,
                heading_count=0,
            ),
            quality=PreprocessingQualityResult(
                score=0.9,
                bucket="silver",
                rejection_reason=None,
                token_count_estimate=1,
                modality="text",
                language="en",
                alignment_score=0.9,
                signals={},
            ),
            exact_duplicate_key="key",
            pages=(page,),
        )


def test_preprocessed_document_validates_page_span_overlap() -> None:
    from preprocessing.preprocessed_document import (
        DocumentPage,
        PreprocessedDocument,
        PreprocessingMetadata,
    )
    from preprocessing.preprocessing_quality import PreprocessingQualityResult

    page_1 = DocumentPage(page_number=1, text_start=0, text_end=2)
    page_2 = DocumentPage(page_number=2, text_start=1, text_end=3)

    with pytest.raises(ValueError, match="page spans overlap"):
        PreprocessedDocument(
            document_id="doc-1",
            source_id="src-1",
            source_url="https://example.test",
            title=None,
            text="ABC",
            markdown="ABC",
            language="en",
            metadata=PreprocessingMetadata(
                char_count=3,
                token_count_estimate=2,
                line_count=1,
                paragraph_count=1,
                heading_count=0,
            ),
            quality=PreprocessingQualityResult(
                score=0.9,
                bucket="silver",
                rejection_reason=None,
                token_count_estimate=2,
                modality="text",
                language="en",
                alignment_score=0.9,
                signals={},
            ),
            exact_duplicate_key="key",
            pages=(page_1, page_2),
        )
