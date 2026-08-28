"""Fail-closed privacy guards at the snapshot publication boundary."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, cast

import pytest

from crawler.curation.snapshots.alignment_rows import CuratedSnapshotRows
from preprocessing.preprocessed_document import (
    DocumentFigure,
    DocumentHeading,
    DocumentPage,
    DocumentTable,
    DocumentTextBlock,
    PreprocessedDocument,
)
from preprocessing.privacy.clearance import (
    ApprovedTextField,
    PrivacyClearance,
    PrivacyClearanceStatus,
)

_BODY = "Approved heading\nApproved public body."
_HEADING = "Approved heading"


@dataclass(frozen=True, slots=True)
class _PreparedStructure:
    text: str
    privacy_clearance: PrivacyClearance
    pages: tuple[DocumentPage, ...]
    text_blocks: tuple[DocumentTextBlock, ...]
    tables: tuple[DocumentTable, ...]
    figures: tuple[DocumentFigure, ...]
    headings: tuple[DocumentHeading, ...]


def _approved_field(name: str, value: str) -> ApprovedTextField:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return ApprovedTextField(
        name=name,
        value=value,
        input_digest=digest,
        output_digest=digest,
    )


def _prepared_structure() -> _PreparedStructure:
    approved_values = {
        "body": _BODY,
        "heading:0": _HEADING,
        "structure:figure:0:caption": "Approved figure",
        "structure:figure:0:figure_id": "figure-1",
        "structure:page:0:text": _BODY,
        "structure:table:0:caption": "Approved table",
        "structure:table:0:cell:0:0": "Approved cell",
        "structure:table:0:table_id": "table-1",
    }
    body_digest = hashlib.sha256(_BODY.encode("utf-8")).hexdigest()
    clearance = PrivacyClearance(
        status=PrivacyClearanceStatus.APPROVED,
        input_digest=body_digest,
        output_digest=body_digest,
        checked_fields=frozenset(approved_values),
        required_fields=frozenset(approved_values),
        approved_text_fields=tuple(
            _approved_field(name, value)
            for name, value in sorted(approved_values.items())
        ),
        inspection_digest="1" * 64,
        assessment_digest="2" * 64,
    )
    return _PreparedStructure(
        text=_BODY,
        privacy_clearance=clearance,
        pages=(
            DocumentPage(
                page_number=1,
                text_start=0,
                text_end=len(_BODY),
            ),
        ),
        text_blocks=(
            DocumentTextBlock(
                block_id="page:1:native",
                page_number=1,
                text=_BODY,
                text_start=0,
                text_end=len(_BODY),
                reading_order=0,
            ),
        ),
        tables=(
            DocumentTable(
                table_id="table-1",
                page_number=1,
                cells=(("Approved cell",),),
                caption="Approved table",
            ),
        ),
        figures=(
            DocumentFigure(
                figure_id="figure-1",
                page_number=1,
                caption="Approved figure",
            ),
        ),
        headings=(
            DocumentHeading(
                heading_id="heading:approved",
                text=_HEADING,
                text_start=0,
                text_end=len(_HEADING),
            ),
        ),
    )


def _document_rows(prepared: _PreparedStructure) -> tuple[dict[str, Any], ...]:
    prepared_map = cast(
        dict[str, PreprocessedDocument],
        {"doc-1": prepared},
    )
    return CuratedSnapshotRows.build_document(
        snapshot_id="snapshot-1",
        schema_version="v1",
        preprocessed_documents_by_id=prepared_map,
    )


def _tamper_structure(
    prepared: _PreparedStructure,
    mutation: str,
) -> _PreparedStructure:
    if mutation == "table_caption":
        return replace(
            prepared,
            tables=(replace(prepared.tables[0], caption="owner@example.com"),),
        )
    if mutation == "table_cell":
        return replace(
            prepared,
            tables=(
                replace(
                    prepared.tables[0],
                    cells=(("owner@example.com",),),
                ),
            ),
        )
    if mutation == "figure_caption":
        return replace(
            prepared,
            figures=(
                replace(prepared.figures[0], caption="owner@example.com"),
            ),
        )
    if mutation == "heading":
        rogue = "owner@example.com"
        return replace(
            prepared,
            headings=(
                replace(
                    prepared.headings[0],
                    text=rogue,
                    text_end=len(rogue),
                ),
            ),
        )
    if mutation == "table_id_email":
        return replace(
            prepared,
            tables=(
                replace(prepared.tables[0], table_id="owner@example.com"),
            ),
        )
    if mutation == "table_id_traversal":
        return replace(
            prepared,
            tables=(replace(prepared.tables[0], table_id="../../private"),),
        )
    if mutation == "figure_id":
        return replace(
            prepared,
            figures=(
                replace(prepared.figures[0], figure_id="owner@example.com"),
            ),
        )
    if mutation == "figure_image_path":
        return replace(
            prepared,
            figures=(
                replace(prepared.figures[0], image_path="figures/raw.png"),
            ),
        )
    raise AssertionError(f"unknown mutation: {mutation}")


@pytest.mark.parametrize(
    "mutation",
    [
        "table_caption",
        "table_cell",
        "figure_caption",
        "heading",
        "table_id_email",
        "table_id_traversal",
        "figure_id",
        "figure_image_path",
    ],
)
def test_document_structure_rows_require_literal_privacy_binding(
    mutation: str,
) -> None:
    prepared = _prepared_structure()
    baseline = _document_rows(prepared)
    assert [row["object_modality"] for row in baseline] == [
        "table",
        "figure",
        "section",
    ]

    rogue = _tamper_structure(prepared, mutation)

    assert rogue.privacy_clearance.output_digest == (
        hashlib.sha256(rogue.text.encode("utf-8")).hexdigest()
    )
    assert _document_rows(rogue) == ()


def test_page_rows_reject_reintroduced_unapproved_rendered_image_path() -> (
    None
):
    prepared = _prepared_structure()
    rogue = replace(
        prepared,
        pages=(
            replace(
                prepared.pages[0],
                rendered_image_path="pages/raw-page.png",
            ),
        ),
    )
    prepared_map = cast(
        dict[str, PreprocessedDocument],
        {"doc-1": rogue},
    )
    curated_document = cast(
        Any,
        SimpleNamespace(document_id="doc-1", title="Approved heading"),
    )

    rows = CuratedSnapshotRows.build_page(
        snapshot_id="snapshot-1",
        schema_version="v1",
        documents=(curated_document,),
        preprocessed_documents_by_id=prepared_map,
    )

    assert rows == ()
