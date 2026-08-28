"""Curated snapshot row building."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

from mmcrawler_datasets.curated.image import CuratedImageRecord
from mmcrawler_datasets.curated.timed_media import (
    CuratedAudioRecord,
    CuratedVideoRecord,
    TimedMediaRecord,
)
from preprocessing.text.document_structure_privacy import (
    approved_structure_identity,
    is_safe_structure_block_type,
    is_safe_structure_identity,
)

if TYPE_CHECKING:
    from mmcrawler_datasets.curated.document import (
        ChunkRecord,
        CuratedDocumentRecord,
    )
    from preprocessing.preprocessed_document import (
        BoundingBox,
        PreprocessedDocument,
    )
    from preprocessing.privacy.clearance import PrivacyClearance


def _build_stable_id(*parts: str, prefix: str | None = None) -> str:
    """Hash ordered parts and optionally prepend a compact record prefix."""

    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return digest if prefix is None else f"{prefix}_{digest[:24]}"


class CuratedSnapshotRows:
    """Build evidence-bearing cross-modal alignment rows."""

    _IMAGE_ALIGNMENT_TYPE_BY_SOURCE = {
        "alt": "alt_text",
        "figcaption": "figcaption",
        "ocr": "ocr",
        "surrounding": "surrounding_text",
        "page_title": "page_title",
    }

    @classmethod
    def build_sync(
        cls,
        *,
        snapshot_id: str,
        images: tuple[CuratedImageRecord, ...],
        audio_rows: tuple[CuratedAudioRecord, ...],
        video_rows: tuple[CuratedVideoRecord, ...],
        schema_version: str,
        documents: tuple[CuratedDocumentRecord, ...] = (),
        chunks: tuple[ChunkRecord, ...] = (),
        preprocessed_documents_by_id: Mapping[str, PreprocessedDocument]
        | None = None,
    ) -> tuple[dict[str, Any], ...]:
        document_map = preprocessed_documents_by_id or {}
        return (
            *cls.build_page(
                snapshot_id=snapshot_id,
                schema_version=schema_version,
                documents=documents,
                preprocessed_documents_by_id=document_map,
            ),
            *cls.build_image(
                snapshot_id=snapshot_id,
                images=images,
                schema_version=schema_version,
                preprocessed_documents_by_id=document_map,
            ),
            *cls.build_audio(
                snapshot_id=snapshot_id,
                rows=audio_rows,
                schema_version=schema_version,
                preprocessed_documents_by_id=document_map,
            ),
            *cls.build_video(
                snapshot_id=snapshot_id,
                rows=video_rows,
                schema_version=schema_version,
                preprocessed_documents_by_id=document_map,
            ),
            *cls.build_document(
                snapshot_id=snapshot_id,
                schema_version=schema_version,
                documents=documents,
                chunks=chunks,
                preprocessed_documents_by_id=document_map,
            ),
        )

    @staticmethod
    def build_page(
        *,
        snapshot_id: str,
        schema_version: str,
        documents: tuple[CuratedDocumentRecord, ...] = (),
        preprocessed_documents_by_id: Mapping[str, PreprocessedDocument]
        | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """Build page-to-document and page-to-text evidence rows."""

        rows: list[dict[str, Any]] = []
        preprocessed = preprocessed_documents_by_id or {}
        for document in documents:
            prepared = preprocessed.get(document.document_id)
            if prepared is None:
                continue
            if not _document_structure_release_is_approved(prepared):
                continue
            for index, page in enumerate(prepared.pages):
                page_number = page.page_number
                if page_number is None:
                    page_number = index + 1
                start_offset = page.text_start
                end_offset = page.text_end
                if (
                    start_offset is None
                    or end_offset is None
                    or end_offset < start_offset
                ):
                    continue
                page_id = f"{document.document_id}:page:{page_number}"
                rows.append(
                    {
                        "schema_version": schema_version,
                        "snapshot_id": snapshot_id,
                        "alignment_id": _build_stable_id(
                            "page",
                            page_id,
                            document.document_id,
                            prefix="align",
                        ),
                        "object_modality": "page",
                        "object_id": page_id,
                        "text_document_id": document.document_id,
                        "alignment_type": "page_text",
                        "relation_type": "page_belongs_to_document",
                        "confidence": 1.0,
                        "source_field": "preprocessed_document.pages",
                        "evidence_source": "canonical_document_structure",
                        "caption_text": None,
                        "surrounding_text": document.title,
                        "text_span_start": start_offset,
                        "text_span_end": end_offset,
                        "timestamp_start": None,
                        "timestamp_end": None,
                        "page_number": page_number,
                        "rendered_image_path": page.rendered_image_path,
                    }
                )
        return tuple(rows)

    @classmethod
    def build_image(
        cls,
        *,
        snapshot_id: str,
        images: tuple[CuratedImageRecord, ...],
        schema_version: str,
        preprocessed_documents_by_id: Mapping[str, PreprocessedDocument]
        | None = None,
    ) -> tuple[dict[str, Any], ...]:
        rows: list[dict[str, Any]] = []
        documents = preprocessed_documents_by_id or {}
        for image in images:
            document_id = image.parent_document_id
            image_id = image.image_id
            if not document_id or not image_id:
                continue
            source_text = cls._image_source_text(image=image)
            span = _find_text_span(
                document_text=(
                    documents[document_id].text
                    if document_id in documents
                    else None
                ),
                evidence_text=source_text,
            )
            rows.append(
                {
                    "schema_version": schema_version,
                    "snapshot_id": snapshot_id,
                    "alignment_id": _build_stable_id(
                        "image",
                        str(image_id),
                        str(document_id),
                        prefix="align",
                    ),
                    "object_modality": "image",
                    "object_id": str(image_id),
                    "text_document_id": str(document_id),
                    "alignment_type": cls._image_alignment_type(
                        source=image.caption_source,
                    ),
                    "relation_type": cls._image_relation_type(
                        source=image.caption_source,
                    ),
                    "confidence": image.context_score,
                    "evidence_source": "curated_image_context",
                    "source_field": image.caption_source,
                    "caption_text": image.caption_text,
                    "surrounding_text": image.surrounding_text,
                    "text_span_start": span[0] if span else None,
                    "text_span_end": span[1] if span else None,
                    "timestamp_start": None,
                    "timestamp_end": None,
                    "ocr_boxes": list(image.ocr_boxes),
                }
            )
        return tuple(rows)

    @classmethod
    def build_audio(
        cls,
        *,
        snapshot_id: str,
        rows: tuple[TimedMediaRecord, ...],
        schema_version: str,
        preprocessed_documents_by_id: Mapping[str, PreprocessedDocument]
        | None = None,
    ) -> tuple[dict[str, Any], ...]:
        return cls.build_media(
            snapshot_id=snapshot_id,
            rows=rows,
            schema_version=schema_version,
            modality="audio",
            preprocessed_documents_by_id=preprocessed_documents_by_id,
        )

    @classmethod
    def build_video(
        cls,
        *,
        snapshot_id: str,
        rows: tuple[TimedMediaRecord, ...],
        schema_version: str,
        preprocessed_documents_by_id: Mapping[str, PreprocessedDocument]
        | None = None,
    ) -> tuple[dict[str, Any], ...]:
        return cls.build_media(
            snapshot_id=snapshot_id,
            rows=rows,
            schema_version=schema_version,
            modality="video",
            preprocessed_documents_by_id=preprocessed_documents_by_id,
        )

    @staticmethod
    def build_document(
        *,
        snapshot_id: str,
        schema_version: str,
        documents: tuple[CuratedDocumentRecord, ...] = (),
        chunks: tuple[ChunkRecord, ...] = (),
        preprocessed_documents_by_id: Mapping[str, PreprocessedDocument]
        | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """Build chunk and canonical document-structure relations."""

        document_ids = {document.document_id for document in documents}
        rows: list[dict[str, Any]] = []
        for chunk in chunks:
            if chunk.document_id not in document_ids:
                continue
            rows.append(
                {
                    "schema_version": schema_version,
                    "snapshot_id": snapshot_id,
                    "alignment_id": _build_stable_id(
                        "document",
                        chunk.document_id,
                        chunk.chunk_id,
                        prefix="align",
                    ),
                    "object_modality": "document",
                    "object_id": chunk.document_id,
                    "text_document_id": chunk.document_id,
                    "alignment_type": "document_chunk",
                    "relation_type": "section_belongs_to_document",
                    "confidence": chunk.quality_score,
                    "source_field": "chunk_text",
                    "evidence_source": "curated_document_chunk",
                    "caption_text": chunk.text[:512],
                    "surrounding_text": " / ".join(chunk.section_path) or None,
                    "text_span_start": chunk.start_char,
                    "text_span_end": chunk.end_char,
                    "timestamp_start": None,
                    "timestamp_end": None,
                    "chunk_id": chunk.chunk_id,
                    "chunk_index": chunk.chunk_index,
                    "section_path": list(chunk.section_path),
                }
            )

        prepared_documents = preprocessed_documents_by_id or {}
        for document_id, prepared in prepared_documents.items():
            if document_ids and document_id not in document_ids:
                continue
            if not _document_structure_release_is_approved(prepared):
                continue
            for table in prepared.tables:
                rows.append(
                    _structure_alignment_row(
                        snapshot_id=snapshot_id,
                        schema_version=schema_version,
                        document_id=document_id,
                        object_id=str(table.table_id),
                        object_modality="table",
                        page_number=int(table.page_number),
                        relation_type="table_belongs_to_page",
                        caption_text=table.caption,
                        bounding_box=table.bounding_box,
                        confidence=table.confidence,
                    )
                )
            for figure in prepared.figures:
                rows.append(
                    _structure_alignment_row(
                        snapshot_id=snapshot_id,
                        schema_version=schema_version,
                        document_id=document_id,
                        object_id=str(figure.figure_id),
                        object_modality="figure",
                        page_number=int(figure.page_number),
                        relation_type=(
                            "figure_belongs_to_section"
                            if not figure.caption
                            else "figure_has_caption"
                        ),
                        caption_text=figure.caption,
                        bounding_box=figure.bounding_box,
                        confidence=figure.confidence,
                    )
                )
            for heading in prepared.headings:
                rows.append(
                    {
                        "schema_version": schema_version,
                        "snapshot_id": snapshot_id,
                        "alignment_id": _build_stable_id(
                            "heading",
                            document_id,
                            heading.heading_id,
                            prefix="align",
                        ),
                        "object_modality": "section",
                        "object_id": heading.heading_id,
                        "text_document_id": document_id,
                        "alignment_type": "document_structure",
                        "relation_type": "section_belongs_to_document",
                        "confidence": 1.0,
                        "source_field": "preprocessed_document.headings",
                        "evidence_source": "canonical_document_structure",
                        "caption_text": heading.text,
                        "surrounding_text": None,
                        "text_span_start": heading.text_start,
                        "text_span_end": heading.text_end,
                        "timestamp_start": None,
                        "timestamp_end": None,
                        "page_number": heading.page_number,
                        "heading_level": heading.level,
                    }
                )
        return tuple(rows)

    @classmethod
    def build_media(
        cls,
        *,
        snapshot_id: str,
        rows: tuple[TimedMediaRecord, ...],
        schema_version: str,
        modality: Literal["audio", "video"],
        preprocessed_documents_by_id: Mapping[str, PreprocessedDocument]
        | None = None,
    ) -> tuple[dict[str, Any], ...]:
        alignment_rows: list[dict[str, Any]] = []
        documents = preprocessed_documents_by_id or {}
        for row in rows:
            media_id = row.media_id
            document_id = row.parent_document_id or ""
            if not media_id or not document_id:
                continue
            document_text = (
                documents[document_id].text
                if document_id in documents
                else None
            )
            if row.transcript_segments:
                for index, segment in enumerate(row.transcript_segments):
                    span = _find_text_span(
                        document_text=document_text,
                        evidence_text=segment.text,
                    )
                    alignment_rows.append(
                        cls._media_row(
                            snapshot_id=snapshot_id,
                            schema_version=schema_version,
                            row=row,
                            modality=modality,
                            suffix=f"segment:{index}",
                            caption_text=segment.text,
                            source_field=segment.source,
                            confidence=(
                                segment.confidence
                                if segment.confidence is not None
                                else row.context_score
                            ),
                            text_span=span,
                            timestamp_start=segment.start_seconds,
                            timestamp_end=segment.end_seconds,
                            segment_index=index,
                        )
                    )
                continue
            evidence_text = row.transcript_preview or row.transcript_text
            span = _find_text_span(
                document_text=document_text,
                evidence_text=evidence_text,
            )
            alignment_rows.append(
                cls._media_row(
                    snapshot_id=snapshot_id,
                    schema_version=schema_version,
                    row=row,
                    modality=modality,
                    suffix="object",
                    caption_text=evidence_text,
                    source_field=cls._alignment_source_field(row=row),
                    confidence=row.context_score,
                    text_span=span,
                    timestamp_start=cls.alignment_timestamp(
                        row=row,
                        boundary="start",
                    ),
                    timestamp_end=cls.alignment_timestamp(
                        row=row,
                        boundary="end",
                    ),
                    segment_index=None,
                )
            )
        return tuple(alignment_rows)

    @classmethod
    def _media_row(
        cls,
        *,
        snapshot_id: str,
        schema_version: str,
        row: TimedMediaRecord,
        modality: Literal["audio", "video"],
        suffix: str,
        caption_text: str | None,
        source_field: str,
        confidence: float | None,
        text_span: tuple[int, int] | None,
        timestamp_start: float | None,
        timestamp_end: float | None,
        segment_index: int | None,
    ) -> dict[str, Any]:
        return {
            "schema_version": schema_version,
            "snapshot_id": snapshot_id,
            "alignment_id": _build_stable_id(
                modality,
                row.media_id,
                row.parent_document_id or "",
                suffix,
                prefix="align",
            ),
            "object_modality": modality,
            "object_id": row.media_id,
            "text_document_id": row.parent_document_id or "",
            "alignment_type": cls._alignment_type(row=row),
            "relation_type": (
                "audio_transcribes_to_text"
                if modality == "audio"
                else "video_segment_transcribes_to_text"
            ),
            "confidence": confidence,
            "source_field": source_field,
            "evidence_source": "curated_timed_transcript",
            "caption_text": caption_text,
            "surrounding_text": row.surrounding_text,
            "text_span_start": text_span[0] if text_span else None,
            "text_span_end": text_span[1] if text_span else None,
            "timestamp_start": timestamp_start,
            "timestamp_end": timestamp_end,
            "segment_index": segment_index,
            "frame_indices": (
                _video_frame_indices(
                    row=row,
                    start=timestamp_start,
                    end=timestamp_end,
                )
                if modality == "video"
                else []
            ),
        }

    @staticmethod
    def alignment_timestamp(
        *,
        row: TimedMediaRecord,
        boundary: Literal["start", "end"],
    ) -> float | None:
        segments = (
            row.transcript_segments
            if boundary == "start"
            else reversed(row.transcript_segments)
        )
        for segment in segments:
            value = (
                segment.start_seconds
                if boundary == "start"
                else segment.end_seconds
            )
            if value is not None:
                return value
        return None

    @staticmethod
    def _alignment_type(*, row: TimedMediaRecord) -> str:
        if row.transcript_text or row.transcript_segments:
            return "transcript"
        if row.html_context:
            return "html_context"
        return "surrounding_text"

    @staticmethod
    def _alignment_source_field(*, row: TimedMediaRecord) -> str:
        if row.transcript_text:
            return "transcript_text"
        if row.html_context:
            return "html_context"
        return "surrounding_text"

    @staticmethod
    def _image_source_text(*, image: CuratedImageRecord) -> str | None:
        source = image.caption_source
        if source == "ocr":
            return image.ocr_text or image.ocr_preview
        if source == "figcaption":
            return image.figcaption
        if source == "alt":
            return image.alt_text
        if source == "page_title":
            return image.page_title
        return image.caption_text or image.surrounding_text

    @classmethod
    def _image_alignment_type(cls, *, source: str | None) -> str:
        return cls._IMAGE_ALIGNMENT_TYPE_BY_SOURCE.get(source or "", "unknown")

    @staticmethod
    def _image_relation_type(*, source: str | None) -> str:
        if source == "ocr":
            return "ocr_originates_from_image"
        if source in {"alt", "figcaption", "surrounding", "page_title"}:
            return "caption_describes_image"
        return "image_has_text_context"


def _document_structure_release_is_approved(
    prepared: PreprocessedDocument,
) -> bool:
    """Bind every releasable structure value to exact privacy evidence."""

    clearance = prepared.privacy_clearance
    text = prepared.text
    if (
        clearance is None
        or clearance.permits_training is not True
        or not isinstance(text, str)
        or clearance.output_digest
        != hashlib.sha256(text.encode("utf-8")).hexdigest()
        or clearance.approved_text("body") != text
    ):
        return False
    return (
        _release_pages_are_approved(prepared, clearance, text)
        and _release_blocks_are_approved(prepared, clearance, text)
        and _release_tables_are_approved(prepared, clearance)
        and _release_figures_are_approved(prepared, clearance)
        and _release_headings_are_approved(prepared, clearance, text)
    )


def _release_pages_are_approved(
    prepared: PreprocessedDocument,
    clearance: PrivacyClearance,
    text: str,
) -> bool:
    for index, page in enumerate(prepared.pages):
        if page.rendered_image_path is not None:
            return False
        start = page.text_start
        end = page.text_end
        if start is None or end is None or not start <= end <= len(text):
            return False
        approved_page = clearance.approved_text(f"structure:page:{index}:text")
        released_page = text[start:end]
        if approved_page is None:
            if released_page:
                return False
        elif released_page != approved_page:
            return False
    return True


def _release_blocks_are_approved(
    prepared: PreprocessedDocument,
    clearance: PrivacyClearance,
    text: str,
) -> bool:
    for index, block in enumerate(prepared.text_blocks):
        block_id = block.block_id
        block_text = block.text
        start = block.text_start
        end = block.text_end
        page_number = block.page_number
        if (
            not isinstance(block_id, str)
            or not is_safe_structure_identity(block_id)
            or not isinstance(block_text, str)
            or start is None
            or end is None
            or not start <= end <= len(text)
            or text[start:end] != block_text
            or page_number is None
            or block.source not in {"native", "ocr"}
            or not is_safe_structure_block_type(block.block_type)
        ):
            return False
        generated_page_id = f"page:{block.page_number}:native"
        if block_id == generated_page_id:
            continue
        approved_id = clearance.approved_text(
            f"structure:block:{index}:block_id"
        )
        if block_id != approved_structure_identity(
            kind="block",
            index=index,
            original=approved_id,
            approved=approved_id,
        ):
            return False
    return True


def _release_tables_are_approved(
    prepared: PreprocessedDocument,
    clearance: PrivacyClearance,
) -> bool:
    for table_index, table in enumerate(prepared.tables):
        table_id = table.table_id
        approved_id = clearance.approved_text(
            f"structure:table:{table_index}:table_id"
        )
        if not isinstance(table_id, str) or table_id != (
            approved_structure_identity(
                kind="table",
                index=table_index,
                original=approved_id,
                approved=approved_id,
            )
        ):
            return False
        caption = table.caption
        if caption != clearance.approved_text(
            f"structure:table:{table_index}:caption"
        ):
            return False
        cells = table.cells
        if not isinstance(cells, tuple):
            return False
        for row_index, row in enumerate(cells):
            if not isinstance(row, tuple):
                return False
            for column_index, cell in enumerate(row):
                if not isinstance(cell, str):
                    return False
                approved_cell = clearance.approved_text(
                    f"structure:table:{table_index}:cell:"
                    f"{row_index}:{column_index}"
                )
                if (cell and cell != approved_cell) or (
                    not cell and approved_cell is not None
                ):
                    return False
    return True


def _release_figures_are_approved(
    prepared: PreprocessedDocument,
    clearance: PrivacyClearance,
) -> bool:
    for figure_index, figure in enumerate(prepared.figures):
        figure_id = figure.figure_id
        approved_id = clearance.approved_text(
            f"structure:figure:{figure_index}:figure_id"
        )
        if not isinstance(figure_id, str) or figure_id != (
            approved_structure_identity(
                kind="figure",
                index=figure_index,
                original=approved_id,
                approved=approved_id,
            )
        ):
            return False
        if figure.caption != clearance.approved_text(
            f"structure:figure:{figure_index}:caption"
        ):
            return False
        if figure.image_path is not None:
            return False
    return True


def _release_headings_are_approved(
    prepared: PreprocessedDocument,
    clearance: PrivacyClearance,
    text: str,
) -> bool:
    for index, heading in enumerate(prepared.headings):
        heading_id = heading.heading_id
        heading_text = heading.text
        start = heading.text_start
        end = heading.text_end
        if (
            not isinstance(heading_id, str)
            or not is_safe_structure_identity(heading_id)
            or not isinstance(heading_text, str)
            or heading_text != clearance.approved_text(f"heading:{index}")
            or start is None
            or end is None
            or not start <= end <= len(text)
            or text[start:end] != heading_text
        ):
            return False
    return True


def _structure_alignment_row(
    *,
    snapshot_id: str,
    schema_version: str,
    document_id: str,
    object_id: str,
    object_modality: str,
    page_number: int,
    relation_type: str,
    caption_text: str | None,
    bounding_box: BoundingBox | None,
    confidence: float | None,
) -> dict[str, Any]:
    return {
        "schema_version": schema_version,
        "snapshot_id": snapshot_id,
        "alignment_id": _build_stable_id(
            object_modality, document_id, object_id, prefix="align"
        ),
        "object_modality": object_modality,
        "object_id": object_id,
        "text_document_id": document_id,
        "alignment_type": "document_structure",
        "relation_type": relation_type,
        "confidence": 1.0 if confidence is None else confidence,
        "source_field": f"preprocessed_document.{object_modality}s",
        "evidence_source": "canonical_document_structure",
        "caption_text": caption_text,
        "surrounding_text": None,
        "text_span_start": None,
        "text_span_end": None,
        "timestamp_start": None,
        "timestamp_end": None,
        "page_number": page_number,
        "bounding_box": list(bounding_box)
        if bounding_box is not None
        else None,
    }


def _video_frame_indices(
    *,
    row: TimedMediaRecord,
    start: float | None,
    end: float | None,
) -> list[int]:
    if not isinstance(row, CuratedVideoRecord):
        return []
    result: list[int] = []
    for index, keyframe in enumerate(row.keyframes):
        raw_timestamp = keyframe.get("timestamp_seconds")
        if not isinstance(raw_timestamp, (int, float)) or isinstance(
            raw_timestamp, bool
        ):
            continue
        timestamp = float(raw_timestamp)
        if start is not None and timestamp < start:
            continue
        if end is not None and timestamp > end:
            continue
        raw_index = keyframe.get("frame_index")
        result.append(int(raw_index) if isinstance(raw_index, int) else index)
    return result


def _find_text_span(
    *,
    document_text: str | None,
    evidence_text: str | None,
) -> tuple[int, int] | None:
    if not document_text or not evidence_text:
        return None
    candidate = evidence_text.strip()
    if not candidate:
        return None
    start = document_text.find(candidate)
    if start < 0:
        start = document_text.casefold().find(candidate.casefold())
    if start < 0:
        return None
    return start, start + len(candidate)


__all__ = [
    "CuratedSnapshotRows",
]
