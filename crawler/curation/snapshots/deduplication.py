"""Stable-id deduplication for curated snapshot records."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from mmcrawler_datasets.curated.document import CuratedDocumentRecord
from mmcrawler_datasets.curated.image import CuratedImageRecord
from mmcrawler_datasets.curated.timed_media import TimedMediaRecord

if TYPE_CHECKING:
    from preprocessing.preprocessed_document import PreprocessedDocument

_TimedMediaRow = TypeVar("_TimedMediaRow", bound=TimedMediaRecord)


class CuratedSnapshotDeduplicator:
    """Deduplicate curated snapshot records by stable identity."""

    @classmethod
    def documents(
        cls,
        *,
        documents: tuple[CuratedDocumentRecord, ...],
        preprocessed_documents_by_id: dict[str, PreprocessedDocument],
    ) -> tuple[
        tuple[CuratedDocumentRecord, ...], dict[str, PreprocessedDocument]
    ]:
        deduped_documents = cls.by_stable_id(documents)
        deduped_preprocessed_documents: dict[str, PreprocessedDocument] = {}
        for document in deduped_documents:
            document_id = document.document_id
            if document_id and document_id in preprocessed_documents_by_id:
                deduped_preprocessed_documents[document_id] = (
                    preprocessed_documents_by_id[document_id]
                )
        return deduped_documents, deduped_preprocessed_documents

    @classmethod
    def images(
        cls,
        *,
        images: tuple[CuratedImageRecord, ...],
    ) -> tuple[CuratedImageRecord, ...]:
        return cls.by_stable_id(images)

    @classmethod
    def media_rows(
        cls,
        *,
        rows: tuple[_TimedMediaRow, ...],
    ) -> tuple[_TimedMediaRow, ...]:
        return cls.by_stable_id(rows)

    @classmethod
    def sync_rows(
        cls,
        *,
        rows: tuple[dict[str, Any], ...],
    ) -> tuple[dict[str, Any], ...]:
        return cls.by_stable_id(rows)

    @classmethod
    def by_stable_id(cls, records: tuple[Any, ...]) -> tuple[Any, ...]:
        seen_ids: set[str] = set()
        deduped: list[Any] = []
        for record in records:
            stable_id = cls._record_stable_id(record)
            if not stable_id or stable_id in seen_ids:
                continue
            seen_ids.add(stable_id)
            deduped.append(record)
        return tuple(deduped)

    @classmethod
    def _record_stable_id(cls, record: Any) -> str:
        if isinstance(record, dict):
            return cls._dict_stable_id(record)
        for attr in (
            "exact_duplicate_key",
            "document_id",
            "image_id",
            "media_id",
            "object_id",
        ):
            value = str(getattr(record, attr, "") or "").strip()
            if value:
                return value
        return ""

    @staticmethod
    def _dict_stable_id(record: dict[str, Any]) -> str:
        for key in (
            "alignment_id",
            "media_id",
            "object_id",
            "document_id",
            "image_id",
        ):
            value = str(record.get(key) or "").strip()
            if value:
                return value
        return ""
