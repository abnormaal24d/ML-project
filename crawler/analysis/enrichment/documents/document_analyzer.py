"""Document analysis enrichment: models, analyzer, and assembler.

Objective document properties come from ``DocumentPayloadExtractor``.
Native text and OCR produce a single ``DocumentText`` result; documents
without text are rejected rather than accepted with a failed status.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from crawler.extraction.payloads.document_payload_extractor import (
    DocumentPayloadExtractor,
)
from preprocessing.media.document.document_text import (
    DocumentText,
    DocumentTextUnavailableError,
)
from preprocessing.media.ocr.ocr_engine import (
    OcrBackendFailure,
    OcrBackendUnavailable,
)

if TYPE_CHECKING:
    from crawler.analysis.enrichment.media_files.media_payload_path_resolver import (
        MediaPayloadPathResolver,
    )
    from crawler.fetching.results.result import FetchResult
    from preprocessing.media.adapters.document_ocr_extractor import (
        DocumentOcrExtractor,
    )
    from preprocessing.media.adapters.document_text_reader import (
        DocumentTextReader,
    )
    from preprocessing.media.ocr.ocr_result import (
        OpticalCharacterRecognitionResult,
    )


@dataclass(frozen=True, slots=True)
class DocumentAnalysisResult:
    """Raw analysis output from DocumentAnalyzer."""

    payload_path: str | None
    page_count: int | None
    metadata: dict[str, Any]
    document_text: DocumentText
    ocr_result: OpticalCharacterRecognitionResult | None = None


class DocumentAnalyzer:
    """Coordinate objective payload extraction and required document text."""

    def __init__(
        self,
        *,
        extract_text: bool,
        run_ocr: bool,
        max_ocr_bytes: int,
        media_file_resolver: MediaPayloadPathResolver,
        payload_extractor: DocumentPayloadExtractor,
        document_text_reader: DocumentTextReader,
        text_extraction_service: DocumentOcrExtractor,
    ) -> None:
        self._extract_text = extract_text
        self._run_ocr = run_ocr
        self._max_ocr_bytes = max_ocr_bytes
        self._media_file_resolver = media_file_resolver
        self._payload_extractor = payload_extractor
        self._document_text_reader = document_text_reader
        self._text_extraction_service = text_extraction_service

    async def analyze(
        self,
        *,
        result: FetchResult,
    ) -> DocumentAnalysisResult:
        path = await self._media_file_resolver.resolve_path(
            result=result,
            suffix=".document",
        )

        try:
            payload = await asyncio.to_thread(
                self._payload_extractor.extract_from_path,
                path=path,
                mime_type=result.mime_type,
            )
            native_text = None
            if self._extract_text and payload is not None:
                native_text = await asyncio.to_thread(
                    self._document_text_reader.read_text,
                    path=path,
                    is_pdf=payload.is_pdf,
                    format_name=payload.format,
                    encoding=result.encoding,
                )

            document_text, ocr_result = await asyncio.to_thread(
                self._resolve_document_text,
                path=path,
                native_text=native_text,
                is_pdf=payload.is_pdf if payload is not None else False,
            )

            page_count = payload.page_count if payload is not None else None
            metadata: dict[str, Any] = {
                "suffix": (
                    payload.file_suffix if payload is not None else path.suffix
                ),
                "format": payload.format if payload is not None else None,
                "is_pdf": bool(payload.is_pdf)
                if payload is not None
                else False,
                "byte_size": (
                    payload.byte_size if payload is not None else None
                ),
                "sha256": payload.sha256 if payload is not None else None,
                "document_text_source": document_text.source,
                "raw_document_preserved": True,
                "pdf_page_sample_rules": "first_middle_last",
            }

            return DocumentAnalysisResult(
                payload_path=str(path),
                page_count=page_count,
                metadata=metadata,
                document_text=document_text,
                ocr_result=ocr_result,
            )
        finally:
            self._media_file_resolver.cleanup_owned_path(path)

    def _resolve_document_text(
        self,
        *,
        path: Path,
        native_text: str | None,
        is_pdf: bool,
    ) -> tuple[DocumentText, OpticalCharacterRecognitionResult | None]:
        if self._extract_text and native_text:
            text = native_text.strip()
            if text:
                return DocumentText(text=text, source="native"), None

        if (
            self._run_ocr
            and is_pdf
            and _is_within_byte_limit(
                path=path,
                max_bytes=self._max_ocr_bytes,
            )
        ):
            try:
                ocr_result = self._text_extraction_service.extract(path=path)
            except (OcrBackendUnavailable, OcrBackendFailure) as exc:
                raise DocumentTextUnavailableError(
                    reason="ocr_failed"
                ) from exc
            if ocr_result is not None and ocr_result.text.strip():
                return (
                    DocumentText(
                        text=ocr_result.text.strip(),
                        source="ocr",
                    ),
                    ocr_result,
                )
            raise DocumentTextUnavailableError(reason="no_text")

        raise DocumentTextUnavailableError(reason="no_text")


def _is_within_byte_limit(
    *,
    path: Path,
    max_bytes: int | None,
) -> bool:
    if max_bytes is None or max_bytes <= 0:
        return True
    try:
        return path.stat().st_size <= max_bytes
    except OSError:
        return False
