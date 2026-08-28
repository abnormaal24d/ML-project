"""Document persisting processor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from config.collection.processors import DocumentProcessorSettings
from config.environment.default_values import (
    ENRICHMENT_PREVIEW_MAX_CHARACTERS,
)
from crawler.analysis.enrichment.documents.document_analyzer import (
    DocumentAnalysisResult as DocumentAnalysis,
)
from crawler.fetching.errors.exceptions import IgnoredFetchError
from crawler.processing.processors.persisting_processor import (
    PersistingProcessor,
)
from logger.project_logger import ProjectLogger
from preprocessing.media.document.document_text import (
    DocumentTextUnavailableError,
)

if TYPE_CHECKING:
    from crawler.analysis.enrichment.documents.document_analyzer import (
        DocumentAnalyzer,
    )
    from crawler.fetching.results.result import FetchResult
    from crawler.processing.processors.processor_failure_handler import (
        ProcessorFailureHandler,
    )
    from crawler.storage.datasets.writing.dataset_writer import DatasetWriter


class DocumentHandler(
    PersistingProcessor[DocumentProcessorSettings, DocumentAnalysis]
):
    """Persisting processor for document fetch results."""

    def __init__(
        self,
        *,
        settings: DocumentProcessorSettings,
        dataset_writer: DatasetWriter,
        logger: ProjectLogger,
        failure_handler: ProcessorFailureHandler,
        analyzer: DocumentAnalyzer | None = None,
    ) -> None:
        super().__init__(
            settings=settings,
            dataset_writer=dataset_writer,
            logger=logger,
            failure_handler=failure_handler,
        )
        if analyzer is None:
            raise ValueError(
                "DocumentHandler requires an injected DocumentAnalyzer"
            )
        self._analyzer = analyzer

    async def prepare_analysis(
        self,
        *,
        result: FetchResult,
    ) -> DocumentAnalysis:
        """Analyze the fetched document result; require extractable text."""
        try:
            return await self._analyzer.analyze(result=result)
        except DocumentTextUnavailableError as exc:
            raise IgnoredFetchError(
                reason=f"document_text_{exc.reason}",
                observed_bytes=result.body_size,
            ) from exc

    async def validate_result(
        self,
        *,
        result: FetchResult,
        analysis: DocumentAnalysis | None,
    ) -> tuple[bool, str | None, dict[str, Any]]:
        """Validate analyzed document quality before persistence."""

        if analysis is None:
            raise ValueError("Document analysis is required for validation")

        return self._evaluate_quality(
            analysis=analysis,
            payload_size=result.body_size,
        )

    async def build_enrichment(
        self,
        *,
        result: FetchResult,
        analysis: DocumentAnalysis | None,
    ) -> Any:
        """Build persisted enrichment fields for the analyzed document."""

        if analysis is None:
            raise ValueError("Document analysis is required for enrichment")

        return self._build_document_enrichment_fields(
            analysis=analysis,
        )

    def _evaluate_quality(
        self,
        *,
        analysis: DocumentAnalysis,
        payload_size: int,
    ) -> tuple[bool, str | None, dict[str, Any]]:
        fields: dict[str, Any] = {
            "payload_bytes": payload_size,
            "quality_modality": "document",
            "document_is_textual": True,
            "document_text_source": analysis.document_text.source,
        }
        if payload_size < self._settings.min_bytes:
            fields["quality_score"] = 0.0
            return False, "document_too_small", fields

        text = analysis.document_text.text
        fields["document_text_preview_chars"] = len(text)
        if len(text) < self._settings.min_text_preview_chars:
            fields["quality_score"] = 0.2
            return False, "document_text_too_short", fields

        fields["quality_score"] = 0.8
        return True, None, fields

    def _build_document_enrichment_fields(
        self,
        *,
        analysis: DocumentAnalysis,
    ) -> dict[str, object]:
        payload: dict[str, object] = {}
        document_text = analysis.document_text
        if bool(self._settings.extract_metadata):
            payload.update(
                {k: v for k, v in analysis.metadata.items() if v is not None}
            )
            payload["page_count"] = analysis.page_count
        payload["document_text"] = document_text.text
        payload["document_text_source"] = document_text.source
        payload["document_text_chars"] = len(document_text.text)
        payload["text_preview"] = document_text.text[
            :ENRICHMENT_PREVIEW_MAX_CHARACTERS
        ]
        if document_text.source == "ocr" and analysis.ocr_result is not None:
            payload["ocr_text"] = document_text.text
            payload["ocr_text_preview"] = document_text.text[
                :ENRICHMENT_PREVIEW_MAX_CHARACTERS
            ]
            payload["ocr_origin"] = analysis.ocr_result.origin.value
            payload["ocr_confidence"] = analysis.ocr_result.confidence
            payload["ocr_language"] = analysis.ocr_result.language
            payload["ocr_engine"] = analysis.ocr_result.engine
            payload["ocr_provenance"] = (
                analysis.ocr_result.provenance.to_dict()
            )
        return payload
