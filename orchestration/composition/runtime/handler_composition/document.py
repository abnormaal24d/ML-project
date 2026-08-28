"""Document handler composition."""

from __future__ import annotations

from typing import TYPE_CHECKING

from config.collection.processors import DocumentProcessorSettings
from config.preprocessing.media_settings import OcrBackendSettings
from crawler.analysis.enrichment.documents.document_analyzer import DocumentAnalyzer
from crawler.analysis.enrichment.media_files.media_payload_path_resolver import (
    MediaPayloadPathResolver,
)
from crawler.analysis.enrichment.media_files.media_temp_file_writer import (
    MediaTempFileWriter,
)
from crawler.extraction.payloads.document_payload_extractor import DocumentPayloadExtractor
from crawler.processing.handlers.document_handler import DocumentHandler
from crawler.storage.datasets.writing.dataset_writer import DatasetWriter
from logger.factory import ProjectLoggerFactory
from orchestration.composition.preprocessing_dependencies import build_ocr_engine
from preprocessing.media.adapters.document_ocr_extractor import DocumentOcrExtractor
from preprocessing.media.adapters.document_text_reader import (
    DocumentTextReader,
    PdfTextReader,
)

if TYPE_CHECKING:
    from crawler.processing.processors.processor_failure_handler import (
        ProcessorFailureHandler,
    )


def _build_media_path_resolver(
    *,
    logs: ProjectLoggerFactory,
) -> MediaPayloadPathResolver:
    """Build the temporary media payload path resolver."""

    return MediaPayloadPathResolver(
        writer=MediaTempFileWriter(
            logger=logs.get_logger_for(MediaTempFileWriter),
        ),
        logger=logs.get_logger_for(MediaPayloadPathResolver),
    )


def build_document_handler(
    *,
    document_settings: DocumentProcessorSettings,
    ocr_settings: OcrBackendSettings,
    writer: DatasetWriter,
    logs: ProjectLoggerFactory,
    failure_handler: ProcessorFailureHandler,
) -> DocumentHandler:
    """Build the document handler with explicit runtime dependencies."""

    return DocumentHandler(
        settings=document_settings,
        dataset_writer=writer,
        logger=logs.get_logger_for(DocumentHandler),
        failure_handler=failure_handler,
        analyzer=_build_document_analyzer(
            document_settings=document_settings,
            ocr_settings=ocr_settings,
            logs=logs,
        ),
    )


def _build_document_analyzer(
    *,
    document_settings: DocumentProcessorSettings,
    ocr_settings: OcrBackendSettings,
    logs: ProjectLoggerFactory,
) -> DocumentAnalyzer:
    """Build the document analyzer with text and OCR extraction."""

    pdf_text_reader = PdfTextReader(
        max_pages=document_settings.pdf_text_extraction.max_pages,
    )
    return DocumentAnalyzer(
        extract_text=document_settings.extract_text,
        run_ocr=document_settings.run_ocr,
        max_ocr_bytes=document_settings.max_ocr_bytes,
        media_file_resolver=_build_media_path_resolver(logs=logs),
        payload_extractor=DocumentPayloadExtractor(),
        document_text_reader=DocumentTextReader(
            pdf_text_reader=pdf_text_reader,
            max_characters=document_settings.native_text.max_characters,
        ),
        text_extraction_service=DocumentOcrExtractor(
            first_page=document_settings.ocr_first_page,
            last_page=document_settings.ocr_last_page,
            ocr_engine=build_ocr_engine(settings=ocr_settings),
        ),
    )


__all__ = ["build_document_handler"]