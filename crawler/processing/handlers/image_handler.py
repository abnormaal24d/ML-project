"""Image persisting processor."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from config.collection.processors import ImageProcessorSettings
from config.environment.default_values import (
    ENRICHMENT_PREVIEW_MAX_CHARACTERS,
)
from crawler.analysis.enrichment.image.image_analyzer import ImageAnalysis
from crawler.processing.processors.persisting_processor import (
    PersistingProcessor,
)
from logger.project_logger import ProjectLogger
from preprocessing.media.ports import ImageNormalizationResult

if TYPE_CHECKING:
    from crawler.analysis.enrichment.image.image_analyzer import ImageAnalyzer
    from crawler.fetching.results.result import FetchResult
    from crawler.processing.processors.processor_failure_handler import (
        ProcessorFailureHandler,
    )
    from crawler.storage.datasets.writing.dataset_writer import DatasetWriter

ImageNormalizer = Callable[..., ImageNormalizationResult]


class ImageHandler(PersistingProcessor[ImageProcessorSettings, ImageAnalysis]):
    """Persisting processor for image fetch results."""

    def __init__(
        self,
        *,
        settings: ImageProcessorSettings,
        dataset_writer: DatasetWriter,
        logger: ProjectLogger,
        failure_handler: ProcessorFailureHandler,
        analyzer: ImageAnalyzer | None = None,
        image_normalizer: ImageNormalizer | None = None,
    ) -> None:
        super().__init__(
            settings=settings,
            dataset_writer=dataset_writer,
            logger=logger,
            failure_handler=failure_handler,
        )
        if analyzer is None:
            raise ValueError("ImageHandler requires an injected ImageAnalyzer")
        if image_normalizer is None:
            raise ValueError(
                "ImageHandler requires an injected image_normalizer"
            )
        self._analyzer = analyzer
        self._image_normalizer = image_normalizer

    async def prepare_analysis(
        self,
        *,
        result: FetchResult,
    ) -> ImageAnalysis:
        """Analyze the fetched image result."""
        return await self._analyzer.analyze(
            result=result,
        )

    async def validate_result(
        self,
        *,
        result: FetchResult,
        analysis: ImageAnalysis | None,
    ) -> tuple[bool, str | None, dict[str, Any]]:
        """Validate analyzed image quality before persistence."""
        if analysis is None:
            raise RuntimeError("image analysis is required for validation")
        return self._evaluate_quality(
            analysis=analysis,
            payload_size=result.body_size,
        )

    async def build_enrichment(
        self,
        *,
        result: FetchResult,
        analysis: ImageAnalysis | None,
    ) -> dict[str, object]:
        """Build persisted enrichment fields for the analyzed image."""
        if analysis is None:
            raise RuntimeError("image analysis is required for enrichment")
        return self._build_image_enrichment_fields(
            analysis=analysis,
            result=result,
        )

    def _evaluate_quality(
        self,
        *,
        analysis: ImageAnalysis,
        payload_size: int,
    ) -> tuple[bool, str | None, dict[str, Any]]:
        fields: dict[str, Any] = {
            "payload_bytes": payload_size,
            "quality_modality": "image",
        }
        if payload_size < self._settings.min_bytes:
            fields["quality_score"] = 0.0
            return False, "image_too_small", fields
        if analysis.width is None or analysis.height is None:
            if self._settings.reject_unknown_dimensions:
                fields["quality_score"] = 0.0
                return False, "image_unknown_dimensions", fields
            fields["quality_score"] = 0.45
            return True, None, fields
        fields["image_width"] = analysis.width
        fields["image_height"] = analysis.height
        if (
            analysis.width < self._settings.min_width
            or analysis.height < self._settings.min_height
        ):
            fields["quality_score"] = 0.2
            return False, "image_dimensions_too_small", fields
        if analysis.aspect_ratio is not None:
            fields["image_aspect_ratio"] = round(analysis.aspect_ratio, 4)
            if analysis.aspect_ratio_magnitude is not None:
                fields["image_aspect_ratio_magnitude"] = round(
                    analysis.aspect_ratio_magnitude, 4
                )
            if analysis.orientation is not None:
                fields["image_orientation"] = analysis.orientation
            extreme_ratio = (
                analysis.aspect_ratio_magnitude
                if analysis.aspect_ratio_magnitude is not None
                else analysis.aspect_ratio
            )
            if extreme_ratio > max(1.0, self._settings.max_aspect_ratio):
                fields["quality_score"] = 0.2
                return False, "image_aspect_ratio_extreme", fields
        if self._settings.detect_blur and analysis.blur_variance is not None:
            fields["image_blur_variance"] = round(analysis.blur_variance, 4)
            if analysis.blur_variance < self._settings.blur_variance_threshold:
                fields["quality_score"] = 0.3
                return False, "image_too_blurry", fields
        fields["quality_score"] = 0.85
        return True, None, fields

    def _build_image_enrichment_fields(
        self,
        *,
        analysis: ImageAnalysis,
        result: FetchResult,
    ) -> dict[str, object]:
        payload: dict[str, object] = {}
        if self._settings.extract_metadata:
            payload.update(
                {k: v for k, v in analysis.metadata.items() if v is not None}
            )
        if self._settings.run_ocr and analysis.extracted_text:
            payload["image_ocr_text"] = analysis.extracted_text
            payload["image_ocr_preview"] = analysis.extracted_text[
                :ENRICHMENT_PREVIEW_MAX_CHARACTERS
            ]
            payload["image_ocr_text_available"] = True

        ocr = getattr(analysis, "ocr_result", None)
        if ocr is not None:
            payload["ocr_confidence"] = getattr(ocr, "confidence", None)
            payload["ocr_language"] = getattr(ocr, "language", None)
            payload["ocr_engine"] = getattr(ocr, "engine", None)
            payload["ocr_origin"] = ocr.origin.value
            payload["ocr_provenance"] = ocr.provenance.to_dict()
            if getattr(ocr, "words", None):
                payload["ocr_boxes"] = [
                    {
                        "text": w.text,
                        "confidence": w.confidence,
                        "box": w.box,
                        "role": "word",
                    }
                    for w in ocr.words
                    if w.text
                ]
            if getattr(ocr, "lines", None):
                payload["ocr_lines"] = [
                    {
                        "text": line.text,
                        "confidence": line.confidence,
                        "box": line.box,
                        "role": "line",
                    }
                    for line in ocr.lines
                    if line.text
                ]
            if getattr(ocr, "confidence", None) is not None:
                payload["ocr_quality_score"] = ocr.confidence

        normalization = self._persist_normalized_image(result=result)
        if normalization is not None:
            normalized_path, normalized = normalization
            payload["normalized_media_path"] = str(normalized_path)
            payload["normalized_image_format"] = normalized.format
            payload["normalized_image_width"] = normalized.width
            payload["normalized_image_height"] = normalized.height
            payload["image_was_oriented"] = normalized.was_oriented
            payload["image_was_converted"] = normalized.was_converted

        payload["image_metrics"] = {
            "ocr_available": bool(payload.get("image_ocr_text")),
            "has_normalized_path": bool(payload.get("normalized_media_path")),
            "normalization_status": "passed"
            if payload.get("normalized_media_path")
            else "unavailable",
            "hash_quality": "real"
            if payload.get("image_difference_hash")
            or payload.get("image_phash")
            else "average_hash_only",
            "task_types_potential": [
                "image_text_pair",
                "vqa",
                "ocr_parse",
                "text_to_image",
            ],
        }
        return payload

    def _persist_normalized_image(
        self,
        *,
        result: FetchResult,
    ) -> tuple[Path, ImageNormalizationResult] | None:
        if result.payload is None:
            return None
        try:
            body = result.read_body_required()
        except (OSError, RuntimeError, ValueError):
            return None
        if not body:
            return None

        normalized = self._image_normalizer(body=body)
        normalized_bytes = getattr(normalized, "normalized_bytes", None)
        if not normalized_bytes:
            return None

        extension = self._normalized_image_extension(
            image_format=getattr(normalized, "format", None),
        )
        normalized_path = result.payload.temp_path.with_name(
            f"{result.payload.temp_path.stem}.normalized{extension}"
        )
        try:
            normalized_path.write_bytes(normalized_bytes)
        except OSError:
            return None
        return normalized_path, normalized

    @staticmethod
    def _normalized_image_extension(*, image_format: object) -> str:
        normalized = str(image_format or "").strip().upper()
        if normalized in {"JPEG", "JPG"}:
            return ".jpg"
        if normalized == "PNG":
            return ".png"
        if normalized == "WEBP":
            return ".webp"
        return ".img"
