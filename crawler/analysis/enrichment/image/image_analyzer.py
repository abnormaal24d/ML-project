"""Image analysis enrichment: payload extraction + optional OCR/blur.

Objective image properties come from ``ImagePayloadExtractor``. Blur
estimation and OCR remain enrichment concerns and stay in this layer.
Perceptual hashing (when enabled) is still assembled via the optional
metadata reader so enrichment fields stay complete without moving scoring
into the payload extractor.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from config.collection.processors import ImageProcessorSettings
from crawler.extraction.payloads.image_payload_extractor import (
    ImagePayloadExtractionResult,
    ImagePayloadExtractor,
)
from crawler.fetching.errors.exceptions import IgnoredFetchError
from logger.project_logger import ProjectLogger

if TYPE_CHECKING:
    from crawler.fetching.results.result import FetchResult
    from preprocessing.media.image.image_blur_score import (
        ImageBlurEstimator,
        ImageBlurScore,
    )
    from preprocessing.media.image.image_metadata_reader import (
        ImageMetadata,
        ImageMetadataReader,
    )
    from preprocessing.media.ocr.ocr_engine import OcrEngine
    from preprocessing.media.ocr.ocr_result import (
        OpticalCharacterRecognitionResult,
    )


@dataclass(frozen=True, slots=True)
class ImageAnalysis:
    width: int | None
    height: int | None
    aspect_ratio: float | None
    aspect_ratio_magnitude: float | None
    orientation: str | None
    blur_variance: float | None
    metadata: dict[str, Any]
    extracted_text: str | None
    ocr_result: "OpticalCharacterRecognitionResult | None" = None


class ImageAnalysisAssembler:
    """Build ImageAnalysis DTOs from payload extraction + enrichment."""

    def build(
        self,
        *,
        payload: ImagePayloadExtractionResult,
        blur_score: ImageBlurScore | None,
        extracted_text: str | None,
        ocr_result: "OpticalCharacterRecognitionResult | None" = None,
        fingerprint: ImageMetadata | None = None,
    ) -> ImageAnalysis:
        return ImageAnalysis(
            width=payload.width,
            height=payload.height,
            aspect_ratio=self._compute_aspect_ratio(
                width=payload.width,
                height=payload.height,
            ),
            aspect_ratio_magnitude=self._compute_aspect_ratio_magnitude(
                width=payload.width,
                height=payload.height,
            ),
            orientation=self._orientation(
                width=payload.width,
                height=payload.height,
            ),
            blur_variance=(
                None if blur_score is None else blur_score.laplacian_variance
            ),
            metadata=self._build_metadata_payload(
                payload=payload,
                fingerprint=fingerprint,
            ),
            extracted_text=extracted_text,
            ocr_result=ocr_result,
        )

    @staticmethod
    def _compute_aspect_ratio(
        *,
        width: int | None,
        height: int | None,
    ) -> float | None:
        if width is None or height is None:
            return None
        if width <= 0 or height <= 0:
            return None
        return width / height

    @staticmethod
    def _compute_aspect_ratio_magnitude(
        *,
        width: int | None,
        height: int | None,
    ) -> float | None:
        if width is None or height is None:
            return None
        if width <= 0 or height <= 0:
            return None
        ratio = width / height
        return max(ratio, 1.0 / ratio)

    @staticmethod
    def _orientation(
        *,
        width: int | None,
        height: int | None,
    ) -> str | None:
        if width is None or height is None or width <= 0 or height <= 0:
            return None
        if width == height:
            return "square"
        return "landscape" if width > height else "portrait"

    @staticmethod
    def _build_metadata_payload(
        *,
        payload: ImagePayloadExtractionResult,
        fingerprint: ImageMetadata | None,
    ) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "image_width": payload.width,
            "image_height": payload.height,
            "image_mode": payload.color_mode,
            "image_format": payload.format,
            "image_payload_bytes": payload.byte_size,
            "sha256": payload.sha256,
            "image_exif_orientation": payload.exif_orientation,
            "image_is_animated": payload.frame_count > 1,
            "image_frame_count": payload.frame_count,
        }
        if fingerprint is None:
            return fields
        fields["image_average_hash"] = fingerprint.average_hash
        fields["image_difference_hash"] = fingerprint.difference_hash
        fields["image_phash"] = fingerprint.phash
        fields["image_icc_profile_sha256"] = fingerprint.icc_profile_sha256
        return fields


class ImageAnalyzer:
    """Coordinate objective payload extraction and image enrichment."""

    def __init__(
        self,
        *,
        settings: ImageProcessorSettings,
        payload_extractor: ImagePayloadExtractor,
        metadata_reader: ImageMetadataReader | None = None,
        blur_estimator: ImageBlurEstimator,
        ocr_engine: OcrEngine | None,
        logger: ProjectLogger,
    ) -> None:
        self._settings = settings
        self._payload_extractor = payload_extractor
        self._metadata_reader = metadata_reader
        self._blur_estimator = blur_estimator
        self._ocr_engine = ocr_engine
        self._logger = logger

    async def analyze(
        self,
        *,
        result: FetchResult,
    ) -> ImageAnalysis:
        from crawler.classification.media_kind_registry import (
            is_supported_image_mime_type,
        )

        if not is_supported_image_mime_type(result.mime_type):
            raise IgnoredFetchError(
                reason="unsupported_image_format",
                observed_bytes=result.body_size,
            )

        body = await asyncio.to_thread(result.read_body_required)

        payload = await asyncio.to_thread(
            self._payload_extractor.extract,
            body=body,
        )
        if payload is None:
            raise IgnoredFetchError(
                reason="image_metadata_unreadable",
                observed_bytes=result.body_size,
            )

        run_ocr = (
            self._ocr_engine is not None
            and self._settings.run_ocr
            and _is_within_optional_byte_limit(
                byte_size=len(body),
                max_bytes=self._settings.max_ocr_bytes,
            )
        )
        ocr_awaitable = (
            asyncio.to_thread(
                self._ocr_engine.extract,
                image_bytes=body,
            )
            if run_ocr and self._ocr_engine is not None
            else asyncio.sleep(0, result=None)
        )
        blur_awaitable = (
            asyncio.to_thread(
                self._blur_estimator.estimate_blur,
                body=body,
            )
            if self._settings.detect_blur
            else asyncio.sleep(0, result=None)
        )
        fingerprint_awaitable = (
            asyncio.to_thread(
                self._metadata_reader.read_metadata,
                body=body,
            )
            if (
                self._settings.extract_metadata
                and self._metadata_reader is not None
            )
            else asyncio.sleep(0, result=None)
        )
        blur_score, ocr_result, fingerprint = await asyncio.gather(
            blur_awaitable,
            ocr_awaitable,
            fingerprint_awaitable,
        )

        extracted_text = ocr_result.text if ocr_result is not None else None

        return ImageAnalysisAssembler().build(
            payload=payload,
            blur_score=blur_score,
            extracted_text=extracted_text,
            ocr_result=ocr_result,
            fingerprint=fingerprint,
        )


def _is_within_optional_byte_limit(
    *,
    byte_size: int,
    max_bytes: int | None,
) -> bool:
    if max_bytes is None or max_bytes <= 0:
        return True
    return max(0, int(byte_size)) <= max_bytes
