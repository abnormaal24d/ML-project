"""Image handler composition."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from config.collection.modality_acceptance import ImageAcceptanceSettings
from config.collection.processors import ImageProcessorSettings
from config.preprocessing.media_settings import OcrBackendSettings
from crawler.analysis.enrichment.image.image_analyzer import ImageAnalyzer
from crawler.extraction.payloads.image_payload_extractor import ImagePayloadExtractor
from crawler.processing.handlers.image_handler import ImageHandler
from crawler.storage.datasets.writing.dataset_writer import DatasetWriter
from logger.factory import ProjectLoggerFactory
from orchestration.composition.preprocessing_dependencies import build_ocr_engine
from preprocessing.media.adapters.opencv_video import OpenCvFrameProcessor
from preprocessing.media.adapters.pillow_image import (
    PillowImageLoader,
    normalize_image_for_training,
)
from preprocessing.media.image.image_blur_score import (
    ImageBlurEstimator,
    PillowGrayscaleArrayConverter,
)
from preprocessing.media.image.image_hashes import PillowImageHashCalculator
from preprocessing.media.image.image_metadata_reader import (
    ImageMetadataReader,
    PillowImageMetadataAssembler,
)

if TYPE_CHECKING:
    from crawler.processing.processors.processor_failure_handler import (
        ProcessorFailureHandler,
    )


def build_image_handler(
    *,
    image_settings: ImageProcessorSettings,
    image_acceptance: ImageAcceptanceSettings,
    ocr_settings: OcrBackendSettings,
    writer: DatasetWriter,
    logs: ProjectLoggerFactory,
    failure_handler: ProcessorFailureHandler,
) -> ImageHandler:
    """Build the image handler with explicit runtime dependencies."""

    return ImageHandler(
        settings=image_settings,
        dataset_writer=writer,
        logger=logs.get_logger_for(ImageHandler),
        failure_handler=failure_handler,
        analyzer=_build_image_analyzer(
            image_settings=image_settings,
            image_acceptance=image_acceptance,
            ocr_settings=ocr_settings,
            logs=logs,
        ),
        image_normalizer=partial(
            normalize_image_for_training,
            resize_images=image_settings.resize_images,
            max_width=image_settings.max_image_width,
            max_height=image_settings.max_image_height,
            max_decode_pixels=image_acceptance.max_decode_pixels,
        ),
    )


def _build_image_analyzer(
    *,
    image_settings: ImageProcessorSettings,
    image_acceptance: ImageAcceptanceSettings,
    ocr_settings: OcrBackendSettings,
    logs: ProjectLoggerFactory,
) -> ImageAnalyzer:
    """Build image payload extraction, blur, and OCR analysis."""

    image_loader = PillowImageLoader(
        max_decode_pixels=image_acceptance.max_decode_pixels,
    )
    ocr_engine = build_ocr_engine(settings=ocr_settings)
    return ImageAnalyzer(
        settings=image_settings,
        payload_extractor=ImagePayloadExtractor(
            max_decode_pixels=image_acceptance.max_decode_pixels,
        ),
        metadata_reader=ImageMetadataReader(
            image_loader=image_loader,
            metadata_assembler=PillowImageMetadataAssembler(
                average_hash_calculator=PillowImageHashCalculator(
                    kind="average",
                    settings=image_settings.average_hash,
                    image_loader=image_loader,
                ),
                difference_hash_calculator=PillowImageHashCalculator(
                    kind="difference",
                    image_loader=image_loader,
                ),
                phash_calculator=PillowImageHashCalculator(
                    kind="perceptual",
                    image_loader=image_loader,
                ),
            ),
        ),
        blur_estimator=ImageBlurEstimator(
            image_loader=image_loader,
            grayscale_converter=PillowGrayscaleArrayConverter(),
            frame_processor=OpenCvFrameProcessor(),
        ),
        ocr_engine=ocr_engine,
        logger=logs.get_logger_for(ImageAnalyzer),
    )


__all__ = ["build_image_handler"]