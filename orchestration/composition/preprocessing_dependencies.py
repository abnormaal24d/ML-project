"""Build concrete preprocessing dependencies for higher layers."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from config.media_toolchain import MediaToolchainSettings
from config.multimodal.generation_settings import (
    AudioTokenizerSettings,
    VideoGeneratorSettings,
)
from config.preprocessing.media_settings import (
    MediaPrivacySettings,
    OcrBackendSettings,
)
from logger.project_logger import ProjectLogger
from orchestration.composition.privacy.privacy_inspection_services import (
    build_privacy_inspection_services,
)
from preprocessing.media.adapters.embedded_metadata import (
    FfmpegEmbeddedMetadataAdapter,
)
from preprocessing.media.adapters.rapidocr_engine import RapidOcrEngine
from preprocessing.media.adapters.tesseract_engine import TesseractOcrEngine
from preprocessing.media.audio.audio_fingerprint import (
    calculate_audio_fingerprint,
)
from preprocessing.media.audio.audio_preprocessor import AudioPreprocessor
from preprocessing.media.image.image_preprocessor import ImagePreprocessor
from preprocessing.media.ocr.ocr_engine import OcrEngine
from preprocessing.media.video.video_preprocessor import VideoPreprocessor
from preprocessing.multimodal_preprocessor import MultimodalPreprocessor
from preprocessing.privacy.inspection.local_content_factories import (
    LocalAudioPrivacyContentFactory,
    LocalDocumentPrivacyContentFactory,
    LocalImagePrivacyContentFactory,
    LocalVideoPrivacyContentFactory,
)
from preprocessing.privacy.inspection.local_visual_analysis import (
    OpenCvVisualPrivacyAnalyzer,
)
from preprocessing.text.text_language import LanguageDetector
from preprocessing.text.text_preparation import TextInputPreparer
from preprocessing.text.text_preprocessor import TextPreprocessor
from preprocessing.text.text_quality import TextQualityScorer
from shared.runtime_primitives import Clock, IdGenerator

if TYPE_CHECKING:
    from config.settings.root import Settings
    from mmcrawler_datasets.materialization.audio_generation import (
        AudioGenerationTargetMaterializer,
    )
    from mmcrawler_datasets.materialization.video_generation import (
        VideoGenerationTargetMaterializer,
    )


def build_ocr_engine(*, settings: OcrBackendSettings) -> OcrEngine | None:
    """Construct the selected OCR adapter at the composition boundary."""

    backend = str(settings.backend).strip().lower()
    if backend == "disabled":
        return None
    if backend == "rapidocr":
        return RapidOcrEngine(settings=settings)
    if backend == "tesseract":
        return TesseractOcrEngine(settings=settings)
    raise AssertionError(f"unreachable OCR backend: {backend!r}")


def build_multimodal_preprocessor(
    *,
    settings: Settings,
    logger: ProjectLogger,
    clock: Clock,
    id_generator: IdGenerator,
) -> MultimodalPreprocessor:
    """Build the multimodal preprocessing orchestrator."""

    from preprocessing.media.adapters.opencv_video import (
        OpenCvFrameProcessor,
        OpenCvVideoReader,
    )

    preprocessing_settings = settings.preprocessing
    modality_acceptance = settings.collection.modality_acceptance
    media_toolchain = settings.media_toolchain
    quality_scorer = TextQualityScorer(
        settings=preprocessing_settings.text_quality_scorer,
        language_detector=LanguageDetector(),
    )
    privacy_services = build_privacy_inspection_services(
        settings=preprocessing_settings.privacy_detection,
    )
    pii_detector = privacy_services.pii_detector
    ocr_engine = build_ocr_engine(settings=preprocessing_settings.ocr)
    visual_analyzer = OpenCvVisualPrivacyAnalyzer()
    video_reader = OpenCvVideoReader()
    frame_processor = OpenCvFrameProcessor()
    embedded_metadata_adapter = FfmpegEmbeddedMetadataAdapter(
        toolchain=media_toolchain,
        settings=preprocessing_settings.media_privacy,
        required=settings.application.environment == "prod",
    )
    image_privacy_factory = LocalImagePrivacyContentFactory(
        ocr_engine=ocr_engine,
        visual_analyzer=visual_analyzer,
        max_decode_pixels=modality_acceptance.image.max_decode_pixels,
    )
    audio_privacy_factory = LocalAudioPrivacyContentFactory()
    video_privacy_factory = LocalVideoPrivacyContentFactory(
        ocr_engine=ocr_engine,
        visual_analyzer=visual_analyzer,
        reader=video_reader,
        frame_processor=frame_processor,
        audio_stream_probe=embedded_metadata_adapter.has_audio_stream,
        max_frames=(
            preprocessing_settings.media_privacy.video_privacy_max_frames
        ),
    )
    document_privacy_factory = LocalDocumentPrivacyContentFactory()
    text_input_preparer = TextInputPreparer(
        pii_detector=pii_detector,
        document_content_factory=document_privacy_factory,
    )
    text_preprocessor = TextPreprocessor(
        quality_scorer=quality_scorer,
        input_validation=preprocessing_settings.input_validation,
        input_preparer=text_input_preparer,
        logger=logger,
    )
    image_preprocessor = ImagePreprocessor(
        logger=logger,
        settings=preprocessing_settings.image_validation,
        modality_acceptance=modality_acceptance.image,
        pii_detector=pii_detector,
        privacy_content_factory=image_privacy_factory,
        embedded_metadata_adapter=embedded_metadata_adapter,
        now=clock.now,
        generate_id=id_generator.generate,
    )
    audio_preprocessor = AudioPreprocessor(
        logger=logger,
        settings=preprocessing_settings.audio_validation,
        modality_acceptance=modality_acceptance.audio,
        max_duration_seconds=(
            settings.collection.processors.audio.max_duration_seconds
        ),
        pii_detector=pii_detector,
        privacy_content_factory=audio_privacy_factory,
        embedded_metadata_adapter=embedded_metadata_adapter,
        now=clock.now,
        generate_id=id_generator.generate,
        audio_fingerprint_calculator=partial(
            calculate_audio_fingerprint,
            executable=(
                preprocessing_settings.audio_validation.chromaprint_executable
            ),
            expected_version=(
                preprocessing_settings.audio_validation.chromaprint_expected_version
            ),
            timeout_seconds=(
                preprocessing_settings.audio_validation.chromaprint_timeout_seconds
            ),
        ),
    )
    video_preprocessor = VideoPreprocessor(
        logger=logger,
        settings=preprocessing_settings.video_validation,
        modality_acceptance=modality_acceptance.video,
        max_duration_seconds=(
            settings.collection.processors.video.max_duration_seconds
        ),
        pii_detector=pii_detector,
        privacy_content_factory=video_privacy_factory,
        video_reader=video_reader,
        embedded_metadata_adapter=embedded_metadata_adapter,
        now=clock.now,
        generate_id=id_generator.generate,
    )
    return MultimodalPreprocessor(
        text_preprocessor=text_preprocessor,
        image_preprocessor=image_preprocessor,
        audio_preprocessor=audio_preprocessor,
        video_preprocessor=video_preprocessor,
        logger=logger,
    )


def build_audio_materializer_factory(
    *,
    settings: AudioTokenizerSettings,
) -> Callable[[Path], "AudioGenerationTargetMaterializer"]:
    """Bind static tokenizer configuration; output_root stays runtime."""

    def build(output_root: Path) -> "AudioGenerationTargetMaterializer":
        from mmcrawler_datasets.materialization.audio_generation import (
            AudioGenerationTargetMaterializer,
        )
        from multimodal.tokenization.audio import AudioTokenizer

        return AudioGenerationTargetMaterializer(
            tokenizer=AudioTokenizer(
                sample_rate=settings.sample_rate,
                frame_ms=settings.frame_ms,
                hop_ms=settings.hop_ms,
                codebook_size=settings.codebook_size,
                n_codebooks=settings.n_codebooks,
                mode=(
                    "discrete"
                    if settings.codec == "discrete"
                    else "continuous"
                ),
            ),
            output_root=output_root,
        )

    return build


def build_video_materializer_factory(
    *,
    settings: VideoGeneratorSettings,
) -> Callable[[Path], "VideoGenerationTargetMaterializer"]:
    """Bind static tokenizer configuration; output_root stays runtime."""

    def build(output_root: Path) -> "VideoGenerationTargetMaterializer":
        from mmcrawler_datasets.materialization.video_generation import (
            VideoGenerationTargetMaterializer,
        )
        from multimodal.tokenization.video import VideoFrameGridTokenizer
        from preprocessing.media.adapters.opencv_video import (
            OpenCvVideoFrameCodec,
        )

        return VideoGenerationTargetMaterializer(
            tokenizer=VideoFrameGridTokenizer(
                frame_codec=OpenCvVideoFrameCodec(),
                vocab_size=settings.video_token_vocab_size,
                grid_height=settings.grid_height,
                grid_width=settings.grid_width,
                frame_count=settings.frames,
                height=settings.resolution,
                width=settings.resolution,
            ),
            output_root=output_root,
        )

    return build
