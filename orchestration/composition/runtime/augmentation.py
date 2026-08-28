from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from augmentation.audio.audio_augmenter import AudioAugmenter
from augmentation.audio.audio_validation import (
    validate_audio_input,
    validate_audio_output,
)
from augmentation.document.document_augmenter import DocumentAugmenter
from augmentation.document.document_page_augmenter import DocumentPageAugmenter
from augmentation.document.document_text_augmenter import DocumentTextAugmenter
from augmentation.generated_artifact_cache import AugmentationCache
from augmentation.image.image_augmentation_validation import (
    validate_image_input,
    validate_image_output,
)
from augmentation.image.image_augmenter import ImageAugmenter
from augmentation.image.image_operation_executor import ImageOperationExecutor
from augmentation.text.text_field_augmenter import (
    TextFieldAugmenter,
)
from augmentation.text.text_variant_assembler import (
    TextVariantAssembler,
)
from augmentation.training_dataset_augmenter import (
    MediaAugmenter,
    TrainingDatasetAugmenter,
)
from augmentation.video.ffmpeg_video_transform import (
    FfmpegVideoTransformBackend,
)
from augmentation.video.video_clip_augmenter import VideoClipAugmenter
from augmentation.video.video_keyframe_augmenter import VideoKeyframeAugmenter
from augmentation.video.video_operations import resolve_video_output_kinds
from config.media_toolchain import MediaToolchainSettings
from crawler.classification.media_kind import MediaKind
from crawler.classification.media_kind_registry import definition_for
from preprocessing.media.adapters.audio_decode import (
    CompositeAudioDecodeBackend,
    SoundFileAudioDecodeBackend,
    WaveAudioDecodeBackend,
)

if TYPE_CHECKING:
    from augmentation.outcomes.augmentation_result import AugmentationRejection
    from config.augmentation.video_settings import VideoAugmentationSettings
    from config.settings.root import Settings
    from logger.project_logger import ProjectLogger
    from mmcrawler_datasets.schema import MultimodalSample


def build_augmentation_workflow(
    *,
    settings: Settings,
    logger: ProjectLogger,
) -> TrainingDatasetAugmenter:
    """Build the augmentation workflow from canonical settings."""

    augmentation_settings = settings.augmentation
    cache = AugmentationCache(
        enabled=augmentation_settings.cache_enabled,
        cache_directory=(
            Path(settings.paths.root) / augmentation_settings.cache_directory
            if not Path(augmentation_settings.cache_directory).is_absolute()
            else augmentation_settings.cache_directory
        ),
        logger=logger,
    )

    image_max_bytes = int(
        settings.collection.modality_acceptance.image.preprocessing_max_bytes
    )
    audio_max_bytes = int(
        settings.collection.modality_acceptance.audio.preprocessing_max_bytes
    )
    image_mime_types = frozenset(definition_for(MediaKind.IMAGE).mime_types)
    audio_mime_types = frozenset(definition_for(MediaKind.AUDIO).mime_types)

    image_settings = augmentation_settings.image
    image_output_validator = partial(
        validate_image_output,
        settings=image_settings,
    )
    image_augmenter = ImageAugmenter(
        settings=image_settings,
        operation_executor=ImageOperationExecutor(
            settings=image_settings,
            cache=cache,
            validate_output=image_output_validator,
        ),
        validate_input=partial(
            validate_image_input,
            settings=image_settings,
            allowed_mime_types=image_mime_types,
            max_input_bytes=image_max_bytes,
        ),
        logger=logger,
    )
    audio_augmenter = AudioAugmenter(
        settings=augmentation_settings.audio,
        decoder=CompositeAudioDecodeBackend(
            wave_backend=WaveAudioDecodeBackend(),
            soundfile_backend_factory=SoundFileAudioDecodeBackend,
        ),
        cache=cache,
        validate_input=partial(
            validate_audio_input,
            allowed_mime_types=audio_mime_types,
            max_input_bytes=audio_max_bytes,
        ),
        validate_output=validate_audio_output,
        max_duration_seconds=(
            settings.collection.processors.audio.max_duration_seconds
        ),
        logger=logger,
    )
    max_video_bytes = int(
        settings.collection.modality_acceptance.video.preprocessing_max_bytes
    )
    video_backend = FfmpegVideoTransformBackend(
        toolchain=settings.media_toolchain,
        settings=augmentation_settings.video,
    )
    video_augmenter = _build_video_augmenter(
        settings=augmentation_settings.video,
        keyframe_augmenter=VideoKeyframeAugmenter(
            settings=augmentation_settings.video,
            max_input_bytes=max_video_bytes,
            cache=cache,
            backend=video_backend,
            logger=logger,
        ),
        clip_augmenter=VideoClipAugmenter(
            settings=augmentation_settings.video,
            max_input_bytes=max_video_bytes,
            cache=cache,
            backend=video_backend,
            logger=logger,
        ),
    )
    document_augmenter = DocumentAugmenter(
        settings=augmentation_settings.document,
        text_augmenter=DocumentTextAugmenter(
            settings=augmentation_settings.document
        ),
        page_augmenter=DocumentPageAugmenter(
            settings=augmentation_settings.document
        ),
        logger=logger,
    )
    sample_augmenter = TextFieldAugmenter(
        settings=augmentation_settings,
        variant_assembler=TextVariantAssembler(
            settings=augmentation_settings,
            logger=logger,
        ),
        logger=logger,
    )
    return TrainingDatasetAugmenter(
        settings=augmentation_settings,
        sample_augmenter=sample_augmenter,
        logger=logger,
        media_augmenters=(
            document_augmenter.augment,
            image_augmenter.augment,
            audio_augmenter.augment,
            video_augmenter,
        ),
    )


def _build_video_augmenter(
    *,
    settings: VideoAugmentationSettings,
    keyframe_augmenter: VideoKeyframeAugmenter,
    clip_augmenter: VideoClipAugmenter,
) -> MediaAugmenter:
    """Return a media-augmenter callable for keyframe and clip workflows."""

    output_kinds = resolve_video_output_kinds(settings.operations)

    def augment(
        *,
        sample: MultimodalSample,
        dataset_root: str | Path | None,
    ) -> tuple[
        tuple[tuple[str, MultimodalSample], ...],
        tuple[AugmentationRejection, ...],
    ]:
        if not settings.enabled:
            return (), ()

        variants: list[tuple[str, MultimodalSample]] = []
        rejections: list[AugmentationRejection] = []

        if "keyframe_view" in output_kinds:
            produced, rejected = keyframe_augmenter.augment(
                sample=sample,
                dataset_root=dataset_root,
            )
            variants.extend(produced)
            rejections.extend(rejected)

        if "video_clip" in output_kinds:
            produced, rejected = clip_augmenter.augment(
                sample=sample,
                dataset_root=dataset_root,
            )
            variants.extend(produced)
            rejections.extend(rejected)

        return tuple(variants), tuple(rejections)

    return augment
