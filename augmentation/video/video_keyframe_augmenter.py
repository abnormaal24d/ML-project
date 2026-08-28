"""Timestamp-selected, aspect-safe video keyframe augmentation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from augmentation.annotations.annotation_safety import (
    non_transformable_annotations,
    rejection_message,
)
from augmentation.generated_artifact_cache import settings_fingerprint
from augmentation.media_variant_support import (
    media_rejection,
    resolve_dataset_root,
    resolve_source_path,
)
from augmentation.outcomes.augmentation_result import AugmentationRejection
from augmentation.outcomes.rejection_reason import AugmentationRejectionReason
from augmentation.variant_lineage import (
    MEDIA_AUGMENTATION_IMPLEMENTATION_HASH,
    file_sha256,
    media_variant_id,
)
from augmentation.video.annotations.annotation_assembler import (
    transform_video_keyframe_sample,
)
from augmentation.video.video_transform import VideoKeyframeReceipt
from augmentation.video.video_transform_backend import (
    SpatialTransform,
    VideoTransformBackend,
)
from augmentation.video.video_variant_assembler import (
    build_video_keyframe_variant_metadata,
    keyframe_timestamp,
)
from logger.project_logger import ProjectLogger
from mmcrawler_datasets.schema import ModalityObject, MultimodalSample
from multimodal.tasks.registry import require_task

if TYPE_CHECKING:
    from augmentation.generated_artifact_cache import AugmentationCache
    from config.augmentation.video_settings import VideoAugmentationSettings

_OPERATION = "video_keyframe_view"


class VideoKeyframeAugmenter:
    """Extract one keyframe at a real source timestamp and validate the image."""

    def __init__(
        self,
        *,
        settings: VideoAugmentationSettings,
        max_input_bytes: int,
        cache: AugmentationCache,
        backend: VideoTransformBackend,
        logger: ProjectLogger,
    ) -> None:
        self._settings = settings
        self._max_input_bytes = max_input_bytes
        self._cache = cache
        self._backend = backend
        self._logger = logger
        self._settings_fingerprint = settings_fingerprint(
            settings.model_dump(mode="json")
        )
        self._logger.debug("video_keyframe_augmenter_initialized")

    def augment(
        self,
        *,
        sample: MultimodalSample,
        dataset_root: str | Path | None,
    ) -> tuple[
        tuple[tuple[str, MultimodalSample], ...],
        tuple[AugmentationRejection, ...],
    ]:
        """Extract, decode, transform annotations and lineage-bind a keyframe."""

        if not self._settings.enabled:
            return (), ()
        root = resolve_dataset_root(dataset_root)
        if root is None:
            return (), (
                _rejection(
                    sample=sample,
                    reason=AugmentationRejectionReason.DATASET_ROOT_MISSING,
                ),
            )
        if sample.video is None or sample.video.path is None:
            return (), ()
        output_task_type = _KEYFRAME_TASK_TYPE_MAP.get(sample.task_type)
        if output_task_type is None:
            return (), (
                _rejection(
                    sample=sample,
                    reason=AugmentationRejectionReason.VIDEO_KEYFRAME_TASK_NOT_SUPPORTED,
                    message=(
                        "single-frame augmentation is not valid for temporal task "
                        f"{sample.task_type!r}"
                    ),
                ),
            )
        unsafe = non_transformable_annotations(
            sample=sample, media_kind="video"
        )
        if unsafe:
            return (), (
                _rejection(
                    sample=sample,
                    reason=AugmentationRejectionReason.MEDIA_ANNOTATIONS_NOT_TRANSFORMABLE,
                    message=rejection_message(fields=unsafe),
                ),
            )
        try:
            source_path = resolve_source_path(
                dataset_root=root,
                value=sample.video.path,
                error_message="video_source_path_escapes_dataset_root",
            )
        except ValueError as exc:
            return (), (
                _rejection(
                    sample=sample,
                    reason=AugmentationRejectionReason.VIDEO_SOURCE_PATH_INVALID,
                    message=str(exc),
                ),
            )
        if not source_path.is_file():
            return (), (
                _rejection(
                    sample=sample,
                    reason=AugmentationRejectionReason.MISSING_VIDEO_FILE,
                ),
            )
        if source_path.stat().st_size > self._max_input_bytes:
            return (), (
                _rejection(
                    sample=sample,
                    reason=AugmentationRejectionReason.VIDEO_TOO_LARGE,
                ),
            )
        if not self._backend.is_available():
            return (), (
                _rejection(
                    sample=sample,
                    reason=AugmentationRejectionReason.VIDEO_BACKEND_MISSING,
                    message="FFmpeg and ffprobe are required for timestamp keyframes.",
                ),
            )

        output_path: Path | None = None
        try:
            source_sha256 = file_sha256(path=source_path)
            source_probe = self._backend.probe(path=source_path)
            timestamp = keyframe_timestamp(
                duration_seconds=source_probe.duration_seconds,
                fps=source_probe.fps,
                fraction=self._settings.keyframe_timestamp_fraction,
            )
            spatial = SpatialTransform.build(
                source_width=source_probe.width,
                source_height=source_probe.height,
                output_width=self._settings.output_width,
                output_height=self._settings.output_height,
                mode=self._settings.resize_mode,
            )
            receipt = VideoKeyframeReceipt(
                source=source_probe,
                timestamp_seconds=timestamp,
                spatial=spatial,
                output_width=self._settings.output_width,
                output_height=self._settings.output_height,
            )
            variant_id = media_variant_id(
                source_sample_id=sample.sample_id,
                operation=_OPERATION,
                source_sha256=source_sha256,
                config_hash=self._settings_fingerprint,
            )
            output_path = (
                root
                / self._settings.keyframe_output_directory
                / f"{variant_id}.jpg"
            )
            cache_key = self._cache.cache_key(
                source_path=source_path,
                operation=_OPERATION,
                settings_digest=self._settings_fingerprint,
            )
            restored = self._cache.restore(
                dataset_root=root,
                cache_key=cache_key,
                output_path=output_path,
                expected_metadata={
                    "source_sha256": source_sha256,
                    "config_hash": self._settings_fingerprint,
                    "implementation_hash": MEDIA_AUGMENTATION_IMPLEMENTATION_HASH,
                    "operation": _OPERATION,
                },
            )
            if not restored:
                self._backend.extract_keyframe(
                    source_path=source_path,
                    output_path=output_path,
                    timestamp_seconds=timestamp,
                    output_width=self._settings.output_width,
                    output_height=self._settings.output_height,
                    resize_mode=self._settings.resize_mode,
                    timeout_seconds=self._settings.command_timeout_seconds,
                )
            validation = self._backend.validate_keyframe(
                path=output_path,
                expected_width=self._settings.output_width,
                expected_height=self._settings.output_height,
                output_max_bytes=self._settings.output_max_bytes,
                timeout_seconds=self._settings.command_timeout_seconds,
            )
            transformed, annotation_receipt = transform_video_keyframe_sample(
                sample=sample,
                spatial=spatial,
                timestamp_seconds=timestamp,
                source_fps=source_probe.fps,
            )
            transformed = _apply_keyframe_task_contract(
                sample=transformed,
                output_task_type=output_task_type,
            )
            output_sha256 = file_sha256(path=output_path)
            if not restored:
                self._cache.store(
                    dataset_root=root,
                    cache_key=cache_key,
                    output_path=output_path,
                    cache_metadata={
                        "operation": _OPERATION,
                        "source_path": source_path.as_posix(),
                        "source_sha256": source_sha256,
                        "config_hash": self._settings_fingerprint,
                        "implementation_hash": MEDIA_AUGMENTATION_IMPLEMENTATION_HASH,
                        "video_keyframe_transform": receipt.to_dict(),
                    },
                )
            metadata = build_video_keyframe_variant_metadata(
                sample=transformed,
                dataset_root=root,
                output_path=output_path,
                source_path=source_path,
                cache_key=cache_key,
                source_sha256=source_sha256,
                output_sha256=output_sha256,
                config_hash=self._settings_fingerprint,
                variant_id=variant_id,
                receipt=receipt.to_dict(),
                validation=validation,
                annotation_receipt=annotation_receipt.to_dict(),
            )
        except (OSError, RuntimeError, ValueError) as exc:
            if output_path is not None:
                output_path.unlink(missing_ok=True)
            return (), (
                _rejection(
                    sample=sample,
                    reason=AugmentationRejectionReason.VIDEO_KEYFRAME_TRANSFORM_FAILED,
                    message=str(exc),
                ),
            )

        variant = replace(
            transformed,
            sample_id=variant_id,
            image=ModalityObject(
                path=output_path,
                mime_type="image/jpeg",
                byte_size=output_path.stat().st_size,
                metadata={
                    "source_video_path": str(source_path),
                    "derived_from_modality": "video",
                    "source_sha256": source_sha256,
                    "output_sha256": output_sha256,
                    "keyframe_timestamp_seconds": timestamp,
                    "width": self._settings.output_width,
                    "height": self._settings.output_height,
                    "audio_status": "not_applicable_keyframe",
                },
            ),
            image_tensor_path=None,
            metadata=metadata,
        )
        return ((_OPERATION, variant),), ()


_KEYFRAME_TASK_TYPE_MAP: dict[str, str] = {
    "representation": "representation",
    "video_text_pair": "image_text_pair",
    "scene_understanding": "image_text_pair",
}


def _apply_keyframe_task_contract(
    *,
    sample: MultimodalSample,
    output_task_type: str,
) -> MultimodalSample:
    definition = require_task(output_task_type)
    output_modalities = tuple(definition.output_modalities)

    task_target = dict(sample.task_target)
    task_target.update(
        {
            "task_type": definition.name,
            "task_family": definition.family,
            "output_modalities": list(output_modalities),
        }
    )

    metadata = dict(sample.metadata)
    metadata["modality"] = "image"
    metadata["task_target"] = dict(task_target)

    return replace(
        sample,
        task_type=definition.name,
        task_family=definition.family,
        output_modalities=output_modalities,
        task_target=task_target,
        metadata=metadata,
    )


def _rejection(
    *,
    sample: MultimodalSample,
    reason: AugmentationRejectionReason,
    message: str | None = None,
) -> AugmentationRejection:
    return media_rejection(
        sample=sample,
        reason=reason,
        message=message,
        operation=_OPERATION,
        modality="video",
    )
