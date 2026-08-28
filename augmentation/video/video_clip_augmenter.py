"""Timestamp-correct, aspect-safe video clip augmentation."""

from __future__ import annotations

import logging
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
    preserved_metadata,
    remove_incomplete_file,
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
    transform_video_clip_sample,
)
from augmentation.video.video_transform import (
    build_clip_receipt,
    deterministic_crop_start,
    validate_clip_receipt,
)
from augmentation.video.video_transform_backend import (
    VideoTransformBackend,
)
from augmentation.video.video_variant_assembler import (
    build_video_clip_variant_metadata,
)
from logger.project_logger import ProjectLogger
from mmcrawler_datasets.schema import ModalityObject, MultimodalSample

if TYPE_CHECKING:
    from augmentation.generated_artifact_cache import AugmentationCache
    from config.augmentation.video_settings import VideoAugmentationSettings

_LOGGER = logging.getLogger(__name__)
_OPERATION = "video_clip_transform"


class VideoClipAugmenter:
    """Create one validated clip while transforming timeline and geometry labels."""

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
        self._logger.debug("video_clip_augmenter_initialized")

    def augment(
        self,
        *,
        sample: MultimodalSample,
        dataset_root: str | Path | None,
    ) -> tuple[
        tuple[tuple[str, MultimodalSample], ...],
        tuple[AugmentationRejection, ...],
    ]:
        """Render, decode, validate and lineage-bind a transformed clip."""

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
                    message="FFmpeg and ffprobe are required for correct video augmentation.",
                ),
            )

        output_path: Path | None = None
        try:
            source_sha256 = file_sha256(path=source_path)
            source_probe = self._backend.probe(path=source_path)
            clip_duration = min(
                source_probe.duration_seconds,
                self._settings.max_clip_duration_seconds,
            )
            if clip_duration <= 0.0:
                raise ValueError("video_duration_invalid")
            variant_id = media_variant_id(
                source_sample_id=sample.sample_id,
                operation=_OPERATION,
                source_sha256=source_sha256,
                config_hash=self._settings_fingerprint,
            )
            crop_start = deterministic_crop_start(
                duration_seconds=source_probe.duration_seconds,
                clip_duration_seconds=clip_duration,
                seed_hex=variant_id.rsplit("_", 1)[-1],
                mode=self._settings.temporal_crop_offset_mode,
            )
            output_directory = root / self._settings.output_directory
            output_path = output_directory / f"{variant_id}.mp4"
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
                self._backend.render_clip(
                    source_path=source_path,
                    output_path=output_path,
                    crop_start_seconds=crop_start,
                    crop_duration_seconds=clip_duration,
                    output_fps=self._settings.output_fps,
                    output_width=self._settings.output_width,
                    output_height=self._settings.output_height,
                    resize_mode=self._settings.resize_mode,
                    audio_policy=self._settings.audio_policy,
                    timeout_seconds=self._settings.command_timeout_seconds,
                )
            self._backend.decode_check(
                path=output_path,
                timeout_seconds=self._settings.command_timeout_seconds,
            )
            output_probe = self._backend.probe(path=output_path)
            receipt = build_clip_receipt(
                source=source_probe,
                output=output_probe,
                crop_start_seconds=crop_start,
                crop_duration_seconds=clip_duration,
                output_fps=self._settings.output_fps,
                output_width=self._settings.output_width,
                output_height=self._settings.output_height,
                resize_mode=self._settings.resize_mode,
                audio_policy=self._settings.audio_policy,
            )
            validation = validate_clip_receipt(
                receipt=receipt,
                output_path=output_path,
                output_max_bytes=self._settings.output_max_bytes,
                duration_tolerance_seconds=self._settings.duration_tolerance_seconds,
                fps_tolerance=self._settings.fps_tolerance,
            )
            transformed, annotation_receipt = transform_video_clip_sample(
                sample=sample,
                spatial=receipt.spatial,
                timeline=receipt.timeline,
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
                        "video_transform": receipt.to_dict(),
                    },
                )
            metadata = build_video_clip_variant_metadata(
                sample=transformed,
                dataset_root=root,
                source_path=source_path,
                output_path=output_path,
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
                remove_incomplete_file(
                    output_path,
                    logger=_LOGGER,
                    event_name="video_clip_augmentation_cleanup_failed",
                )
            return (), (
                _rejection(
                    sample=sample,
                    reason=AugmentationRejectionReason.VIDEO_CLIP_TRANSFORM_FAILED,
                    message=str(exc),
                ),
            )

        original_metadata = _preserved_metadata(
            transformed.video.metadata
            if transformed.video is not None
            else {},
            self._settings.metadata_policy,
        )
        video_metadata = {
            **original_metadata,
            "source_video_path": str(source_path),
            "source_sha256": source_sha256,
            "output_sha256": output_sha256,
            "fps": output_probe.fps,
            "duration_seconds": output_probe.duration_seconds,
            "width": output_probe.width,
            "height": output_probe.height,
            "frame_count": output_probe.frame_count,
            "has_audio": output_probe.has_audio,
            "audio_status": receipt.output_audio_status,
            "audio_codec": output_probe.audio_codec,
            "audio_sample_rate": output_probe.audio_sample_rate,
            "audio_channels": output_probe.audio_channels,
            "audio_duration_seconds": output_probe.audio_duration_seconds,
            "crop_start_seconds": receipt.timeline.crop_start_seconds,
            "crop_end_seconds": receipt.timeline.crop_end_seconds,
            "resize_mode": receipt.spatial.mode,
        }
        variant = replace(
            transformed,
            sample_id=variant_id,
            video=ModalityObject(
                path=output_path,
                mime_type="video/mp4",
                byte_size=output_path.stat().st_size,
                metadata=video_metadata,
            ),
            metadata=metadata,
        )
        return ((_OPERATION, variant),), ()


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


_VIDEO_SAFE_METADATA_FIELDS = frozenset(
    {
        "language",
        "fps",
        "duration_seconds",
        "width",
        "height",
        "codec",
        "color_space",
    }
)


def _preserved_metadata(
    metadata: dict[str, object], policy: str
) -> dict[str, object]:
    return preserved_metadata(metadata, policy, _VIDEO_SAFE_METADATA_FIELDS)
