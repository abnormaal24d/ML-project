"""Audio media augmentation workflow."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from augmentation.annotations.annotation_safety import (
    non_transformable_annotations,
    rejection_message,
)
from augmentation.audio.audio_operations import resolve_audio_operations
from augmentation.audio.audio_stream_transformer import (
    prepare_audio_transform,
    write_prepared_wav,
)
from augmentation.audio.audio_transform_builder import (
    apply_audio_timed_sample,
    build_audio_transform_parameters,
)
from augmentation.audio.audio_variant_metadata import (
    build_audio_variant_metadata,
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
from logger.project_logger import ProjectLogger
from mmcrawler_datasets.schema import ModalityObject, MultimodalSample

if TYPE_CHECKING:
    from collections.abc import Callable

    from augmentation.generated_artifact_cache import AugmentationCache
    from augmentation.outcomes.media_validation_outcome import (
        MediaValidationOutcome,
    )
    from config.augmentation.audio_settings import AudioAugmentationSettings
    from preprocessing.media.ports import AudioDecodeBackend

_LOGGER = logging.getLogger(__name__)
_OPERATION = "audio_media_transform"


class AudioAugmenter:
    """Create deterministic, validated PCM WAV variants and timed annotations."""

    def __init__(
        self,
        *,
        settings: AudioAugmentationSettings,
        decoder: AudioDecodeBackend,
        cache: AugmentationCache,
        validate_input: Callable[..., MediaValidationOutcome],
        validate_output: Callable[..., MediaValidationOutcome],
        max_duration_seconds: float,
        logger: ProjectLogger,
    ) -> None:
        self._settings = settings
        self._operations = resolve_audio_operations(settings.operations)
        self._decoder = decoder
        self._validate_input = validate_input
        self._validate_output = validate_output
        self._max_duration_seconds = max_duration_seconds
        self._cache = cache
        self._logger = logger
        self._logger.debug("audio_augmenter_initialized")
        self._settings_fingerprint = settings_fingerprint(
            settings.model_dump(mode="json")
        )

    def enabled_operations(self) -> tuple[str, ...]:
        """Return configured audio operations when media transforms are on."""

        return self._settings.operations if self._settings.enabled else ()

    def augment(
        self,
        *,
        sample: MultimodalSample,
        dataset_root: str | Path | None,
    ) -> tuple[
        tuple[tuple[str, MultimodalSample], ...],
        tuple[AugmentationRejection, ...],
    ]:
        """Create one validated, lineage-bound transformed WAV variant."""

        if not self._settings.enabled:
            return (), ()
        dataset_root_path = resolve_dataset_root(dataset_root)
        if dataset_root_path is None:
            return (), (
                _rejection(
                    sample=sample,
                    reason=AugmentationRejectionReason.DATASET_ROOT_MISSING,
                ),
            )
        if sample.audio is None or sample.audio.path is None:
            return (), ()

        unsafe_fields = non_transformable_annotations(
            sample=sample,
            media_kind="audio",
        )
        if unsafe_fields:
            return (), (
                _rejection(
                    sample=sample,
                    reason=AugmentationRejectionReason.MEDIA_ANNOTATIONS_NOT_TRANSFORMABLE,
                    message=rejection_message(fields=unsafe_fields),
                ),
            )

        try:
            source_path = resolve_source_path(
                dataset_root=dataset_root_path,
                value=sample.audio.path,
                error_message="relative audio source path escapes dataset root",
            )
        except ValueError as exc:
            return (), (
                _rejection(
                    sample=sample,
                    reason=AugmentationRejectionReason.AUDIO_SOURCE_PATH_INVALID,
                    message=str(exc),
                ),
            )

        validation = self._validate_input(
            path=source_path,
            declared_mime_type=sample.audio.mime_type,
            declared_byte_size=sample.audio.byte_size,
        )
        if not validation.accepted:
            return (), (
                _rejection(
                    sample=sample,
                    reason=AugmentationRejectionReason(
                        validation.rejection_reason or "invalid_audio"
                    ),
                    message=str(validation.signals),
                ),
            )

        try:
            source_sha256 = file_sha256(path=source_path)
        except OSError as exc:
            return (), (
                _rejection(
                    sample=sample,
                    reason=AugmentationRejectionReason.AUDIO_SOURCE_HASH_FAILED,
                    message=str(exc),
                ),
            )

        variant_id = media_variant_id(
            source_sample_id=sample.sample_id,
            operation=_OPERATION,
            source_sha256=source_sha256,
            config_hash=self._settings_fingerprint,
        )
        output_directory = dataset_root_path / self._settings.output_directory
        output_path = output_directory / f"{variant_id}.wav"
        cache_key = self._cache.cache_key(
            source_path=source_path,
            operation=_OPERATION,
            settings_digest=self._settings_fingerprint,
        )

        try:
            decoded_audio = self._decoder.decode(
                path=source_path,
                chunk_frames=self._settings.chunk_frames,
            )
            if decoded_audio.duration_sec > self._max_duration_seconds:
                raise ValueError("audio_duration_too_long")
            parameters = build_audio_transform_parameters(
                settings=self._settings,
                operations=self._operations,
                variant_id=variant_id,
            )
            prepared = prepare_audio_transform(
                decoded_audio=decoded_audio,
                parameters=parameters,
            )
            if (
                prepared.receipt.input_duration_seconds
                > self._max_duration_seconds
            ):
                raise ValueError("audio_duration_too_long")
            if (
                prepared.receipt.output_duration_seconds
                > self._max_duration_seconds
            ):
                raise ValueError("augmented_audio_duration_too_long")
            if len(prepared.pcm_bytes) + 44 > self._settings.output_max_bytes:
                raise ValueError(
                    "generated_audio_would_exceed_output_max_bytes"
                )
            transformed_sample = apply_audio_timed_sample(
                sample=sample,
                receipt=prepared.receipt,
            )

            output_directory.mkdir(parents=True, exist_ok=True)
            restored = self._cache.restore(
                dataset_root=dataset_root_path,
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
                write_prepared_wav(prepared=prepared, output_path=output_path)
                self._cache.store(
                    dataset_root=dataset_root_path,
                    cache_key=cache_key,
                    output_path=output_path,
                    cache_metadata={
                        "operation": _OPERATION,
                        "source_path": source_path.as_posix(),
                        "source_sha256": source_sha256,
                        "config_hash": self._settings_fingerprint,
                        "implementation_hash": MEDIA_AUGMENTATION_IMPLEMENTATION_HASH,
                        "audio_transform": prepared.receipt.to_dict(),
                    },
                )
            output_validation = self._validate_output(
                path=output_path,
                expected_sample_rate=prepared.receipt.output_sample_rate,
                expected_channels=prepared.receipt.output_channels,
                expected_duration_seconds=prepared.receipt.output_duration_seconds,
                output_max_bytes=self._settings.output_max_bytes,
                max_clipping_fraction=self._settings.max_clipping_fraction,
                duration_tolerance_seconds=self._settings.duration_tolerance_seconds,
            )
            if not output_validation.accepted:
                remove_incomplete_file(
                    output_path,
                    logger=_LOGGER,
                    event_name="audio_augmentation_cleanup_failed",
                )
                return (), (
                    _rejection(
                        sample=sample,
                        reason=AugmentationRejectionReason(
                            output_validation.rejection_reason
                            or "generated_audio_invalid"
                        ),
                        message=str(output_validation.signals),
                    ),
                )
            output_byte_size = output_path.stat().st_size
            output_sha256 = file_sha256(path=output_path)
            metadata = build_audio_variant_metadata(
                sample=transformed_sample,
                dataset_root=dataset_root_path,
                source_path=source_path,
                output_path=output_path,
                output_byte_size=output_byte_size,
                cache_key=cache_key,
                source_sha256=source_sha256,
                output_sha256=output_sha256,
                config_hash=self._settings_fingerprint,
                variant_id=variant_id,
                receipt=prepared.receipt,
                operations=self._operations,
                validation_signals=output_validation.signals,
            )
        except (OSError, ValueError) as exc:
            remove_incomplete_file(
                output_path,
                logger=_LOGGER,
                event_name="audio_augmentation_cleanup_failed",
            )
            return (), (
                _rejection(
                    sample=sample,
                    reason=AugmentationRejectionReason.AUDIO_TRANSFORM_FAILED,
                    message=str(exc),
                ),
            )

        original_audio_metadata = _preserved_metadata(
            transformed_sample.audio.metadata
            if transformed_sample.audio is not None
            else {},
            self._settings.metadata_policy,
        )
        audio_metadata = {
            **original_audio_metadata,
            "source_path": str(source_path),
            "source_sha256": source_sha256,
            "output_sha256": output_sha256,
            "sample_rate": prepared.receipt.output_sample_rate,
            "channels": prepared.receipt.output_channels,
            "duration_seconds": prepared.receipt.output_duration_seconds,
            "frame_count": prepared.receipt.output_frame_count,
            "sample_width": 2,
            "clipping_fraction": prepared.receipt.clipping_fraction,
        }
        variant = replace(
            transformed_sample,
            sample_id=variant_id,
            audio=ModalityObject(
                path=output_path,
                mime_type="audio/wav",
                byte_size=output_byte_size,
                metadata=audio_metadata,
            ),
            audio_tensor_path=None,
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
        modality="audio",
    )


_AUDIO_SAFE_METADATA_FIELDS = frozenset(
    {"language", "channels", "sample_rate", "duration_seconds", "codec"}
)


def _preserved_metadata(
    metadata: dict[str, object], policy: str
) -> dict[str, object]:
    return preserved_metadata(metadata, policy, _AUDIO_SAFE_METADATA_FIELDS)
