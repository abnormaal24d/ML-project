"""Build deterministic audio transform parameters and timed sample updates."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import TYPE_CHECKING

from augmentation.audio.audio_operations import AudioTransformParameters
from augmentation.audio.audio_stream_transformer import AudioTransformReceipt
from augmentation.audio.audio_timestamps import (
    AudioTimelineTransform,
    transform_audio_annotation_mapping,
    transform_speaker_segments,
)
from mmcrawler_datasets.schema import MultimodalSample

if TYPE_CHECKING:
    from config.augmentation.audio_settings import AudioAugmentationSettings


def build_audio_transform_parameters(
    *,
    settings: AudioAugmentationSettings,
    operations: tuple[str, ...],
    variant_id: str,
) -> AudioTransformParameters:
    seed = int(hashlib.sha256(variant_id.encode("utf-8")).hexdigest()[:16], 16)
    return AudioTransformParameters(
        operations=operations,
        target_sample_rate=settings.target_sample_rate,
        target_channels=settings.target_channels,
        gain_db=settings.gain_db,
        noise_std_fraction=settings.noise_std_fraction,
        trim_silence_threshold_dbfs=settings.trim_silence_threshold_dbfs,
        trim_padding_seconds=settings.trim_padding_seconds,
        speed_factor=settings.speed_factor,
        noise_seed=seed,
    )


def apply_audio_timed_sample(
    *,
    sample: MultimodalSample,
    receipt: AudioTransformReceipt,
) -> MultimodalSample:
    timeline = AudioTimelineTransform(
        trim_start_seconds=receipt.trim_start_seconds,
        trim_end_seconds=receipt.trim_end_seconds,
        speed_factor=receipt.speed_factor,
        output_duration_seconds=receipt.output_duration_seconds,
    )
    transformed_metadata = transform_audio_annotation_mapping(
        value=sample.metadata or {},
        timeline=timeline,
    )
    transformed_task_target = transform_audio_annotation_mapping(
        value=sample.task_target or {},
        timeline=timeline,
    )
    transformed_audio = sample.audio
    if transformed_audio is not None:
        transformed_audio = replace(
            transformed_audio,
            metadata=transform_audio_annotation_mapping(
                value=transformed_audio.metadata or {},
                timeline=timeline,
            ),
        )
    return replace(
        sample,
        audio=transformed_audio,
        speaker_segments=transform_speaker_segments(
            segments=sample.speaker_segments,
            timeline=timeline,
        ),
        task_target=transformed_task_target,
        metadata=transformed_metadata,
    )
