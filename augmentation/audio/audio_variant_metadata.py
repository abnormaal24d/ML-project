"""Build lineage-bound metadata for accepted audio variants."""

from __future__ import annotations

from pathlib import Path

from augmentation.audio.audio_stream_transformer import AudioTransformReceipt
from augmentation.variant_lineage import (
    MEDIA_AUGMENTATION_IMPLEMENTATION_HASH,
    MEDIA_AUGMENTATION_IMPLEMENTATION_VERSION,
)
from mmcrawler_datasets.schema import MultimodalSample

_OPERATION = "audio_media_transform"


def build_audio_variant_metadata(
    *,
    sample: MultimodalSample,
    dataset_root: Path,
    source_path: Path,
    output_path: Path,
    output_byte_size: int,
    cache_key: str,
    source_sha256: str,
    output_sha256: str,
    config_hash: str,
    variant_id: str,
    receipt: AudioTransformReceipt,
    operations: tuple[str, ...],
    validation_signals: dict[str, object],
) -> dict[str, object]:
    from augmentation.variant_metadata import media_variant_metadata
    from mmcrawler_datasets.training_samples.artifact_path import (
        relative_dataset_path,
    )

    metadata = media_variant_metadata(
        sample=sample,
        variant_id=variant_id,
        augmentation_name=_OPERATION,
        augmentation_type="media_transform",
        modifies=("audio", "speaker_segments", "transcript_segments"),
    )
    metadata.update(
        {
            "augmentation_cache_key": cache_key,
            "augmentation_source_path": relative_dataset_path(
                dataset_root=dataset_root,
                output_path=source_path,
            ),
            "augmentation_output_path": relative_dataset_path(
                dataset_root=dataset_root,
                output_path=output_path,
            ),
            "augmentation_output_mime_type": "audio/wav",
            "augmentation_output_byte_size": output_byte_size,
            "augmentation_source_sha256": source_sha256,
            "augmentation_output_sha256": output_sha256,
            "augmentation_config_hash": config_hash,
            "augmentation_implementation_version": (
                MEDIA_AUGMENTATION_IMPLEMENTATION_VERSION
            ),
            "augmentation_implementation_hash": MEDIA_AUGMENTATION_IMPLEMENTATION_HASH,
            "augmentation_audio_operations": list(operations),
            "augmentation_audio_transform": receipt.to_dict(),
            "augmentation_audio_validation": dict(validation_signals),
            "audio_duration_seconds": receipt.output_duration_seconds,
            "audio_sample_rate": receipt.output_sample_rate,
            "audio_channels": receipt.output_channels,
        }
    )
    return metadata
