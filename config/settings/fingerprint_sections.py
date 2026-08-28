"""Canonical Settings payloads shared by datachecker and manifests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from config.collection.training_input_gate import DataCheckerSettings
from config.settings.root import Settings

SettingsPayload = Mapping[str, object]


@dataclass(frozen=True, slots=True)
class SettingsPayloads:
    """Group settings payloads passed to workflow runtime services."""

    crawl: SettingsPayload
    preprocessing: SettingsPayload
    normalization: SettingsPayload
    deduplication: SettingsPayload
    splitting: SettingsPayload
    validation: SettingsPayload
    augmentation: SettingsPayload
    augmentation_strategy: SettingsPayload
    training: SettingsPayload
    model: SettingsPayload


def build_settings_payloads(
    *,
    settings: Settings,
    checker_settings: DataCheckerSettings,
) -> SettingsPayloads:
    """Build settings payloads used for workflow fingerprinting.

    Delegates to section-specific builders in the same file.
    """
    crawl = _build_crawl_payload(settings)
    pre = _build_preprocessing_payload(settings)
    ds = _build_dataset_payload(settings, checker_settings)
    aug = _build_augmentation_payload(settings)
    tr = _build_training_payload(settings, checker_settings)
    model = _build_model_payload(settings)

    return SettingsPayloads(
        crawl=crawl,
        preprocessing=pre,
        normalization=pre,
        deduplication={
            "raw_dataset": settings.datasets.raw.model_dump(mode="json"),
            "near_deduper": settings.datasets.curation.near_deduper.model_dump(
                mode="json"
            ),
        },
        splitting=settings.datasets.splits.model_dump(mode="json"),
        validation=settings.datasets.training.dataset_validator.model_dump(
            mode="json"
        ),
        augmentation=_mapping_payload(aug.get("augmentation")),
        augmentation_strategy=_mapping_payload(
            aug.get("augmentation_strategy")
        ),
        training={**ds, **tr},
        model=model,
    )


def _build_crawl_payload(settings: Settings) -> dict[str, object]:
    processors = settings.collection.processors

    payload: dict[str, object] = {
        "crawler": settings.crawler.model_dump(mode="json"),
        "classification": settings.classification.model_dump(mode="json"),
        "coverage": settings.coverage.model_dump(mode="json"),
        "fetcher": settings.collection.fetcher.model_dump(mode="json"),
        "response_body_reader": settings.collection.response_body_reader.model_dump(
            mode="json"
        ),
        "scheduling": settings.collection.scheduling.model_dump(mode="json"),
        "url_filter": settings.collection.url_filter.model_dump(mode="json"),
        "robots": settings.collection.robots.model_dump(mode="json"),
        "url_normalizer": settings.collection.url_normalizer.model_dump(
            mode="json"
        ),
        "url_extractor": settings.collection.url_extractor.model_dump(
            mode="json"
        ),
        "link_extractor": settings.collection.link_extractor.model_dump(
            mode="json"
        ),
        "asset_extractor": settings.collection.asset_extractor.model_dump(
            mode="json"
        ),
        "host_extractor": settings.collection.host_extractor.model_dump(
            mode="json"
        ),
        "processors": processors.model_dump(mode="json"),
    }

    if (
        processors.audio.run_transcription
        or processors.video.run_transcription
    ):
        payload["transcription_recipe"] = (
            settings.preprocessing.transcription.model_dump(mode="json")
        )

    return payload


def _build_preprocessing_payload(settings: Settings) -> dict[str, object]:
    return settings.preprocessing.model_dump(mode="json")


def _build_dataset_payload(
    settings: Settings, checker_settings: DataCheckerSettings
) -> dict[str, object]:
    return {
        "dataset_paths": settings.datasets.paths.model_dump(mode="json"),
        "training": {
            "snapshot_builder": settings.datasets.training.snapshot_builder.model_dump(
                mode="json"
            ),
        },
        "curation": settings.datasets.curation.model_dump(mode="json"),
        "training_input_mode": checker_settings.training_input_mode.value,
        "datasets_raw": settings.datasets.raw.model_dump(mode="json"),
    }


def _build_augmentation_payload(settings: Settings) -> dict[str, object]:
    return {
        "augmentation": settings.augmentation.model_dump(mode="json"),
        "augmentation_strategy": {
            "workflow": "multimodal_augmenter",
            "sample_augmenter": "multimodal_sample_augmenter",
            "scope": "train",
            "val_test_rules": "copy_without_augmentation",
            "variant_id_strategy": "source_name_config_hash",
            "media_transform_rules": "enabled_by_preset_and_operation_registry",
            "operation_registry_validated": True,
            "cache_rules": "content_hash_operation_settings_reuse",
            "augmenters": (
                "title_text_augmenter",
                "context_prefix_augmenter",
                "text_span_focus_augmenter",
            ),
            "media_augmenters": {
                "document": "document_augmenter",
                "image": "validated_image_augmenter",
                "audio": "streaming_audio_augmenter",
                "video": ("video_clip_augmenter+video_keyframe_augmenter"),
            },
            "video_toolchain": {
                "ffmpeg_expected_version": (
                    settings.media_toolchain.ffmpeg_expected_version
                ),
                "ffprobe_expected_version": (
                    settings.media_toolchain.ffprobe_expected_version
                ),
            },
        },
    }


def _build_training_payload(
    settings: Settings, checker_settings: DataCheckerSettings
) -> dict[str, object]:
    return {
        "checkpoint_root": settings.datasets.paths.training_checkpoint_directory,
        "checkpoint_filename": settings.datasets.paths.training_checkpoint_filename,
        "metrics_filename": settings.datasets.paths.training_metrics_filename,
        "training_input_mode": checker_settings.training_input_mode.value,
        "dataset_validator": settings.datasets.training.dataset_validator.model_dump(
            mode="json"
        ),
        "acceptance_evaluation_schema": {
            "version": 3,
            "statuses": ("FAILED", "PIPELINE_ACCEPTED", "MODEL_ACCEPTED"),
            "model_quality_failures_are_sanity_only": True,
            "loss_ratio_scope": "model_acceptance",
            "unlabeled_evaluation_status": "PIPELINE_ACCEPTED",
        },
        "snapshot_builder": settings.datasets.training.snapshot_builder.model_dump(
            mode="json"
        ),
        "runtime": settings.training.model_dump(mode="json"),
    }


def _build_model_payload(settings: Settings) -> dict[str, object]:
    return settings.multimodal.model_dump(mode="json")


def _mapping_payload(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}
