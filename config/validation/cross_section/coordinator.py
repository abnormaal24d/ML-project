"""Coordinate domain-oriented cross-field settings validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from config.validation.cross_section.basic import (
    _validate_coverage,
    _validate_dataset_splits,
)
from config.validation.cross_section.collection import _validate_media_limits
from config.validation.cross_section.composition import validate_composition_config
from config.validation.cross_section.multimodal import (
    _validate_generation_loss_backends,
    _validate_multimodal_training_configuration_shape,
)
from config.validation.cross_section.preprocessing import (
    _validate_preprocessing_configuration,
    _validate_video_transcription_flags,
)
from config.validation.cross_section.release import (
    _validate_production_configuration_guarantees,
    _validate_release_stage,
)

if TYPE_CHECKING:
    from config.settings.root import Settings


def validate_structural_settings(settings: Settings) -> None:
    """Run config-owned cross-field checks without domain registries."""

    _validate_dataset_splits(settings)
    _validate_coverage(settings)
    _validate_media_limits(settings)
    _validate_multimodal_training_configuration_shape(settings)
    _validate_generation_loss_backends(settings)
    _validate_preprocessing_configuration(settings)
    _validate_video_transcription_flags(settings)
    _validate_release_stage(settings)
    _validate_production_configuration_guarantees(settings)
    validate_composition_config(settings)
