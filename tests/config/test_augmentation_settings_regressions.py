"""Regression tests for centralized augmentation settings validation."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from config.augmentation import augmentation_settings, common
from config.augmentation.augmentation_settings import AugmentationSettings
from config.load import load_settings
from config.validation.cross_section.coordinator import validate_settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_default_augmentation_settings_can_be_constructed() -> None:
    settings = AugmentationSettings()

    assert settings.enabled is True
    assert settings.text.enabled is True

    assert settings.document.enabled is True
    assert settings.document.mode == "text_field_only"

    assert settings.image.enabled is False
    assert settings.audio.enabled is False
    assert settings.video.enabled is False

    assert settings.cache_directory == "data/interim/augmentation_cache"


def test_removed_preset_field_is_rejected() -> None:
    with pytest.raises(ValidationError, match="preset"):
        AugmentationSettings.model_validate(
            {"preset": "removed_value"},
        )


def test_generated_schema_has_no_preset_contract() -> None:
    schema = json.loads(
        (PROJECT_ROOT / "docs/configuration_schema.json").read_text(
            encoding="utf-8",
        )
    )

    properties = schema["$defs"]["AugmentationSettings"]["properties"]
    assert "preset" not in properties


def test_production_augmentation_policy_is_explicit(
    production_whisper_env: None,
) -> None:
    payload = tomllib.loads(
        (PROJECT_ROOT / "config/profiles/prod.toml").read_text(
            encoding="utf-8",
        )
    )
    augmentation = payload["augmentation"]

    assert augmentation["enabled"] is True
    assert augmentation["text"] == {
        "enabled": True,
        "title_prefix_enabled": True,
        "context_prefix_enabled": True,
        "text_span_focus_enabled": True,
        "max_variants_per_sample": 2,
    }
    assert augmentation["document"] == {
        "enabled": True,
        "mode": "text_field_only",
    }
    assert augmentation["image"] == {"enabled": False}
    assert augmentation["audio"] == {"enabled": False}
    assert augmentation["video"] == {"enabled": False}

    settings = load_settings(
        "prod",
        project_root=PROJECT_ROOT,
        config_root=PROJECT_ROOT,
        environment="prod",
    )
    resolved = settings.augmentation
    assert resolved.enabled is True
    assert resolved.text.enabled is True
    assert resolved.text.title_prefix_enabled is True
    assert resolved.text.context_prefix_enabled is True
    assert resolved.text.text_span_focus_enabled is True
    assert resolved.text.max_variants_per_sample == 2
    assert resolved.document.enabled is True
    assert resolved.document.mode == "text_field_only"
    assert resolved.image.enabled is False
    assert resolved.audio.enabled is False
    assert resolved.video.enabled is False


@pytest.mark.parametrize(
    "modality",
    ("document", "image", "audio", "video"),
)
def test_production_rejects_media_augmentation(
    production_whisper_env: None,
    modality: str,
) -> None:
    settings = load_settings(
        "prod",
        project_root=PROJECT_ROOT,
        config_root=PROJECT_ROOT,
        environment="prod",
    )
    section = getattr(settings.augmentation, modality)
    update = (
        {"mode": "document_media"}
        if modality == "document"
        else {"enabled": True}
    )
    invalid = settings.model_copy(
        update={
            "augmentation": settings.augmentation.model_copy(
                update={modality: section.model_copy(update=update)}
            )
        }
    )
    with pytest.raises(
        ValueError,
        match=rf"media augmentation is enabled for: .*{modality}",
    ):
        validate_settings(invalid)


def test_production_allows_inert_media_when_root_disabled(
    production_whisper_env: None,
) -> None:
    settings = load_settings(
        "prod",
        project_root=PROJECT_ROOT,
        config_root=PROJECT_ROOT,
        environment="prod",
    )
    inert = settings.model_copy(
        update={
            "augmentation": settings.augmentation.model_copy(
                update={
                    "enabled": False,
                    "image": settings.augmentation.image.model_copy(
                        update={"enabled": True}
                    ),
                }
            )
        }
    )

    validate_settings(inert)

    assert inert.augmentation.enabled is False
    assert inert.augmentation.image.enabled is True


def test_augmentation_settings_use_centralized_validators() -> None:
    assert (
        augmentation_settings.normalize_modalities
        is common.normalize_modalities
    )
    assert (
        augmentation_settings.validate_output_directory
        is common.validate_output_directory
    )


@pytest.mark.parametrize(
    "path",
    (
        "",
        "   ",
        ".",
        "../escape",
        "../../outside-project",
        "/absolute/path",
        r"C:\sensitive-location",
        r"C:sensitive-location",
        r"\\server\share\output",
    ),
)
def test_output_directory_rejects_unsafe_paths(path: str) -> None:
    with pytest.raises(ValueError, match="non-empty relative path"):
        common.validate_output_directory(path)


@pytest.mark.parametrize(
    "path",
    (
        "data/interim/augmentation_cache",
        "artifacts/augmentation",
        r"data\interim\augmentation_cache",
        ".cache/augmentation",
    ),
)
def test_output_directory_accepts_safe_relative_paths(path: str) -> None:
    common.validate_output_directory(path)
